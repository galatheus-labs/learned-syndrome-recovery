"""Surface Attention Decoder: compute scales with surface area, not volume.

Instead of processing the full 32³ grid uniformly, this decoder:
1. Identifies surface voxels from the DCT prior
2. Extracts local patches around each surface voxel
3. Each patch cross-attends to its syndrome bits and DCT context
4. Refined patches are scattered back to the full grid

Compute is O(surface_area) ≈ O(d²) for a d³ grid, not O(d³).
For a 2% fill rate object: ~2000 surface voxels → ~250 patches
instead of 32³ = 32768 voxels.

Each surface patch has a corresponding syndrome bit from the
overlapping 2³ constraint — the attention token and the constraint
are the same spatial object.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def sinusoidal_3d_encoding(coords, D, n_freqs=8):
    """NeRF-style sinusoidal positional encoding for 3D coordinates.

    Args:
        coords: (n_patches, 3) integer coordinates
        D: grid dimension (for normalization)
        n_freqs: number of frequency bands

    Returns:
        encoding: (n_patches, 3 * 2 * n_freqs) float tensor
    """
    # Normalize to [0, 1]
    normalized = coords.float() / D  # (n_patches, 3)
    freqs = 2.0 ** torch.arange(n_freqs, device=coords.device).float()  # (n_freqs,)
    # (n_patches, 3, 1) * (1, 1, n_freqs) → (n_patches, 3, n_freqs)
    angles = normalized.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0) * math.pi
    encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (n_patches, 3, 2*n_freqs)
    return encoding.reshape(coords.size(0), -1)  # (n_patches, 3 * 2 * n_freqs)


class RelativePositionBias(nn.Module):
    """Learned relative position bias for 3D spatial attention.

    Discretizes the relative displacement between patches into bins
    and learns a bias per bin per attention head. Two patches 3 voxels
    apart get the same bias regardless of absolute position.
    """

    def __init__(self, n_heads=4, max_dist=8):
        super().__init__()
        self.n_heads = n_heads
        self.max_dist = max_dist
        # Separate per-axis bias embeddings, clamped to local neighborhood
        n_bins = 2 * max_dist + 1
        self.bias_x = nn.Embedding(n_bins, n_heads)
        self.bias_y = nn.Embedding(n_bins, n_heads)
        self.bias_z = nn.Embedding(n_bins, n_heads)

    def forward(self, coords):
        """Compute pairwise relative position bias.

        Uses chunked computation to avoid allocating the full n×n×n_heads
        tensor at once. For n=4000 patches this saves ~750MB.

        Args:
            coords: (n_patches, 3) integer coordinates

        Returns:
            bias: (n_heads, n_patches, n_patches) attention bias
        """
        n = coords.size(0)
        device = coords.device
        chunk_size = 512  # process 512 rows at a time

        bias = torch.zeros(self.n_heads, n, n, device=device)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            for dim, embed in enumerate([self.bias_x, self.bias_y, self.bias_z]):
                d = coords[start:end, dim].unsqueeze(1) - coords[:, dim].unsqueeze(0)
                d = (d.clamp(-self.max_dist, self.max_dist) + self.max_dist).long()
                bias[:, start:end, :] += embed(d).permute(2, 0, 1)

        return bias


class PatchExtractor(nn.Module):
    """Extract and embed local patches around surface voxels."""

    def __init__(self, patch_size=2, embed_dim=64, pos_encoding='normalized',
                 n_pos_freqs=8):
        super().__init__()
        self.patch_size = patch_size
        self.n_voxels = patch_size ** 3  # 8 for 2³
        self.pos_encoding = pos_encoding
        self.n_pos_freqs = n_pos_freqs
        # Position dimension depends on encoding type
        if pos_encoding == 'sinusoidal':
            pos_dim = 3 * 2 * n_pos_freqs  # 48 for n_freqs=8
        else:
            pos_dim = 3  # normalized xyz
        # Project patch voxels + syndrome bit + position → embedding
        self.proj = nn.Linear(self.n_voxels + 1 + pos_dim, embed_dim)

    def forward(self, dct_3d, syndrome_bits, patch_coords, grid_size):
        """Extract patches and embed them.

        Args:
            dct_3d: (B, D, D, D) DCT reconstruction
            syndrome_bits: (B, n_patches) syndrome bit per patch
            patch_coords: (n_patches, 3) integer coords of patch corners
            grid_size: D

        Returns:
            tokens: (B, n_patches, embed_dim)
        """
        B = dct_3d.size(0)
        D = grid_size
        s = self.patch_size
        n_patches = patch_coords.size(0)
        device = dct_3d.device

        # Extract patch values from DCT field via gather
        dct_flat = dct_3d.view(B, -1)  # (B, N)
        # Compute flat indices for each patch's voxels
        flat_indices = SurfaceAttentionDecoder._patch_flat_indices(
            patch_coords, D, s, device)  # (n_patches, n_voxels)
        valid = flat_indices >= 0
        safe_idx = flat_indices.clamp(min=0)  # clamp for gather, mask invalid later
        # Gather: (B, n_patches * n_voxels)
        gather_idx = safe_idx.reshape(1, -1).expand(B, -1)
        gathered = torch.gather(dct_flat, 1, gather_idx).reshape(B, n_patches, self.n_voxels)
        # Zero out invalid positions
        gathered[:, ~valid] = 0.0
        patch_vals = gathered

        # Position encoding
        if self.pos_encoding == 'sinusoidal':
            pos = sinusoidal_3d_encoding(patch_coords, D, self.n_pos_freqs)  # (n_patches, 3*2*n_freqs)
        else:
            pos = patch_coords.float().to(device) / D  # (n_patches, 3)
        pos = pos.unsqueeze(0).expand(B, -1, -1)  # (B, n_patches, pos_dim)

        # Syndrome bit per patch
        syn = syndrome_bits.unsqueeze(-1)  # (B, n_patches, 1)

        # Concatenate and project
        features = torch.cat([patch_vals, syn, pos], dim=-1)  # (B, n_patches, n_voxels+4)
        return self.proj(features)  # (B, n_patches, embed_dim)


class SurfaceAttentionBlock(nn.Module):
    """Self-attention over surface patches + cross-attention to global context."""

    def __init__(self, embed_dim=64, n_heads=4, ff_dim=128):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        # Manual self-attention to support attn_bias
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, tokens, context, attn_bias=None):
        """
        Args:
            tokens: (B, n_patches, embed_dim)
            context: (B, n_context, embed_dim)
            attn_bias: (n_heads, n_patches, n_patches) optional relative position bias
        """
        B, T, E = tokens.shape
        h = self.norm1(tokens)

        # Self-attention with optional relative position bias
        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if attn_bias is not None:
            # F.scaled_dot_product_attention supports attn_mask as additive bias
            # Expand bias for batch: (1, n_heads, T, T)
            bias = attn_bias.unsqueeze(0).expand(B, -1, -1, -1)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        else:
            out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, T, E)
        h = tokens + self.o_proj(out)

        # Cross-attention to DCT context
        h2 = self.norm2(h)
        h = h + self.cross_attn(h2, context, context, need_weights=False)[0]

        # Feed-forward
        h = h + self.ff(self.norm3(h))
        return h


class SurfaceAttentionDecoder(nn.Module):
    """Decode voxels by attending to surface patches.

    Architecture:
      1. DCT → global context tokens (downsampled DCT as context)
      2. Surface patches extracted and embedded with syndrome bits
      3. N attention blocks: self-attn over patches + cross-attn to DCT
      4. Each patch predicts its 8 voxel occupancies
      5. Predictions scattered back to full grid, averaged where overlapping

    Params: ~500K-2M depending on embed_dim and depth.
    """

    def __init__(self, cube_size=32, patch_size=2, embed_dim=64,
                 n_heads=4, depth=4, ff_dim=128, context_pool=4,
                 pos_encoding='normalized', use_rel_bias=False):
        super().__init__()
        self.cube_size = cube_size
        self.patch_size = patch_size
        self.n_voxels = patch_size ** 3
        self.use_rel_bias = use_rel_bias

        # Patch extraction and embedding
        self.patch_embed = PatchExtractor(patch_size, embed_dim,
                                          pos_encoding=pos_encoding)

        # Relative position bias (optional)
        self.rel_bias = RelativePositionBias(n_heads, max_dist=cube_size) if use_rel_bias else None

        # Global context: downsample DCT to context_pool³ tokens
        self.context_pool_size = context_pool
        n_context = context_pool ** 3
        self.context_proj = nn.Linear(1, embed_dim)  # per-cell DCT value → embedding
        self.context_pos = nn.Parameter(
            torch.randn(1, n_context, embed_dim) * 0.02)

        # Attention blocks
        self.blocks = nn.ModuleList([
            SurfaceAttentionBlock(embed_dim, n_heads, ff_dim)
            for _ in range(depth)
        ])

        # Output head: predict 8 voxel logits per patch
        self.head = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, self.n_voxels),
        )

    def _make_context(self, dct_3d):
        """Downsample DCT field to context tokens."""
        B = dct_3d.size(0)
        cp = self.context_pool_size
        # Average pool to cp³
        pooled = F.adaptive_avg_pool3d(
            dct_3d.unsqueeze(1), cp).view(B, -1, 1)  # (B, cp³, 1)
        context = self.context_proj(pooled) + self.context_pos  # (B, cp³, embed_dim)
        return context

    def forward(self, dct_3d, syndrome_bits, patch_coords):
        """
        Args:
            dct_3d: (B, D, D, D) DCT reconstruction (soft prior)
            syndrome_bits: (B, n_patches) float — syndrome bit per patch
            patch_coords: (n_patches, 3) int — corner coords of each patch

        Returns:
            recon: (B, N) float logits for each voxel
        """
        B = dct_3d.size(0)
        D = self.cube_size
        N = D ** 3
        s = self.patch_size
        n_patches = patch_coords.size(0)
        device = dct_3d.device

        # Build context from DCT
        context = self._make_context(dct_3d)

        # Extract and embed surface patches
        tokens = self.patch_embed(dct_3d, syndrome_bits, patch_coords, D)

        # Compute relative position bias if enabled
        attn_bias = self.rel_bias(patch_coords) if self.rel_bias is not None else None

        # Attention blocks
        for block in self.blocks:
            tokens = block(tokens, context, attn_bias=attn_bias)

        # Predict per-patch voxel logits
        patch_logits = self.head(tokens)  # (B, n_patches, 8)

        # Scatter back to full grid using precomputed index map
        # Build flat index mapping: (n_patches, n_voxels) → flat grid indices
        flat_indices = self._patch_flat_indices(patch_coords, D, s, device)
        # flat_indices: (n_patches, n_voxels) long tensor, -1 for out of bounds

        valid = flat_indices >= 0  # (n_patches, n_voxels)
        # Expand for batch: scatter_add over (B, N)
        logit_sum = torch.zeros(B, N, device=device)
        count = torch.zeros(1, N, device=device)

        # Flatten patches: (B, n_patches * n_voxels)
        flat_logits = patch_logits.reshape(B, -1)  # (B, n_patches * n_voxels)
        flat_idx = flat_indices.reshape(-1)  # (n_patches * n_voxels,)
        flat_valid = valid.reshape(-1)  # (n_patches * n_voxels,)

        # Only scatter valid indices
        valid_idx = flat_idx[flat_valid]  # (n_valid,)
        valid_idx_batch = valid_idx.unsqueeze(0).expand(B, -1)  # (B, n_valid)
        valid_logits = flat_logits[:, flat_valid]  # (B, n_valid)

        logit_sum.scatter_add_(1, valid_idx_batch, valid_logits)
        count.scatter_add_(1, valid_idx.unsqueeze(0),
                           torch.ones(1, valid_idx.size(0), device=device))

        # Average where multiple patches overlap; use DCT prior for uncovered
        covered = count.squeeze(0) > 0  # (N,)
        recon = torch.zeros(B, N, device=device)
        recon[:, covered] = logit_sum[:, covered] / count[:, covered]
        dct_flat = dct_3d.view(B, -1)
        dct_logits = torch.log(dct_flat.clamp(1e-6) / (1 - dct_flat).clamp(1e-6))
        recon[:, ~covered] = dct_logits[:, ~covered]

        return recon

    @staticmethod
    def _patch_flat_indices(patch_coords, D, s, device):
        """Precompute flat grid indices for each voxel in each patch. Fully vectorized."""
        n_patches = patch_coords.size(0)
        n_voxels = s ** 3

        # Local offsets within a patch: (n_voxels, 3)
        dz, dy, dx = torch.meshgrid(
            torch.arange(s, device=device),
            torch.arange(s, device=device),
            torch.arange(s, device=device),
            indexing='ij')
        offsets = torch.stack([dx.reshape(-1), dy.reshape(-1), dz.reshape(-1)], dim=1)

        # All patch corners: (n_patches, 1, 3) + (1, n_voxels, 3) → (n_patches, n_voxels, 3)
        corners = patch_coords.to(device).unsqueeze(1)  # (n_patches, 1, 3)
        all_coords = corners + offsets.unsqueeze(0)      # (n_patches, n_voxels, 3)

        # Check bounds
        valid = (all_coords[:, :, 0] < D) & (all_coords[:, :, 1] < D) & (all_coords[:, :, 2] < D)

        # Flat index: x + y*D + z*D*D
        flat = all_coords[:, :, 0] + all_coords[:, :, 1] * D + all_coords[:, :, 2] * D * D
        flat[~valid] = -1

        return flat


def get_surface_patches(dct_3d, cube_size, threshold=0.3, patch_size=2):
    """Detect surface and return patch coordinates + constraint matrix.

    Returns:
        patch_coords: (n_patches, 3) int tensor of patch corner coordinates
        H: (n_patches, N) uint8 constraint matrix (one check per patch)
    """
    from surface_constraints import detect_surface

    D = cube_size
    N = D ** 3
    s = patch_size

    surface = detect_surface(dct_3d, threshold=threshold)
    surface_coords = torch.nonzero(surface, as_tuple=False)

    seen = set()
    patch_list = []
    H_rows = []

    for coord in surface_coords:
        z, y, x = coord[0].item(), coord[1].item(), coord[2].item()
        # Patch corner (can go negative, clamp)
        x0 = max(0, x - s//2)
        y0 = max(0, y - s//2)
        z0 = max(0, z - s//2)
        key = (x0, y0, z0)
        if key in seen:
            continue
        seen.add(key)

        x1 = min(D, x0 + s)
        y1 = min(D, y0 + s)
        z1 = min(D, z0 + s)

        patch_list.append([x0, y0, z0])

        # Build constraint row
        row = torch.zeros(N, dtype=torch.uint8)
        for dz in range(z1 - z0):
            for dy in range(y1 - y0):
                for dx in range(x1 - x0):
                    row[(x0+dx) + (y0+dy)*D + (z0+dz)*D*D] = 1
        H_rows.append(row)

    if not patch_list:
        return torch.zeros(0, 3, dtype=torch.long), torch.zeros(0, N, dtype=torch.uint8)

    patch_coords = torch.tensor(patch_list, dtype=torch.long)
    H = torch.stack(H_rows)
    return patch_coords, H
