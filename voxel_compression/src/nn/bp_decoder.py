"""Neural Belief Propagation decoder on the Tanner graph.

Each voxel is a variable node, each parity check is a factor node.
Messages pass between them iteratively. Unlike standard BP, the
message update functions are learned (small MLPs).

Additionally, voxel nodes are spatially connected to their 6 neighbors,
enabling the network to learn spatial coherence.

Architecture:
  - Variable nodes: 32^3 voxels, each with a hidden state
  - Factor nodes: M parity checks, each with a hidden state
  - Spatial edges: 6-connected voxel grid
  - Inputs: syndrome bits, DCT field values, coarse map values
  - Output: per-voxel logit (probability of being 1)

Each iteration:
  1. Variable → Factor messages (via learned function)
  2. Factor → Variable messages (via learned function)
  3. Variable → Variable spatial messages (via learned function)
  4. Update variable hidden states
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralBPDecoder(nn.Module):
    """Neural Belief Propagation decoder.

    Operates on the Tanner graph defined by the parity check matrix H.
    Uses learned message-passing instead of fixed BP rules.

    Args:
        n_vars: number of variable nodes (32^3 = 32768)
        n_checks: number of check nodes
        hidden_dim: hidden state dimension per node
        n_iters: number of BP iterations
        use_spatial: include 6-connected spatial messages
    """

    def __init__(self, n_vars=32768, n_checks=2977, hidden_dim=32,
                 n_iters=10, use_spatial=True, cube_size=32,
                 n_extra_features=0):
        super().__init__()
        self.n_vars = n_vars
        self.n_checks = n_checks
        self.hidden_dim = hidden_dim
        self.n_iters = n_iters
        self.use_spatial = use_spatial
        self.cube_size = cube_size

        # Variable node initial embedding
        # Input features per variable: 1 (initial logit from canonical base)
        #   + n_extra_features (DCT field, coarse map, etc.)
        n_var_input = 1 + n_extra_features
        self.var_embed = nn.Sequential(
            nn.Linear(n_var_input, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Check node initial embedding (from syndrome bit)
        self.check_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Message functions (shared across iterations for memory efficiency)
        # Variable → Check message
        self.var_to_check = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Check → Variable message
        self.check_to_var = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Spatial message (variable → variable, 6-connected)
        if use_spatial:
            self.spatial_msg = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # Variable state update (GRU-like)
        self.var_update = nn.GRUCell(hidden_dim * (2 if use_spatial else 1), hidden_dim)

        # Output: variable hidden → logit
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, var_features, syndrome, H_sparse, spatial_neighbors=None):
        """Run neural BP.

        Args:
            var_features: (B, N, F) per-variable input features
            syndrome: (B, M) target syndrome bits (float 0/1)
            H_sparse: sparse parity matrix info for message routing
                       dict with 'var_indices', 'check_indices' (edge lists)
            spatial_neighbors: precomputed 6-neighbor index tensor (optional)

        Returns:
            logits: (B, N) per-variable output logits
        """
        B = var_features.size(0)
        device = var_features.device

        # Initialize hidden states
        var_h = self.var_embed(var_features)  # (B, N, H)
        check_h = self.check_embed(syndrome.unsqueeze(-1))  # (B, M, H)

        var_idx = H_sparse['var_indices'].to(device)    # (E,) variable index per edge
        check_idx = H_sparse['check_indices'].to(device) # (E,) check index per edge
        n_edges = var_idx.size(0)

        for iteration in range(self.n_iters):
            # 1. Variable → Check messages
            # For each edge: concat var_h[var_idx] and check_h[check_idx]
            var_msgs = var_h[:, var_idx]      # (B, E, H)
            check_ctx = check_h[:, check_idx]  # (B, E, H)
            v2c = self.var_to_check(torch.cat([var_msgs, check_ctx], dim=-1))  # (B, E, H)

            # Aggregate at check nodes: sum messages per check
            check_agg = torch.zeros(B, self.n_checks, self.hidden_dim, device=device)
            check_agg.scatter_add_(1, check_idx.unsqueeze(0).unsqueeze(-1).expand(B, n_edges, self.hidden_dim), v2c)

            # 2. Check → Variable messages
            check_msgs = check_h[:, check_idx] + check_agg[:, check_idx]  # (B, E, H) — include aggregated info
            var_ctx = var_h[:, var_idx]
            c2v = self.check_to_var(torch.cat([check_msgs, var_ctx], dim=-1))  # (B, E, H)

            # Aggregate at variable nodes: sum messages per variable
            var_agg_checks = torch.zeros(B, self.n_vars, self.hidden_dim, device=device)
            var_agg_checks.scatter_add_(1, var_idx.unsqueeze(0).unsqueeze(-1).expand(B, n_edges, self.hidden_dim), c2v)

            # 3. Spatial messages (6-connected)
            if self.use_spatial:
                # Use 3D conv as efficient spatial message passing
                n = self.cube_size
                var_3d = var_h.view(B, n, n, n, self.hidden_dim).permute(0, 4, 1, 2, 3)  # (B, H, n, n, n)
                spatial_out = self.spatial_msg[0](var_h)  # linear
                spatial_3d = spatial_out.view(B, n, n, n, self.hidden_dim).permute(0, 4, 1, 2, 3)
                # Average pool from neighbors (3x3x3 conv with center=0)
                kernel = torch.zeros(self.hidden_dim, 1, 3, 3, 3, device=device)
                kernel[:, 0, 0, 1, 1] = 1/6  # x-1
                kernel[:, 0, 2, 1, 1] = 1/6  # x+1
                kernel[:, 0, 1, 0, 1] = 1/6  # y-1
                kernel[:, 0, 1, 2, 1] = 1/6  # y+1
                kernel[:, 0, 1, 1, 0] = 1/6  # z-1
                kernel[:, 0, 1, 1, 2] = 1/6  # z+1
                spatial_agg = F.conv3d(spatial_3d, kernel, padding=1, groups=self.hidden_dim)
                var_agg_spatial = spatial_agg.permute(0, 2, 3, 4, 1).reshape(B, self.n_vars, self.hidden_dim)
                var_agg_spatial = self.spatial_msg[2](F.silu(var_agg_spatial))  # second linear + activation
            else:
                var_agg_spatial = torch.zeros_like(var_agg_checks)

            # 4. Update variable hidden states
            if self.use_spatial:
                update_input = torch.cat([var_agg_checks, var_agg_spatial], dim=-1)
            else:
                update_input = var_agg_checks

            # GRU update (flatten batch*vars for GRUCell)
            var_h = self.var_update(
                update_input.view(B * self.n_vars, -1),
                var_h.view(B * self.n_vars, self.hidden_dim),
            ).view(B, self.n_vars, self.hidden_dim)

            # Update check hidden states too
            check_h = check_h + check_agg * 0.1  # gentle residual update

        # Output logits
        logits = self.output(var_h).squeeze(-1)  # (B, N)
        return logits


def build_tanner_edges(parity_matrix):
    """Extract edge lists from parity check matrix H.

    Args:
        parity_matrix: (M, N) uint8 sparse parity check matrix

    Returns:
        dict with 'var_indices' and 'check_indices' — (E,) long tensors
        where E = number of 1s in H.
    """
    H = parity_matrix
    if H.is_sparse:
        H = H.to_dense()
    check_idx, var_idx = torch.where(H > 0)
    return {
        'var_indices': var_idx.long(),
        'check_indices': check_idx.long(),
    }


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    """Quick test: create BP decoder and run on random data."""
    from parity import load_constraint_ops, ConstraintConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ConstraintConfig(axis_parity=True, plane_parity=True, global_parity=True)
    _, _, parity = load_constraint_ops(32, cfg)

    M, N = parity.shape
    print(f"Parity matrix: {M} checks × {N} variables")

    edges = build_tanner_edges(parity)
    print(f"Tanner graph: {edges['var_indices'].size(0)} edges")
    print(f"Avg edges per variable: {edges['var_indices'].size(0) / N:.1f}")
    print(f"Avg edges per check: {edges['var_indices'].size(0) / M:.1f}")

    # Create model
    model = NeuralBPDecoder(
        n_vars=N, n_checks=M, hidden_dim=32,
        n_iters=5, use_spatial=True, cube_size=32,
        n_extra_features=2,  # DCT + coarse
    ).to(device)
    print(f"Parameters: {count_params(model):,}")

    # Test forward pass
    B = 2
    var_features = torch.randn(B, N, 3, device=device)  # 1 base + 2 extra
    syndrome = torch.randint(0, 2, (B, M), device=device).float()

    logits = model(var_features, syndrome, edges)
    print(f"Output shape: {logits.shape}")
    print(f"Output range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
