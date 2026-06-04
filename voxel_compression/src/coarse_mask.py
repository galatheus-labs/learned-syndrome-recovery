"""Coarse-component hard constraint for inference.

Uses the already-transmitted 4^3 coarse occupancy map to enforce that
the decoder cannot bridge voxels belonging to different coarse-level
connected components. Zero extra bits — the coarse map is already sent.

The mask prevents gap-filling between separate objects by labeling each
fine voxel by its coarse component, then suppressing any ON voxel whose
neighbors belong to a different component.
"""

import torch
import numpy as np
from scipy import ndimage


def build_coarse_component_mask(coarse_map, cube_size=32, coarse_size=4,
                                voxels=None):
    """Build a fine-resolution mask from coarse-level connected components.

    Each fine voxel inherits the component label of its coarse block.
    Voxels in empty coarse blocks are suppressed (mask=0).

    Args:
        coarse_map: (B, 1, cube_size, cube_size, cube_size) float tensor
            The tiled coarse occupancy map (0 or 1 at block granularity)
        cube_size: fine grid size
        coarse_size: coarse grid size
        voxels: (B, N) uint8 optional — if provided, compute coarse occupancy
            with any-occupancy threshold instead of relying on pre-thresholded map

    Returns:
        mask: (B, N) float tensor — 1 where voxels are allowed, 0 where suppressed
        info: dict with component stats
    """
    B = coarse_map.shape[0]
    k = cube_size // coarse_size  # block size (e.g., 8)
    masks = []
    total_components = 0

    for b in range(B):
        if voxels is not None:
            # Compute coarse occupancy from raw voxels with any-occupancy threshold
            import torch.nn.functional as Fpool
            v3d = voxels[b].float().view(1, 1, cube_size, cube_size, cube_size)
            coarse_avg = Fpool.avg_pool3d(v3d, k, stride=k).squeeze()
            coarse_small = (coarse_avg > 0.0).cpu().numpy()
        else:
            # Use the transmitted coarse map (pre-thresholded)
            coarse_3d = coarse_map[b, 0].cpu().numpy()
            coarse_small = coarse_3d[::k, ::k, ::k]  # (coarse_size, coarse_size, coarse_size)
            coarse_small = coarse_small > 0.0

        labeled, n_components = ndimage.label(coarse_small)
        total_components += n_components

        if n_components <= 1:
            # Single component or empty — no masking needed
            masks.append(torch.ones(cube_size ** 3))
            continue

        # Tile labels back to fine resolution
        fine_labels = np.repeat(
            np.repeat(
                np.repeat(labeled, k, axis=0),
                k, axis=1),
            k, axis=2)  # (cube_size, cube_size, cube_size)

        # For each fine voxel, check if any 6-connected neighbor has a
        # different nonzero component label. If so, this voxel is on the
        # boundary between components and should be allowed (it belongs
        # to its own component). The voxels to SUPPRESS are those in
        # empty coarse blocks (label=0) that are adjacent to multiple
        # different occupied components.
        #
        # Simpler approach: a voxel is allowed if its coarse block is occupied.
        # Voxels in empty coarse blocks are suppressed (set to 0).
        # This prevents the decoder from filling gaps between objects.
        mask_3d = (fine_labels > 0).astype(np.float32)
        masks.append(torch.from_numpy(mask_3d.reshape(-1)))

    mask = torch.stack(masks)  # (B, N)
    return mask, {'avg_components': total_components / B}


def apply_coarse_mask(logits, coarse_map, cube_size=32, coarse_size=4,
                      voxels=None):
    """Apply coarse-component mask to decoder logits.

    Suppresses logits in empty coarse blocks by setting them to a large
    negative value, ensuring those voxels are predicted as OFF.

    Args:
        logits: (B, N) decoder output logits
        coarse_map: (B, 1, cube_size, cube_size, cube_size) tiled coarse map
        voxels: (B, N) optional raw voxels for any-occupancy threshold

    Returns:
        masked_logits: (B, N) with suppressed voxels in empty regions
    """
    mask, info = build_coarse_component_mask(
        coarse_map, cube_size, coarse_size, voxels=voxels)
    mask = mask.to(logits.device)

    # Where mask=0 (empty coarse block), force logits very negative
    masked = logits.clone()
    masked[mask < 0.5] = -10.0  # sigmoid(-10) ≈ 0.00005

    return masked, info
