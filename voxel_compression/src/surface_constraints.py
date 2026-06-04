"""Surface-aware parity constraints for voxel compression.

Key insight: reconstruction errors concentrate at the object surface.
The interior (all 0) and occupied bulk (all 1) are easy to predict.
The hard part is the boundary — which voxels flip from 0 to 1.

We design constraints that target the surface:

1. **Neighbor-pair parity**: For each pair of 6-connected adjacent voxels
   near the predicted surface, transmit their XOR. This tells the decoder
   whether adjacent voxels are same/different — encoding the surface edges.

2. **Surface patch parity**: Small overlapping patches (2³) centered on
   predicted surface voxels. Like the subcube constraints but only where
   they matter.

3. **Gradient parity**: Parity of the discrete gradient (differences
   between adjacent voxels along each axis) in surface regions.

All constraint locations are derived from the DCT prior, so both encoder
and decoder generate the same constraint matrix without extra bits.

Bit budget analysis for a 2% fill rate 32³ object:
  - ~655 non-zero voxels
  - ~1500-2000 surface voxels
  - Neighbor pairs along surface: ~3000-5000 constraints
  - Each constraint = 1 syndrome bit
  - Total: 3-5K bits — viable!
"""

import torch
import torch.nn.functional as F


def detect_surface(prior_3d, threshold=0.3):
    """Identify surface voxels from a soft probability map.

    Surface = voxels where the probability gradient is high,
    i.e., the boundary between likely-occupied and likely-empty regions.

    Returns a binary mask of surface voxels.
    """
    D = prior_3d.shape[0]
    p = prior_3d.unsqueeze(0).unsqueeze(0)  # (1,1,D,D,D)

    # Compute gradient magnitude via Sobel-like finite differences
    # Dilate occupied region
    dilated = F.max_pool3d(p, kernel_size=3, stride=1, padding=1)
    # Erode occupied region
    eroded = -F.max_pool3d(-p, kernel_size=3, stride=1, padding=1)

    # Surface = where dilated > threshold but eroded < threshold
    surface = ((dilated.squeeze() > threshold) &
               (eroded.squeeze() < 1 - threshold))

    return surface


def build_neighbor_pair_constraints(surface_mask, cube_size):
    """Build parity constraints for adjacent voxel pairs near the surface.

    For each surface voxel, create constraints with its 6-connected neighbors.
    Each constraint has exactly 2 nonzeros (the pair) — their XOR tells
    the decoder if adjacent voxels are same or different.

    Returns: (M, N) uint8 constraint matrix
    """
    D = cube_size
    N = D ** 3
    rows = []
    seen = set()

    surface_coords = torch.nonzero(surface_mask, as_tuple=False)  # (K, 3)
    offsets = [(1,0,0), (0,1,0), (0,0,1)]  # only positive direction to avoid duplicates

    for coord in surface_coords:
        z, y, x = coord[0].item(), coord[1].item(), coord[2].item()
        for dz, dy, dx in offsets:
            nz, ny, nx = z+dz, y+dy, x+dx
            if nz >= D or ny >= D or nx >= D:
                continue
            idx1 = x + y*D + z*D*D
            idx2 = nx + ny*D + nz*D*D
            pair = (min(idx1, idx2), max(idx1, idx2))
            if pair in seen:
                continue
            seen.add(pair)
            row = torch.zeros(N, dtype=torch.uint8)
            row[idx1] = 1
            row[idx2] = 1
            rows.append(row)

    if not rows:
        return torch.zeros((0, N), dtype=torch.uint8)
    return torch.stack(rows)


def build_surface_patch_constraints(surface_mask, cube_size, patch_size=2):
    """Build overlapping patch parity constraints centered on surface voxels.

    Like 2³ subcube constraints but only placed where the surface is.
    Each patch is a small cube centered on a surface voxel.

    Returns: (M, N) uint8 constraint matrix
    """
    D = cube_size
    N = D ** 3
    s = patch_size
    rows = []
    seen = set()

    surface_coords = torch.nonzero(surface_mask, as_tuple=False)

    for coord in surface_coords:
        z, y, x = coord[0].item(), coord[1].item(), coord[2].item()
        # Place a patch_size³ cube with corner at (x-s//2, y-s//2, z-s//2)
        x0 = max(0, x - s//2)
        y0 = max(0, y - s//2)
        z0 = max(0, z - s//2)
        x1 = min(D, x0 + s)
        y1 = min(D, y0 + s)
        z1 = min(D, z0 + s)

        key = (x0, y0, z0, x1, y1, z1)
        if key in seen:
            continue
        seen.add(key)

        row = torch.zeros(N, dtype=torch.uint8)
        for pz in range(z0, z1):
            for py in range(y0, y1):
                for px in range(x0, x1):
                    row[px + py*D + pz*D*D] = 1
        rows.append(row)

    if not rows:
        return torch.zeros((0, N), dtype=torch.uint8)
    return torch.stack(rows)


def build_gradient_constraints(surface_mask, cube_size):
    """Build gradient parity constraints along each axis near the surface.

    For each surface voxel, create a constraint over a short run of
    voxels along each axis (length 4). This captures local directional
    patterns — essentially encoding whether the surface crosses this
    axis segment.

    Returns: (M, N) uint8 constraint matrix
    """
    D = cube_size
    N = D ** 3
    run_length = 4
    rows = []
    seen = set()

    surface_coords = torch.nonzero(surface_mask, as_tuple=False)

    for coord in surface_coords:
        z, y, x = coord[0].item(), coord[1].item(), coord[2].item()

        # X-axis run centered on this voxel
        x0 = max(0, x - run_length//2)
        x1 = min(D, x0 + run_length)
        if x1 - x0 >= 2:
            key = ('x', y, z, x0)
            if key not in seen:
                seen.add(key)
                row = torch.zeros(N, dtype=torch.uint8)
                for px in range(x0, x1):
                    row[px + y*D + z*D*D] = 1
                rows.append(row)

        # Y-axis run
        y0 = max(0, y - run_length//2)
        y1 = min(D, y0 + run_length)
        if y1 - y0 >= 2:
            key = ('y', x, z, y0)
            if key not in seen:
                seen.add(key)
                row = torch.zeros(N, dtype=torch.uint8)
                for py in range(y0, y1):
                    row[x + py*D + z*D*D] = 1
                rows.append(row)

        # Z-axis run
        z0 = max(0, z - run_length//2)
        z1 = min(D, z0 + run_length)
        if z1 - z0 >= 2:
            key = ('z', x, y, z0)
            if key not in seen:
                seen.add(key)
                row = torch.zeros(N, dtype=torch.uint8)
                for pz in range(z0, z1):
                    row[x + y*D + pz*D*D] = 1
                rows.append(row)

    if not rows:
        return torch.zeros((0, N), dtype=torch.uint8)
    return torch.stack(rows)


def build_surface_constraints(dct_3d, cube_size, mode='all',
                               surface_threshold=0.3):
    """Build all surface-aware constraints from DCT prior.

    Args:
        dct_3d: (D, D, D) DCT reconstruction
        cube_size: grid dimension
        mode: 'neighbor', 'patch', 'gradient', or 'all'
        surface_threshold: threshold for surface detection

    Returns:
        H: (M, N) uint8 constraint matrix
        stats: dict with constraint counts per type
    """
    surface = detect_surface(dct_3d, threshold=surface_threshold)
    n_surface = surface.sum().item()

    parts = []
    stats = {'n_surface_voxels': n_surface}

    if mode in ('neighbor', 'all'):
        H_neigh = build_neighbor_pair_constraints(surface, cube_size)
        parts.append(H_neigh)
        stats['neighbor_pairs'] = H_neigh.size(0)

    if mode in ('patch', 'all'):
        H_patch = build_surface_patch_constraints(surface, cube_size)
        parts.append(H_patch)
        stats['surface_patches'] = H_patch.size(0)

    if mode in ('gradient', 'all'):
        H_grad = build_gradient_constraints(surface, cube_size)
        parts.append(H_grad)
        stats['gradient_runs'] = H_grad.size(0)

    if not parts:
        N = cube_size ** 3
        return torch.zeros((0, N), dtype=torch.uint8), stats

    H = torch.cat(parts, dim=0)
    stats['total_constraints'] = H.size(0)
    return H, stats
