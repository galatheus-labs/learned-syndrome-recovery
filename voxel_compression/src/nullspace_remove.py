"""Parity-aware satellite removal.

Key discovery: decoder errors are always null-space perturbations (because
the decoder + projection guarantees parity). So removing false-positive
voxels and re-projecting is essentially free — the projection cost is
minimal because the errors were null-space structured to begin with.

Two approaches:
  1. CC removal: connected-component analysis, remove small components, project
  2. Denoiser-guided: score voxels with denoiser, remove low-confidence, project

Both maintain parity via project_with_left_inverse after removal.
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage


def cc_cleanup(x, parity, left_inv, syndrome, cube_size=32,
               min_component_size=4, device='cuda'):
    """Remove small connected components and re-project to parity.

    Small disconnected blobs are likely satellites (false positives).
    Removing them + projecting is cheap because FP errors are null-space
    structured, so projection introduces minimal new errors.

    Args:
        x: (1, N) uint8 reconstruction (must satisfy parity)
        parity: (M, N) parity matrix
        left_inv: (N, M) left inverse
        syndrome: (1, M) uint8 target syndrome
        cube_size: grid size
        min_component_size: components smaller than this are removed
        device: torch device

    Returns:
        x_cleaned: (1, N) uint8 cleaned reconstruction
        info: dict with cleanup stats
    """
    from parity import project_with_left_inverse

    x_np = x.cpu().numpy().reshape(cube_size, cube_size, cube_size)
    labeled, n_components = ndimage.label(x_np)

    if n_components <= 1:
        return x, {'n_components': n_components, 'removed': 0, 'kept': n_components}

    # Get component sizes
    component_sizes = ndimage.sum(x_np, labeled, range(1, n_components + 1))

    # Remove small components
    cleaned = x_np.copy()
    removed_count = 0
    kept_count = 0
    for comp_id, size in enumerate(component_sizes, 1):
        if size < min_component_size:
            cleaned[labeled == comp_id] = 0
            removed_count += 1
        else:
            kept_count += 1

    if removed_count == 0:
        return x, {'n_components': n_components, 'removed': 0, 'kept': n_components}

    # Re-project to maintain parity
    cleaned_t = torch.from_numpy(cleaned.reshape(1, -1).astype(np.uint8)).to(device)
    cleaned_t, _ = project_with_left_inverse(cleaned_t, parity, left_inv, syndrome)

    voxels_removed = int(x.sum().item() - cleaned_t.sum().item())

    return cleaned_t, {
        'n_components': n_components,
        'removed_components': removed_count,
        'kept_components': kept_count,
        'voxels_removed': voxels_removed,
    }


def denoiser_cleanup(x, denoiser, parity, left_inv, syndrome,
                     cube_size=32, threshold=0.3, device='cuda'):
    """Remove voxels the denoiser thinks should be OFF, then re-project.

    Uses the denoiser as a learned "plausibility" scorer. Voxels where
    the denoiser output is below threshold are removed.

    Args:
        x: (1, N) uint8 reconstruction
        denoiser: VoxelDenoiser model
        parity, left_inv, syndrome: parity constraint ops
        cube_size: grid size
        threshold: remove ON voxels where denoiser prob < threshold
        device: torch device

    Returns:
        x_cleaned: (1, N) uint8
        info: dict with stats
    """
    from parity import project_with_left_inverse

    with torch.no_grad():
        x_3d = x.float().view(1, 1, cube_size, cube_size, cube_size).to(device)
        clean_prob = torch.sigmoid(denoiser(x_3d)).view(-1)

    x_flat = x.view(-1).to(device)

    # Remove ON voxels where denoiser confidence is low
    remove_mask = (x_flat == 1) & (clean_prob < threshold)
    n_candidates = remove_mask.sum().item()

    if n_candidates == 0:
        return x, {'candidates': 0, 'removed': 0}

    cleaned = x_flat.clone()
    cleaned[remove_mask] = 0

    # Re-project
    cleaned = cleaned.unsqueeze(0)
    cleaned, _ = project_with_left_inverse(cleaned, parity, left_inv, syndrome)

    voxels_removed = int(x.sum().item() - cleaned.sum().item())

    return cleaned, {
        'candidates': n_candidates,
        'voxels_removed': voxels_removed,
    }


def combined_cleanup(x, denoiser, parity, left_inv, syndrome,
                     cube_size=32, min_component_size=4,
                     denoiser_threshold=0.3, device='cuda'):
    """Combined: CC removal then denoiser-guided removal, with one final projection.

    1. Remove small connected components (structural satellites)
    2. Remove low-confidence voxels per denoiser (learned artifacts)
    3. Single projection at the end

    Returns:
        x_cleaned: (1, N) uint8
        info: dict with combined stats
    """
    from parity import project_with_left_inverse

    x_np = x.cpu().numpy().reshape(cube_size, cube_size, cube_size)
    cleaned = x_np.copy()

    # Step 1: CC removal
    labeled, n_components = ndimage.label(cleaned)
    cc_removed = 0
    if n_components > 1:
        sizes = ndimage.sum(cleaned, labeled, range(1, n_components + 1))
        for comp_id, size in enumerate(sizes, 1):
            if size < min_component_size:
                cleaned[labeled == comp_id] = 0
                cc_removed += 1

    # Step 2: Denoiser scoring on remaining voxels
    cleaned_t = torch.from_numpy(cleaned.reshape(1, -1).astype(np.uint8)).to(device)
    denoiser_removed = 0

    if denoiser is not None:
        with torch.no_grad():
            x_3d = cleaned_t.float().view(1, 1, cube_size, cube_size, cube_size)
            clean_prob = torch.sigmoid(denoiser(x_3d)).view(-1)

        remove_mask = (cleaned_t.view(-1) == 1) & (clean_prob < denoiser_threshold)
        denoiser_removed = remove_mask.sum().item()
        if denoiser_removed > 0:
            cleaned_t.view(-1)[remove_mask] = 0

    # Step 3: Single projection
    cleaned_t, _ = project_with_left_inverse(cleaned_t, parity, left_inv, syndrome)

    return cleaned_t, {
        'n_components': n_components,
        'cc_removed': cc_removed,
        'denoiser_removed': denoiser_removed,
        'total_voxels_removed': int(x.sum().item() - cleaned_t.sum().item()),
    }
