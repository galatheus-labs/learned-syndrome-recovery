"""Voxel-to-mesh conversion and watertightness analysis.

Uses scikit-image marching cubes for isosurface extraction and trimesh
for mesh quality validation.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
import trimesh


def voxels_to_mesh(
    voxels: np.ndarray,
    smooth_sigma: float = 0.5,
    level: float = 0.5,
) -> trimesh.Trimesh | None:
    """Convert a binary voxel grid to a triangle mesh via marching cubes.

    Parameters
    ----------
    voxels : np.ndarray
        Shape ``(N, N, N)``, binary (0/1) values.
    smooth_sigma : float
        Gaussian smoothing sigma applied before marching cubes.
        Set to 0 for no smoothing.
    level : float
        Isosurface level for marching cubes.

    Returns
    -------
    trimesh.Trimesh or None
        The extracted mesh, or ``None`` if the grid is empty / full
        (no isosurface exists).
    """
    volume = voxels.astype(np.float32)
    if volume.sum() == 0 or volume.sum() == volume.size:
        return None

    if smooth_sigma > 0:
        volume = gaussian_filter(volume, sigma=smooth_sigma)

    try:
        verts, faces, _, _ = marching_cubes(volume, level=level)
    except (ValueError, RuntimeError):
        return None

    if len(faces) == 0:
        return None

    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


def mesh_quality_metrics(mesh: trimesh.Trimesh | None) -> dict:
    """Compute mesh quality metrics.

    Parameters
    ----------
    mesh : trimesh.Trimesh or None
        If ``None``, returns a dict with all metrics set to failure values.

    Returns
    -------
    dict
        Keys: ``watertight``, ``is_volume``, ``num_components``,
        ``num_faces``, ``euler_number``, ``volume``, ``surface_area``.
    """
    if mesh is None:
        return {
            "watertight": False,
            "is_volume": False,
            "num_components": 0,
            "num_faces": 0,
            "euler_number": 0,
            "volume": float("nan"),
            "surface_area": float("nan"),
        }

    bodies = mesh.split(only_watertight=False)
    num_components = len(bodies) if bodies else 1

    wt = mesh.is_watertight
    vol = mesh.is_volume if hasattr(mesh, "is_volume") else wt

    return {
        "watertight": bool(wt),
        "is_volume": bool(vol),
        "num_components": num_components,
        "num_faces": len(mesh.faces),
        "euler_number": int(mesh.euler_number),
        "volume": float(mesh.volume) if wt else float("nan"),
        "surface_area": float(mesh.area),
    }


def batch_watertightness(
    voxel_batch: np.ndarray,
    smooth_sigma: float = 0.5,
) -> dict:
    """Run watertightness analysis on a batch of voxel grids.

    Parameters
    ----------
    voxel_batch : np.ndarray
        Shape ``(B, N, N, N)``, binary values.
    smooth_sigma : float
        Gaussian smoothing before marching cubes.

    Returns
    -------
    dict
        Keys: ``watertight_rate``, ``volume_rate``, ``per_sample`` (list of
        :func:`mesh_quality_metrics` dicts).
    """
    per_sample = []
    for i in range(voxel_batch.shape[0]):
        mesh = voxels_to_mesh(voxel_batch[i], smooth_sigma=smooth_sigma)
        per_sample.append(mesh_quality_metrics(mesh))

    n = len(per_sample)
    if n == 0:
        return {"watertight_rate": 0.0, "volume_rate": 0.0, "per_sample": []}

    wt_count = sum(1 for s in per_sample if s["watertight"])
    vol_count = sum(1 for s in per_sample if s["is_volume"])

    return {
        "watertight_rate": wt_count / n,
        "volume_rate": vol_count / n,
        "per_sample": per_sample,
    }
