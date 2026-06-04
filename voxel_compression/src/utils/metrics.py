import torch
import math


def shannon_entropy(x: torch.Tensor) -> float:
    p = x.float().mean().clamp(1e-6, 1 - 1e-6)
    return float(-p * math.log2(p) - (1 - p) * math.log2(1 - p))


def hamming_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamming distance between two binary tensors (per sample in a batch).

    Parameters
    ----------
    a, b : torch.Tensor
        Shape ``(batch, n)`` or ``(n,)``.  Values should be 0 or 1.

    Returns
    -------
    torch.Tensor
        Scalar or shape ``(batch,)`` — number of differing bits.
    """
    return (a ^ b).sum(dim=-1)


def iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Intersection over union for binary voxel grids.

    Parameters
    ----------
    a, b : torch.Tensor
        Shape ``(batch, n)`` or ``(n,)``.  Values should be 0 or 1.

    Returns
    -------
    torch.Tensor
        Scalar or shape ``(batch,)`` — IoU in [0, 1].
    """
    a_f = a.float()
    b_f = b.float()
    intersection = (a_f * b_f).sum(dim=-1)
    union = ((a_f + b_f) > 0).float().sum(dim=-1)
    return intersection / union.clamp(min=1.0)


def voxel_density(x: torch.Tensor) -> torch.Tensor:
    """Fraction of filled voxels.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(batch, n)`` or ``(n,)``.  Values should be 0 or 1.

    Returns
    -------
    torch.Tensor
        Scalar or shape ``(batch,)`` — density in [0, 1].
    """
    return x.float().mean(dim=-1)


def constraint_satisfaction(x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Fraction of parity checks satisfied per sample.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(batch, n)`` — binary voxels (0 or 1).
    H : torch.Tensor
        Shape ``(m, n)`` — parity check matrix over GF(2).

    Returns
    -------
    torch.Tensor
        Shape ``(batch,)`` — fraction of checks satisfied, in [0, 1].
    """
    if H.size(0) == 0:
        return torch.ones(x.size(0), device=x.device)
    syndrome = (x.float() @ H.float().t()) % 2  # (batch, m)
    satisfied = (syndrome == 0).float().mean(dim=-1)
    return satisfied
