"""Syndrome refinement: apply parity constraints to neural net output via BP.

Instead of hard projection (flip minimum-weight bits), use belief propagation
with the neural net's soft output as prior. BP nudges uncertain voxels to
satisfy the syndrome while respecting confident predictions.

This replaces project_with_left_inverse as the post-processing step.

Usage:
    from syndrome_refine import refine_with_syndrome

    # logits: (B, N) raw neural net output
    # H: (M, N) parity check matrix
    # syndrome: (B, M) target syndrome
    refined = refine_with_syndrome(logits, H, syndrome, n_iters=20)
    # refined: (B, N) uint8 — satisfies Hx = s better while staying close to net prediction
"""

import torch
from bp_marginals import compute_bp_marginals_sparse
from parity import gf2_matmul


def refine_with_syndrome(logits, H, syndrome_uint8, n_iters=20,
                         damping=0.5, device=None):
    """Refine neural net predictions using syndrome constraints via BP.

    Args:
        logits: (B, N) float — raw logits from neural net
        H: (M, N) uint8 — parity check matrix
        syndrome_uint8: (B, M) uint8 — target syndrome
        n_iters: BP iterations
        damping: BP damping (higher = more conservative, respects net more)

    Returns:
        refined: (B, N) uint8 — refined binary prediction
        n_violations_before: constraint violations before refinement
        n_violations_after: constraint violations after refinement
        n_flipped: bits changed by refinement
    """
    if device is None:
        device = logits.device
    B, N = logits.shape

    # Neural net's soft prediction as BP prior
    prior = torch.sigmoid(logits).clamp(0.01, 0.99)

    # Before refinement: check violations
    hard_before = (prior > 0.5).to(torch.uint8)
    r_before = gf2_matmul(H.to(device), hard_before.t()).t() ^ syndrome_uint8.to(device)
    n_violations_before = r_before.sum(dim=1).float().mean().item()

    # Run BP with neural net output as prior
    marginals = compute_bp_marginals_sparse(
        H, syndrome_uint8, n_iters=n_iters,
        damping=damping, prior=prior, device=device)

    # Threshold refined marginals
    refined = (marginals > 0.5).to(torch.uint8)

    # After refinement: check violations
    r_after = gf2_matmul(H.to(device), refined.t()).t() ^ syndrome_uint8.to(device)
    n_violations_after = r_after.sum(dim=1).float().mean().item()

    # How many bits changed
    n_flipped = (hard_before ^ refined).sum(dim=1).float().mean().item()

    return refined, n_violations_before, n_violations_after, n_flipped


def confidence_weighted_project(logits, H, left_inv, syndrome_uint8, device=None):
    """Alternative: flip least-confident bits to satisfy constraints.

    Greedy iterative approach:
    1. Find violated constraints
    2. For each violated constraint, identify the least-confident variable
    3. Flip that variable
    4. Repeat until no violations remain

    This is simpler than BP and may be faster for sparse violations.
    """
    if device is None:
        device = logits.device
    B, N = logits.shape
    M = H.size(0)

    confidence = logits.abs()  # higher = more confident
    x = (torch.sigmoid(logits) > 0.5).to(torch.uint8)

    for iteration in range(100):  # max iterations
        r = gf2_matmul(H.to(device), x.t()).t() ^ syndrome_uint8.to(device)
        n_violations = r.sum(dim=1)
        if n_violations.max().item() == 0:
            break  # all constraints satisfied

        # For each batch element, find a violated constraint and flip least-confident bit
        for b in range(B):
            violated = torch.nonzero(r[b], as_tuple=True)[0]
            if len(violated) == 0:
                continue
            # Pick first violated constraint
            c = violated[0].item()
            # Find variables in this constraint
            vars_in_check = torch.nonzero(H[c], as_tuple=True)[0]
            # Find least confident among them
            confs = confidence[b, vars_in_check]
            min_idx = vars_in_check[confs.argmin()]
            # Flip it
            x[b, min_idx] ^= 1

    return x
