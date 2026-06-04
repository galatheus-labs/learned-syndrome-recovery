"""Belief Propagation marginals for GF(2) parity constraints.

Given a parity check matrix H and syndrome s, compute per-voxel
marginal probabilities P(x_i = 1 | Hx = s) using standard BP on
the Tanner graph. No learned parameters — pure message passing.

This produces a soft prior map that encodes what the constraints
alone tell us about each voxel, replacing the hard binary x_base
with a probabilistic "3D Sudoku" solution.

The tanh-domain formulation is used for numerical stability:
  check→var messages use the product-of-tanh rule for GF(2) checks.

Usage:
    from bp_marginals import compute_bp_marginals

    # H: (M, N) uint8 parity check matrix
    # syndrome: (B, M) uint8
    marginals = compute_bp_marginals(H, syndrome, n_iters=20)
    # marginals: (B, N) float in [0, 1]
"""

import torch
from typing import Optional


def _build_adjacency(H: torch.Tensor):
    """Build sparse adjacency lists from parity check matrix.

    Returns:
        check_to_vars: list of M tensors, each containing variable indices
        var_to_checks: list of N tensors, each containing check indices
    """
    M, N = H.shape
    check_to_vars = []
    var_to_checks = [[] for _ in range(N)]

    for c in range(M):
        var_indices = torch.nonzero(H[c], as_tuple=True)[0]
        check_to_vars.append(var_indices)
        for v in var_indices.tolist():
            var_to_checks[v].append(c)

    var_to_checks = [torch.tensor(checks, dtype=torch.long) for checks in var_to_checks]
    return check_to_vars, var_to_checks


def compute_bp_marginals(
    H: torch.Tensor,           # (M, N) uint8 parity check matrix
    syndrome: torch.Tensor,    # (B, M) uint8 syndrome
    n_iters: int = 20,
    damping: float = 0.5,
    prior: Optional[torch.Tensor] = None,  # (B, N) prior probability of each bit being 1
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute BP marginals P(x_i = 1 | Hx = s).

    Args:
        H: parity check matrix (M checks × N variables)
        syndrome: target syndrome (batch × M)
        n_iters: number of BP iterations
        damping: message damping factor (0 = no damping, 1 = freeze)
        prior: prior probability for each bit (default: uniform 0.5)
        device: computation device

    Returns:
        marginals: (B, N) float tensor of probabilities in [0, 1]
    """
    M, N = H.shape
    B = syndrome.size(0)
    if device is None:
        device = syndrome.device

    H = H.to(device)
    syndrome = syndrome.to(device).float()
    # Convert syndrome to ±1: s_c = (-1)^{syndrome_c}
    s_sign = 1.0 - 2.0 * syndrome  # (B, M), +1 if syndrome=0, -1 if syndrome=1

    # Build adjacency
    check_to_vars, var_to_checks = _build_adjacency(H.cpu())

    # Initialize log-likelihood ratios (LLR)
    # LLR_i = log(P(x_i=0) / P(x_i=1))
    if prior is not None:
        prior = prior.to(device).clamp(1e-7, 1 - 1e-7)
        llr_prior = torch.log((1 - prior) / prior)  # (B, N)
    else:
        llr_prior = torch.zeros(B, N, device=device)  # uniform prior

    # Messages: check→var LLR updates
    # Store as dense (M, N) for simplicity. Sparse would be more efficient,
    # but the dense form is easier to inspect for small/medium check sets.
    mu_c2v = torch.zeros(B, M, N, device=device)  # check-to-var messages (LLR)
    mu_v2c = torch.zeros(B, M, N, device=device)  # var-to-check messages (LLR)

    for iteration in range(n_iters):
        # --- Variable → Check messages ---
        # mu_v2c[c, v] = LLR_prior[v] + sum_{c' != c} mu_c2v[c', v]
        # Total belief at each variable
        total_belief = llr_prior.unsqueeze(1) + mu_c2v.sum(dim=1, keepdim=True)  # (B, 1, N)
        # Subtract the message from the target check (extrinsic)
        mu_v2c_new = total_belief - mu_c2v  # (B, M, N)

        # --- Check → Variable messages ---
        # Product-of-tanh rule for GF(2) parity checks:
        # tanh(mu_c2v[c,v]/2) = s_sign[c] * prod_{v' != v} tanh(mu_v2c[c,v']/2)
        tanh_v2c = torch.tanh(mu_v2c_new / 2).clamp(-0.9999999, 0.9999999)  # (B, M, N)

        # Mask to only include edges that exist in H
        H_float = H.float().unsqueeze(0)  # (1, M, N)
        tanh_v2c_masked = tanh_v2c * H_float + (1 - H_float)  # non-edges contribute 1 to product

        # Product over all variables per check
        log_abs_tanh = torch.log(tanh_v2c_masked.abs().clamp(min=1e-30))
        sign_tanh = tanh_v2c_masked.sign()

        # Sum of log-abs for each check (all vars)
        sum_log = (log_abs_tanh * H_float).sum(dim=2, keepdim=True)  # (B, M, 1)
        # Product of signs for each check (all vars)
        # Count negative signs per check
        neg_count = ((sign_tanh < 0).float() * H_float).sum(dim=2, keepdim=True)
        prod_sign = (1 - 2 * (neg_count % 2))  # (B, M, 1)

        # Extrinsic: divide out variable v's contribution
        extrinsic_log = sum_log - log_abs_tanh * H_float  # (B, M, N)
        extrinsic_sign = prod_sign * sign_tanh.sign()  # flip if v had negative sign

        # Include syndrome sign
        s_expanded = s_sign.unsqueeze(2)  # (B, M, 1)
        product = s_expanded * extrinsic_sign * torch.exp(extrinsic_log.clamp(-30, 30))
        product = product.clamp(-0.9999999, 0.9999999)

        mu_c2v_new = 2 * torch.atanh(product) * H_float  # (B, M, N)

        # Damping
        if damping > 0 and iteration > 0:
            mu_c2v = damping * mu_c2v + (1 - damping) * mu_c2v_new
            mu_v2c = damping * mu_v2c + (1 - damping) * mu_v2c_new
        else:
            mu_c2v = mu_c2v_new
            mu_v2c = mu_v2c_new

    # Final marginals
    total_llr = llr_prior + mu_c2v.sum(dim=1)  # (B, N)
    marginals = torch.sigmoid(-total_llr)  # P(x_i = 1)

    return marginals


def compute_bp_marginals_sparse(
    H: torch.Tensor,
    syndrome: torch.Tensor,
    n_iters: int = 20,
    damping: float = 0.5,
    prior: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Memory-efficient BP using sparse edge-indexed message storage.

    Same algorithm as compute_bp_marginals but stores only messages
    on edges that exist in H, reducing memory from O(B*M*N) to
    O(B*nnz(H)). Uses vectorized scatter/gather for speed.
    """
    M, N = H.shape
    B = syndrome.size(0)
    if device is None:
        device = syndrome.device

    syndrome = syndrome.to(device).float()
    s_sign = 1.0 - 2.0 * syndrome  # (B, M)

    # Build edge list
    edges = torch.nonzero(H.cpu(), as_tuple=False)  # (E, 2)
    E = edges.size(0)
    check_idx = edges[:, 0].to(device)  # (E,)
    var_idx = edges[:, 1].to(device)    # (E,)

    # LLR prior
    if prior is not None:
        prior = prior.to(device).clamp(1e-7, 1 - 1e-7)
        llr_prior = torch.log((1 - prior) / prior)  # (B, N)
    else:
        llr_prior = torch.zeros(B, N, device=device)

    # Messages stored per-edge: (B, E)
    mu_c2v = torch.zeros(B, E, device=device)

    for iteration in range(n_iters):
        # --- Variable → Check messages ---
        # mu_v2c[e] = llr_prior[var[e]] + sum of all c2v into var[e] - mu_c2v[e]
        # First: sum all c2v messages per variable using scatter_add
        total_c2v_per_var = torch.zeros(B, N, device=device)
        total_c2v_per_var.scatter_add_(1, var_idx.unsqueeze(0).expand(B, -1), mu_c2v)
        # Extrinsic: total minus this edge's own c2v
        mu_v2c = llr_prior[:, var_idx] + total_c2v_per_var[:, var_idx] - mu_c2v  # (B, E)

        # --- Check → Variable messages ---
        # Product-of-tanh rule: for each edge e with check c,
        # tanh(mu_c2v[e]/2) = s_sign[c] * prod_{e' in check c, e'!=e} tanh(mu_v2c[e']/2)
        tanh_v2c = torch.tanh(mu_v2c / 2).clamp(-0.9999999, 0.9999999)  # (B, E)

        # Compute log-abs and sign per edge
        log_abs = torch.log(tanh_v2c.abs().clamp(min=1e-30))  # (B, E)
        signs = tanh_v2c.sign()  # (B, E)

        # Sum log-abs per check, count negative signs per check
        sum_log_per_check = torch.zeros(B, M, device=device)
        sum_log_per_check.scatter_add_(1, check_idx.unsqueeze(0).expand(B, -1), log_abs)

        neg_mask = (signs < 0).float()
        neg_count_per_check = torch.zeros(B, M, device=device)
        neg_count_per_check.scatter_add_(1, check_idx.unsqueeze(0).expand(B, -1), neg_mask)
        prod_sign_per_check = 1.0 - 2.0 * (neg_count_per_check % 2)  # (B, M)

        # Extrinsic: subtract this edge's contribution
        extrinsic_log = sum_log_per_check[:, check_idx] - log_abs  # (B, E)
        # Flip sign if this edge was negative
        extrinsic_sign = prod_sign_per_check[:, check_idx] * signs  # (B, E)

        # Include syndrome sign
        s_per_edge = s_sign[:, check_idx]  # (B, E)
        product = s_per_edge * extrinsic_sign * torch.exp(extrinsic_log.clamp(-30, 30))
        product = product.clamp(-0.9999999, 0.9999999)

        mu_c2v_new = 2 * torch.atanh(product)  # (B, E)

        # Damping
        if damping > 0 and iteration > 0:
            mu_c2v = damping * mu_c2v + (1 - damping) * mu_c2v_new
        else:
            mu_c2v = mu_c2v_new

    # Final marginals
    total_c2v = torch.zeros(B, N, device=device)
    total_c2v.scatter_add_(1, var_idx.unsqueeze(0).expand(B, -1), mu_c2v)
    total_llr = llr_prior + total_c2v
    marginals = torch.sigmoid(-total_llr)

    return marginals
