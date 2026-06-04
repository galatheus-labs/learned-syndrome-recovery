"""BP (+ OSD-0) baseline decoder for the X-cube fracton code, made directly
comparable to the learned decoder's per-sector / block logical-Z metric.

Algorithm (numpy, CPU; no heavy deps):
  - Tanner graph from the Z-cube check matrix HZ: variables = edges (degree 4 in
    the cube checks), checks = cubes (degree 12). Channel LLR Lv = log((1-p)/p).
  - Log-domain sum-product BP, flooding schedule, tanh check rule with the
    syndrome sign (1 - 2 s_c). MANDATORY magnitude-preserving NaN fix (see below).
  - Posterior LLR -> hard decision; stop when HZ @ ehat == s (mod 2); max_iter=64.
  - OSD-0: order columns by reliability (|posterior LLR|), GF(2)-eliminate HZ to
    find a pivot set, set FREE columns to the BP HARD DECISION (NOT to 0), then
    back-substitute the pivots so HZ @ ehat == s exactly. HZ is rank-deficient
    (by 3L-2), so we guard: every all-zero reduced row must have RHS bit 0.

Critical NaN fix: the naive `np.clip(t, -1+eps, 1-eps)` does NOT move t==0 away
from 0, so `log(0) = -inf` and `excl_log = (-inf) - (-inf) = NaN` on the
symmetric-tie configurations the X-cube produces constantly. We clip by
MAGNITUDE preserving sign: sgn = where(t>=0,1,-1); t = sgn * clip(|t|, eps, 1-eps).

RE-FRAMING (load-bearing): empirically BP+OSD-0 is a WEAK baseline for this
sector/metric, not a strong one. BP converges on only a minority of shots and
OSD's minimum-weight representative lands in the WRONG logical coset for
full-torus wrapping-line operators -- so per-sector LER for BP+OSD is typically
WORSE than the trivial baseline. We report it as a diagnostic of why
matching-free min-weight decoding fails on wrapping lines, and predeclare that
the success criterion is BEATING TRIVIAL (with BP+OSD reported alongside). We do
NOT reimplement arXiv:1901.08061 unless the learned decoder first clears trivial.

Eval fairness: we reuse the EXACT test sampler `rng.random((n,N)) < p` at seed
`seed*1000+800` (regenerating the raw errors e, since OSD scores residual on e),
derive predicted logical-Z parities as logical_Z @ ehat (mod 2) -> the SAME
(6L-3,) per-sector metric on the SAME shots as the learned decoder -> paired.

Run as __main__ to execute the 100%-syndrome-consistency unit test and a small
multi-seed {42,43,44} smoke comparison vs trivial.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from xcube_code import make_xcube, logical_operators


# ----------------------------------------------------------------------------
# Belief propagation (log-domain sum-product, flooding schedule)
# ----------------------------------------------------------------------------
def _bp_decode_batch(HZ, syndromes, p, max_iter=64):
    """Run BP for a batch of syndromes.

    Args:
      HZ         : (m, N) uint8 check matrix
      syndromes  : (n, m) uint8 syndromes (flat, cube order matching HZ rows)
      p          : channel flip rate
      max_iter   : BP iteration cap

    Returns:
      hard       : (n, N) uint8 BP hard decision
      post_llr   : (n, N) float posterior LLR (positive => bit likely 0)
      converged  : (n,) bool whether HZ @ hard == s
    """
    m, N = HZ.shape
    n = syndromes.shape[0]
    eps = 1e-12

    # build adjacency lists
    check_vars = [np.nonzero(HZ[c])[0] for c in range(m)]  # vars in each check
    var_checks = [np.nonzero(HZ[:, v])[0] for v in range(N)]  # checks per var

    Lv = np.log((1 - p) / p)  # channel LLR per variable (prior bit=0 more likely)
    Lch = np.full((n, N), Lv, dtype=np.float64)

    hard = np.zeros((n, N), dtype=np.uint8)
    post = np.zeros((n, N), dtype=np.float64)
    converged = np.zeros(n, dtype=bool)
    remaining = np.arange(n)

    # messages: var->check (M_vc) and check->var (M_cv), stored as dense (n,m,N)
    # would be huge; instead store as dict keyed by (check, var) is slow. Use
    # ragged numpy via per-check arrays. For CPU-smoke sizes (L<=4 -> N<=192,
    # m<=64) dense (n, m, N) is fine and vectorizes cleanly.
    M_vc = np.zeros((n, m, N), dtype=np.float64)  # var->check messages
    mask = (HZ > 0)  # (m, N) edges
    for c in range(m):
        M_vc[:, c, check_vars[c]] = Lv  # init with channel LLR

    s_sign = (1 - 2 * syndromes.astype(np.float64))  # (n, m) in {+1,-1}

    for _ in range(max_iter):
        # ---- check update (tanh rule) ----
        # t_cv = product over v' != v of tanh(M_vc/2), times syndrome sign
        # compute per check the product of tanh(M/2) over its variables, then
        # divide out each variable's own contribution (exclusive product).
        th = np.tanh(0.5 * M_vc)  # (n, m, N); off-edges have M_vc=0 -> tanh=0
        # set off-edge entries to 1 so they don't kill the product
        th_edge = np.where(mask[None, :, :], th, 1.0)
        prod = np.prod(th_edge, axis=2, keepdims=True)  # (n, m, 1) full product
        # exclusive product = full / own (guard own==0)
        own = th_edge
        safe_own = np.where(np.abs(own) < eps, 1.0, own)
        excl = prod / safe_own  # (n, m, N) exclusive tanh product
        excl = excl * s_sign[:, :, None]
        # magnitude-preserving clip (the NaN fix)
        sgn = np.where(excl >= 0, 1.0, -1.0)
        excl_c = sgn * np.clip(np.abs(excl), eps, 1 - eps)
        M_cv = 2.0 * np.arctanh(excl_c)  # (n, m, N) check->var messages
        M_cv = np.where(mask[None, :, :], M_cv, 0.0)

        # ---- variable update ----
        sum_cv = M_cv.sum(axis=1)  # (n, N) sum over checks
        post = Lch + sum_cv
        # M_vc[c,v] = post[v] - M_cv[c,v] (exclusive)
        M_vc = (post[:, None, :] - M_cv)
        M_vc = np.where(mask[None, :, :], M_vc, 0.0)

        # ---- hard decision + convergence ----
        hard = (post < 0).astype(np.uint8)  # post<0 => bit=1 more likely
        synd_hat = (hard @ HZ.T) % 2  # (n, m)
        ok = np.all(synd_hat == syndromes, axis=1)
        newly = ~converged & ok
        converged[newly] = True
        if converged.all():
            break

    assert not np.isnan(post).any(), "NaN in BP posterior LLR (NaN fix failed)"
    return hard, post, converged


# ----------------------------------------------------------------------------
# OSD-0 on rank-deficient HZ
# ----------------------------------------------------------------------------
def _osd0_single(HZ, syndrome, post_llr, hard):
    """OSD-0 for one shot. Order columns by reliability (|post_llr| ascending =
    least reliable first is the classic OSD ordering: we pivot on the MOST
    reliable columns). Free columns set to BP hard decision; back-substitute
    pivots so HZ @ ehat == syndrome exactly.

    HZ is rank-deficient (by 3L-2); guard that every all-zero reduced row has
    RHS bit 0 (syndrome-consistent), else raise.
    """
    m, N = HZ.shape
    # reliability: most reliable (large |LLR|) should be pivots -> sort columns
    # by DECREASING reliability so elimination prefers reliable pivots.
    order = np.argsort(-np.abs(post_llr))  # most reliable first
    A = HZ[:, order].astype(np.uint8).copy()  # (m, N) permuted
    b = syndrome.astype(np.uint8).copy()      # (m,)

    # GF(2) elimination, tracking which permuted columns are pivots
    pivot_cols = []
    r = 0
    for c in range(N):
        # find pivot row at/below r with A[row,c]==1
        piv = -1
        for i in range(r, m):
            if A[i, c]:
                piv = i
                break
        if piv < 0:
            continue
        if piv != r:
            A[[r, piv]] = A[[piv, r]]
            b[r], b[piv] = b[piv], b[r]
        # eliminate column c from all other rows
        for i in range(m):
            if i != r and A[i, c]:
                A[i] ^= A[r]
                b[i] ^= b[r]
        pivot_cols.append(c)
        r += 1
        if r == m:
            break

    # consistency guard: all-zero reduced rows must have RHS 0
    for i in range(r, m):
        if b[i] != 0:
            raise AssertionError(
                "OSD-0 GF(2) elimination hit inconsistent rank-deficient row "
                f"(row {i} RHS={b[i]}); syndrome not in column space of HZ")

    # free columns (in permuted index space) := BP hard decision
    hard_perm = hard[order].astype(np.uint8)
    e_perm = np.zeros(N, dtype=np.uint8)
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(N) if c not in pivot_set]
    for c in free_cols:
        e_perm[c] = hard_perm[c]

    # back-substitute pivots: pivot row r_i has leading col pivot_cols[r_i];
    # e[pivot] = b[r_i] xor sum_{free in row} A[r_i, free]*e[free]
    for ri in range(len(pivot_cols)):
        pc = pivot_cols[ri]
        acc = b[ri]
        # all columns in this reduced row beyond pivots are free (RREF guarantees
        # pivot columns are unit columns), so xor free contributions
        row = A[ri]
        contrib = np.bitwise_and(row, e_perm)
        contrib[pc] = 0  # exclude the pivot itself
        acc ^= int(contrib.sum() % 2)
        e_perm[pc] = acc

    # un-permute
    ehat = np.zeros(N, dtype=np.uint8)
    ehat[order] = e_perm
    return ehat


def _osd0_batch(HZ, syndromes, post_llr, hard):
    n, N = hard.shape
    out = np.zeros((n, N), dtype=np.uint8)
    for i in range(n):
        out[i] = _osd0_single(HZ, syndromes[i], post_llr[i], hard[i])
    return out


# ----------------------------------------------------------------------------
# Top-level eval (mirrors xcube_decoder.evaluate keys)
# ----------------------------------------------------------------------------
def evaluate_bp_osd(L, p, n, seed, logical_Z=None, max_iter=64, return_per_shot=True):
    """BP+OSD-0 evaluation on the SAME test shots as the learned decoder.

    Returns dict with the same keys as xcube_decoder.evaluate plus
    bp_convergence_frac and syndrome_consistent_frac (sanity calibration).
    """
    code = make_xcube(L)
    HZ = code["HZ"]
    if logical_Z is None:
        logical_Z = logical_operators(code)["logical_Z"]
    k = logical_Z.shape[0]

    # byte-identical test shots: regenerate raw errors at seed*1000+800
    rng = np.random.default_rng(seed * 1000 + 800)
    errors = (rng.random((n, code["N"])) < p).astype(np.uint8)
    syn_flat = (errors @ HZ.T) % 2  # (n, m) in cube-row order
    true_labels = ((errors @ logical_Z.T) % 2).astype(np.uint8)  # (n, k)

    # BP+OSD in shot-chunks. _bp_decode_batch builds several dense (chunk, m, N)
    # float64 message arrays at once, so decoding all n shots together is
    # O(n * L^3 * 3L^3) and OOMs for L>=8 (e.g. ~63 GB at L=8, n=2000). Each
    # shot's BP+OSD is independent, so chunking over shots is numerically
    # identical (verified) while capping each array at ~1 GB regardless of L.
    m = HZ.shape[0]
    chunk = max(1, min(n, int(1e9 // (m * code["N"] * 8))))
    ehat = np.zeros((n, code["N"]), dtype=np.uint8)
    converged = np.zeros(n, dtype=bool)
    for c0 in range(0, n, chunk):
        c1 = min(c0 + chunk, n)
        h_c, po_c, cv_c = _bp_decode_batch(HZ, syn_flat[c0:c1], p, max_iter=max_iter)
        converged[c0:c1] = cv_c
        ehat[c0:c1] = _osd0_batch(HZ, syn_flat[c0:c1], po_c, h_c)

    # second-defense syndrome consistency
    synd_consistent = np.all((ehat @ HZ.T) % 2 == syn_flat, axis=1)

    pred_labels = ((ehat @ logical_Z.T) % 2).astype(np.uint8)  # (n, k)
    corr = (pred_labels == true_labels)  # (n, k)
    per_sector = 1.0 - corr.mean(0)
    block_correct = corr.all(1)

    res = {
        "per_sector_ler_mean": float(per_sector.mean()),
        "per_sector_ler_max": float(per_sector.max()),
        "block_failure": float(1.0 - block_correct.mean()),
        "bp_convergence_frac": float(converged.mean()),
        "syndrome_consistent_frac": float(synd_consistent.mean()),
    }
    if return_per_shot:
        res["per_shot_correct"] = corr
        res["block_correct"] = block_correct
    return res


def trivial_baseline(L, p, n, seed, logical_Z=None):
    """Trivial 'predict majority class per sector' decoder, on the SAME shots."""
    code = make_xcube(L)
    if logical_Z is None:
        logical_Z = logical_operators(code)["logical_Z"]
    rng = np.random.default_rng(seed * 1000 + 800)
    errors = (rng.random((n, code["N"])) < p).astype(np.uint8)
    true_labels = ((errors @ logical_Z.T) % 2).astype(np.uint8)
    base_rate = true_labels.mean(0)
    pred = (base_rate > 0.5).astype(np.uint8)  # majority class per sector
    corr = (true_labels == pred[None, :])
    per_sector = 1.0 - corr.mean(0)
    block_correct = corr.all(1)
    return {
        "per_sector_ler_mean": float(per_sector.mean()),
        "per_sector_ler_max": float(per_sector.max()),
        "block_failure": float(1.0 - block_correct.mean()),
        "per_shot_correct": corr,
        "block_correct": block_correct,
    }


# ----------------------------------------------------------------------------
# Unit test + multi-seed driver
# ----------------------------------------------------------------------------
def syndrome_consistency_unit_test(L=3, p=0.05, n=200, seed=42):
    """Assert OSD-0 produces a 100%-syndrome-consistent ehat (HZ @ ehat == s)."""
    code = make_xcube(L)
    HZ = code["HZ"]
    rng = np.random.default_rng(seed)
    errors = (rng.random((n, code["N"])) < p).astype(np.uint8)
    syn = (errors @ HZ.T) % 2
    hard, post, conv = _bp_decode_batch(HZ, syn, p)
    ehat = _osd0_batch(HZ, syn, post, hard)
    consistent = np.all((ehat @ HZ.T) % 2 == syn, axis=1)
    frac_inconsistent = float(1.0 - consistent.mean())
    assert frac_inconsistent == 0.0, (
        f"OSD-0 produced {frac_inconsistent:.4f} syndrome-inconsistent shots "
        "(must be 0.000)")
    assert not np.isnan(post).any(), "NaN in BP posterior"
    print(f"  [syndrome-consistency] L={L} p={p} n={n}: 0.000 inconsistent, "
          f"BP converged {conv.mean():.3f}  OK")
    return frac_inconsistent


def main():
    ap = argparse.ArgumentParser(description="X-cube BP(+OSD-0) baseline")
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--max-iter", type=int, default=64)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    print("xcube_baseline_bposd:")
    # GATE 1: 100%-syndrome-consistency unit test
    syndrome_consistency_unit_test(L=args.L, p=args.p, n=min(200, args.n_test))

    code = make_xcube(args.L)
    logical_Z = logical_operators(code)["logical_Z"]
    rows = []
    for s in args.seeds:
        bp = evaluate_bp_osd(args.L, args.p, args.n_test, s, logical_Z,
                             max_iter=args.max_iter, return_per_shot=False)
        tr = trivial_baseline(args.L, args.p, args.n_test, s, logical_Z)
        rows.append({"seed": s, "bp_osd": bp,
                     "trivial": {k_: tr[k_] for k_ in
                                 ("per_sector_ler_mean", "per_sector_ler_max", "block_failure")}})
        print(f"  seed {s}: BP+OSD per-sector LER {bp['per_sector_ler_mean']:.4f} "
              f"(block {bp['block_failure']:.4f}, conv {bp['bp_convergence_frac']:.3f}, "
              f"synd-consistent {bp['syndrome_consistent_frac']:.3f})  |  "
              f"trivial per-sector LER {tr['per_sector_ler_mean']:.4f} "
              f"(block {tr['block_failure']:.4f})")

    bp_means = np.array([r["bp_osd"]["per_sector_ler_mean"] for r in rows])
    tr_means = np.array([r["trivial"]["per_sector_ler_mean"] for r in rows])
    print(f"  ACROSS SEEDS: BP+OSD per-sector LER {bp_means.mean():.4f} +/- {bp_means.std():.4f}  "
          f"trivial {tr_means.mean():.4f} +/- {tr_means.std():.4f}  "
          f"(BP+OSD minus trivial = {bp_means.mean()-tr_means.mean():+.4f}; "
          "positive => BP+OSD WORSE than trivial, the wrong-coset diagnostic)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"L": args.L, "p": args.p, "n_test": args.n_test, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
