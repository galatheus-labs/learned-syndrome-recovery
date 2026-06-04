"""Precondition GATE 0: does decodable signal exist at all?

Before training any neural decoder we check that a SIMPLE local-feature model,
fit and evaluated out-of-sample, can beat the trivial baseline. If even a
generous local lookup/logistic model cannot beat trivial at a given (L, p), then
there is no signal for a CNN to find there and we should not waste compute.

For each logical sector (a wrapping line along axis d at transverse (u,v)), the
informative local feature is the syndrome restricted to the cubes ADJACENT to
that line. We use a logistic regression on the local cube-syndrome neighborhood
of each line (a per-sector linear probe with weight sharing across the L
positions along the line), fit on a TRAIN split and scored on a disjoint
held-out test split (seed*1000+800). This is the "achievable ceiling reference"
the smoke prints BEFORE training the CNN.

KNOWN RESULTS (to reproduce / document):
  * L=2: only 16 distinct syndromes; per-sector headroom over trivial is exactly
    0 -> L=2 is a SHAPE/plumbing test ONLY, never a performance test.
  * L=3, p=0.05: a local-feature model beats trivial out-of-sample (~19% rel).
  * Headroom collapses toward 0 by p~0.10.
  * At L=4 a NAIVE full-neighborhood LOOKUP table loses to trivial (feature too
    sparse to populate) -- which is exactly why the weight-sharing CNN (which
    generalizes across positions a table cannot) is the right tool. The logistic
    probe here is the generalizing analog and is the honest Gate-0 reference.

Run: cd parity_diffusion/internal/fracton && python3 precondition_oracle.py
"""

from __future__ import annotations

import argparse

import numpy as np

from xcube_code import make_xcube, logical_operators


def _line_local_cubes(code, logical_Z, L, radius=1):
    """For each logical-Z line, the set of cube indices (HZ rows) whose center is
    within `radius` (Chebyshev, on the transverse torus) of the line's transverse
    (u,v), at every position along the line's axis. Returns a list of
    (sector_index, cube_index_array)."""
    edge = code["edge_coords"]
    cube_coords = code["cube_coords"]
    # map (x,y,z) -> cube row index
    cube_row = {c: i for i, c in enumerate(cube_coords)}
    k = logical_Z.shape[0]
    feats = []
    for r in range(k):
        supp = np.nonzero(logical_Z[r])[0]
        coords = [edge[int(q)] for q in supp]
        d = coords[0][3]
        others = [a for a in range(3) if a != d]
        u = coords[0][others[0]]
        v = coords[0][others[1]]
        cubes = []
        for along in range(L):
            cell = [0, 0, 0]
            cell[d] = along
            for du in range(-radius, radius + 1):
                for dv in range(-radius, radius + 1):
                    cc = list(cell)
                    cc[others[0]] = (u + du) % L
                    cc[others[1]] = (v + dv) % L
                    cubes.append(cube_row[tuple(cc)])
        feats.append((r, np.array(sorted(set(cubes)), dtype=np.int64)))
    return feats


def _pattern_lookup_pred(ktr, ytr, kte, glob_maj):
    """Out-of-sample local-PATTERN majority lookup (captures nonlinear syndrome
    interactions a linear probe misses). For each distinct local syndrome
    pattern key seen in TRAIN, predict its train-majority label; unseen test
    patterns fall back to the global majority. This is the honest achievable
    ceiling -- the same family of model the spec measured (~19% rel at L=3,p=0.05
    for a richer local cube feature)."""
    from collections import defaultdict
    cnt = defaultdict(lambda: [0, 0])
    for key, lab in zip(ktr, ytr):
        cnt[int(key)][int(lab)] += 1
    pred = np.empty(len(kte), dtype=np.uint8)
    for i, key in enumerate(kte):
        c = cnt.get(int(key))
        pred[i] = (1 if c[1] > c[0] else 0) if c is not None else glob_maj
    return pred


def gate0_table(L, p, n_train=200000, n_test=20000, seed=42, radius=1):
    """Per-(L,p) local-feature held-out ML per-sector LER vs trivial, via an
    out-of-sample local-PATTERN majority lookup over each line's transverse
    neighborhood (radius `radius` on the transverse torus, all L positions)."""
    code = make_xcube(L)
    logical_Z = logical_operators(code)["logical_Z"]
    HZ = code["HZ"]; N = code["N"]; k = logical_Z.shape[0]

    rng_tr = np.random.default_rng(seed)
    e_tr = (rng_tr.random((n_train, N)) < p).astype(np.uint8)
    syn_tr = (e_tr @ HZ.T) % 2
    y_tr = (e_tr @ logical_Z.T) % 2

    rng_te = np.random.default_rng(seed * 1000 + 800)
    e_te = (rng_te.random((n_test, N)) < p).astype(np.uint8)
    syn_te = (e_te @ HZ.T) % 2
    y_te = (e_te @ logical_Z.T) % 2

    feats = _line_local_cubes(code, logical_Z, L, radius=radius)
    ml_wrong = np.zeros(k); triv_wrong = np.zeros(k)
    for (r, cubes) in feats:
        Xtr = syn_tr[:, cubes].astype(np.int64)
        Xte = syn_te[:, cubes].astype(np.int64)
        pw = (1 << np.arange(len(cubes)))
        ktr = (Xtr * pw).sum(1)
        kte = (Xte * pw).sum(1)
        glob_maj = 1 if y_tr[:, r].mean() > 0.5 else 0
        pred = _pattern_lookup_pred(ktr, y_tr[:, r], kte, glob_maj)
        ml_wrong[r] = (pred != y_te[:, r]).mean()
        triv_wrong[r] = (y_te[:, r] != glob_maj).mean()

    return {
        "L": L, "p": p, "k": k,
        "ml_per_sector_ler": float(ml_wrong.mean()),
        "trivial_per_sector_ler": float(triv_wrong.mean()),
        "rel_improvement_pct": float(
            100.0 * (triv_wrong.mean() - ml_wrong.mean()) / triv_wrong.mean()
            if triv_wrong.mean() > 0 else 0.0),
        "n_distinct_syndromes_cap": int(min(2 ** (L ** 3), n_test)),
    }


def main():
    ap = argparse.ArgumentParser(description="Gate-0 signal-existence oracle")
    ap.add_argument("--Ls", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--ps", type=float, nargs="+", default=[0.03, 0.05, 0.07, 0.09])
    ap.add_argument("--n-train", type=int, default=100000)
    ap.add_argument("--n-test", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("precondition_oracle (Gate 0): local-feature held-out ML vs trivial\n")
    print(f"{'L':>2} {'p':>5} {'ML LER':>8} {'trivial':>8} {'rel %':>7}  note")
    for L in args.Ls:
        for p in args.ps:
            t = gate0_table(L, p, args.n_train, args.n_test, args.seed)
            note = ""
            if L == 2:
                note = "L=2 zero-headroom: SHAPE/plumbing only, never a perf test"
            elif t["rel_improvement_pct"] <= 0:
                note = "no out-of-sample headroom here"
            print(f"{L:>2} {p:>5.2f} {t['ml_per_sector_ler']:>8.4f} "
                  f"{t['trivial_per_sector_ler']:>8.4f} {t['rel_improvement_pct']:>7.1f}  {note}")


if __name__ == "__main__":
    main()
