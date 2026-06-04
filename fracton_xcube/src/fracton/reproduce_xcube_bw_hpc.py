"""HPC (CPU) driver for the BW-naive baseline of the X-cube learned decoder.

BW is single-threaded per shot and the wall-clock long pole at L>=8, so it does
NOT belong on a GPU node. This driver fans the per-shot decode across all cores
(multiprocessing) and is meant to run on the cluster's CPU/HPC nodes
(`node-class=hpc`: milan / tinkara, 128 cores) while the learned decoder trains
on a GPU node via `reproduce_xcube_results.py --skip-bw`.

BW uses the SAME seeded test shots (sampler seed*1000+800) as the learned /
trivial / BP+OSD decoders, so the block-failure and per-sector-LER numbers are
directly comparable and merge into reproduce_xcube_results.py's output as the
`bw_naive` decoder column. Output JSON:

    {"config": {...}, "workers": N,
     "bw": {"L8_p0.03": {"block_failure": {mean,ci_lo,ci_hi,vals},
                          "per_sector_ler": {mean,ci_lo,ci_hi,vals},
                          "syndrome_consistent_frac": float}, ...}}

Merge with merge_xcube_l810.py (or by hand) into the GPU job's decoder_tables.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from xcube_code import make_xcube, logical_operators
from xcube_bw_decoder import evaluate_bw_parallel


def seed_bootstrap_ci(vals, iters=10000, seed=0):
    """Bootstrap 95% CI of the mean across seed-level values. Verbatim copy of
    reproduce_xcube_results.seed_bootstrap_ci, inlined to keep this CPU driver
    torch-free (it runs on GPU-less HPC nodes)."""
    vals = np.asarray(vals, dtype=np.float64)
    if len(vals) <= 1:
        return float(vals.mean()), float(vals.mean()), float(vals.mean())
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(vals, size=len(vals), replace=True).mean()
                      for _ in range(iters)])
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _agg(vals):
    m, lo, hi = seed_bootstrap_ci(vals)
    return {"mean": m, "ci_lo": lo, "ci_hi": hi, "vals": list(map(float, vals))}


def main():
    ap = argparse.ArgumentParser(description="Parallel BW-naive baseline (HPC CPU nodes)")
    ap.add_argument("--Ls", type=int, nargs="+", default=[8, 10])
    ap.add_argument("--ps", type=float, nargs="+", default=[0.03, 0.05])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-test", type=int, default=30000,
                    help="MUST match the GPU run's --n-test so BW pairs with learned/trivial.")
    ap.add_argument("--workers", type=int, default=0, help="0 = (cores - 1)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    print(f"BW-naive parallel HPC driver: cores={os.cpu_count()} workers={workers} "
          f"Ls={args.Ls} ps={args.ps} seeds={args.seeds} n_test={args.n_test}", flush=True)

    out = {"config": vars(args), "workers": workers, "bw": {}}
    t_all = time.time()
    for L in args.Ls:
        code = make_xcube(L)
        logical_Z = logical_operators(code)["logical_Z"]
        for p in args.ps:
            key = f"L{L}_p{p}"
            bfs, pss, scs = [], [], []
            for s in args.seeds:
                t0 = time.time()
                r = evaluate_bw_parallel(L, p, args.n_test, s, logical_Z, workers=workers)
                bfs.append(r["block_failure"])
                pss.append(r["per_sector_ler_mean"])
                scs.append(r["syndrome_consistent_frac"])
                print(f"  {key} seed {s}: BW bf {r['block_failure']:.4f} "
                      f"ps {r['per_sector_ler_mean']:.4f} "
                      f"synd-ok {r['syndrome_consistent_frac']:.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            out["bw"][key] = {
                "block_failure": _agg(bfs),
                "per_sector_ler": _agg(pss),
                "syndrome_consistent_frac": float(np.mean(scs)),
            }
            tb = out["bw"][key]
            print(f"    -> {key} BW AGG: bf {tb['block_failure']['mean']:.4f} "
                  f"[{tb['block_failure']['ci_lo']:.4f},{tb['block_failure']['ci_hi']:.4f}]  "
                  f"ps {tb['per_sector_ler']['mean']:.4f}", flush=True)
            if args.out:  # checkpoint after every cell
                with open(args.out, "w") as f:
                    json.dump(out, f, indent=2)
    print(f"done in {time.time() - t_all:.0f}s", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
