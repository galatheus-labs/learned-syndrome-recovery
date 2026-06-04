"""Brown--Williamson (naive-weight cube-defect matching) baseline for the X-cube
fracton code, evaluated on the SAME test split as the learned decoder and BP+OSD.

This produces the BW column of Table 1 in the fracton paper. BW is run on the
byte-identical test shots (sampler seed*1000+800, same n_test) used by
reproduce_xcube_results.py, so block-failure and per-sector LER are directly
comparable to the learned/trivial/BP+OSD numbers already in xcube_L346_gen.json.

Decoder: faithful naive-weight realization of Brown--Williamson's cube-defect
("planeon") matching decoder (arXiv:1901.08061, App. B). It is provably
syndrome-consistent and recovers isolated single-edge errors exactly; its
matching ignores the X-cube wrapping-line degeneracy, so above ~3% per-sector
LER can dip toward/above trivial even while block-failure stays well below
trivial (BW's BP edge-reweighting, not implemented here, is what lifts the
threshold 3.7%->4.3%). Reported honestly as the naive-weight variant.

Usage:
    python reproduce_xcube_bw.py --Ls 3 4 6 --ps 0.03 0.05 --seeds 42 43 44 \
        --n-test 40000 --out xcube_bw_baseline.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from xcube_code import make_xcube, logical_operators
from xcube_bw_decoder import evaluate_bw, trivial_baseline


def agg(vals):
    """3-seed summary: mean and [min, max] across seeds (matches Table 1 style)."""
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()), "ci_lo": float(a.min()), "ci_hi": float(a.max()),
            "vals": [float(x) for x in a]}


def main():
    ap = argparse.ArgumentParser(description="BW baseline at the paper's (L,p) cells")
    ap.add_argument("--Ls", type=int, nargs="+", default=[3, 4, 6])
    ap.add_argument("--ps", type=float, nargs="+", default=[0.03, 0.05])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-test", type=int, default=40000)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    out = {"config": vars(args), "cells": {}}
    print(f"BW baseline | Ls={args.Ls} ps={args.ps} seeds={args.seeds} n_test={args.n_test}",
          flush=True)
    for L in args.Ls:
        # precompute logical_Z once per L (expensive nullspace) and reuse across seeds/ps
        logical_Z = logical_operators(make_xcube(L))["logical_Z"]
        for p in args.ps:
            bw_bf, bw_ps, tr_bf, tr_ps, sc = [], [], [], [], []
            for s in args.seeds:
                rb = evaluate_bw(L, p, args.n_test, s, logical_Z=logical_Z)
                rt = trivial_baseline(L, p, args.n_test, s, logical_Z=logical_Z)
                bw_bf.append(rb["block_failure"]); bw_ps.append(rb["per_sector_ler_mean"])
                tr_bf.append(rt["block_failure"]); tr_ps.append(rt["per_sector_ler_mean"])
                sc.append(rb["syndrome_consistent_frac"])
                print(f"  L{L} p{p} seed{s}: BW bf {rb['block_failure']:.4f} "
                      f"ps {rb['per_sector_ler_mean']:.4f} (sc {rb['syndrome_consistent_frac']:.3f}) "
                      f"| trivial bf {rt['block_failure']:.4f} ps {rt['per_sector_ler_mean']:.4f}",
                      flush=True)
            key = f"L{L}_p{p}"
            out["cells"][key] = {
                "L": L, "p": p,
                "bw": {"block_failure": agg(bw_bf), "per_sector_ler": agg(bw_ps)},
                "trivial": {"block_failure": agg(tr_bf), "per_sector_ler": agg(tr_ps)},
                "syndrome_consistent_frac": agg(sc),
            }
            a = out["cells"][key]
            print(f"  -> {key} AGG: BW bf {a['bw']['block_failure']['mean']:.4f} "
                  f"[{a['bw']['block_failure']['ci_lo']:.4f},{a['bw']['block_failure']['ci_hi']:.4f}] "
                  f"ps {a['bw']['per_sector_ler']['mean']:.4f} | trivial bf "
                  f"{a['trivial']['block_failure']['mean']:.4f} ps {a['trivial']['per_sector_ler']['mean']:.4f}",
                  flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
