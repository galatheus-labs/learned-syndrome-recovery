"""Single reproducible script for the X-cube fracton learned-decoder results.

Mirrors reproduce_qec_results.py. Emits every numerical claim so a reviewer can
verify with one command. Sections:
  1. deterministic verify(L): CSS commutation + k == 6L-3 (from xcube_code).
  2. precondition Gate-0 oracle: local-pattern held-out ML LER vs trivial per (L,p).
  3. per-decoder tables over seeds {42,43,44}: {trivial, BP+OSD, learned A1}
     block-failure (BASIS-INVARIANT headline) + per-sector LER (basis-relative
     diagnostic), with seed-bootstrap CIs.
  4. learned-vs-trivial and learned-vs-BP+OSD paired McNemar on identical shots.
  5. ablation table: {globalpool, a1, coordconv} block-failure.

PREDECLARED WIN CONDITION (write before reading numbers): learned block-failure
< trivial block-failure with non-overlapping seed-bootstrap CIs on >=3 seeds at
L=3, p in {0.05}. A TIE or LOSS is a reportable null result, not a reason to
keep tuning.

Usage (CPU smoke):
  cd parity_diffusion/internal/fracton
  python3 reproduce_xcube_results.py --smoke --out xcube_reproduction.json
Full (design-only / GPU): raise --n-train, --epochs, add --ps and --Ls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "fracton"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # parity_diffusion/ for eval helpers

from xcube_code import make_xcube, logical_operators, verify, sample_xerror_decoding_data
from xcube_decoder import XCubeDecoder, evaluate
from xcube_baseline_bposd import evaluate_bp_osd, trivial_baseline
from xcube_bw_decoder import evaluate_bw
from precondition_oracle import gate0_table

try:
    from eval_test_split import binomial_ci_95, mcnemar_test
except Exception:  # fallback if path import fails
    import math

    def binomial_ci_95(p_hat, n):
        if n <= 0:
            return 0.0
        z = 1.96
        denom = 1 + z * z / n
        center = (p_hat + z * z / (2 * n)) / denom
        half = z * math.sqrt((p_hat * (1 - p_hat) / n) + (z * z / (4 * n * n))) / denom
        return float(max(abs(center - half - p_hat), abs(center + half - p_hat)))

    def mcnemar_test(neural_correct, mwpm_correct):
        b = int((neural_correct & (~mwpm_correct)).sum())
        c = int(((~neural_correct) & mwpm_correct).sum())
        if b + c == 0:
            return 0.0, 1.0, b, c
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        try:
            from scipy.stats import chi2 as chi2_dist
            pv = 1 - chi2_dist.cdf(chi2, df=1)
        except Exception:
            pv = float("nan")
        if b < c:
            pv = 1 - pv
        return float(chi2), float(pv), int(b), int(c)


def seed_bootstrap_ci(vals, iters=10000, seed=0):
    """Bootstrap 95% CI of the mean across a small number of seed-level values."""
    vals = np.asarray(vals, dtype=np.float64)
    if len(vals) <= 1:
        return float(vals.mean()), float(vals.mean()), float(vals.mean())
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(vals, size=len(vals), replace=True).mean()
                      for _ in range(iters)])
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def train_learned(L, p, seed, arch, n_train, n_val, n_test, width, depth,
                  epochs, batch_size, lr, device):
    """Train one learned decoder, predeclared val-epoch selection, return test
    metrics + per-shot correctness (for paired McNemar)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    code = make_xcube(L)
    logical_Z = logical_operators(code)["logical_Z"]
    k = logical_Z.shape[0]

    syn_tr, y_tr = sample_xerror_decoding_data(L, p, n_train, seed, logical_Z)
    syn_va, y_va = sample_xerror_decoding_data(L, p, n_val, seed * 1000 + 500, logical_Z)
    syn_te, y_te = sample_xerror_decoding_data(L, p, n_test, seed * 1000 + 800, logical_Z)

    model = XCubeDecoder(code, logical_Z, width, depth, arch=arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossfn = nn.BCEWithLogitsLoss()

    best_val, best_state = float("inf"), None
    idx = np.arange(n_train)
    for ep in range(epochs):
        model.train()
        np.random.shuffle(idx)
        for i in range(0, n_train, batch_size):
            b = idx[i:i + batch_size]
            xb = torch.from_numpy(syn_tr[b]).to(device)
            yb = torch.from_numpy(y_tr[b]).to(device)
            opt.zero_grad()
            loss = lossfn(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        vm = evaluate(model, syn_va, y_va, device)
        if vm["per_sector_ler_mean"] < best_val:
            best_val = vm["per_sector_ler_mean"]
            best_state = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return evaluate(model, syn_te, y_te, device, return_per_shot=True)


def run(args):
    device = torch.device(args.device)
    out = {"config": vars(args), "sections": {}}

    # ---- 1. deterministic verify ----
    ver = {L: verify(L) for L in args.Ls}
    out["sections"]["verify"] = {str(L): v for L, v in ver.items()}
    print("== Section 1: deterministic verify(L) ==")
    for L, v in ver.items():
        print(f"  L={L}: N={v['N']} k={v['k']} (expected {v['expected_k']}) "
              f"css_ok={v['css_ok']} ok={v['ok']}")
        assert v["ok"], f"verify(L={L}) failed"

    # --verify-only: emit just the deterministic code-parameter verification
    # (CSS commutation + k=6L-3) over the requested --Ls and stop. CPU-cheap
    # (GF(2) rank only, no training); used to source the k-verification claim
    # for L in {2,3,4,5,6,8,10} in the paper and tutorial without a GPU run.
    if getattr(args, "verify_only", False):
        if args.out:
            with open(args.out, "w") as f:
                json.dump(out, f, indent=1)
            print(f"\nwrote {args.out}")
        return out

    # ---- 2. Gate-0 oracle ---- (skipped under --tables-only)
    if not args.tables_only:
        print("\n== Section 2: Gate-0 signal-existence oracle (local-pattern held-out ML) ==")
        g0 = {}
        for L in args.Ls:
            for p in args.ps:
                t = gate0_table(L, p, args.oracle_train, args.oracle_test, 42)
                g0[f"L{L}_p{p}"] = t
                print(f"  L={L} p={p}: ML LER {t['ml_per_sector_ler']:.4f} "
                      f"trivial {t['trivial_per_sector_ler']:.4f} rel {t['rel_improvement_pct']:.1f}%")
        out["sections"]["gate0_oracle"] = g0

    # ---- 3. per-decoder tables over seeds ----
    print("\n== Section 3: per-decoder tables over seeds {} ==".format(args.seeds))
    tables = {}
    pershot = {}  # (L,p) -> {seed: {decoder: block_correct}}
    for L in args.Ls:
        code = make_xcube(L)
        logical_Z = logical_operators(code)["logical_Z"]
        for p in args.ps:
            key = f"L{L}_p{p}"
            triv_bf, triv_ps, bp_bf, bp_ps, bp_conv = [], [], [], [], []
            learn_bf, learn_ps, bw_bf, bw_ps = [], [], [], []
            pershot[key] = {}
            for s in args.seeds:
                tr = trivial_baseline(L, p, args.n_test, s, logical_Z)
                bp = evaluate_bp_osd(L, p, args.bp_test, s, logical_Z,
                                     max_iter=args.bp_iter, return_per_shot=True)
                # BW-naive on the SAME n_test shots (sampler seed*1000+800) as the
                # learned decoder -> paired McNemar (learned vs BW-naive) is exact.
                # --skip-bw runs it separately on HPC CPU nodes (single-threaded, the
                # wall-clock long pole at L>=8); see reproduce_xcube_bw_hpc.py.
                bw = None if args.skip_bw else evaluate_bw(
                    L, p, args.n_test, s, logical_Z, return_per_shot=True)
                ln = train_learned(L, p, s, "a1", args.n_train, args.n_val, args.n_test,
                                   args.width, args.depth, args.epochs, args.batch_size,
                                   args.lr, device)
                triv_bf.append(tr["block_failure"]); triv_ps.append(tr["per_sector_ler_mean"])
                bp_bf.append(bp["block_failure"]); bp_ps.append(bp["per_sector_ler_mean"])
                bp_conv.append(bp["bp_convergence_frac"])
                if bw is not None:
                    bw_bf.append(bw["block_failure"]); bw_ps.append(bw["per_sector_ler_mean"])
                learn_bf.append(ln["block_failure"]); learn_ps.append(ln["per_sector_ler_mean"])
                pershot[key][s] = {
                    "trivial": tr["block_correct"],
                    "bp": bp["block_correct"],
                    "learned": ln["block_correct"],
                }
                if bw is not None:
                    pershot[key][s]["bw_naive"] = bw["block_correct"]
                bwstr = ("" if bw is None else
                         f"BW-naive bf {bw['block_failure']:.4f} "
                         f"(ps {bw['per_sector_ler_mean']:.4f}) | ")
                print(f"  {key} seed {s}: learned bf {ln['block_failure']:.4f} "
                      f"(ps {ln['per_sector_ler_mean']:.4f}) | {bwstr}"
                      f"trivial bf {tr['block_failure']:.4f} "
                      f"(ps {tr['per_sector_ler_mean']:.4f}) | BP+OSD bf {bp['block_failure']:.4f} "
                      f"(ps {bp['per_sector_ler_mean']:.4f}, conv {bp['bp_convergence_frac']:.3f})")

            def agg(vals):
                m, lo, hi = seed_bootstrap_ci(vals)
                return {"mean": m, "ci_lo": lo, "ci_hi": hi, "vals": list(map(float, vals))}

            tables[key] = {
                "trivial": {"block_failure": agg(triv_bf), "per_sector_ler": agg(triv_ps)},
                "bp_osd": {"block_failure": agg(bp_bf), "per_sector_ler": agg(bp_ps),
                           "bp_convergence_frac": agg(bp_conv)},
                "learned_a1": {"block_failure": agg(learn_bf), "per_sector_ler": agg(learn_ps)},
            }
            if not args.skip_bw:
                tables[key]["bw_naive"] = {"block_failure": agg(bw_bf),
                                           "per_sector_ler": agg(bw_ps)}
            tb = tables[key]
            print(f"    -> {key} AGG: learned bf {tb['learned_a1']['block_failure']['mean']:.4f} "
                  f"[{tb['learned_a1']['block_failure']['ci_lo']:.4f},{tb['learned_a1']['block_failure']['ci_hi']:.4f}]  "
                  f"trivial bf {tb['trivial']['block_failure']['mean']:.4f} "
                  f"[{tb['trivial']['block_failure']['ci_lo']:.4f},{tb['trivial']['block_failure']['ci_hi']:.4f}]  "
                  f"BP+OSD bf {tb['bp_osd']['block_failure']['mean']:.4f}")
    out["sections"]["decoder_tables"] = tables

    # ---- 4. paired McNemar (per-sector pooled across seeds on identical shots) ----
    print("\n== Section 4: paired McNemar (learned vs trivial, learned vs BP+OSD) ==")
    mcn = {}
    for key, perseed in pershot.items():
        ln_all = np.concatenate([perseed[s]["learned"] for s in args.seeds])
        tr_all = np.concatenate([perseed[s]["trivial"] for s in args.seeds])
        # BP eval uses bp_test shots (subset); align by min length per seed
        bp_chunks, ln_bp_chunks = [], []
        for s in args.seeds:
            nbp = len(perseed[s]["bp"])
            bp_chunks.append(perseed[s]["bp"])
            ln_bp_chunks.append(perseed[s]["learned"][:nbp])
        bp_all = np.concatenate(bp_chunks)
        ln_bp_all = np.concatenate(ln_bp_chunks)
        chi_t, p_t, b_t, c_t = mcnemar_test(ln_all, tr_all)
        chi_b, p_b, b_b, c_b = mcnemar_test(ln_bp_all, bp_all)
        mcn[key] = {
            "learned_vs_trivial": {"chi2": chi_t, "p": p_t, "b": b_t, "c": c_t,
                                   "direction": "learned better" if b_t > c_t else "trivial better"},
            "learned_vs_bp": {"chi2": chi_b, "p": p_b, "b": b_b, "c": c_b,
                              "direction": "learned better" if b_b > c_b else "BP+OSD better"},
        }
        bwstr = ""
        if not args.skip_bw:
            # BW-naive uses the same full n_test shots as learned -> directly paired.
            bw_all = np.concatenate([perseed[s]["bw_naive"] for s in args.seeds])
            chi_w, p_w, b_w, c_w = mcnemar_test(ln_all, bw_all)
            mcn[key]["learned_vs_bw_naive"] = {
                "chi2": chi_w, "p": p_w, "b": b_w, "c": c_w,
                "direction": "learned better" if b_w > c_w else "BW-naive better"}
            bwstr = (f" | learned-vs-BW b={b_w} c={c_w} p={p_w:.2e} "
                     f"({mcn[key]['learned_vs_bw_naive']['direction']})")
        print(f"  {key}: learned-vs-trivial b={b_t} c={c_t} chi2={chi_t:.2f} p={p_t:.2e} "
              f"({mcn[key]['learned_vs_trivial']['direction']}) | "
              f"learned-vs-BP b={b_b} c={c_b} p={p_b:.2e} "
              f"({mcn[key]['learned_vs_bp']['direction']}){bwstr}")
    out["sections"]["mcnemar"] = mcn

    # ---- 5. ablation table ---- (skipped under --tables-only)
    if not args.tables_only:
        print("\n== Section 5: ablation {globalpool, a1, coordconv} (block-failure, seed 42) ==")
        abl = {}
        L_ab = args.Ls[0]
        p_ab = args.ps[0]
        code = make_xcube(L_ab)
        logical_Z = logical_operators(code)["logical_Z"]
        triv = trivial_baseline(L_ab, p_ab, args.n_test, 42, logical_Z)
        for arch in args.ablation_archs:
            ln = train_learned(L_ab, p_ab, 42, arch, args.n_train, args.n_val, args.n_test,
                               args.width, args.depth, args.epochs, args.batch_size, args.lr, device)
            abl[arch] = {"block_failure": ln["block_failure"],
                         "per_sector_ler_mean": ln["per_sector_ler_mean"]}
            print(f"  arch {arch:>10}: block_failure {ln['block_failure']:.4f} "
                  f"per-sector LER {ln['per_sector_ler_mean']:.4f} (trivial bf {triv['block_failure']:.4f})")
        abl["trivial"] = {"block_failure": triv["block_failure"],
                          "per_sector_ler_mean": triv["per_sector_ler_mean"]}
        out["sections"]["ablation"] = abl

    # ---- predeclared win-condition check at L=3,p=0.05 if present ----
    win_key = "L3_p0.05"
    if win_key in tables and not args.tables_only:
        tb = tables[win_key]
        learned_hi = tb["learned_a1"]["block_failure"]["ci_hi"]
        triv_lo = tb["trivial"]["block_failure"]["ci_lo"]
        won = learned_hi < triv_lo
        out["win_condition"] = {
            "key": win_key, "learned_bf_ci_hi": learned_hi, "trivial_bf_ci_lo": triv_lo,
            "non_overlapping_learned_below_trivial": bool(won),
        }
        print(f"\n== PREDECLARED WIN CONDITION ({win_key}) ==")
        print(f"  learned block-failure CI hi {learned_hi:.4f} < trivial CI lo {triv_lo:.4f} ? {won}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Reproduce X-cube fracton learned-decoder results")
    ap.add_argument("--Ls", type=int, nargs="+", default=[3])
    ap.add_argument("--ps", type=float, nargs="+", default=[0.03, 0.05])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-train", type=int, default=300000)
    ap.add_argument("--n-val", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=40000)
    ap.add_argument("--bp-test", type=int, default=2000, help="BP+OSD shots (slower; subset)")
    ap.add_argument("--bp-iter", type=int, default=64)
    ap.add_argument("--oracle-train", type=int, default=200000)
    ap.add_argument("--oracle-test", type=int, default=20000)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ablation-archs", type=str, nargs="+",
                    default=["globalpool", "a1", "coordconv"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--verify-only", action="store_true",
                    help="Emit only the deterministic code-parameter verification "
                         "(CSS commutation + k=6L-3) over --Ls, then stop. CPU-only "
                         "(GF(2) rank, no training). Sources the k-verification claim "
                         "for the full set of simulated sizes.")
    ap.add_argument("--tables-only", action="store_true",
                    help="Run only verify + decoder tables + McNemar (skip the Gate-0 "
                         "oracle, the ablation, and the win-condition). For task-parallel "
                         "per-cell sharding across GPUs: each cell job stays lean and its "
                         "JSON merges into the full grid. Oracle/ablation come from the "
                         "main L<=6 run.")
    ap.add_argument("--skip-bw", action="store_true",
                    help="Skip the BW-naive baseline (single-threaded, the wall-clock "
                         "long pole at L>=8). Run it separately on HPC CPU nodes via "
                         "reproduce_xcube_bw_hpc.py and merge. Omits bw_naive from "
                         "decoder_tables and the learned_vs_bw_naive McNemar.")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU-verified-feasible config: L=3, p=0.03 (most signal), "
                         "cheap backbone (w16d2), ~36s/epoch on CPU. Reproduces the "
                         "smoke numbers in the writeup. Each cell ~6min CPU; the full "
                         "3-seed run with ablation is ~50min CPU.")
    args = ap.parse_args()
    if args.smoke:
        # Verified CPU-feasible (~36s/epoch). p=0.03 is the regime with the most
        # decodable signal (Gate-0 oracle: ~34% rel headroom over trivial); the
        # signal collapses by p~0.07, so this is the predeclared smoke regime.
        args.Ls = [3]
        args.ps = [0.03]
        args.n_train = 60000
        args.n_val = 10000
        args.n_test = 20000
        args.bp_test = 1500
        args.width = 16
        args.depth = 2
        args.epochs = 10
        args.batch_size = 4096
        args.lr = 2e-3
    run(args)


if __name__ == "__main__":
    main()
