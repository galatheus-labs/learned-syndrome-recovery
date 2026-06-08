#!/usr/bin/env python3
"""Emit fig_threshold.png: learned-decoder LER vs physical error rate p, one
curve per code size L -- the classic threshold-crossing view.

Reads ONLY the checked-in reproduce_xcube_results.py output JSONs (recomputes
nothing). Left panel: block failure (basis-invariant). Right panel: per-sector
LER (basis-relative). A threshold appears as the point where the L-ordering
inverts: below it, larger L => lower error (sub-threshold); above it, larger L
=> higher error (super-threshold).

NOTE ON RESOLUTION: the learned decoder is only available at p in {0.03, 0.05}
for every L (plus a fine sweep at L=4). So for L != 4 each curve is two points;
the crossing estimate is a 2-point linear interpolation, not a dense scan. A
publication-grade threshold plot needs the learned decoder retrained on a fine p
grid at each L (GPU). This figure shows what the released data already implies.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = {3: "data/xcube_L346_gen.json", 4: "data/xcube_L346_gen.json",
       6: "data/xcube_L346_gen.json", 8: "data/xcube_L810_gen.json",
       10: "data/xcube_L810_gen.json"}
PSWEEP = "data/xcube_L4_psweep.json"
LS = [3, 4, 6, 8, 10]
COLORS = {3: "#c7e9b4", 4: "#7fcdbb", 6: "#41b6c4", 8: "#2c7fb8", 10: "#253494"}
MARK = {3: "o", 4: "s", 6: "^", 8: "D", 10: "v"}

cache = {f: json.load(open(f))["sections"]["decoder_tables"] for f in set(SRC.values())}
sweep = json.load(open(PSWEEP))["sections"]["decoder_tables"]


def series(L, metric):
    """Return sorted (ps, means, ci_lo_err, ci_hi_err) for learned decoder at L."""
    cells = {}
    for cell, v in {**cache[SRC[L]], **(sweep if L == 4 else {})}.items():
        cl = int(cell.split("_")[0][1:])
        if cl != L:
            continue
        p = float(cell.split("_p")[1])
        s = v["learned_a1"][metric]
        cells[p] = (s["mean"], s["mean"] - s["ci_lo"], s["ci_hi"] - s["mean"])
    ps = sorted(cells)
    return ps, [cells[p][0] for p in ps], [cells[p][1] for p in ps], [cells[p][2] for p in ps]


def crossing(La, Lb, metric):
    """Linear-interp crossing p between curves La,Lb on the [0.03,0.05] segment."""
    pa, ya, *_ = series(La, metric)
    pb, yb, *_ = series(Lb, metric)
    a0, a1 = ya[pa.index(0.03)], ya[pa.index(0.05)]
    b0, b1 = yb[pb.index(0.03)], yb[pb.index(0.05)]
    denom = (a1 - a0) - (b1 - b0)
    if denom == 0:
        return None
    t = (b0 - a0) / denom
    if 0 <= t <= 1:
        return 0.03 + t * 0.02
    return None


fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
panels = [("block_failure", "Block failure (basis-invariant)", axes[0]),
          ("per_sector_ler", "Per-sector LER (basis-relative)", axes[1])]

for metric, title, ax in panels:
    for L in LS:
        ps, ys, lo, hi = series(L, metric)
        ax.errorbar(ps, ys, yerr=[lo, hi], label=f"$L={L}$", color=COLORS[L],
                    marker=MARK[L], linestyle="-", linewidth=1.8, markersize=6,
                    capsize=2.5, zorder=3)
    # crossing band (use L=4 vs L=10, the cleanest extremes; L=3 has small-code effects)
    xs = [crossing(a, 10, metric) for a in (4, 6, 8)]
    xs = [x for x in xs if x is not None]
    if xs:
        ax.axvspan(min(xs), max(xs), color="0.85", alpha=0.6, zorder=0)
        pc = sum(xs) / len(xs)
        ax.axvline(pc, color="0.4", linestyle=":", linewidth=1.2, zorder=1)
        ax.annotate(f"crossing\n$p^*\\approx{pc:.3f}$", xy=(pc, 0.5),
                    xytext=(pc + 0.012, 0.42), fontsize=8.5, color="0.25",
                    xycoords=("data", "axes fraction"))
    ax.set_xlabel("Physical error rate $p$")
    ax.set_ylabel(title)
    ax.set_xlim(0.025, 0.135)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.4)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", ncol=2)

axes[1].set_yscale("log")
fig.suptitle("Learned X-cube decoder: LER vs $p$ (threshold-crossing view, 3-seed means, 95% CI)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig_threshold.png", dpi=140, bbox_inches="tight")
fig.savefig("fig_threshold.pdf", bbox_inches="tight")
print("wrote fig_threshold.png / .pdf")

for metric, title, _ in panels:
    xs = [crossing(a, 10, metric) for a in (4, 6, 8)]
    xs = [x for x in xs if x is not None]
    print(f"  {title:36s} crossings (Lx vs L10) = {[round(x,4) for x in xs] if xs else 'none in [0.03,0.05]'}")
