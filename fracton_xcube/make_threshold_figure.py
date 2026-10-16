#!/usr/bin/env python3
"""Emit fig_threshold.png: learned-decoder LER vs physical error rate p, one
curve per code size L. The historical filename is retained for compatibility.

Reads ONLY the checked-in reproduce_xcube_results.py output JSONs (recomputes
nothing). Left panel: block failure (basis-invariant). Right panel: per-sector
LER (basis-relative). Points are measurements; dashed connectors are guides.

NOTE ON RESOLUTION: the learned decoder is only available at p in {0.03, 0.05}
for every L (plus a fine sweep at L=4). So for L != 4 each curve is two points;
these points cannot resolve a threshold. Establishing a threshold requires
a dense common p grid and a finite-size analysis. No crossing band is drawn.
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


fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
panels = [("block_failure", "Block failure (basis-invariant)", axes[0]),
          ("per_sector_ler", "Per-sector LER (basis-relative)", axes[1])]

for metric, title, ax in panels:
    for L in LS:
        ps, ys, lo, hi = series(L, metric)
        ax.errorbar(ps, ys, yerr=[lo, hi], label=f"$L={L}$", color=COLORS[L],
                    marker=MARK[L], linestyle="--", linewidth=1.3, markersize=6,
                    capsize=2.5, zorder=3)
    ax.set_xlabel("Physical error rate $p$")
    ax.set_ylabel(title)
    ax.set_xlim(0.025, 0.135)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.4)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", ncol=2)

axes[1].set_yscale("log")
fig.suptitle("X-cube noise dependence: measured points, dashed guides (three seeds)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig_threshold.png", dpi=140, bbox_inches="tight")
fig.savefig("fig_threshold.pdf", bbox_inches="tight")
print("wrote fig_threshold.png / .pdf")

print("No threshold estimate: all sizes except L=4 have only two noise points.")

