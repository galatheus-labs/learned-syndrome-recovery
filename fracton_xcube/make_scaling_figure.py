#!/usr/bin/env python3
"""Emit fig_scaling.pdf: per-sector LER vs code size L at p=0.03.

Reads the reproduce_xcube_results.py output JSONs (does NOT recompute any
number) and plots the four decoders' per-sector LER with seed-bootstrap 95% CI
error bars. This is the visual of the monotone divergence that Table 1 reports:
the learned decoder improves with L while every baseline degrades.

Usage:
    python make_scaling_figure.py            # writes fig_scaling.pdf
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = "0.03"
LS = [3, 4, 6, 8, 10]

# Which JSON holds each L, and the decoder keys -> display config.
SRC = {
    3: "data/xcube_L346_gen.json", 4: "data/xcube_L346_gen.json",
    6: "data/xcube_L346_gen.json", 8: "data/xcube_L810_gen.json",
    10: "data/xcube_L810_gen.json",
}
SERIES = [
    ("learned_a1", "Learned (AxisLineonHead)", "#1a6fdf", "o", "-"),
    ("trivial",    "Trivial floor",            "#888888", "s", "--"),
    ("bp_osd",     "BP+OSD-0",                 "#d1495b", "^", "-."),
    ("bw_naive",   "BW-naive",                 "#2e8b57", "D", ":"),
]

cache = {f: json.load(open(f))["sections"]["decoder_tables"] for f in set(SRC.values())}

def cell(L):
    return cache[SRC[L]][f"L{L}_p{P}"]

fig, ax = plt.subplots(figsize=(5.0, 3.4))
for key, label, color, marker, ls in SERIES:
    ys, lo, hi = [], [], []
    for L in LS:
        s = cell(L)[key]["per_sector_ler"]
        ys.append(s["mean"]); lo.append(s["mean"] - s["ci_lo"]); hi.append(s["ci_hi"] - s["mean"])
    ax.errorbar(LS, ys, yerr=[lo, hi], label=label, color=color, marker=marker,
                linestyle=ls, linewidth=1.8, markersize=5.5, capsize=2.5, zorder=3)

ax.set_xlabel("Code size $L$")
ax.set_ylabel("Per-sector LER  ($p=0.03$)")
ax.set_yscale("log")
ax.set_xticks(LS)
ax.grid(True, which="both", linewidth=0.4, alpha=0.4)
ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig("fig_scaling.pdf", bbox_inches="tight")
print("wrote fig_scaling.pdf")

# Echo the plotted numbers so they can be eyeballed against Table 1.
for key, label, *_ in SERIES:
    vals = [round(cell(L)[key]["per_sector_ler"]["mean"], 4) for L in LS]
    print(f"  {label:28s} {vals}")
