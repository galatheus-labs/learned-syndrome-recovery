#!/usr/bin/env python3
"""Emit fig_scaling.pdf/png: block and per-sector error vs L at p=0.03.

Reads the reproduce_xcube_results.py output JSONs (does NOT recompute any
number) and plots both metrics with seed-bootstrap 95% CI error bars.
Per-sector error improves at every tested size; block failure improves from
L=4 onward. Models are trained separately at each size. No threshold is inferred.

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
    ("trivial",    "All-zero baseline",        "#888888", "s", "--"),
    ("bp_osd",     "BP+OSD-0",                 "#d1495b", "^", "-."),
    ("bw_naive",   "BW-naive",                 "#2e8b57", "D", ":"),
]

cache = {f: json.load(open(f))["sections"]["decoder_tables"] for f in set(SRC.values())}

def cell(L):
    return cache[SRC[L]][f"L{L}_p{P}"]

fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
for ax, metric, label in zip(axes, ["block_failure", "per_sector_ler"],
                            ["Block failure", "Per-sector LER (fixed basis)"]):
    for key, name, color, marker, ls in SERIES:
        ys, lo, hi = [], [], []
        for L in LS:
            stats = cell(L)[key][metric]
            ys.append(stats["mean"])
            lo.append(stats["mean"] - stats["ci_lo"])
            hi.append(stats["ci_hi"] - stats["mean"])
        ax.errorbar(LS, ys, yerr=[lo, hi], label=name, color=color, marker=marker,
                    linestyle=ls, linewidth=1.8, markersize=5.5, capsize=2.5)
    ax.set_xlabel("Code size $L$")
    ax.set_ylabel(label)
    ax.set_xticks(LS)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.4)
axes[0].set_ylim(0, 1.05)
axes[1].set_yscale("log")
axes[1].legend(frameon=False, fontsize=8, loc="lower left")
fig.suptitle("X-cube finite-size results at $p=0.03$ (three seeds)", fontsize=11)
fig.tight_layout()
fig.savefig("fig_scaling.pdf", bbox_inches="tight")
fig.savefig("fig_scaling.png", dpi=160, bbox_inches="tight")
print("wrote fig_scaling.pdf / .png")

# Echo the plotted numbers so they can be eyeballed against Table 1.
for key, label, *_ in SERIES:
    vals = [round(cell(L)[key]["per_sector_ler"]["mean"], 4) for L in LS]
    print(f"  {label:28s} {vals}")

