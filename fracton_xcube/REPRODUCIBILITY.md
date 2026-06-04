# Reproducibility — Fracton X-cube learned decoder

Every numerical claim in `main.tex` comes from one reproducible pipeline built
on `parity_diffusion/internal/fracton/reproduce_xcube_results.py`. For L<=6 this
is the single command below; the L=8/10 scaling rows additionally use per-cell
shards of the same script merged with a parallel CPU BW-naive run (see
"L=8/10 scaling extension" below). Canonical run outputs are checked in under
`parity_diffusion/papers/fracton_arxiv/data/` (e.g. `data/xcube_L346_gen.json`,
`data/xcube_L810_gen.json`).

## One command

```bash
# from parity_diffusion/ with internal/fracton on PYTHONPATH
python internal/fracton/reproduce_xcube_results.py \
  --Ls 3 4 6 --ps 0.03 0.05 --seeds 42 43 44 \
  --out papers/fracton_arxiv/data/xcube_L346_gen.json
```

On the cluster this is the detached job `fracton-xcube-l346-gen` (tag
`fracton-gen-may31`), launched via the standard clone-mode Job manifest on a
single GPU node. Deps beyond the base PyTorch image: `stim pymatching scipy`
(BP+OSD-0 is pure NumPy, no extra solver). Wall-clock is dominated by the L=6
BP+OSD-0 baseline (BP convergence collapses to ~0.03 at L=6, p=0.05). The
default run uses `n_test=40000` for learned/trivial/BW-naive and `bp_test=2000`
for BP+OSD-0.

## What the script verifies / emits (sections of the JSON)

- `verify`: CSS commutation + `k = 6L-3` for each L (machine-checked). The
  paper's `§2` claim that `k = 6L-3` holds for `L ∈ {2,3,4,5,6,8,10}` (`k =
  9,15,21,27,33,45,57`) is sourced from `data/xcube_verify_all.json`, emitted by
  `reproduce_xcube_results.py --verify-only --Ls 2 3 4 5 6 8 10` (CPU-only GF(2)
  rank, no training).
- `gate0_oracle`: local-pattern held-out ML LER vs trivial per (L,p) — the
  empirical radius-1 locality control used for the mechanism analysis.
- `decoder_tables`: per (L,p), 3-seed block-failure and per-sector LER with
  seed-bootstrap 95% CIs for {trivial, bp_osd, bw_naive, learned_a1}.
- `mcnemar`: paired McNemar (learned vs trivial, learned vs BP+OSD-0, learned vs
  BW-naive) on identical shots — chi2, p, b, c, direction. The learned-vs-BW-naive
  comparison is present only for L<=6; at L>=8 BW is compared by aggregate
  failure rate, not paired shot-by-shot.
- `ablation`: block failure / per-sector LER for {globalpool, a1 (AxisLineonHead),
  coordconv, trivial} at L=3, p=0.03, seed 42.
- `win_condition`: predeclared L3_p0.05 non-overlapping-CI check.

## Numbers in the paper → JSON keys

| Paper claim | JSON path |
|---|---|
| Table 1 block failure + CI | `sections.decoder_tables.L{L}_p{p}.{learned_a1,bw_naive,bp_osd,trivial}.block_failure` |
| Table 1 per-sector LER | `…per_sector_ler.mean` |
| McNemar p / b / c (App. A) | `sections.mcnemar.L{L}_p{p}.{learned_vs_trivial,learned_vs_bp,learned_vs_bw_naive}` (`learned_vs_bw_naive` only for L<=6; BW is compared by aggregate failure rate, not paired, at L>=8 — see line 74) |
| Ablation (Table 2) | `sections.ablation.{trivial,globalpool,a1,coordconv}` |
| Gate-0 oracle advantages | `sections.gate0_oracle` |
| Win condition | `win_condition` |

## Status of paper claims

- **Table 1 (all 4 decoders) / Table 2 / oracle / win-condition / App. A (three
  paired-McNemar comparisons incl. learned-vs-BW-naive)**: final, from the single
  consolidated run `data/xcube_L346_gen.json` (job `fracton-l346-bw-rerun`, 3-seed
  {42,43,44}, 2026-06-02). ONE command (above) now emits all four decoders
  {trivial, bp_osd, bw_naive, learned_a1} and all three McNemar comparisons on
  identical shots. Supersedes the earlier split (`fracton-xcube-l346-gen` +
  standalone `xcube_bw_baseline.json` + `reproduce_xcube_bw.py`); learned numbers
  match the earlier run within seed-bootstrap CIs.
- **L=8/L=10 scaling extension (Table 1, App. A rows, scaling text)**:
  `data/xcube_L810_gen.json`. Produced by **task-parallel per-cell GPU shards** —
  `reproduce_xcube_results.py --Ls {8,10} --ps {0.03,0.05} --seeds 42 43 44
  --batch-size {768,384} --n-test 30000 --skip-bw --tables-only` (jobs
  `fracton-cell-l8-p03`/`-p05`, `fracton-cell-l10-p03`/`-p05`, one cell per GPU) —
  **merged** with the parallel HPC BW-naive run `reproduce_xcube_bw_hpc.py`
  (`xcube_L810_bw_hpc.json`). Per-sector LER (learned @ p=0.03): 0.008 (L=8) →
  0.003 (L=10) — the trend continues 0.042→0.035→0.018→0.008→0.003 (L=3,4,6,8,10),
  a ~24× advantage over trivial at L=8 widening to ~66× at L=10. At L≥8, BP+OSD-0
  and BW-naive per-sector exceed the trivial floor; only the learned decoder
  improves on trivial. learned-vs-trivial McNemar (72574, 0) at L=8, (79351, 0)
  at L=10 — c=0 (trivial never once beats learned). Protocol deviations from L≤6
  (noted in the paper): batch 768 (L=8) / 384 (L=10) for memory; n_test 30000;
  BP+OSD-0 on 2000 test shots at L=8 and 300 at L=10 (`--bp-test 300`);
  BW on CPU nodes;
  learned-vs-BW-naive McNemar tabulated only for L≤6 (BW-naive compared by aggregate failure rate, not paired, at L≥8).
- **High-p sweep (§Discussion)**: `data/xcube_L4_psweep.json` (job
  `fracton-xcube-psweep`, `--Ls 4 --ps 0.07 0.09 0.11 0.13 --seeds 42 43 44`,
  n_test 40000, same test split). Per-sector LER (learned/trivial) vs p at L=4:
  0.07 0.198/0.227, 0.09 0.267/0.274, 0.11 0.314/0.315, 0.13 0.350/0.350 — the
  learned advantage decays to a tie by p~0.11–0.13 (~ the 7.5% optimal
  threshold); BP+OSD-0 per-sector 0.42–0.48 throughout. Folded into §Discussion.
- **Brown–Williamson (BW-naive) baseline (Table 1, App. A, §Baselines)**: now
  emitted by the consolidated run into `data/xcube_L346_gen.json`
  (`sections.decoder_tables.*.bw_naive` + `sections.mcnemar.*.learned_vs_bw_naive`);
  standalone `data/xcube_bw_baseline.json` (job `fracton-bw-baseline`) retained for
  provenance and matches it. Naive-weight cube-defect matching
  (`internal/fracton/xcube_bw_decoder.py`, faithful to arXiv:1901.08061 App. B;
  100% syndrome-consistent). Learned beats BW-naive on both metrics at every cell
  (paired McNemar $b\gg c$, $p<10^{-12}$ for $L\le 6$; compared by aggregate
  failure rate, not paired, at $L\ge 8$,
  where BW-naive fails near-totally so no pairing is computed). BW-naive per-sector / block-failure by cell:
  L3 0.098/0.391, 0.204/0.703; L4 0.169/0.682, 0.301/0.937; L6 0.242/0.935, 0.392/0.999.
  For $L\ge 8$ BW-naive is run separately on HPC CPU nodes via
  `internal/fracton/reproduce_xcube_bw_hpc.py` (parallelized over shots) and merged.
- **Transfer to other codes (§Discussion, "future work")**: not started.

## Provenance / honesty notes

- `arch=a1` in the JSON is the `AxisLineonHead` decoder reported in the paper.
- The Gate-0 oracle is a *local-pattern* ML decoder; the learned decoder beats
  its per-sector LER at L≥4 precisely because it is not restricted to local
  patterns — this is the mechanism claim, not a contradiction.
- Block failure is high in absolute terms at large (L,p) because it is the
  all-of-6L−3-sectors-correct event; per-sector LER is the per-logical metric
  and is where the scaling-with-L improvement is clearest.
