# Reproducibility Guide (QEC paper)

This document traces every numerical claim in the QEC paper to its source
code and data. The governing rules are in the repo-root
[`CLAUDE.md`](../../../CLAUDE.md) under the *Rigorous paper data* section.

All test-set claims are verified by **`reproduce_qec_results.py`**, which
aggregates per-run JSONs produced by **`eval_test_split.py`**.

The post-training prep plan (phases, tooling TODOs, timeline, paper
structure) lives in the companion [`PAPER_PLAN.md`](PAPER_PLAN.md).

## Current release status (2026-06-04)

The paper's main multi-seed claims are backed by checked-in aggregate and
per-seed JSON artifacts under `data/qec_clean_knee_multiseed.json` and
`data/qec_clean_knee_multiseed_json/`. The d=9 data-arm follow-up is backed by
`data/d9_dataarm_reproduction.json`. The active provenance rule is unchanged:
paper numbers must come from `eval_test_split.py` outputs aggregated by
`reproduce_qec_results.py`, with strict seed coverage for seeds 42, 43, and 44.

The April 2026 audit sections below are retained as history because they explain
why earlier pre-pivot runs were rejected. Statements in those sections saying
that multi-seed results were incomplete are superseded by the June artifacts
listed above.

---

## Tooling

| Script | Role |
|---|---|
| `qec_train_ddp.py` (via `bin/qec_ddp`) | DDP training on alpha+cpm (large models at d=9, d=11) |
| `surface_code_resnet.py` (via `bin/qec_job`) | Single-GPU training on any node (all distances) |
| `eval_test_split.py` | Per-run test-set evaluation, writes JSON per run |
| `reproduce_qec_results.py` | Aggregates JSONs across seeds, emits paper tables |
| `bin/qec_capacity_scan` | Launch a (width, depth) grid per distance for knee-finding |
| `analyze_capacity_scan.py` | Read scan results, emit sizing recommendation |

Architecture: `ResNetDecoder` in `surface_code_resnet.py`, with two
`block_type` variants:
- `classic` — the original block: conv-norm-silu-cond-conv-norm-silu skip.
- `lc0` — LC0-style: pre-activation, dual-output SE (ratio=8), per-channel
  FiLM conditioning, optional ReZero scalar for identity-at-init.

The `zero_init_residual=True` flag:
- For `classic`: zeros `gn2.weight` in every block.
- For `lc0`: adds a per-block learnable `rezero` scalar initialized to zero
  (full residual branch starts at zero → tower is literal identity at init).

Both flags are auto-detected from checkpoint state dicts by `eval_test_split.py`
(via `detect_arch_from_state`), so checkpoints load without additional config.

---

## Methodology — per-distance model sizing

Model capacity is chosen per distance, not held fixed across distances.
A one-size-fits-all model overfits the tiny d=3 syndrome space (~5
stabilizers) while underfitting large distances. Sizing follows an
**information-theoretic scaling argument**, calibrated by an empirical
capacity scan.

### Scaling argument

At distance d with circuit noise, the number of independent detectors is
M ~ d^3 (spatial area d^2 times ~d syndrome-extraction rounds). The
representational capacity of a flat 3D ResNet per forward pass is

        capacity ~ w * M * depth

where w is the channel width. To capture 2-body correlated error
patterns (the regime where neural decoders beat MWPM) we need

        w * M * depth >= M^2           =>       w * depth >= M ~ d^3

So **w * depth scales as alpha * d^3** for some constant alpha.

### Capacity scan (calibrates alpha)

For each distance d ∈ {3, 5, 7, 9, 11} we train a small grid of (w,
depth) pairs at short epochs (10 by default, via `bin/qec_capacity_scan`).
`analyze_capacity_scan.py` reads the per-run `epochs.log`, extracts best
val LER per cell, and recommends the smallest cell whose best val LER is
within 1.1× of the distance's best. The knee cells' w*depth / d^3 gives
an empirical alpha.

The resulting per-distance (w, depth) is used for the main sweep. Not
using a fixed-size model is explicit: the paper will motivate the
sizing rule in the methods section, citing the scaling argument and
showing the calibration data.

### Launching the scan

```bash
bin/qec_capacity_scan                 # full grid (20 runs, tag=capacity-scan-apr17)
bin/qec_capacity_scan --distances 3   # one distance only
```

### Analyzing

```bash
python parity_diffusion/analyze_capacity_scan.py \
  --tag capacity-scan-apr17 --recommend
```

Emits a per-distance table of (w, depth, params, val LER) sorted by
params, then prints the recommended (w, depth) per distance and the
fitted alpha range. Paste the recommendations into the main sweep
launcher.

---

## Methodology — clean test-split protocol

Every run follows the same `eval_test_split.py` protocol to prevent the
usual methodological holes:

1. **Disjoint seeds**: training uses `seed * 1000 + <base>`; val uses
   `seed * 1000 + 500 + i`; test uses `seed * 1000 + 800 + i`. No
   sampler seed ever appears in more than one split.

2. **Predeclared checkpoint selection rule**: pick the training checkpoint
   that minimizes **mean val LER across eval rates, equally weighted**. One
   scalar, one rule, written down before evaluation. No per-rate cherry
   picking.

3. **Paired MWPM comparison**: MWPM is decoded on the same test shots as the
   neural decoder. Reported p-values come from **McNemar's test** (paired
   binary outcomes on identical shots), not an unpaired two-sample test.

4. **95% CIs on every rate**: Wilson interval on each LER, seed-level
   bootstrap on multi-seed aggregates.

---

## Experimental grid

Every cell in every paper table corresponds to a (distance, seed, block_type,
width, depth) tuple. For the paper to be defensible, every cell needs
multi-seed coverage **and** a test_results.json written by `eval_test_split.py`.

Current approach: **run the scaling-law scan first, in isolation**. If the
law `w*depth ≈ α·d³` holds, pick knee (w, depth) per distance, then run the
classic/LC0 sweeps and ablations at the knee. Everything downstream
(paired comparison, ablation, CIs) is gated on the scan producing a clean
fit.

| Table | distances | seeds | block_type | Status |
|---|---|---|---|---|
| Capacity scan (`capscan-clean-*`) | 3, 5, 7, 9, 11 | 42 | classic, lc0 | Seed-42 eval JSONs available for selected cells |
| Classic right-sized main sweep | 3, 5, 7, 9, 11 | 42, 43, 44 | classic | Seeds 43/44 queued under `qec-knee-clean-apr26`; eval pending |
| LC0 right-sized main sweep | 3, 5, 7, 9, 11 | 42, 43, 44 | lc0 | Seeds 43/44 queued under `qec-knee-clean-apr26`; eval pending |
| LC0 vs classic paired | 3, 5, 7, 9, 11 | 42, 43, 44 | both | Requires `eval_paired_comparison.py` (TODO, Phase 2) |
| Ablation | knee(d) | 42, 43, 44 | lc0 component variants | Requires ablation block variants (TODO, Phase 2) |

### Deprecated sweeps (2026-04-18)

- **`lc0-sweep-apr16`** (fixed w=128/d=12 LC0, 15 cells) — killed and
  archived. All seed=42 cells were overwritten by capacity-scan (see
  audit below); seed=43/44 checkpoints exist in
  `/work/qec_runs/_archived_20260418_*` but will not be reused for the
  paper because the overall approach pivoted to per-distance sizing.
- **`classic-parity-apr16`** (fixed w=128/d=12 classic, 15 cells) —
  killed before completion. Only d3_s42 logged 18 epochs. Dir archived.

The prior (non-aug, non-focal) classic sweep stored in memory under
`surface_code_results_apr7.md` is NOT used for the paper — its training
regime differs from the LC0 sweep, which would confound the comparison
(see CLAUDE.md "baseline parity" rule).

---

## Checkpoint provenance audit (2026-04-18)

Snapshot of what exists under `/work/qec_runs` and whether it can defend
each number currently in `main.tex`. The authoritative index lives at
`/work/qec_runs/_index.json`, emitted by
`parity_diffusion/index_checkpoints.py`.

### Bug fixed today

`bin/qec_job` previously computed `results_dir = /work/qec_runs/d{D}_s{S}{suffix}`
without width or depth. Any two jobs at the same `(d, s, suffix)` but
different `(w, depth)` wrote to the same directory and overwrote each
other's `ckpts/` and `run_meta.json`. This silently destroyed:

- **All capacity-scan-apr17 seed=42 cells.** Four `(w, depth)` cells per
  distance collapsed into one surviving checkpoint per distance — whichever
  job finished last.
- **lc0-sweep-apr16 seed=42 cells** at the default (w=128, d=12), since the
  capacity-scan ran over the same directories.

Fix: `results_dir` now includes `_w{W}_dep{DEP}`. Shipped in commit with
`bin/index_checkpoints.py` which detects overwrites by comparing
`run_meta.json` to checkpoint tensor shapes and flags mismatches.

### Legacy classic-block checkpoints (pre-paper runs) — ARCHIVED

Per user directive "start fresh", all legacy dirs (fifteen `d{D}_s{S}/`
classic runs plus seven ad-hoc `ddp_d9_s42_*` experiment dirs and
pre-scan u-net/resnet dirs) were moved to
`/work/qec_runs/_archived_20260418_032904/` and are **not used for any
paper number**. Every paper number will come from post-archive runs
under the fixed `_w{W}_dep{DEP}` naming.

Snapshot of what *was* there before archival (verified against
`torch.load` on `epoch30.pt`):

| d | s=42 w/dep | s=43 w/dep | s=44 w/dep |
|---|---|---|---|
| 3 | 128/12 ✓ | 128/12 ✓ | 128/12 ✓ |
| 5 | 128/12 ✓ | 128/12 ✓ | 128/12 ✓ |
| 7 | 128/12 ✓ | 128/12 ✓ | 128/12 ✓ |
| 9 | **192/12 ✗** | 128/12 ✓ | 128/12 ✓ |
| 11 | 128/12 ✓ | 128/12 ✓ | 128/12 ✓ |

**`d9_s42` is w=192, not w=128.** The dir was overwritten by an ad-hoc
`qec-d9-w192-b64` experiment. The current paper's "−53% at d=9" number,
if computed from this dir, mixes a w=192 seed with w=128 seeds — a
confounded mean that cannot be published.

### Test-set evaluation JSONs (backing for paper tables)

Audit of `/work/qec_runs/d*_s*/test_results.json` against `main.tex` claims:

| Cell | w/dep | neural LER @ p=0.003 | MWPM @ p=0.003 | impr | backing |
|---|---|---|---|---|---|
| d=3 s=42 | — | — | — | — | **missing** |
| d=3 s=43 | 128/12 | 0.00688 | 0.00666 | −3.3% | present |
| d=3 s=44 | 128/12 | 0.00562 | 0.00658 | +14.6% | present |
| d=5 s=42 | 128/12 | 0.00292 | 0.00330 | +11.4% | present (ckpt=epoch10) |
| d=5 s=43 | 128/12 | 0.00336 | 0.00327 | −2.7% | present |
| d=5 s=44 | — | — | — | — | **missing** |
| d=7 s=42 | 128/12 | 0.00123 | 0.00136 | +9.8% | present |
| d=7 s=43 | 128/12 | 0.00110 | 0.00143 | +22.8% | present |
| d=7 s=44 | 128/12 | 0.00112 | 0.00140 | +20.4% | present |
| d=9 s=42 | **192/12** | — | — | — | **ckpt wrong width** |
| d=9 s=43 | 128/12 | — | — | — | **no test JSON** |
| d=9 s=44 | 128/12 | — | — | — | **no test JSON** |
| d=11 s=42 | 128/12 | — | — | — | **no test JSON** |
| d=11 s=43 | 128/12 | — | — | — | **no test JSON** |
| d=11 s=44 | 128/12 | — | — | — | **no test JSON** |

### Reconciliation with main.tex

The current paper table (`main.tex` §4.2) at p=0.003 claims:

- d=3: **mean +8.3%**, range [0.0056, 0.0067] — backed cells: s=43 (−3.3%), s=44 (+14.6%); **s=42 missing**. Two-cell mean = +5.65%, does not match.
- d=5: **mean +6.8%**, range [0.0029, 0.0035] — backed cells: s=42 (+11.4%), s=43 (−2.7%); **s=44 missing**. Two-cell mean = +4.35%, does not match.
- d=7: **mean +17.9%**, range [0.0011, 0.0013] — all three seeds present (+9.8%, +22.8%, +20.4%). Computed mean = +17.67%, matches paper within rounding. **This is the only currently-defensible row.**
- d=9: **mean −53%** — zero backing JSONs; d9_s42 checkpoint is w=192 which could not participate in a w=128 mean.
- d=11: **mean −92%** — zero backing JSONs.

**Conclusion**: 4 of 5 rows in the paper's main result table are not
traceable to `eval_test_split.py` outputs. Numbers were transcribed from
training `epochs.log`/`val.log` at some point, violating the CLAUDE.md
rule. The d=7 row is the only one defensible as written.

### Remediation path to 100% provenance

1. **Capacity scan validates scaling law.** Running now. Expect completion
   in ~2–3 days given 6-GPU throughput and per-cell runtimes.
2. **Pick knee (w, depth) per distance** from `analyze_capacity_scan.py
   --recommend`.
3. **Relaunch both sweeps** (classic and LC0) at the knee size, seeds
   {42, 43, 44}, full 30 epochs. Total 30 runs. Tag them
   `classic-knee-<date>` and `lc0-knee-<date>`.
4. **Run `eval_test_split.py` on every resulting checkpoint** (helper
   script `bin/run_evals` needed — listed as TODO in §"Known gaps").
5. **Rewrite `main.tex` Tables 1–3** from `reproduce_qec_results.py`
   output. No transcription from logs.
6. **Keep legacy test JSONs as archive** under
   `/work/qec_runs/_archived_20260418_*/legacy_test_results/` for
   historical reference but do not cite in paper.

Until step 5 lands, the paper cannot be submitted with the current
numbers.

---

## Per-run evaluation

After each training run finishes, evaluate to JSON:

```bash
python eval_test_split.py \
  --ckpt-dir /work/qec_runs/d9_s42_w128_dep12_btlc0_zi1_aug_f1/ckpts \
  --distance 9 --seed 42 \
  --width 128 --depth 12 \
  --n-test 1000000 --n-test-low-p 10000000 \
  --out test_results/d9_s42_lc0.json
```

JSON schema (abbreviated):
```json
{
  "distance": 9, "seed": 42,
  "width": 128, "depth": 12,
  "block_type": "lc0", "zero_init_residual": true,
  "selected_checkpoint": "qec_d9_epoch22.pt",
  "selection_criterion": "min mean val error across eval rates (equal weight)",
  "test_results": {
    "0.003": {
      "n_shots": 1000000,
      "mwpm_err": 0.000460, "mwpm_ci_95": 0.000042,
      "neural_err": 0.000218, "neural_ci_95": 0.000029,
      "improvement_pct": 52.6,
      "mcnemar_chi2": 18.2, "mcnemar_p": 2.0e-05,
      "mcnemar_b": 234, "mcnemar_c": 452
    },
    ...
  }
}
```

---

## Aggregate reproduction

Once all per-run JSONs land in `test_results/`:

```bash
python reproduce_qec_results.py --results-dir test_results/ \
  --distances 3,5,7,9,11 --seeds 42,43,44 \
  --out qec_paper_aggregates.json
```

Output sections:

- **§1 Deterministic** — architectural invariants (param counts per
  (distance, width, depth, block_type), grid dimensions per distance).
  Independent of trained weights or data.
- **§2 Classic baseline** — per-distance test-set LER (mean ± SEM across
  seeds) at each rate.
- **§3 LC0 results** — same structure as §2.
- **§4 LC0 vs classic** — aggregate comparison table. Note: **this is not a
  paired comparison**. For paired (same-shot) McNemar between LC0 and
  classic, run `eval_paired_comparison.py` (TODO — not yet implemented).
- **§5 Ablation** — per-component attribution (pre-activation, dual-output
  SE, FiLM conditioning, ReZero). Not yet collected; skeleton only.

---

## Provenance chain (per claim)

| Paper claim | Source | Notes |
|---|---|---|
| Architecture | `surface_code_resnet.py:ResNetDecoder` | `block_type` ∈ {classic, lc0} |
| Param counts | `reproduce_qec_results.py` §1 | Derived, no data needed |
| Classic baseline table | `reproduce_qec_results.py` §2 | Aggregates prior sweep JSONs |
| LC0 results table | `reproduce_qec_results.py` §3 | Requires LC0 sweep JSONs |
| LC0 improvement over MWPM | `test_results[rate]['improvement_pct']` per run | Paired on same shots |
| McNemar p-values | `test_results[rate]['mcnemar_p']` per run | Paired on same shots |
| Multi-seed CIs | `reproduce_qec_results.py` bootstrap | Seed-level |

Never transcribe from training `epochs.log` files into the paper — those are
val metrics, not test-set results. The LER numbers in the paper must come from
`eval_test_split.py` via `reproduce_qec_results.py`.

---

## Known gaps and historical audit status

This section contains the April audit trail plus remaining future-work tooling.
Gap 0 below is superseded by the June 2026 `qec_clean_knee_multiseed` artifacts;
the LC0-vs-classic paired comparison and component ablations remain future work.

0. **Superseded historical gap: paper numbers not backed by test_results.json**
   (2026-04-18). This was true at the April pivot and is preserved as an audit
   record. The current paper tables are backed by the June
   `qec_clean_knee_multiseed` artifacts listed at the top of this file.

1. **`eval_paired_comparison.py`** — not yet implemented. Needed for proper
   LC0-vs-classic paired McNemar on identical test shots. Currently §4
   reports aggregate comparison only. See PAPER_PLAN Phase 2.

2. **Ablation block variants** — require `block_type` values that isolate
   each LC0 component (`lc0-no-reze`, `lc0-no-se`, `lc0-post-act`,
   `lc0-no-film`). Today only `classic` and `lc0` exist; components are
   bundled. Not blocking the main tables but blocks §5. See PAPER_PLAN
   Phase 2 and Phase 3.

3. **Batch-eval launcher** — once training finishes, 30+ runs need
   `eval_test_split.py` run per checkpoint. A parallelization helper
   (`bin/run_evals` or similar) would schedule these as k8s Jobs across
   any GPU node. Without it, evaluation becomes a manual chore.

4. **d=3 regime interpretation** — LC0 at w=128/d=12 underperforms MWPM at
   d=3. This is mostly an overparameterization problem: w*depth = 1536 vs
   d^3 = 27 means ~57× capacity overhead, which invites overfitting on the
   tiny syndrome space. The capacity scan + right-sized sweep should
   resolve it. If LC0 (sized via the knee) still loses to MWPM at d=3,
   that becomes an honest limitation in the paper — but the win-everywhere
   story is worth pursuing first. See PAPER_PLAN §"The d=3 regime issue".

Addressing 0–3 is a prerequisite for venue submission.

## Addendum (2026-06-01): d=9 boundary is under-resourcing, not fundamental

A follow-up capacity probe (k8s tag `qec-bnd-may30`) re-ran d=9 with the same
architecture (classic, w=96, depth=12) but 5x training data, evaluated on the
disjoint test split (eval_test_split.py, sampler seed*1000+800), 3 seeds 42/43/44.
The data-arm BEATS MWPM at every rate:

- p=0.003: neural 0.000232 vs MWPM 0.000537 (+56.5%), per-seed McNemar p ~1e-7..1e-11
- p=0.005: neural 0.00346 vs MWPM 0.006785 (+49%)
- p=0.007: neural 0.0201 vs MWPM 0.0312 (+36%)
- p=0.001: neural 1e-6 vs MWPM 3e-6 (too few events for significance)

So the d=9 "capacity boundary" of Table~\ref{tab:boundary} is an
under-training/under-data artifact of the selected knee, not a fundamental limit.
Reproduce: stage the 3 data-arm test_results.json (run dirs
/work/qec_runs/qec-bnd-may30/d9_s{42,43,44}_w96_dep12/) into a results-dir and run
`python reproduce_qec_results.py --results-dir <dir> --distances 9 --seeds
42,43,44 --width 96 --depth 12`. Canonical output: `data/d9_dataarm_reproduction.json`.
