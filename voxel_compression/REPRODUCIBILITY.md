# Reproducibility Guide (clean build, 2026-05-02)

This document records the authoritative numerical provenance for
*"Constraint Locality in Learned Syndrome Decoding for 3D Occupancy Grids"*.
Submission tables must use the May 2 clean-build artifacts listed here, not
the historical v4 archive later in this file.

## Clean-Build Definition

The current compression experiments use the corrected fixed constraint
definition:

- Axis parity includes true x/y/z line checks.
- GF(2) row reduction preserves original independent rows instead of using
  Gauss-Jordan-transformed rows as checks.
- Constraint caches are versioned so old left-inverse/nullspace tensors are
  not silently reused.

For a `32^3` grid with fixed axis + plane + global checks, the corrected
matrix has `M = 2,977` syndrome bits and `N = 32,768` variables. Historical
v4 compression results used `M = 2,016` and are not submission evidence.

## Rate Accounting Scope

The paper's rate-distortion table is a nominal spectral-bandwidth diagnostic,
not a completed entropy-coded codec benchmark. The bit totals count:

- `2,977` fixed global syndrome bits;
- `8` coarse occupancy bits;
- `8` nominal bits per retained low-frequency spectral coefficient.

The evaluator reconstructs the shared low-frequency spectral side field used
by the learned proposal and surface-constraint placement. It does not yet
serialize, quantize, or entropy-code a coefficient bitstream. Treat the table
as controlled side-information bandwidth evidence for the learned syndrome
decoding claim.

## Authoritative Artifacts

| Purpose | Location |
|---|---|
| Seed 42 reproduce JSON | `data/clean-v2-may02-rsfix/s42_results.json` |
| Seed 43 reproduce JSON | `data/clean-v2-may02-rsfix/s43_results.json` |
| Seed 44 reproduce JSON | `data/clean-v2-may02-rsfix/s44_results.json` |
| Main aggregate summary | `data/clean-v2-may02-rsfix/aggregate_summary.json` |
| Seed 42 strata JSON | `data/clean-v2-may02-strata/s42_strata.json` |
| Seed 43 strata JSON | `data/clean-v2-may02-strata/s43_strata.json` |
| Seed 44 strata JSON | `data/clean-v2-may02-strata/s44_strata.json` |
| Strata aggregate summary | `data/clean-v2-may02-strata/strata_summary.json` |
| Strata evaluator snapshot | `data/clean-v2-may02-strata/stratify_compression_results.py` |

The corresponding trained checkpoints are:

| Seed | Config | Checkpoint |
|---|---|---|
| 42 | `configs/compression_clean_v2_s42.yaml` | `/work/ablation_ckpts/compression_clean_v2_s42/ckpt_294000.pt` |
| 43 | `configs/compression_clean_v2_s43.yaml` | `/work/ablation_ckpts/compression_clean_v2_s43/ckpt_294000.pt` |
| 44 | `configs/compression_clean_v2_s44.yaml` | `/work/ablation_ckpts/compression_clean_v2_s44/ckpt_294000.pt` |

## Paper Table Mapping

| Paper table | Source |
|---|---|
| Projection diagnosis | `aggregate_summary.json`, key `projection` |
| Constraint locality ablation | `aggregate_summary.json`, key `surface` |
| Nominal rate-distortion diagnostic | `aggregate_summary.json`, key `levels` |
| Occupancy strata | `strata_summary.json`, key `strata` |
| Auxiliary temporal diagnostic | `aggregate_summary.json`, key `surface_bp_accumulation` |

## Reproduction Commands

Run from `parity_diffusion/`.

```bash
python reproduce_results.py \
  --ckpt /work/ablation_ckpts/compression_clean_v2_s42/ckpt_294000.pt \
  --cfg configs/compression_clean_v2_s42.yaml \
  --data data/shapenet/val --val-data data/shapenet/val \
  --n-samples 200 --n-frames 20 \
  --out papers/compression_arxiv/data/clean-v2-may02-rsfix/s42_results.json

python reproduce_results.py \
  --ckpt /work/ablation_ckpts/compression_clean_v2_s43/ckpt_294000.pt \
  --cfg configs/compression_clean_v2_s43.yaml \
  --data data/shapenet/val --val-data data/shapenet/val \
  --n-samples 200 --n-frames 20 \
  --out papers/compression_arxiv/data/clean-v2-may02-rsfix/s43_results.json

python reproduce_results.py \
  --ckpt /work/ablation_ckpts/compression_clean_v2_s44/ckpt_294000.pt \
  --cfg configs/compression_clean_v2_s44.yaml \
  --data data/shapenet/val --val-data data/shapenet/val \
  --n-samples 200 --n-frames 20 \
  --out papers/compression_arxiv/data/clean-v2-may02-rsfix/s44_results.json
```

```bash
python stratify_compression_results.py \
  --seed 42 \
  --ckpt /work/ablation_ckpts/compression_clean_v2_s42/ckpt_294000.pt \
  --cfg configs/compression_clean_v2_s42.yaml \
  --data data/shapenet/val --n-samples 200 --level 4 --n-strata 4 \
  --surface-modes patch,neighbor,gradient \
  --out papers/compression_arxiv/data/clean-v2-may02-strata/s42_strata.json

python stratify_compression_results.py \
  --seed 43 \
  --ckpt /work/ablation_ckpts/compression_clean_v2_s43/ckpt_294000.pt \
  --cfg configs/compression_clean_v2_s43.yaml \
  --data data/shapenet/val --n-samples 200 --level 4 --n-strata 4 \
  --surface-modes patch,neighbor,gradient \
  --out papers/compression_arxiv/data/clean-v2-may02-strata/s43_strata.json

python stratify_compression_results.py \
  --seed 44 \
  --ckpt /work/ablation_ckpts/compression_clean_v2_s44/ckpt_294000.pt \
  --cfg configs/compression_clean_v2_s44.yaml \
  --data data/shapenet/val --n-samples 200 --level 4 --n-strata 4 \
  --surface-modes patch,neighbor,gradient \
  --out papers/compression_arxiv/data/clean-v2-may02-strata/s44_strata.json
```

## Historical v4 Archive

Everything below this line documents the old run family. Keep it only for
audit trail context; do not use it for submission tables.

## Prerequisites

```bash
cd parity_diffusion
pip install torch numpy scipy pyyaml tqdm matplotlib DracoPy
```

### Required artifacts

These artifacts are historical v4 artifacts. For a clean build, retrain a
checkpoint after regenerating constraints with the corrected matrix.

| Artifact | Location | Notes |
|---|---|---|
| Trained checkpoint (w=192) | `ckpts/resnet_shapenet_w192/ckpt_18000.pt` | 25M params, Tables 1--2 |
| ShapeNet val data | `data/shapenet/val/` | ShapeNetVox32 validation split (seed=42, 90/10) |

Table 3 (temporal accumulation) requires **no checkpoint** — it uses
DCT + surface BP only (`reproduce_results.py` Section 9).

### Provenance summary

| Paper table | Model | Script section | Log file |
|---|---|---|---|
| Table 1 (projection diagnosis) | w=192, 25M params | Section 5 | `reproduce_v2.log` |
| Table 2 (constraint ablation) | w=192, 25M params | Section 6 | `reproduce_v2.log` |
| Table 3 (temporal accumulation) | No model (DCT+BP) | Section 9 | (inline) |

### Training configuration

The w=192 model was trained on ShapeNet with the standard config:

| Flag | Effect |
|---|---|
| `disable_x_base=True` | Zero out the canonical-base channel (no GF(2) base solution) |
| `disable_syndrome_cond=True` | Zero out the syndrome conditioning vector |
| `disable_projection=True` | Skip hard left-inverse projection after prediction |

This yields a **sideinfo-only** model: the decoder sees only the coarse
occupancy map and the DCT field, with no parity algebra at inference time.

---

## Authoritative reproduction commands

### Full eval (all DCT levels + surface BP, GPU recommended, ~20 min)

Use a clean-build checkpoint trained with the current syndrome dimension.
The historical v4 checkpoints are not authoritative for the corrected
constraint set.

```bash
python eval_w48.py \
  --ckpt /work/ckpts/resnet_w48_matched/ckpt_48000.pt \
  --data data/shapenet/val \
  --n-samples 200
```

### Full reproduce_results.py (all 8 sections, GPU recommended, ~40 min)

```bash
python reproduce_results.py \
  --ckpt /work/ckpts/resnet_w48_matched/ckpt_48000.pt \
  --data data/shapenet/val --val-data data/shapenet/val \
  --n-samples 200 --n-frames 20
```

Results are saved to `reproduction_results.json`.

### Quick smoke test (CPU, ~10 min)

```bash
CUDA_VISIBLE_DEVICES="" python eval_w48.py \
  --ckpt /work/ckpts/resnet_w48_matched/ckpt_48000.pt \
  --data data/shapenet/val \
  --n-samples 10
```

### Baselines only (no checkpoint, ~5 min)

```bash
python reproduce_results.py --baselines-only --n-samples 200 \
  --data data/shapenet/val
```

---

## Script sections

`reproduce_results.py` has 10 sections:

| Section | Name | Needs |
|---|---|---|
| 1 | Deterministic claims | Nothing |
| 2 | Baseline benchmarks (Draco / Octree / RLE) | ShapeNet val data |
| 3 | Parity codec variable-rate evaluation | Checkpoint + val data |
| 4 | Bayesian temporal accumulation | Checkpoint |
| 5 | Projection diagnostics | Checkpoint + val data |
| 6 | Surface constraint ablation | Checkpoint + val data |
| 7 | Surface Attention evaluation | Surface Attention checkpoint (optional) |
| 8 | Architecture comparison | Multiple checkpoints (optional) |
| 9 | Surface BP temporal accumulation | Nothing (DCT+BP only) |
| 10 | RS text recovery over GF(256) | Nothing (NLTK corpus, ~15 min CPU) |

`eval_w48.py` is a focused script that evaluates the w=48 model at
DCT levels L2, L4, L5, L6 with surface BP refinement at each level.

---

## Claim-by-claim verification

### 1. Syndrome dimensions (Section 3.1)

**Clean-build claim**: M = 2,977 for a 32^3 grid with corrected x/y/z
axis parity, plane parity, global parity, and locality-preserving independent
row selection.

**Historical v4 claim**: M = 2,016. This came from a duplicated-axis
construction plus Gauss-Jordan-transformed rows and is superseded.

**Source**: `parity.py:build_constraint_matrix`

**Reproduced by**: Section 1 of `reproduce_results.py` (function
`verify_deterministic`, item 1a).

**Reproduce standalone**:
```bash
python -c "
from parity import build_constraint_matrix, ConstraintConfig
cfg = ConstraintConfig(axis_parity=True, plane_parity=True, global_parity=True)
H = build_constraint_matrix(32, cfg)
print(f'H shape: {H.shape}')  # expect (2977, 32768)
"
```

**Status**: Deterministic. No data or checkpoint needed.

---

### 2. Model parameters

**Claim**: parameter count at width 48, depth 8 (sideinfo-only config) is
deterministic for a fixed syndrome dimension.

**Source**: `nn/resnet3d.py:ResNet3D`

**Reproduced by**: Section 1, item 1d (verifies multiple widths);
`eval_w48.py` prints parameter count on load.

**Reproduce standalone**:
```bash
python -c "
from parity import build_constraint_matrix, ConstraintConfig
from nn.resnet3d import ResNet3D
cfg = ConstraintConfig(axis_parity=True, plane_parity=True, global_parity=True)
M = build_constraint_matrix(32, cfg).shape[0]
m = ResNet3D(in_ch=3, width=48, depth=8, cond_dim=M)
print(f'{sum(p.numel() for p in m.parameters()):,} params')
"
```

**Status**: Deterministic, but historical exact counts are tied to the old
syndrome dimension.

---

### 3. Variable-rate IoU results (Table / main results)

**Claims** (200 ShapeNet val samples, eval_w48.py):

| Level | Raw IoU | + BP IoU | Syndrome bits |
|---|---|---|---|
| L2 | 0.569 | 0.604 | 13,587 |
| L4 | 0.815 | 0.878 | 4,205 |
| L5 | 0.882 | 0.944 | 3,206 |
| L6 | 0.931 | 0.978 | 2,848 |

**Source**: `eval_w48.py` (authoritative), also reproducible via
`reproduce_results.py` Section 3.

**Code path**:
1. `dataset.py:VoxelDataset` -- loads ShapeNet val data with constraint matrix
2. `adaptive_dct.py:make_adaptive_dct_field` -- constructs DCT side info at each level
3. `nn/resnet3d.py:ResNet3D` -- forward pass (sideinfo-only: x_base=0, cond=0)
4. `surface_constraints.py:build_surface_constraints` -- builds surface parity checks (patch mode)
5. `syndrome_refine.py:refine_with_syndrome` -- BP refinement (30 iterations, damping 0.5)
6. `utils/metrics.py:iou` -- IoU computation

**Key detail**: The sideinfo-only model sets `x_base = zeros` and
`cond = zeros`, so the decoder only receives the coarse map (channel 1)
and the DCT field (channel 2). Surface BP refinement computes the true
syndrome from ground-truth voxels via the surface constraint matrix and
uses model logits as the prior.

**Status**: Fully reproducible. Requires checkpoint + val data.

---

### 4. Surface constraint ablation (Section 4 / Table)

**Claims** (200 samples, L4):

| Method | IoU | Delta | Avg bits |
|---|---|---|---|
| Raw (no projection) | 0.815 | -- | 0 |
| Hard projection | varies | negative | 0 |
| Surface BP: patch | 0.878 | +0.063 | 4,205 |
| Surface BP: neighbor | higher | higher | ~12,800 |
| Surface BP: gradient | higher | higher | ~12,700 |

**Source**: `reproduce_results.py:run_surface_ablation` (Section 6).

**Code path**:
1. `surface_constraints.py:build_surface_constraints` -- builds surface-aware
   parity check matrices in three modes (`patch`, `neighbor`, `gradient`)
2. `syndrome_refine.py:refine_with_syndrome` -- belief propagation refinement
   using the surface constraints and model logits as prior
3. `bp_marginals.py` -- BP message-passing internals
4. `bp_codec.py` -- codec integration for syndrome-based refinement

**Logic**: For each sample, the script runs the model to get raw logits,
then applies post-processing strategies: (1) no projection, (2) hard
left-inverse projection, (3-5) surface-aware BP refinement in three modes.
The surface constraints are built from the DCT field at the current level,
identifying surface voxels (threshold 0.3) and constructing local parity
checks. BP uses model logits as the prior and iteratively refines to
satisfy the surface constraints.

**Status**: Fully reproducible. Requires checkpoint + val data.

---

### 5. Baseline benchmarks (Table 1)

**Claims**:
- Draco qb=4: 3,305 bits, IoU 0.116
- Draco qb=5: 7,019 bits, IoU 1.0
- Octree depth 4: 1,722 bits, IoU 0.612
- RLE (lossless): 3,854 bits, IoU 1.0

**Source**: `reproduce_results.py:run_baselines` (Section 2), which calls
`benchmark_draco.py`.

**Reproduce standalone**:
```bash
python benchmark_draco.py --n-samples 200 --data data/shapenet/val --out benchmark_results.json
```

**Runtime**: ~5 minutes, CPU only, no checkpoint needed.

**Status**: Fully reproducible. Numbers are deterministic given the same
200 val samples (seed=42 selection).

**Data dependency**: `data/shapenet/val/` (ShapeNetVox32 validation split,
downloaded from `http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz`, split
with seed=42, 90/10).

---

### 6. Fill rate (Section 4)

**Claim**: Mean fill rate 8.4% on 200 ShapeNet val samples.

**Source**: Printed by `run_baselines` in Section 2.

**Status**: Reproducible. Same data dependency as claim 5.

---

### 7. DCT bit counts and total bit budgets

**Historical v4 claim**: Level 2 = 216 DCT bits (2,240 total), Level 3 =
1,000 (3,024), Level 4 = 2,744 (4,768). Total = syndrome (2,016) +
coarse map (8) + DCT.

**Clean-build formula**: total bits = current syndrome bits + 8 + DCT bits.
With the corrected fixed constraint set, the deterministic totals are
3,201 (L2), 3,985 (L3), and 5,729 (L4) before any surface-BP variable
syndrome accounting.

**Source**: `adaptive_dct.py:FREQ_LEVELS, FREQ_BITS`. Verified in Section
1, items 1b and 1c.

**Status**: Deterministic. Formula-based. Historical v4 totals are
superseded for submission.

---

### 8. Projection diagnostics (Section 4 / Table)

**Claims**:
- Parity violations on average before projection
- Most flipped bits are wrong (harmful flips)
- IoU drops after hard projection

**Source**: `reproduce_results.py:run_projection_diagnostics` (Section 5).

**Code path**:
1. `parity.py:make_canonical_base` -- builds the base solution from syndrome
2. `nn/resnet3d.py:ResNet3D` -- predicts residual
3. `parity.py:gf2_matmul` -- counts constraint violations
4. `parity.py:project_with_left_inverse` -- hard projection (left-inverse method)
5. IoU and flip analysis computed in the section function

**Key logic**: For each sample, the script computes `recon_before` (model
output XORed with canonical base), counts syndrome violations, then runs
hard projection to get `recon_after`. It tracks which bits were flipped
and whether each flip helped or hurt relative to ground truth.

**Status**: Fully reproducible. Requires checkpoint + val data.

---

### 9. Surface Attention (negative result)

**Claim**: Surface Attention (`nn/surface_attention.py`) was explored as
an alternative decoder architecture but is not competitive with the
matched-parameter ResNet at 32^3 resolution.

**Source**: `reproduce_results.py:run_surface_attention_eval` (Section 7)
and `reproduce_results.py:run_architecture_comparison` (Section 8).

**Code path**: `nn/surface_attention.py:SurfaceAttentionDecoder`

**Details**: The Surface Attention decoder uses patch-based self-attention
over surface voxels identified from the DCT field. At 32^3, the patch
count is small enough that attention overhead does not pay off versus
the ResNet's dense convolutions. Section 8 of `reproduce_results.py`
compares architectures head-to-head at L4 when provided with both
checkpoints.

**Status**: Negative result. Included for completeness. Requires the
Surface Attention checkpoint (`--sa-ckpt` flag).

---

### 10. Bayesian temporal accumulation

**Claim**: Bayesian accumulation over multiple frames improves IoU
relative to independent per-frame reconstruction.

**Source**: `reproduce_results.py:run_bayes_benchmark` (Section 4).

**Code path**: `temporal_bayes.py:BayesAccumulator`

**Status**: Fully reproducible. Requires checkpoint.

---

## Cross-check results (2026-04-12, v3/v4 historical)

This table is retained only as provenance for the superseded run family.

| # | Paper claim | Reproduction | Status |
|---|---|---|---|
| 1 | M = 2,016 | 2016 | Superseded by clean-build M=2,977 |
| 2 | 1.68M params at w=48 d=8 | from `eval_w48.py` / `verify_deterministic` | Exact |
| 3 | Draco qb=4: 3,305 bits, IoU 0.116 | 3305, 0.116 | Exact |
| 4 | RLE: 3,854 bits, IoU 1.0 | 3854, 1.0 | Exact |
| 5 | Octree d=4: 1,722 bits, IoU 0.612 | 1722, 0.612 | Exact |
| 6 | Fill rate 8.4% | 8.4% | Exact |
| 7 | L2=2240, L3=3024, L4=4768 bits | 2240, 3024, 4768 | Superseded by clean-build totals |
| 8 | L2: Raw 0.569, +BP 0.604, 13587 syn bits | `eval_w48.py` output | Reproduced |
| 9 | L4: Raw 0.815, +BP 0.878, 4205 syn bits | `eval_w48.py` output | Reproduced |
| 10 | L5: Raw 0.882, +BP 0.944, 3206 syn bits | `eval_w48.py` output | Reproduced |
| 11 | L6: Raw 0.931, +BP 0.978, 2848 syn bits | `eval_w48.py` output | Reproduced |
| 12 | Surface Attention not competitive at 32^3 | Section 7/8 | Negative result |

| 13 | GF(256) text: global+prior ≤ prior at all error rates | `rs_text_experiment.py` | Reproduced |
| 14 | GF(256) text: local overlap+prior +0.147 at 50% | `rs_text_experiment.py` | Reproduced |
| 15 | GF(256) text: overlap > disjoint at ≥30% | `rs_text_experiment.py` | Reproduced |
| 16 | Theorem 1: P_1(w,p) = w(1-p)p^{w-1} | Analytical | Exact |
| 17 | Theorem 2: k≥2 errors create syndrome ambiguity | Analytical | Exact |

Items 1 and 7 are superseded by the clean-build constraint definition.
The remaining rows are historical provenance and must be rerun before
submission under the corrected code path.
Items 8-11 are reproduced from the authoritative run (200 samples,
w=48 sideinfo-only, ckpt_48000).
Item 12 is a documented negative result.
Items 13-15 are from `rs_text_experiment.py` (200 blocks, seed=42,
NLTK Gutenberg corpus, results in `rs_experiment_results.json`).
Items 16-17 are analytical results (no computation needed).

---

## Quick reproduction commands

**eval_w48.py (primary results, GPU recommended, ~20 min):**
```bash
python eval_w48.py \
  --ckpt /work/ckpts/resnet_w48_matched/ckpt_48000.pt \
  --data data/shapenet/val --n-samples 200
```

**Full reproduce_results.py (all 8 sections, GPU recommended, ~40 min):**
```bash
python reproduce_results.py \
  --ckpt /work/ckpts/resnet_w48_matched/ckpt_48000.pt \
  --data data/shapenet/val --val-data data/shapenet/val \
  --n-samples 200 --n-frames 20
```

**Quick smoke test (CPU, ~10 min):**
```bash
CUDA_VISIBLE_DEVICES="" python eval_w48.py \
  --ckpt /work/ckpts/resnet_w48_matched/ckpt_48000.pt \
  --data data/shapenet/val --n-samples 10
```

**Baselines only (no checkpoint, ~5 min):**
```bash
python reproduce_results.py --baselines-only --n-samples 200 \
  --data data/shapenet/val
```

**RS text recovery (GF(256), CPU only, ~15 min):**
```bash
python rs_text_experiment.py --n-blocks 200 --out rs_experiment_results.json
```

**RS text recovery with LLM prior (requires llama-cpp-python + GGUF model):**
```bash
python rs_text_experiment.py --n-blocks 200 --llm-model /path/to/model.gguf
```

---

## Source file index

| File | Role |
|---|---|
| `reproduce_results.py` | Single reproduction script (8 sections) |
| `eval_w48.py` | Authoritative w=48 eval: all DCT levels + surface BP |
| `parity.py` | Constraint matrix, syndrome computation, projection |
| `adaptive_dct.py` | DCT frequency levels and bit counts |
| `nn/resnet3d.py` | ResNet3D model architecture |
| `nn/surface_attention.py` | Surface Attention decoder (negative result at 32^3) |
| `benchmark_draco.py` | Draco / Octree / RLE baselines |
| `surface_constraints.py` | Surface-aware constraint construction (patch, neighbor, gradient) |
| `syndrome_refine.py` | Belief propagation refinement with surface constraints |
| `bp_marginals.py` | GF(2) BP message-passing marginals |
| `bp_codec.py` | Codec integration for BP-based syndrome refinement |
| `temporal_bayes.py` | Bayesian temporal accumulation |
| `gen_qualitative_figs.py` | Qualitative figure generation |
| `dataset.py` | VoxelDataset loader |
| `gf256.py` | GF(256) finite field arithmetic |
| `rs_text_experiment.py` | RS text recovery experiment (Section 10) |
| `rs_experiment_results.json` | Authoritative 200-block GF(256) results |

---

## Capacity scan plan (scaling-law unification paper, 2026-04-18)

This section describes a *new* experiment being set up for the
**unification paper** that combines QEC and compression under a single
"reconstructive prior scales as `w·depth ≈ α·N³`" claim. It is not part
of the currently-submitted compression paper. See
`parity_diffusion/papers/qec_arxiv/REPRODUCIBILITY.md` §"Checkpoint
provenance audit" for the QEC side.

### Motivation

The compression paper demonstrates that 3D CNNs can invert local binary
parity syndromes. The QEC paper shows the same architectural family
works for surface-code syndrome decoding. The unification claim: both
are instances of the same problem (inferring underlying state from a
sparse binary syndrome on a spatial grid), and the reconstructive prior
obeys a shared capacity scaling law

        w · depth  ≈  α · N³     (N = spatial scale)

To defend this claim with matched methodology, both domains need a
capacity scan at multiple spatial scales with consistent (w, depth)
cells.

### Audit findings (compression side)

Two bugs in the current compression training pipeline prevent a clean
capacity scan as-is:

1. **Checkpoint dir collision.** `train.py` saves to
   `ckpts/{cfg.logging.run_name}/`. `run_name` is human-set via YAML
   with no auto-encoding of (N, w, depth, block_type). Two runs with
   the same config overwrite; a scan varying (N, w, depth) requires
   either a unique YAML per cell (tedious, error-prone) or a new
   launcher that auto-encodes.
2. **Data resolution binding.** `dataset.py:VoxelDataset` does a raw
   `.reshape(cube_size^3)` on binvox files. ShapeNet is voxelized at
   32³ only — varying N for scaling requires multi-resolution data,
   which ShapeNet does not provide at source. Re-voxelizing from the
   original meshes is out of scope.

### Plan

**Data regeneration** — synthetic shapes at multiple resolutions.
`generate_shapes.py` is deterministic given `--seed 42` and supports
`--n` for resolution. Generate:

```
/work/data/shapes_N16/{train,val}/{sphere,box,torus,cylinder,blob,composite}/*.npy
/work/data/shapes_N24/...
/work/data/shapes_N32/...
/work/data/shapes_N48/...
/work/data/shapes_N64/...
```

~5,000 train + 500 val per resolution; same shape distribution across
resolutions. This gives a controlled problem distribution for the
scaling-law fit, cleanly matched to the QEC surface-code setup (fixed
problem shape, varying spatial scale).

**New training entry-point** — `parity_diffusion/syndrome_reconstructor.py`.
Simplified, matched to QEC methodology:
- Input: flat syndrome (reshape into 3D like detector volume)
- Architecture: `nn.resnet3d.ResNet3D` (existing) — same tower as QEC,
  clean (width, depth, block_type) parameters
- Output head: per-voxel occupancy via BCE loss
- No diffusion, no DCT fields, no coarse maps, no surface BP

This is a *controlled* reconstructor for scaling-law measurements, not
a replacement for the compression paper's full system. The complex
production system has too many confounders to cleanly measure
architecture-scaling.

**New launcher** — `bin/compression_job` (analog of `bin/qec_job`):
- Takes `--cube-size N --width W --depth D --block-type --seed --epochs`
- Writes to `/work/compression_runs/N{N}_s{S}_w{W}_dep{DEP}_{block}{suffix}/`
  (collision-proof by construction, same fix as qec_job)
- Drops `run_meta.json` sidecar with every provenance field
- Runs `python syndrome_reconstructor.py --cube-size N ...` in a k8s Job

**Scan launcher** — `bin/compression_capacity_scan`:
- Per-resolution (w, depth) grids matched to the QEC scan structure:

| N  | (w, depth) cells |
|----|---|
| 16 | (16,4), (32,4), (64,8), (128,12) |
| 24 | (32,4), (64,8), (96,12), (128,12) |
| 32 | (64,8), (96,12), (128,12), (192,12) |
| 48 | (96,12), (128,12), (192,12), (256,12) |
| 64 | (128,12), (192,12), (256,12), (384,16) |

- Short training per cell (10–20 epochs) to find the knee
- Same for LC0 and classic block variants (two scans)

**Analysis** — extend `analyze_capacity_scan.py` to accept a `--metric`
flag (`val_ler` for QEC, `val_iou` for compression). Per-domain α fit,
then compare α_QEC vs α_compression.

### Expected outcomes

- **If α_QEC ≈ α_compression**: the scaling law holds at the abstract
  "reconstructive prior for local binary syndromes" level, regardless
  of whether the underlying state is a logical qubit or a voxel grid.
  Strong unification claim.
- **If α's differ materially**: the problem-distribution structure
  affects required capacity. Still publishable — the law characterizes
  "what about the problem forces more prior capacity" — but the
  unification is softer.

### Execution order

1. Let QEC scans (`capacity-scan-apr17` + `capacity-scan-classic-apr18`)
   drain the GPU queue (~2–3 days). No point queueing compression
   cells on top; they'd just sit Pending.
2. While QEC runs: implement `syndrome_reconstructor.py`, `bin/compression_job`,
   `bin/compression_capacity_scan`, extend `analyze_capacity_scan.py`,
   extend `index_checkpoints.py`.
3. Data regen (cheap, can run on the jumpbox CPU in parallel with GPU
   training): `generate_shapes.py` for each N.
4. After QEC queue drains, launch compression scan (LC0 + classic,
   ~40 cells total, similar 2–3 day runtime).
5. Analyze both domains; fit α per (architecture, domain).

### Load-bearing provenance rule

Same as QEC: every number in the unification paper must come from a
single reproducer script, traceable to a specific checkpoint dir with
`run_meta.json` sidecar. No transcription from training logs.
The `index_checkpoints.py` tool will be extended to walk both
`/work/qec_runs/` and `/work/compression_runs/` and emit a unified
index.
