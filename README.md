# Geometry-Aware Syndrome Decoding

This repository contains a four-paper research program connecting classical
binary syndrome source coding, 3D voxel compression, fracton gauge structure,
and topological quantum error correction (QEC).

The common problem is recovery from constraints. A syndrome rarely specifies a
single answer; it defines a large equivalence class. The practical question is
how to choose the right representative. Across these papers, the answer is that
geometry matters: neural proposals help when their inductive bias matches the
constraint geometry, while hard global algebra can be actively harmful when it
destroys local spatial information.

![X-cube scaling plot: learned AxisLineonHead decoder per-sector LER improves with code size at p = 0.03](papers/figures/fracton/fig_scaling.png)

## Executive Summary

Across four companion papers, we study how algebraic constraints and learned
neural proposals interact on 3D lattices:

1. **Fracton QEC scaling.** We introduce a structure-aware neural decoder for
   the X-cube fracton code using an `AxisLineonHead` line-aggregation layer. In
   the retained scaling experiment, learned per-sector logical error rate
   improves from `L=3` through `L=10` at `p=0.03`, while standard baselines do
   not show the same scaling trend.
2. **Constraint locality.** In 3D voxel compression, high-weight global parity
   projections can degrade strong learned spatial proposals. Local overlapping
   surface checks and targeted belief propagation provide the useful
   refinements.
3. **Gauge bridge.** We prove that the corrected axial-parity voxel codec has
   codespace `image(d_x d_y d_z)`, a discrete `Z_2` fracton gauge structure.
   This explains why the compression and fracton-QEC settings share more than a
   superficial syndrome-decoding form.
4. **Surface-code capacity boundaries.** We map where pure spatial 3D
   convolutional decoders beat paired MWPM in the studied surface-code setting,
   and where the same architectural family runs into data/capacity boundaries.

## Research Narrative

The papers are intended to be read as a sequence:

| Paper | Setting | Main Question | Core Lesson |
|---|---|---|---|
| [`compression.pdf`](papers/compression.pdf) | 3D voxel occupancy grids | Can a neural decoder reconstruct from parity syndromes and side information? | Local constraint structure improves learned proposals; global projections can erase useful geometry. |
| [`gauge.pdf`](papers/gauge.pdf) | Axial parity codes | What algebraic structure explains the codec? | The codespace is exactly a fracton-like `Z_2` gauge structure, but exact gauge enforcement is not automatically a good architecture. |
| [`fracton.pdf`](papers/fracton.pdf) | X-cube fracton QEC | Can a learned decoder exploit subdimensional logical structure? | A line-aware neural readout can scale favorably with code size where generic baselines struggle. |
| [`qec.pdf`](papers/qec.pdf) | Surface-code QEC | How far do pure spatial CNN decoders go against MWPM? | Right-sized spatial decoders can beat MWPM in some regimes, but the capacity boundary is visible and measurable. |

Together, the papers argue for a disciplined version of learned syndrome
decoding: keep the algebraic constraints visible, but let the model use the
right geometric representation instead of forcing exact global structure into
the architecture.

## Papers and Pipelines

### 1. Structure-Aware Neural Decoding for the X-Cube Fracton Code

- Paper: [`papers/fracton.pdf`](papers/fracton.pdf)
- Code: [`fracton_xcube/`](fracton_xcube/)
- Tutorial: [`papers/tutorials/fracton.html`](papers/tutorials/fracton.html)
- Example: [`papers/examples/fracton_examples.html`](papers/examples/fracton_examples.html)

Core idea: the X-cube code has subdimensional lineon structure. The decoder uses
an `AxisLineonHead` to aggregate evidence along the logical line geometry rather
than flattening the problem into a generic 3D classifier.

```bash
cd fracton_xcube
python3 reproduce_xcube_results.py --verify-only --Ls 2 3 --out /tmp/xcube_verify.json
```

The learned-decoder smoke run is more expensive because it trains a small model:

```bash
cd fracton_xcube
python3 reproduce_xcube_results.py --smoke --skip-bw --out /tmp/xcube_smoke.json
```

### 2. Sizing and Capacity Boundaries for Surface-Code QEC

- Paper: [`papers/qec.pdf`](papers/qec.pdf)
- Code: [`surface_code_qec/`](surface_code_qec/)
- Tutorial: [`papers/tutorials/qec.html`](papers/tutorials/qec.html)

Core idea: pure spatial 3D convolutional decoders have measurable capacity
knees. The retained multi-seed results compare classic and LC0-style residual
blocks against paired MWPM across distances and physical error rates.

```bash
cd surface_code_qec
python3 reproduce_qec_results.py --results-dir data/qec_clean_knee_multiseed_json \
  --distances 3,5,7,9,11 --seeds 42,43,44 \
  --out /tmp/qec_reproduction.json
```

### 3. Gauge Structure and Architectural Inductive Bias

- Paper: [`papers/gauge.pdf`](papers/gauge.pdf)
- Code: [`fracton_xcube/src/gauge_paper/`](fracton_xcube/src/gauge_paper/)
- Tutorial: [`papers/tutorials/gauge.html`](papers/tutorials/gauge.html)

Core idea: the corrected axial-parity codec has a clean gauge-theoretic
description, but hard-coding that exact algebraic structure into a neural
architecture underperforms a more flexible multi-scale bottleneck prior.

```bash
cd fracton_xcube
python3 verify_fracton_theorem.py
```

### 4. Constraint Locality in Learned Syndrome Decoding

- Paper: [`papers/compression.pdf`](papers/compression.pdf)
- Code: [`voxel_compression/`](voxel_compression/)
- Tutorial: [`papers/tutorials/compression.html`](papers/tutorials/compression.html)

Core idea: the useful constraints are local, overlapping, and content-aware.
Large global row checks are algebraically valid but weak for local neural
reconstruction.

```bash
cd voxel_compression
python3 reproduce_results.py --baselines-only --n-samples 1 \
  --out /tmp/voxel_reproduction.json
```

Full compression reproduction requires external ShapeNet validation data and
trained checkpoints; the checked-in JSONs are the paper artifacts.

## Repository Layout

```text
papers/
  fracton.pdf
  qec.pdf
  gauge.pdf
  compression.pdf
  tutorials/
  examples/
  figures/

fracton_xcube/
  reproduce_xcube_results.py
  make_scaling_figure.py
  verify_fracton_theorem.py
  src/
  data/

surface_code_qec/
  reproduce_qec_results.py
  src/
  data/

voxel_compression/
  reproduce_results.py
  stratify_compression_results.py
  configs/
  src/
  data/
```

## Reproducibility

Install Python dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

The top-level command map is [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). Each
module has its own compact reproducibility guide:

- [`fracton_xcube/REPRODUCIBILITY.md`](fracton_xcube/REPRODUCIBILITY.md)
- [`fracton_xcube/GAUGE_REPRODUCIBILITY.md`](fracton_xcube/GAUGE_REPRODUCIBILITY.md)
- [`surface_code_qec/REPRODUCIBILITY.md`](surface_code_qec/REPRODUCIBILITY.md)
- [`voxel_compression/REPRODUCIBILITY.md`](voxel_compression/REPRODUCIBILITY.md)

Large trained checkpoints and external datasets are not committed. The release
repository contains source code, preprint PDFs, tutorials, selected figures,
canonical result JSONs, and exact commands for the retained paper artifacts.

## Why This Matters

This program sits at the intersection of scientific machine learning, error
correcting codes, quantum fault tolerance, and 3D geometric inference. The
technical theme is deliberately narrow and reusable: when recovery is defined
by algebraic constraints, the winning architecture is often the one that
respects the geometry of the constraint system without mistaking algebraic
exactness for a useful learning bias.
