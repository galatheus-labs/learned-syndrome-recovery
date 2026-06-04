# Learned Recovery Over Constraint Systems

This repository accompanies a four-paper release on learned syndrome decoding
across classical 3D compression, surface-code QEC, gauge-structured parity
codes, and X-cube fracton QEC.

The common theme is geometry-aware recovery. A syndrome constrains the answer to
a large equivalence class; a useful decoder must choose the right representative
using an inductive bias matched to the constraint geometry.

## Papers

- `papers/compression.pdf` establishes the constraint-locality principle in 3D
  voxel syndrome reconstruction.
- `papers/gauge.pdf` proves the axial-parity codespace theorem and shows that
  exact gauge enforcement is not automatically a useful neural architecture.
- `papers/qec.pdf` measures the capacity boundary for pure spatial
  convolutional surface-code decoders.
- `papers/fracton.pdf` tests a structure-aware neural decoder on the X-cube
  fracton code.

## Code Layout

- `fracton_xcube/`: X-cube decoder experiments and the axial-parity gauge
  theorem verifier.
- `surface_code_qec/`: surface-code QEC capacity-knee aggregation and
  evaluation code.
- `voxel_compression/`: 3D occupancy-grid syndrome reconstruction experiments.

Each module includes its own reproducibility notes and checked-in result JSONs.
The top-level `REPRODUCIBILITY.md` gives the release-wide command map.

## Status

This repository is initially private while the paper family is finalized. The
intended public release is source code, preprint PDFs, result JSONs, and exact
commands. Large trained checkpoints and external datasets are not committed;
module reproducibility docs identify where those artifacts are expected.
