# Voxel Compression Code

This module contains the code for `papers/compression.pdf`, a study of
constraint locality in learned syndrome decoding for 3D occupancy grids.

## Contents

- `src/`: parity checks, DCT side information, residual network code, BP
  refinement, and evaluation helpers.
- `reproduce_results.py`: paper reproducer for deterministic checks, baselines,
  and checkpoint-dependent metrics.
- `stratify_compression_results.py`: occupancy-quartile stratification.
- `data/`: checked-in aggregate JSON artifacts used by the paper.

## Quick Check

```bash
python3 reproduce_results.py --baselines-only
```

Full reproduction requires ShapeNet validation data and trained checkpoints; see
`REPRODUCIBILITY.md`.
