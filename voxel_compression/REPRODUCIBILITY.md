# Reproducibility Guide: Voxel Compression

This directory supports `papers/compression.pdf`, the constraint-locality study
for learned syndrome reconstruction on 3D occupancy grids.

## Release Artifacts

| Purpose | Path |
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

Only the clean-v2 configs for seeds 42, 43, and 44 are retained in `configs/`.
Historical ablation configs and intermediate sweeps are intentionally omitted
from this release repository.

## Clean-build Definition

The retained results use the corrected fixed constraint definition:

- true x/y/z axis parity checks;
- GF(2) row reduction that preserves original independent rows;
- versioned constraint caches, avoiding stale left-inverse/nullspace tensors.

For a `32^3` grid with fixed axis + plane + global checks, the corrected matrix
has `M = 2,977` syndrome bits and `N = 32,768` variables.

## Quick Check

Run from `voxel_compression/`:

```bash
python3 reproduce_results.py --baselines-only --n-samples 1
```

If ShapeNet validation data are absent, data-dependent baseline sections report
that they were skipped. Deterministic syndrome-dimension and bit-accounting
checks still run.

## Full Reproduction Shape

Full reproduction requires external ShapeNet validation data and trained
checkpoints. The checkpoints are not committed.

```bash
python3 reproduce_results.py \
  --ckpt /path/to/compression_clean_v2_s42/ckpt_294000.pt \
  --cfg configs/compression_clean_v2_s42.yaml \
  --data /path/to/shapenet/val --val-data /path/to/shapenet/val \
  --n-samples 200 --n-frames 20 \
  --out data/clean-v2-may02-rsfix/s42_results.json
```

Repeat with `compression_clean_v2_s43.yaml` and
`compression_clean_v2_s44.yaml` for the other paper seeds.

Occupancy-strata reproduction uses:

```bash
python3 stratify_compression_results.py \
  --seed 42 \
  --ckpt /path/to/compression_clean_v2_s42/ckpt_294000.pt \
  --cfg configs/compression_clean_v2_s42.yaml \
  --data /path/to/shapenet/val --n-samples 200 --level 4 --n-strata 4 \
  --surface-modes patch,neighbor,gradient \
  --out data/clean-v2-may02-strata/s42_strata.json
```

## Rate Accounting Scope

The paper's rate-distortion table is a nominal spectral-bandwidth diagnostic,
not a completed entropy-coded codec benchmark. The bit totals count:

- `2,977` fixed global syndrome bits;
- `8` coarse occupancy bits;
- `8` nominal bits per retained low-frequency spectral coefficient.

The evaluator reconstructs the low-frequency side field used by the learned
proposal and surface-constraint placement. It does not serialize, quantize, or
entropy-code a coefficient bitstream.
