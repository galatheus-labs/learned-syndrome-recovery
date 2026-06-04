# Reproducibility Guide: Surface-code QEC

This directory supports `papers/qec.pdf`, the capacity-boundary study for
spatial convolutional surface-code decoders.

## Release Artifacts

The paper tables are backed by checked-in JSON, not by training logs:

| Purpose | Path |
|---|---|
| Main multi-seed aggregate | `data/qec_clean_knee_multiseed.json` |
| Per-distance/per-seed eval JSONs | `data/qec_clean_knee_multiseed_json/` |
| d=9 data-arm follow-up | `data/d9_dataarm_reproduction.json` |

The larger capacity-scan directories and historical cluster runs are omitted
from this release repository. They were used only to select the retained
right-sized evaluation cells.

## Aggregate Reproduction

Run from `surface_code_qec/`:

```bash
python3 reproduce_qec_results.py --results-dir data/qec_clean_knee_multiseed_json \
  --distances 3,5,7,9,11 --seeds 42,43,44 \
  --out /tmp/qec_reproduction.json
```

The script performs three checks:

- deterministic architecture checks for detector-grid dimensions and parameter
  counts;
- aggregation of the classic and LC0 block eval JSONs across seeds 42, 43, and
  44;
- strict seed-coverage validation unless `--allow-missing-seeds` is supplied.

`stim` and `pymatching` are required for the deterministic surface-code circuit
check. If they are absent, the script still aggregates the checked-in eval JSONs
and reports that Section 1 was skipped.

## Per-run Evaluation

The checked-in JSONs are outputs of `src/eval_test_split.py`, one file per
distance, seed, and block type. The general command shape is:

```bash
python3 src/eval_test_split.py \
  --ckpt-dir /path/to/qec_run/ckpts \
  --distance 9 --seed 42 \
  --width 96 --depth 12 \
  --out data/qec_clean_knee_multiseed_json/d9_s42_lc0.json
```

External checkpoints are intentionally not committed. Re-running these
evaluations requires the trained checkpoint directories used for the paper.

## Provenance Rule

Paper numbers must be derived from `src/eval_test_split.py` outputs aggregated
by `reproduce_qec_results.py`. Do not transcribe from training `epochs.log` or
validation logs into paper tables.
