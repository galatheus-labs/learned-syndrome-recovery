# Reproducibility Index

Run commands from the module directories unless noted. The checked-in JSONs are
the canonical paper artifacts; full reruns may require GPUs, checkpoints, and
external datasets described in the module-level guides.

## Fracton X-cube

```bash
cd fracton_xcube
python3 verify_fracton_theorem.py
```

The X-cube learned-decoder smoke trains a small model and is not instant. It
also requires `pymatching` unless the Brown-Williamson baseline is skipped:

```bash
python3 reproduce_xcube_results.py --smoke --skip-bw --out /tmp/xcube_smoke.json
```

Paper provenance:

- `fracton_xcube/REPRODUCIBILITY.md`
- `fracton_xcube/data/`

## Gauge Theorem and Negative Result

```bash
cd fracton_xcube
python3 verify_fracton_theorem.py
```

The gauge paper's architecture experiments are retained under
`fracton_xcube/src/gauge_paper/`; see `fracton_xcube/GAUGE_REPRODUCIBILITY.md`.

## Surface-code QEC

```bash
cd surface_code_qec
python3 reproduce_qec_results.py --results-dir data/qec_clean_knee_multiseed_json \
  --distances 3,5,7,9,11 --seeds 42,43,44 \
  --out /tmp/qec_reproduction.json
```

Paper provenance:

- `surface_code_qec/REPRODUCIBILITY.md`
- `surface_code_qec/data/`

## Voxel Compression

```bash
cd voxel_compression
python3 reproduce_results.py --baselines-only
```

Full reproduction requires ShapeNet validation data and trained checkpoints, as
described in `voxel_compression/REPRODUCIBILITY.md`. Checked-in aggregate JSONs
live under `voxel_compression/data/`.

## Dependencies

Install Python dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

The QEC deterministic architecture check requires `stim` and `pymatching`.
Without them, `surface_code_qec/reproduce_qec_results.py` still aggregates the
checked-in paper JSONs and explicitly skips Section 1.
