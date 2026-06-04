# Reproducibility Guide: X-cube Fracton Decoder

This directory supports `papers/fracton.pdf`.

## Release Artifacts

| Purpose | Path |
|---|---|
| L=3/4/6 main decoder tables | `data/xcube_L346_gen.json` |
| L=8/10 scaling extension | `data/xcube_L810_gen.json` |
| High-p sweep | `data/xcube_L4_psweep.json` |
| Brown-Williamson baseline provenance | `data/xcube_bw_baseline.json` |
| CSS/rank verification sweep | `data/xcube_verify_all.json` |

## Quick Checks

Run from `fracton_xcube/`:

```bash
python3 verify_fracton_theorem.py
```

The learned-decoder smoke run trains a small model, so it is not instant:

```bash
python3 reproduce_xcube_results.py --smoke --skip-bw --out /tmp/xcube_smoke.json
```

Omit `--skip-bw` only in an environment with `pymatching` installed.

## Full Reproduction Shape

The main L=3/4/6 artifact is produced by:

```bash
python3 reproduce_xcube_results.py \
  --Ls 3 4 6 --ps 0.03 0.05 --seeds 42 43 44 \
  --out data/xcube_L346_gen.json
```

The L=8/10 scaling rows were produced as parallel per-cell runs with
`--skip-bw --tables-only`, then merged with the CPU Brown-Williamson baseline
run. The retained JSONs are the release artifacts; full reruns are substantially
more expensive than the smoke command.

## JSON Sections

`reproduce_xcube_results.py` emits:

- CSS commutation and `k = 6L - 3` verification;
- local-pattern Gate-0 oracle controls;
- seed-bootstrap decoder tables for trivial, BP+OSD, Brown-Williamson, and
  learned A1 decoders;
- paired McNemar comparisons where the same shots are available;
- ablations for global pooling, AxisLineonHead, and CoordConv variants.
