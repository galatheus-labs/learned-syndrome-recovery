# Fracton X-cube and Gauge-Theorem Code

This module contains the code for `papers/fracton.pdf` and the theorem verifier
used by `papers/gauge.pdf`.

## Contents

- `src/fracton/`: X-cube code construction, learned decoder, BP+OSD baseline,
  Brown-Williamson baseline, and reproducibility pipeline.
- `src/gauge_paper/`: axial-parity theorem verifier and gauge-architecture
  negative-result experiment code.
- `data/`: checked-in JSON artifacts used by the fracton paper.

## Quick Checks

```bash
python3 verify_fracton_theorem.py
python3 reproduce_xcube_results.py --smoke --skip-bw --out /tmp/xcube_smoke.json
```

The smoke run trains a small learned decoder and is not instant. Omit
`--skip-bw` only in an environment with `pymatching` installed.

See `REPRODUCIBILITY.md` and `GAUGE_REPRODUCIBILITY.md` for full provenance.
