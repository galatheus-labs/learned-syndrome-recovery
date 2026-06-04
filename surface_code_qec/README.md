# Surface-code QEC Code

This module contains the code for `papers/qec.pdf`, a multi-seed
sizing-and-boundary study of pure 3D convolutional surface-code decoders.

## Contents

- `src/surface_code_resnet.py`: decoder architecture and data generation.
- `src/eval_test_split.py`: per-run test-set evaluation.
- `reproduce_qec_results.py`: aggregate reproducer for paper tables.
- `data/`: checked-in JSON artifacts used by the paper.

## Aggregate Reproduction

```bash
python3 reproduce_qec_results.py --results-dir data/qec_clean_knee_multiseed_json \
  --distances 3,5,7,9,11 --seeds 42,43,44 \
  --out /tmp/qec_reproduction.json
```

Full training/evaluation requires the environment described in
`REPRODUCIBILITY.md`.
