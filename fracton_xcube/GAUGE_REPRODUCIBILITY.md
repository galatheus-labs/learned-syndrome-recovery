# Reproducibility Guide: Gauge Structure Paper

This guide supports `papers/gauge.pdf`. The theorem verifier is retained in the
release repository; the larger architecture runs require external ShapeNet data
and cluster training infrastructure.

## Theorem Verification

Run from `fracton_xcube/`:

```bash
python3 verify_fracton_theorem.py
```

Expected checks:

```text
n=4:  rank(H_correct)=37   = 3n^2-3n+1   rank(T)=27  = N-(3n^2-3n+1)   codespace==image(T)
n=6:  rank(H_correct)=91   = 3n^2-3n+1   rank(T)=125 = N-(3n^2-3n+1)   codespace==image(T)
n=8:  rank(H_correct)=169  = 3n^2-3n+1   rank(T)=343 = N-(3n^2-3n+1)   codespace==image(T)
```

The `n=32` arithmetic gives `3*1024 - 96 + 1 = 2977`, matching the
compression paper's corrected syndrome dimension.

## Architecture Runs

The experiment code lives under `src/gauge_paper/`:

- `train_phi_prior_compare.py`: phi-prior, x-prior, and autoencoder comparison
  trainer;
- `gauge_transporter.py`: soft U(1) transporter architecture;
- `phi_prior_decoder.py`: axial-parity prior architecture;
- `experiments.tsv`: experiment ledger.

The retained paper claims are:

| Claim | Source |
|---|---|
| Axial-parity theorem | `verify_fracton_theorem.py` |
| Phi-prior plateau | `experiments.tsv` rows for the phi-prior sweep |
| X-prior baseline | matched ResNet rows in `experiments.tsv` |
| Gauge transporter and ablations | `experiments.tsv` rows exp17, exp22, exp25 |
| Three-seed autoencoder row | `experiments.tsv` rows exp32 seeds 42, 43, 44 |

The exploratory QEC cross-domain scripts from the internal investigation are
not included in this release repo because they are not paper claims. The QEC
paper has its own standalone release path under `surface_code_qec/`.

Private cluster launchers are also omitted. The retained code is the model and
training logic; full reruns require adapting `train_phi_prior_compare.py` to the
target compute environment and dataset location.
