# Reproducibility — gauge paper

This file lists the exact commands required to reproduce every numeric
claim in `main.tex`. All code lives in
`parity_diffusion/internal/gauge_paper/`. Experiment IDs (e.g. `exp17`,
`qec-ae-d5-s42`) refer to rows of `experiments.tsv`.

> **Scope note.** The shipped note reports only: the theorem (\S3), the
> phi-prior negative result (\S5, Table 1), and the architecture
> comparison (\S6, Table 2: phi-prior / x-prior / soft-U(1) transporter
> + its ablation / multi-scale autoencoder). The DCT rate--distortion
> sweep and the cross-domain QEC experiments are **not reported as
> results** in the note — the QEC material appears only as a deferred,
> hypothesis-level discussion (\S7). The commands for those exploratory
> experiments are retained at the end of this file as provenance, but
> they are not paper claims and are mostly single-seed.

## Theorem (\S3, Table-free)

Closed-form theorem verification on small `n`. Pure-Python F₂ Gaussian
elimination, runs in ~20 s on CPU.

```
cd parity_diffusion/internal/gauge_paper
python3 verify_fracton_theorem.py
```

Expected output (verified):

```
n=4:  rank(H_correct)=37   = 3n^2-3n+1   rank(T)=27  = N-(3n^2-3n+1)   codespace==image(T)
n=6:  rank(H_correct)=91   = 3n^2-3n+1   rank(T)=125 = N-(3n^2-3n+1)   codespace==image(T)
n=8:  rank(H_correct)=169  = 3n^2-3n+1   rank(T)=343 = N-(3n^2-3n+1)   codespace==image(T)
```

The `n=32` arithmetic gives `3*1024 - 96 + 1 = 2977`. The buggy
duplicate-axis construction (the deployed `M=2016` regime) gives
`rank in {28, 66, 120}` at `n in {4,6,8}` — strictly smaller, hence a
different, lower-rank code. This is the bug the theorem caught.

## Phi-prior plateau (\S5, Table 1)

All compression-side cells were fired via `run_compare_job.sh` against
the `ghcr.io/briangu/galareon:main` image on the galatheus k8s cluster.
The launcher stages the gauge_paper folder to `/work/gauge_paper_code`
and submits a Job with the appropriate flags. All numbers are **best
validation IoU**, single-seed per cell; the robustness of the ~0.29
plateau across the 14-cell sweep — not any single cell — is the claim.

```
# smoke1b: tanh-product XOR, 3 epochs, val_iou 0.002
./run_compare_job.sh --tag smoke1 --variant phi_prior --epochs 3 \
    --width 48 --depth 6 --batch-size 8

# exp02: bilinear XOR + bias init -5, 5 epochs, val_iou 0.021
./run_compare_job.sh --tag exp02 --variant phi_prior --width 96 --depth 8 \
    --epochs 5 --lr 5e-4

# exp11: BOTH variants, 20 epochs, w=96, pos_weight=19
#   x_prior val_iou 0.481, phi_prior val_iou 0.283  (matched-config row)
./run_compare_job.sh --tag exp11 --variant both --width 96 --depth 8 \
    --epochs 20 --lr 5e-4 --pos-weight 19

# exp12: phi-prior w=128 d=12 (12M), val_iou 0.276 (capacity not the bottleneck)
./run_compare_job.sh --tag exp12 --variant phi_prior --width 128 --depth 12 \
    --epochs 20 --lr 5e-4 --pos-weight 19

# exp14: phi-prior w=96, 30 epochs, val_iou 0.291 (plateau confirmed)
./run_compare_job.sh --tag exp14 --variant phi_prior --width 96 --depth 8 \
    --epochs 30 --lr 5e-4 --pos-weight 19

# exp15: phi-prior + 8K data + 25 epochs, val_iou 0.292
./run_compare_job.sh --tag exp15 --variant phi_prior --width 96 --depth 8 \
    --epochs 25 --lr 5e-4 --pos-weight 19 --train-subset 8192
```

x-prior baseline rows in Table 1 (5.0M=0.481 [exp11], 12M=0.489 [exp20],
17.9M=0.474 [exp23]) come from the matched ResNet3D runs.

## Architecture comparison (\S6, Table 2)

```
# exp17: soft U(1) gauge_transporter (4.4M), 20ep, val_iou 0.509
./run_compare_job.sh --tag exp17 --variant gauge_transporter --epochs 20 \
    --gt-base-ch 64 --gt-latent-dim 32

# exp22: ABLATION — 4.4M transporter, NO aux gauge losses, val_iou 0.518
#   (aux curvature/transport/latent-recon losses are NOT load-bearing)
./run_compare_job.sh --tag exp22 --variant gauge_transporter --epochs 20 \
    --gt-base-ch 64 --gt-latent-dim 32 --no-gauge-aux

# exp25: ABLATION — drop parallel transport (plain autoencoder, 4.1M), val_iou 0.554
./run_compare_job.sh --tag exp25 --variant latent_autoencoder --epochs 20 \
    --gt-base-ch 64 --gt-latent-dim 32

# exp27/exp32: 16M latent autoencoder, 3-seed verification (the headline row)
./run_compare_job.sh --tag exp32 --variant latent_autoencoder --epochs 30 \
    --gt-base-ch 128 --gt-latent-dim 64 --seed 42  # 0.5764
./run_compare_job.sh --tag exp32 --variant latent_autoencoder --epochs 30 \
    --gt-base-ch 128 --gt-latent-dim 64 --seed 43  # 0.5692
./run_compare_job.sh --tag exp32 --variant latent_autoencoder --epochs 30 \
    --gt-base-ch 128 --gt-latent-dim 64 --seed 44  # 0.5710
```

3-seed autoencoder mean: **0.5722, stdev 0.0038** (the only error-barred
row in Table 2). The phi-prior (0.283), x-prior (0.481), and transporter
(0.509/0.518) rows are single-seed.

Single-seed bottleneck-size sweep (mentioned in text, not tabulated):
`16^3` bottleneck (exp31) 0.542; `4^3` bottleneck (exp30) 0.562; `8^3`
(the default) is the near-optimum.

## Caveats and gaps (paper claims)

- **Validation, not test.** Every reported IoU is best validation IoU on
  a 512-sample held-back ShapeNet split, not a disjoint held-out test
  split. Stated explicitly in `main.tex` \S4 and \S9 (Limitations).
- **Single-seed except the autoencoder headline.** Only the 16M
  autoencoder + no-DCT row (3 seeds {42,43,44} → {0.5764, 0.5692,
  0.5710}) is multi-seed. The x-prior / phi-prior / transporter rows and
  the bottleneck sweep are single-seed (seed 42). The headline
  conclusions rest on gaps far larger than seed noise and on the
  directionality of the two ablations, not on single point estimates.
- **Matched budget, not exactly matched params.** Table 2 rows are
  matched in training budget but differ in parameter count across rows.

---

## Exploratory experiments NOT reported as paper results

The following were run during the investigation but are **not** claims in
the shipped note. They are retained here as provenance only. The QEC
cells in particular are single-seed (except d=5, which is 3-seed but
statistically tied with MWPM at ±8%) and were compared against a
deployed-decoder baseline that was recalled from a different run rather
than measured at matched config — which is exactly why they were
demoted to a hypothesis-level discussion (\S7) with no results table.

### DCT side-info sweep (exploratory)

```
# 2K data, level k:
./run_compare_job.sh --tag expK_2K --variant latent_autoencoder --epochs 30 \
    --gt-base-ch 128 --gt-latent-dim 64 --dct-level <k>
# 8K data, level k:
./run_compare_job.sh --tag expK_8K --variant latent_autoencoder --epochs 20 \
    --gt-base-ch 128 --gt-latent-dim 64 --dct-level <k> --train-subset 8192
```

3-seed cell available: 16M autoencoder + DCT5 + 8K → {0.9318, 0.9313,
0.9298}, mean 0.9310, stdev 0.001. Other DCT cells are single-seed. Not
in the note.

### QEC cross-domain (exploratory, hypothesis only — \S7)

```
./run_qec_compare_job.sh --distance 5 --width 64 --down-stages 2 \
    --epochs 30 --n-per-rate 100000 --n-val 200000 --batch-size 1024 --seed 42
./run_qec_compare_job.sh --distance 7 --width 64 --down-stages 1 \
    --epochs 30 --n-per-rate 100000 --n-val 200000 --batch-size 1024 --seed 42
./run_qec_compare_job.sh --distance 9 --width 64 --down-stages 2 \
    --epochs 30 --n-per-rate 100000 --n-val 200000 --batch-size 512 --seed 42
```

QEC d=5 autoencoder, 3 seeds {42,43,44} → {+13.2%, +3.9%, −3.0%} over
MWPM, mean +4.7%, stdev ≈ 8% (statistically tied). A defensible
cross-domain claim would require a disjoint test split, multi-seed
coverage at every distance, and a deployed-ResNet baseline measured at
matched config — none of which is met, so the note states only the
hypothesis (\S7) and reports no QEC numbers.
