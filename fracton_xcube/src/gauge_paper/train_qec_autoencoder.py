"""
QEC autoencoder training: drop-in replacement for surface_code_resnet's
ResNet3D backbone with our multi-scale LatentEncoderQEC.

Hypothesis test: does the multi-scale latent + smoothness prior that wins
for sparse-binary voxel reconstruction also help QEC syndrome decoding?

Strategy: monkey-patch surface_code_resnet.ResNetDecoder to point at our
LatentEncoderQEC, then call surface_code_resnet.train_mixed() as-is. This
reuses *all* the existing data generation, evaluation, MWPM baseline, and
training-loop logic — only the model class changes.

Usage:
  python train_qec_autoencoder.py --distance 5 --width 64 --down-stages 2 \
      --epochs 40 --n-per-rate 100000 --seed 42 --save-dir ckpts/qec_ae_d5_s42
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Resolve the parity_diffusion root via env var or directory walk (mirrors
# the compression-side setup in train_phi_prior_compare.py).
_THIS = Path(__file__).resolve()
_DEFAULT_PARITY_ROOT = _THIS.parent.parent.parent
_PARITY_ROOT = Path(os.environ.get("GALAREON_PD_ROOT", str(_DEFAULT_PARITY_ROOT)))
if not (_PARITY_ROOT / "parity.py").exists():
    _alt = Path("/app/parity_diffusion")
    if (_alt / "parity.py").exists():
        _PARITY_ROOT = _alt
if str(_PARITY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PARITY_ROOT))
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))
print(f"[setup] parity_diffusion root: {_PARITY_ROOT}", flush=True)

import surface_code_resnet as scr  # noqa: E402

from qec_autoencoder import LatentEncoderQEC  # noqa: E402


def make_factory(down_stages: int, blocks_per_stage: int):
    """Return a constructor that surface_code_resnet.train_mixed will call."""

    def factory(grid_size, width=64, depth=8, cond_dim=32, in_channels=1,
                zero_init_residual=False, block_type="classic", se_ratio=8):
        # Ignore depth, block_type, zero_init_residual, se_ratio — those are
        # ResNet-specific options. The autoencoder uses width + down_stages.
        return LatentEncoderQEC(
            grid_size=grid_size,
            width=width,
            cond_dim=cond_dim,
            in_channels=in_channels,
            down_stages=down_stages,
            blocks_per_stage=blocks_per_stage,
        )

    return factory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=int, default=5)
    ap.add_argument("--noise", type=str, default="circuit")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=8,
                    help="Ignored by the autoencoder backbone but accepted "
                         "for matched-CLI compatibility with surface_code_resnet.")
    ap.add_argument("--down-stages", type=int, default=2,
                    help="Number of factor-2 downsamples (autoencoder-specific)")
    ap.add_argument("--blocks-per-stage", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-per-rate", type=int, default=200000)
    ap.add_argument("--n-val", type=int, default=1000000)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--save-dir", type=str, default="ckpts/qec_ae")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local-prior", action="store_true")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--focal-loss", action="store_true")
    ap.add_argument("--oversample", type=int, default=1)
    ap.add_argument("--log-file", type=str, default=None)
    # --block-type and --zero-init-residual accepted but ignored
    ap.add_argument("--block-type", type=str, default="classic")
    ap.add_argument("--zero-init-residual", action="store_true")
    args = ap.parse_args()

    # Monkey-patch ResNetDecoder so train_mixed instantiates our autoencoder.
    scr.ResNetDecoder = make_factory(args.down_stages, args.blocks_per_stage)
    print(f"[setup] backbone = LatentEncoderQEC (down_stages={args.down_stages})", flush=True)

    scr.train_mixed(
        distance=args.distance,
        noise=args.noise,
        n_per_rate=args.n_per_rate,
        n_val=args.n_val,
        epochs=args.epochs,
        width=args.width,
        depth=args.depth,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        seed=args.seed,
        local_prior=args.local_prior,
        augment=args.augment,
        use_focal_loss=args.focal_loss,
        oversample_positive=args.oversample,
        log_file=args.log_file,
        block_type=args.block_type,
        zero_init_residual=args.zero_init_residual,
    )


if __name__ == "__main__":
    main()
