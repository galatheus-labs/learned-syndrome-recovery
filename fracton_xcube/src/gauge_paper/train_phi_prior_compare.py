"""
Minimal A/B comparison: phi-prior vs x-prior decoder at matched parameter
count, on ShapeNet 32^3 voxels with the bug-fixed axial parity code.

Both networks share architecture (ResNet3D backbone), training data,
training schedule, optimizer, conditioning, and loss. The only
difference is whether the output is interpreted as residual logits
(x-prior baseline) or as a phi potential whose discrete third
derivative produces the residual logits (phi-prior).

Per the gauge theorem, the phi-prior is by construction restricted to
the codespace; the x-prior must learn that constraint from data.

Hypothesis: phi-prior reaches the same val BCE / IoU faster, or
attains better final IoU, at matched parameters and matched epochs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


_THIS = Path(__file__).resolve()
_DEFAULT_PARITY_ROOT = _THIS.parent.parent.parent
# Allow the parity_diffusion root to be overridden via env var for cluster
# runs where this script is staged outside the source tree.
import os as _os
_PARITY_ROOT = Path(_os.environ.get("GALAREON_PD_ROOT", str(_DEFAULT_PARITY_ROOT)))
if not (_PARITY_ROOT / "parity.py").exists():
    # Fallback: try the cluster-baked location.
    _alt = Path("/app/parity_diffusion")
    if (_alt / "parity.py").exists():
        _PARITY_ROOT = _alt
if str(_PARITY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PARITY_ROOT))
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))
print(f"[setup] parity_diffusion root: {_PARITY_ROOT}")

from dataset import VoxelDataset  # noqa: E402
from nn.resnet3d import ResNet3D  # noqa: E402
from adaptive_dct import make_adaptive_dct_field  # noqa: E402
from parity import (  # noqa: E402
    _gf2_left_inverse,
    _gf2_null_space,
    build_parity_matrix,
    gf2_matmul,
    make_canonical_base,
)

from phi_prior_decoder import PhiPriorResNet3D, PhiPriorResNet3DEscape  # noqa: E402
from gauge_transporter import GaugeTransporter, LatentAutoencoder  # noqa: E402


def load_parity_ops_workcache(n: int, cache_dir: str = "runs/parity_cache"):
    """Same semantics as parity.load_parity_ops but caches the null-space and
    left-inverse computation between runs."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    ns_path = cache / f"nullspace_v2_true_axes_{n}.pt"
    li_path = cache / f"leftinv_v2_true_axes_{n}.pt"
    A = build_parity_matrix(n)
    if ns_path.exists() and li_path.exists():
        print(f"[cache] loading null-space + left-inverse from {cache}")
        null = torch.load(ns_path)
        left = torch.load(li_path)
        return null, left, A
    print(f"[cache] miss — building null-space + left-inverse for n={n} (~6 min)")
    null = _gf2_null_space(A)
    left = _gf2_left_inverse(A)
    torch.save(null, ns_path)
    torch.save(left, li_path)
    print(f"[cache] saved to {cache}")
    return null, left, A


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------


def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dct_field(voxels_uint8: torch.Tensor, cube_size: int, level: int) -> torch.Tensor:
    """Compute a DCT-style smooth prior field by zeroing all but the lowest
    `level` frequencies in each axis. Returns (B, 1, n, n, n) float in [0, 1]."""
    B = voxels_uint8.shape[0]
    voxels_flat = voxels_uint8.reshape(B, -1)
    # one-hot selection picks index 0 of FREQ_LEVELS=[2,3,4,5,6,7,8] when level=2 etc.
    from adaptive_dct import FREQ_LEVELS
    if level not in FREQ_LEVELS:
        raise ValueError(f"dct_level must be one of {FREQ_LEVELS}, got {level}")
    K = len(FREQ_LEVELS)
    sel = torch.zeros(B, K, device=voxels_uint8.device)
    sel[:, FREQ_LEVELS.index(level)] = 1.0
    field = make_adaptive_dct_field(voxels_flat, cube_size, sel)
    return field  # (B, 1, n, n, n)


def build_inputs(voxels_uint8: torch.Tensor, syndrome_uint8: torch.Tensor,
                 parity: torch.Tensor, left_inv: torch.Tensor,
                 device: torch.device):
    """Compute (x_base, residual_target) for a batch.

    Args:
      voxels_uint8: (B, n, n, n) uint8 in {0, 1}
      syndrome_uint8: (B, M) uint8
      parity, left_inv: precomputed GF(2) ops, on `device`.
    Returns:
      x_base_float: (B, 1, n, n, n) float
      residual_target: (B, 1, n, n, n) float
    """
    B, n1, n2, n3 = voxels_uint8.shape
    x_base = make_canonical_base(syndrome_uint8.to(device), parity, left_inv, device)
    voxels_flat = voxels_uint8.reshape(B, -1).to(device)
    residual = (voxels_flat.int() ^ x_base.int()).to(torch.uint8)
    x_base_3d = x_base.reshape(B, 1, n1, n2, n3).float()
    residual_3d = residual.reshape(B, 1, n1, n2, n3).float()
    return x_base_3d, residual_3d


def compute_iou(pred_voxels_uint8: torch.Tensor, true_voxels_uint8: torch.Tensor) -> torch.Tensor:
    """Binary IoU per sample, shape (B,)."""
    p = pred_voxels_uint8.bool().reshape(pred_voxels_uint8.shape[0], -1)
    t = true_voxels_uint8.bool().reshape(true_voxels_uint8.shape[0], -1)
    inter = (p & t).sum(dim=1).float()
    union = (p | t).sum(dim=1).float().clamp(min=1.0)
    return inter / union


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, parity: torch.Tensor,
             left_inv: torch.Tensor, cube_size: int, device: torch.device,
             dct_level: int = 0):
    model.eval()
    total_bce = 0.0
    total_iou = 0.0
    n_samples = 0
    for voxels, syndrome in loader:
        B = voxels.shape[0]
        voxels = voxels.reshape(B, cube_size, cube_size, cube_size).to(torch.uint8)
        syndrome = syndrome.to(torch.uint8)
        x_base_3d, residual_3d = build_inputs(voxels, syndrome, parity, left_inv, device)
        if dct_level > 0:
            dct_field = build_dct_field(voxels.to(device), cube_size, dct_level)
            x_in = torch.cat([x_base_3d, dct_field], dim=1)
        else:
            x_in = x_base_3d
        t = torch.zeros(B, dtype=torch.long, device=device)
        cond = syndrome.float().to(device)
        logits = model(x_in, t, cond)
        bce = F.binary_cross_entropy_with_logits(logits, residual_3d, reduction="mean")
        total_bce += bce.item() * B
        # decoded voxels = x_base XOR (residual_logits > 0)
        pred_residual = (logits > 0).to(torch.uint8).reshape(B, -1)
        pred_voxels = (x_base_3d.to(torch.uint8).reshape(B, -1) ^ pred_residual)
        ious = compute_iou(
            pred_voxels.reshape(B, cube_size, cube_size, cube_size),
            voxels.to(device),
        )
        total_iou += ious.sum().item()
        n_samples += B
    model.train()
    return total_bce / max(n_samples, 1), total_iou / max(n_samples, 1)


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------


@dataclass
class TrainConfig:
    seed: int
    width: int
    depth: int
    epochs: int
    batch_size: int
    lr: float
    cube_size: int
    train_path: str
    val_path: str
    train_subset: int  # 0 = full
    val_subset: int    # 0 = full
    num_workers: int
    out_dir: Path
    pos_weight: float = 1.0  # >1 weights positive (1-class) samples in BCE
    gauge_aux: bool = True   # if False, drop the soft-gauge aux losses (ablation)
    dct_level: int = 0       # if >0, add DCT side-info (k = dct_level) as input channel


def train_one_model(name: str, model: nn.Module, cfg: TrainConfig,
                    train_loader: DataLoader, val_loader: DataLoader,
                    parity: torch.Tensor, left_inv: torch.Tensor,
                    device: torch.device):
    print(f"\n=== Training {name} ===")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * len(train_loader))

    history = []
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        running_bce = 0.0
        n_batches = 0
        for voxels, syndrome in train_loader:
            B = voxels.shape[0]
            voxels = voxels.reshape(B, cfg.cube_size, cfg.cube_size, cfg.cube_size).to(torch.uint8)
            syndrome = syndrome.to(torch.uint8)
            x_base_3d, residual_3d = build_inputs(voxels, syndrome, parity, left_inv, device)
            if cfg.dct_level > 0:
                dct_field = build_dct_field(voxels.to(device), cfg.cube_size, cfg.dct_level)
                x_in = torch.cat([x_base_3d, dct_field], dim=1)
            else:
                x_in = x_base_3d
            t = torch.zeros(B, dtype=torch.long, device=device)
            cond = syndrome.float().to(device)
            logits = model(x_in, t, cond)
            if cfg.pos_weight != 1.0:
                pw = torch.full_like(residual_3d, cfg.pos_weight)
                loss = F.binary_cross_entropy_with_logits(logits, residual_3d, pos_weight=pw)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, residual_3d)
            # Auxiliary gauge regularisers for the soft-gauge transporter.
            if hasattr(model, "extra_losses_for_last_forward") and getattr(cfg, "gauge_aux", True):
                aux = model.extra_losses_for_last_forward()
                if aux:
                    loss = (
                        loss
                        + 1.0e-3 * aux["curvature"]
                        + 5.0e-2 * aux["local_transport"]
                        + 1.0e-1 * aux["latent_recon"]
                    )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running_bce += loss.item()
            n_batches += 1
        train_bce = running_bce / max(n_batches, 1)
        val_bce, val_iou = evaluate(model, val_loader, parity, left_inv, cfg.cube_size, device,
                                     dct_level=cfg.dct_level)
        elapsed = time.time() - t0
        rec = {
            "epoch": epoch,
            "train_bce": train_bce,
            "val_bce": val_bce,
            "val_iou": val_iou,
            "epoch_seconds": elapsed,
        }
        history.append(rec)
        print(f"  epoch {epoch:2d}: train_bce={train_bce:.4f}  val_bce={val_bce:.4f}  val_iou={val_iou:.4f}  ({elapsed:.0f}s)")

    out = cfg.out_dir / f"{name}_history.json"
    out.write_text(json.dumps({"name": name, "n_params": n_params, "history": history}, indent=2))
    ckpt_path = cfg.out_dir / f"{name}_final.pt"
    torch.save({"state_dict": model.state_dict(), "config": vars(cfg), "name": name}, ckpt_path)
    return history, n_params


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--cube-size", type=int, default=32)
    ap.add_argument("--train-path", default="data/shapenet/train")
    ap.add_argument("--val-path", default="data/shapenet/val")
    ap.add_argument("--train-subset", type=int, default=0,
                    help="if >0, use only this many training samples (for fast smoke test)")
    ap.add_argument("--val-subset", type=int, default=512,
                    help="if >0, use only this many validation samples")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out-dir", default="runs/gauge_paper_runs/run_default")
    ap.add_argument("--variant", choices=["both", "x_prior", "phi_prior", "phi_escape", "gauge_transporter", "latent_autoencoder"], default="both")
    ap.add_argument("--gt-base-ch", type=int, default=64,
                    help="GaugeTransporter base channel width (controls param count)")
    ap.add_argument("--gt-latent-dim", type=int, default=32,
                    help="GaugeTransporter latent channels (must be even)")
    ap.add_argument("--no-gauge-aux", action="store_true",
                    help="Disable curvature/transport/latent-recon aux losses (ablation)")
    ap.add_argument("--dct-level", type=int, default=0,
                    help="If >0, use DCT side-info at level k (one of [2,3,4,5,6,7,8]) as input channel")
    ap.add_argument("--down-stages", type=int, default=2,
                    help="Number of factor-2 downsamples in encoder (default 2: 32->8). "
                         "Use 3 for 32->4 (smaller bottleneck), 1 for 32->16 (bigger bottleneck).")
    ap.add_argument("--pos-weight", type=float, default=1.0,
                    help="BCE pos_weight to upweight positive (1-class) samples")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(
        seed=args.seed, width=args.width, depth=args.depth, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, cube_size=args.cube_size,
        train_path=args.train_path, val_path=args.val_path,
        train_subset=args.train_subset, val_subset=args.val_subset,
        num_workers=args.num_workers, out_dir=out_dir,
        pos_weight=args.pos_weight,
        gauge_aux=not args.no_gauge_aux,
        dct_level=args.dct_level,
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Constraint ops (bug-fixed axial parity, M = 3n^2 - 3n + 1).
    print(f"Loading parity ops for n={cfg.cube_size}...")
    null_space, left_inv, parity = load_parity_ops_workcache(cfg.cube_size)
    parity = parity.to(device)
    left_inv = left_inv.to(device)
    M = parity.size(0)
    expected_M = 3 * cfg.cube_size * cfg.cube_size - 3 * cfg.cube_size + 1
    print(f"  parity matrix: {M} rows  (expected {expected_M})  match={M == expected_M}")

    # Datasets.
    parity_cpu = parity.cpu()
    train_dataset = VoxelDataset(cfg.train_path, cfg.cube_size, constraint_matrix=parity_cpu)
    val_dataset = VoxelDataset(cfg.val_path, cfg.cube_size, constraint_matrix=parity_cpu)
    if cfg.train_subset > 0:
        train_dataset = Subset(train_dataset, list(range(min(cfg.train_subset, len(train_dataset)))))
    if cfg.val_subset > 0:
        val_dataset = Subset(val_dataset, list(range(min(cfg.val_subset, len(val_dataset)))))
    print(f"  train: {len(train_dataset)} samples  val: {len(val_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    in_ch = 2 if cfg.dct_level > 0 else 1
    common_kwargs = dict(in_ch=in_ch, width=cfg.width, depth=cfg.depth, cond_dim=M,
                         block_type="classic", zero_init_residual=False)

    summary = {"config": vars(args), "M": M}
    histories = {}

    if args.variant in ("x_prior", "both"):
        set_seed(cfg.seed)  # match init seed across variants
        x_model = ResNet3D(**common_kwargs)
        history, np_x = train_one_model("x_prior", x_model, cfg, train_loader,
                                         val_loader, parity, left_inv, device)
        histories["x_prior"] = {"n_params": np_x, "history": history}

    if args.variant in ("phi_prior", "both"):
        set_seed(cfg.seed)
        phi_model = PhiPriorResNet3D(**common_kwargs)
        history, np_phi = train_one_model("phi_prior", phi_model, cfg, train_loader,
                                           val_loader, parity, left_inv, device)
        histories["phi_prior"] = {"n_params": np_phi, "history": history}

    if args.variant == "phi_escape":
        set_seed(cfg.seed)
        phi_model = PhiPriorResNet3DEscape(**common_kwargs)
        history, np_phi = train_one_model("phi_escape", phi_model, cfg, train_loader,
                                           val_loader, parity, left_inv, device)
        histories["phi_escape"] = {"n_params": np_phi, "history": history}

    if args.variant == "gauge_transporter":
        set_seed(cfg.seed)
        gt_model = GaugeTransporter(
            in_ch=1, out_ch=1,
            latent_dim=args.gt_latent_dim,
            base_ch=args.gt_base_ch,
            cond_dim=M,
        )
        history, np_gt = train_one_model("gauge_transporter", gt_model, cfg, train_loader,
                                          val_loader, parity, left_inv, device)
        histories["gauge_transporter"] = {"n_params": np_gt, "history": history}

    if args.variant == "latent_autoencoder":
        set_seed(cfg.seed)
        la_model = LatentAutoencoder(
            in_ch=in_ch, out_ch=1,
            latent_dim=args.gt_latent_dim,
            base_ch=args.gt_base_ch,
            cond_dim=M,
            down_stages=args.down_stages,
        )
        history, np_la = train_one_model("latent_autoencoder", la_model, cfg, train_loader,
                                          val_loader, parity, left_inv, device)
        histories["latent_autoencoder"] = {"n_params": np_la, "history": history}

    summary["histories"] = histories

    # Compare final IoU and best val_iou.
    print("\n=== Comparison ===")
    for name, h in histories.items():
        if not h["history"]:
            continue
        last = h["history"][-1]
        best_iou = max(r["val_iou"] for r in h["history"])
        print(f"  {name}: params={h['n_params']:,}  final_val_iou={last['val_iou']:.4f}  "
              f"best_val_iou={best_iou:.4f}  final_val_bce={last['val_bce']:.4f}")

    if "x_prior" in histories and "phi_prior" in histories:
        x_final = histories["x_prior"]["history"][-1]["val_iou"]
        phi_final = histories["phi_prior"]["history"][-1]["val_iou"]
        delta = phi_final - x_final
        verdict = "PHI WINS" if delta > 0.005 else "X WINS" if delta < -0.005 else "TIE"
        print(f"  delta(phi - x) on final val_iou = {delta:+.4f}   --> {verdict}")
        summary["verdict"] = {"x_final_iou": x_final, "phi_final_iou": phi_final,
                              "delta": delta, "verdict": verdict}

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote summary to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
