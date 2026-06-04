"""Distributed QEC training across multiple GPU nodes.

Loads pre-computed data from qec_prep_data.py and trains with
PyTorch DDP across multiple GPUs connected via NCCL.

Architecture:
    CPU nodes: run qec_prep_data.py into a local or mounted data directory.
    GPU nodes: run this script with torchrun.

Usage (single node, for testing):
    torchrun --nproc_per_node=1 qec_train_ddp.py \
        --data-dir runs/qec_data/d9_aug --width 192 --depth 12

Usage (2 nodes over 100G):
    # On alpha (rank 0):
    torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
        --master_addr=192.168.68.232 --master_port=29500 \
        qec_train_ddp.py --data-dir runs/qec_data/d9_aug

    # On cpm (rank 1):
    torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 \
        --master_addr=192.168.68.232 --master_port=29500 \
        qec_train_ddp.py --data-dir runs/qec_data/d9_aug
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from surface_code_resnet import ResNetDecoder, focal_loss


def parse_binary_bool(value):
    """Parse 0/1 CLI flags into bools for artifact validation."""
    if value == "0":
        return False
    if value == "1":
        return True
    raise argparse.ArgumentTypeError("expected 0 or 1")


def setup_ddp():
    """Initialize distributed process group."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def log(msg, rank, log_file=None):
    """Print from rank 0 only, optionally write to file."""
    if rank == 0:
        print(msg, flush=True)
        if log_file:
            with open(log_file, "a") as f:
                f.write(msg + "\n")


def train(args):
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    log(f"DDP: rank={rank}/{world_size}, device={device}, "
        f"host={os.uname().nodename}", rank, args.log_file)

    # Load pre-computed data
    log(f"Loading data from {args.data_dir}...", rank, args.log_file)
    train_data = torch.load(Path(args.data_dir) / "train.pt",
                            weights_only=False)
    val_data = torch.load(Path(args.data_dir) / "val.pt",
                          weights_only=False)

    train_vols = train_data["volumes"]
    train_labs = train_data["labels"]
    train_rates = train_data["rates"]
    grid_size = train_data["grid_size"]
    distance = train_data["distance"]
    train_augmented = bool(train_data.get("augmented", False))

    if args.expected_augmented is not None and train_augmented != args.expected_augmented:
        expected = "augmented" if args.expected_augmented else "non-augmented"
        actual = "augmented" if train_augmented else "non-augmented"
        raise ValueError(
            f"Expected {expected} training data in {args.data_dir}, found {actual}"
        )
    if len(train_vols) != len(train_labs) or len(train_vols) != len(train_rates):
        raise ValueError(
            "Prepared training tensors have mismatched lengths: "
            f"volumes={len(train_vols)}, labels={len(train_labs)}, "
            f"rates={len(train_rates)}"
        )

    val_vols = val_data["volumes"]
    val_labs = val_data["labels"]
    val_rate_list = val_data["val_rates"]
    val_boundaries = val_data["rate_boundaries"]
    val_mwpm = val_data["mwpm"]

    log(f"Train: {len(train_vols)} samples, Val: {len(val_vols)} samples",
        rank, args.log_file)
    log(f"d={distance}, grid={grid_size}", rank, args.log_file)

    for p, err in val_mwpm.items():
        log(f"  p={p}: MWPM={err:.6f}", rank, args.log_file)

    # Oversample positive examples
    if args.oversample > 1:
        pos_mask = train_labs > 0.5
        pos_idx = torch.where(pos_mask)[0]
        extra_idx = pos_idx.repeat(args.oversample - 1)
        all_idx = torch.cat([torch.arange(len(train_vols)), extra_idx])
        train_vols = train_vols[all_idx]
        train_labs = train_labs[all_idx]
        train_rates = train_rates[all_idx]
        n_pos = (train_labs > 0.5).sum().item()
        log(f"Oversampled {args.oversample}x: {len(train_vols)} samples "
            f"({n_pos} pos, {len(train_vols)-n_pos} neg)",
            rank, args.log_file)

    # DataLoader with DistributedSampler
    train_ds = TensorDataset(train_vols, train_labs, train_rates)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                       rank=rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=2,
                              pin_memory=True)

    # Model
    model = ResNetDecoder(grid_size, width=args.width, depth=args.depth,
                          zero_init_residual=args.zero_init_residual,
                          block_type=args.block_type).to(device)
    model = DDP(model, device_ids=[local_rank])
    n_params = sum(p.numel() for p in model.parameters())
    log(f"ResNet3D: {n_params:,} params (w={args.width}, d={args.depth})",
        rank, args.log_file)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    if use_amp:
        log("Using bf16 AMP", rank, args.log_file)
    log(f"Focal loss: {'enabled' if args.focal_loss else 'disabled'}",
        rank, args.log_file)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        n_batches = 0
        t0 = time.time()

        for vol, label, p_rate in train_loader:
            vol = vol.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            p_rate = p_rate.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(vol, p_rate)
                if args.focal_loss:
                    loss = focal_loss(logits, label)
                else:
                    loss = F.binary_cross_entropy_with_logits(logits, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_time = time.time() - t0

        # Eval (rank 0 only to avoid duplicate work)
        if rank == 0:
            model.eval()
            line = f"  Epoch {epoch}: loss={total_loss/n_batches:.4f} " \
                   f"({epoch_time:.0f}s)"

            with torch.no_grad():
                offset = 0
                for i, p in enumerate(val_rate_list):
                    n = val_boundaries[i]
                    v = val_vols[offset:offset + n]
                    l = val_labs[offset:offset + n]
                    offset += n

                    all_preds = []
                    for j in range(0, len(v), args.batch_size):
                        bv = v[j:j + args.batch_size].to(device)
                        bp = torch.full((len(bv),), p, device=device)
                        with torch.amp.autocast("cuda", dtype=amp_dtype,
                                                enabled=use_amp):
                            logits = model.module(bv, bp)
                        all_preds.append((logits > 0).cpu())
                    pred = torch.cat(all_preds).float()
                    err = (pred != l).float().mean().item()
                    line += f"  p={p}:{err:.5f}"

            log(line, rank, args.log_file)

            # Save checkpoint
            if args.save_dir:
                save_path = Path(args.save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model": model.module.state_dict(),
                    "epoch": epoch,
                    "grid_size": grid_size,
                    "width": args.width,
                    "depth": args.depth,
                    "distance": distance,
                }, save_path / f"qec_d{distance}_epoch{epoch}.pt")

        # Sync all ranks before next epoch
        dist.barrier()

    # Final summary
    if rank == 0:
        log(f"\nTraining complete: {args.epochs} epochs", rank, args.log_file)

    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description="DDP QEC training")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with train.pt and val.pt from qec_prep_data.py")
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--focal-loss", action="store_true")
    parser.add_argument("--zero-init-residual", action="store_true",
                        help="Zero-init gn2.weight in each block so tower "
                             "starts as identity (helps at high depth)")
    parser.add_argument("--block-type", type=str, default="classic",
                        choices=["classic", "lc0"],
                        help="Residual block type: classic (current) or "
                             "lc0 (pre-activation + dual-output SE + FiLM)")
    parser.add_argument("--oversample", type=int, default=1)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument(
        "--expected-augmented",
        type=parse_binary_bool,
        default=None,
        help="Assert that train.pt matches the requested augmentation mode (0 or 1)",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
