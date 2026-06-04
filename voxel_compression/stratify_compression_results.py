"""Stratify clean compression results by occupancy density.

This is a focused companion to reproduce_results.py. It evaluates the
checkpoint-backed L4 reconstruction path and records per-sample metrics needed
to test whether the constraint-locality result holds across object-density
strata rather than only on the average ShapeNet validation distribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def load_model_and_ops(ckpt_path: str, cfg_path: str, device: torch.device):
    from config import load_config
    from parity import ConstraintConfig, load_constraint_ops
    from nn.resnet3d import ResNet3D

    if cfg_path and Path(cfg_path).exists():
        cfg = load_config(cfg_path)
        constraint_cfg = cfg.constraints
        coarse_size = getattr(cfg.train, "coarse_size", 2)
        coarse_threshold = getattr(cfg.train, "coarse_threshold", 0.42)
    else:
        constraint_cfg = ConstraintConfig(axis_parity=True, plane_parity=True, global_parity=True)
        coarse_size = 2
        coarse_threshold = 0.42

    _, left_inv, parity = load_constraint_ops(32, constraint_cfg)
    parity = parity.to(device)
    left_inv = left_inv.to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("ema", ckpt.get("model", ckpt))
    first_w = state.get("input_conv.weight")
    width = first_w.shape[0] if first_w is not None else 192

    model = ResNet3D(in_ch=3, width=width, depth=12, cond_dim=parity.shape[0]).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    return model, parity, left_inv, coarse_size, coarse_threshold


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": None, "sem": None}
    arr = np.asarray(values, dtype=np.float64)
    sem = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {"mean": float(arr.mean()), "sem": sem}


def make_rank_strata(samples: list[dict], n_strata: int) -> list[dict]:
    order = sorted(range(len(samples)), key=lambda i: samples[i]["fill"])
    chunks = np.array_split(order, n_strata)
    strata = []
    for idx, chunk in enumerate(chunks, start=1):
        rows = [samples[int(i)] for i in chunk]
        fills = [r["fill"] for r in rows]
        strata.append(
            {
                "name": f"Q{idx}",
                "description": f"occupancy rank quartile {idx}",
                "n": len(rows),
                "fill_min": float(min(fills)),
                "fill_max": float(max(fills)),
                "fill_mean": float(np.mean(fills)),
                "sample_indices": [int(i) for i in chunk],
            }
        )
    return strata


def summarize_strata(samples: list[dict], strata: list[dict]) -> dict:
    metrics = [
        "raw_iou",
        "hard_proj_iou",
        "hard_proj_delta",
        "surface_patch_iou",
        "surface_patch_delta",
        "surface_neighbor_iou",
        "surface_neighbor_delta",
        "surface_gradient_iou",
        "surface_gradient_delta",
    ]
    out = {}
    for s in strata:
        rows = [samples[i] for i in s["sample_indices"]]
        entry = {k: v for k, v in s.items() if k != "sample_indices"}
        for metric in metrics:
            entry[metric] = summarize([float(r[metric]) for r in rows])
        out[s["name"]] = entry
    return out


def evaluate(args: argparse.Namespace) -> dict:
    from adaptive_dct import FREQ_LEVELS, make_adaptive_dct_field
    from dataset import VoxelDataset
    from parity import gf2_matmul, make_canonical_base, project_with_left_inverse
    from surface_constraints import build_surface_constraints
    from syndrome_refine import refine_with_syndrome
    from train import make_coarse_map
    from utils.metrics import iou

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, parity, left_inv, coarse_size, coarse_threshold = load_model_and_ops(
        args.ckpt, args.cfg, device
    )
    dataset = VoxelDataset(args.data, 32, constraint_matrix=parity.cpu())
    n = min(args.n_samples, len(dataset))
    level_idx = FREQ_LEVELS.index(args.level)
    modes = [m.strip() for m in args.surface_modes.split(",") if m.strip()]
    samples = []

    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Config: {args.cfg}")
    print(f"Samples: {n}")
    print(f"Level: L{args.level}")
    print(f"Surface modes: {modes}")

    with torch.no_grad():
        for idx in range(n):
            voxels_t, syndrome_t = dataset[idx]
            voxels = voxels_t.to(device).view(1, -1).to(torch.uint8)
            syndrome = syndrome_t.to(device).unsqueeze(0).float()
            syndrome_uint8 = syndrome.to(torch.uint8)

            x_base = make_canonical_base(syndrome_uint8, parity, left_inv, device)
            coarse = make_coarse_map(voxels, 32, coarse_size, coarse_threshold)
            sel = torch.zeros(1, len(FREQ_LEVELS), device=device)
            sel[0, level_idx] = 1.0
            dct = make_adaptive_dct_field(voxels, 32, sel)
            dct_3d = dct.squeeze(0).squeeze(0)

            x_input = torch.cat([x_base.float().view(1, 1, 32, 32, 32), coarse, dct], dim=1)
            t = torch.full((1,), 1000, device=device)
            logits = model(x_input, t, syndrome)
            logits_flat = logits.view(1, -1)

            pred = (torch.sigmoid(logits_flat) > 0.5).to(torch.uint8)
            recon_raw = x_base ^ pred
            recon_proj, _ = project_with_left_inverse(
                recon_raw.clone(), parity, left_inv, syndrome_uint8
            )

            row = {
                "index": idx,
                "path": str(dataset.paths[idx]),
                "fill": float(voxels.float().mean().item()),
                "raw_iou": float(iou(recon_raw, voxels).item()),
                "hard_proj_iou": float(iou(recon_proj, voxels).item()),
            }
            row["hard_proj_delta"] = row["hard_proj_iou"] - row["raw_iou"]

            full_probs = torch.sigmoid(logits_flat)
            base_f = x_base.float()
            full_p = base_f * (1 - full_probs) + (1 - base_f) * full_probs
            full_logits = torch.log(full_p.clamp(1e-7) / (1 - full_p).clamp(1e-7))

            for mode in modes:
                H_surf, _ = build_surface_constraints(
                    dct_3d, 32, mode=mode, surface_threshold=args.surface_threshold
                )
                n_constraints = H_surf.size(0)
                row[f"surface_{mode}_bits"] = int(n_constraints)
                if n_constraints == 0:
                    refined_iou = row["raw_iou"]
                else:
                    if n_constraints > 15000:
                        surf_syn = gf2_matmul(H_surf, voxels.cpu().t()).t().to(device)
                        H_surf = H_surf.to(device)
                    else:
                        H_surf = H_surf.to(device)
                        surf_syn = gf2_matmul(H_surf, voxels.t()).t()
                    refined, _, _, _ = refine_with_syndrome(
                        full_logits,
                        H_surf,
                        surf_syn,
                        n_iters=args.bp_iters,
                        damping=args.bp_damping,
                        device=device,
                    )
                    refined_iou = float(iou(refined, voxels).item())
                row[f"surface_{mode}_iou"] = refined_iou
                row[f"surface_{mode}_delta"] = refined_iou - row["raw_iou"]

            samples.append(row)
            if (idx + 1) % args.progress_every == 0:
                print(f"  {idx + 1}/{n} done")

    strata = make_rank_strata(samples, args.n_strata)
    return {
        "seed": args.seed,
        "checkpoint": args.ckpt,
        "config": args.cfg,
        "data": args.data,
        "n_samples": n,
        "level": args.level,
        "surface_modes": modes,
        "strata_method": "rank occupancy quartiles",
        "strata": summarize_strata(samples, strata),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--data", default="data/shapenet/val")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--level", type=int, default=4)
    parser.add_argument("--n-strata", type=int, default=4)
    parser.add_argument("--surface-modes", default="patch,neighbor,gradient")
    parser.add_argument("--surface-threshold", type=float, default=0.3)
    parser.add_argument("--bp-iters", type=int, default=30)
    parser.add_argument("--bp-damping", type=float, default=0.5)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    out = evaluate(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
