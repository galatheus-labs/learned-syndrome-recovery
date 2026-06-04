"""Pre-compute QEC training data with label-preserving D4 augmentation.

Run this on CPU-only nodes (tinkara/milan) to generate augmented
training data and save to /work for GPU nodes to load.

Usage:
    python qec_prep_data.py --distance 9 --n-per-rate 50000 \
        --out-dir /work/qec_data/d9_s42_aug --augment --seed 42
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import stim
import pymatching

from surface_code_resnet import (
    generate_data, get_detector_grid_mapping, detectors_to_volume,
    d4_augment_all,
)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute QEC data")
    parser.add_argument("--distance", type=int, default=9)
    parser.add_argument("--noise", default="circuit")
    parser.add_argument("--n-per-rate", type=int, default=50000)
    parser.add_argument("--n-val", type=int, default=100000)
    parser.add_argument("--augment", action="store_true",
                        help="Apply the label-preserving D4 subgroup (4x data)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--train-rates", type=float, nargs="+",
                        default=[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007])
    parser.add_argument("--val-rates", type=float, nargs="+",
                        default=[0.001, 0.003, 0.005, 0.007])
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Get grid mapping
    _, _, ref = generate_data(args.distance, args.train_rates[0], 1,
                              args.noise)
    grid_size, det_to_grid = get_detector_grid_mapping(ref)
    n_det = ref.num_detectors
    print(f"d={args.distance}, noise={args.noise}, detectors={n_det}, "
          f"grid={grid_size}")

    # --- Training data ---
    t0 = time.time()
    print(f"\nGenerating {args.n_per_rate} train samples x "
          f"{len(args.train_rates)} rates...")

    all_vols = []
    all_labs = []
    all_rates = []
    augmentation_factor = 1

    for i, p in enumerate(args.train_rates):
        det, obs, _ = generate_data(args.distance, p, args.n_per_rate,
                                    args.noise,
                                    sampler_seed=args.seed * 1000 + i)
        vol = detectors_to_volume(
            torch.from_numpy(det.astype(np.float32)), grid_size, det_to_grid)
        lab = torch.from_numpy(obs.astype(np.float32)).squeeze(-1)

        n_pos = (lab > 0.5).sum().item()
        print(f"  p={p}: {args.n_per_rate} samples, "
              f"{n_pos} positive ({100*n_pos/args.n_per_rate:.2f}%)")

        if args.augment:
            base_count = len(lab)
            vol = d4_augment_all(vol)
            aug_factor, remainder = divmod(len(vol), base_count)
            if remainder != 0:
                raise ValueError(
                    f"D4 augmentation produced {len(vol)} volumes for "
                    f"{base_count} labels"
                )
            augmentation_factor = aug_factor
            lab = lab.repeat(aug_factor)
            print(f"    D4 augmented ({aug_factor}x): {len(vol)} samples")

        all_vols.append(vol)
        all_labs.append(lab)
        all_rates.extend([p] * len(vol))

    # Concatenate and save
    train_vols = torch.cat(all_vols, dim=0)
    train_labs = torch.cat(all_labs, dim=0)
    train_rates = torch.tensor(all_rates, dtype=torch.float32)

    print(f"\nTotal train: {len(train_vols)} samples "
          f"({train_vols.nbytes / 1e9:.1f} GB)")

    train_path = out / "train.pt"
    torch.save({
        "volumes": train_vols,
        "labels": train_labs,
        "rates": train_rates,
        "grid_size": grid_size,
        "det_to_grid": det_to_grid,
        "distance": args.distance,
        "noise": args.noise,
        "seed": args.seed,
        "augmented": args.augment,
        "augmentation_factor": augmentation_factor,
    }, train_path)
    print(f"Saved train data: {train_path} "
          f"({train_path.stat().st_size / 1e9:.1f} GB)")

    # --- Validation data ---
    print(f"\nGenerating {args.n_val} val samples x "
          f"{len(args.val_rates)} rates...")

    val_vols = []
    val_labs = []
    val_rates_list = []
    val_mwpm = {}

    for i, p in enumerate(args.val_rates):
        det, obs, circuit = generate_data(
            args.distance, p, args.n_val, args.noise,
            sampler_seed=args.seed * 1000 + 500 + i)
        vol = detectors_to_volume(
            torch.from_numpy(det.astype(np.float32)), grid_size, det_to_grid)
        lab = torch.from_numpy(obs.astype(np.float32)).squeeze(-1)

        # MWPM baseline
        dem = circuit.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(dem)
        mwpm_pred = matcher.decode_batch(det)
        mwpm_err = float((mwpm_pred != obs).sum() / args.n_val)
        val_mwpm[p] = mwpm_err

        print(f"  p={p}: MWPM={mwpm_err:.6f}")

        val_vols.append(vol)
        val_labs.append(lab)
        val_rates_list.extend([p] * len(vol))

    val_path = out / "val.pt"
    torch.save({
        "volumes": torch.cat(val_vols, dim=0),
        "labels": torch.cat(val_labs, dim=0),
        "rates": torch.tensor(val_rates_list, dtype=torch.float32),
        "rate_boundaries": [len(v) for v in val_vols],
        "val_rates": args.val_rates,
        "mwpm": val_mwpm,
        "grid_size": grid_size,
    }, val_path)
    print(f"Saved val data: {val_path} "
          f"({val_path.stat().st_size / 1e9:.1f} GB)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
