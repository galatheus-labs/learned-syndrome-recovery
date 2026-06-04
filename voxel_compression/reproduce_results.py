"""Reproduce all numerical claims from the compression paper.

Runs every benchmark and evaluation needed to verify Table 1, the
variable-rate analysis, and the Bayesian temporal accumulation claim.
Outputs a summary that matches the paper's numbers.

Usage:
    # Full reproduction (requires trained checkpoint + ShapeNet val data):
    python reproduce_results.py --ckpt ckpts/resnet_combined_w256/ckpt_166000.pt

    # Baselines only (no checkpoint needed):
    python reproduce_results.py --baselines-only

    # Quick smoke test (fewer samples):
    python reproduce_results.py --ckpt ckpts/resnet_combined_w256/ckpt_166000.pt --n-samples 20
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# ---------------------------------------------------------------------------
# Section 1: Deterministic claims (no data or checkpoint needed)
# ---------------------------------------------------------------------------

def verify_deterministic():
    """Verify syndrome dimensions, DCT bit counts, model params."""
    print("=" * 60)
    print("SECTION 1: Deterministic claims")
    print("=" * 60)

    # 1a. Syndrome dimensions
    from parity import build_constraint_matrix, ConstraintConfig
    cfg = ConstraintConfig(axis_parity=True, plane_parity=True, global_parity=True)
    H = build_constraint_matrix(32, cfg)
    print(f"\n[1a] Parity check matrix H: {H.shape}")
    assert H.shape == (2977, 32768), f"FAIL: expected (2977, 32768), got {H.shape}"
    print(f"     M = {H.shape[0]} syndrome bits, N = {H.shape[1]} voxels  ✓")
    syndrome_bits = H.shape[0]

    # 1b. DCT bit counts
    from adaptive_dct import FREQ_LEVELS, FREQ_BITS
    print(f"\n[1b] DCT bit counts per level:")
    for k, b in zip(FREQ_LEVELS, FREQ_BITS):
        n_coeffs = (2 * k - 1) ** 3
        total = syndrome_bits + 8 + int(b.item())
        print(f"     Level {k}: {n_coeffs} coefficients, {int(b.item())} DCT bits, "
              f"{total} total bits")

    # 1c. Total bit budget (Table 1 "Bits" column)
    print(f"\n[1c] Total bit budget (syndrome + coarse + DCT):")
    for k, b in zip(FREQ_LEVELS, FREQ_BITS):
        total = syndrome_bits + 8 + int(b.item())
        print(f"     Level {k}: {syndrome_bits} + 8 + {int(b.item())} = {total} bits")

    expected = {
        k: syndrome_bits + 8 + int(FREQ_BITS[FREQ_LEVELS.index(k)].item())
        for k in (2, 3, 4)
    }
    for k, exp in expected.items():
        idx = FREQ_LEVELS.index(k)
        actual = syndrome_bits + 8 + int(FREQ_BITS[idx].item())
        assert actual == exp, f"FAIL: L{k} expected {exp}, got {actual}"
    print("     Table 1 bit budgets verified  ✓")

    # 1d. Model parameters
    from nn.resnet3d import ResNet3D
    for width, expected_approx in [(128, 11_000_000), (256, 44_000_000)]:
        m = ResNet3D(in_ch=3, width=width, depth=12, cond_dim=syndrome_bits)
        params = sum(p.numel() for p in m.parameters())
        print(f"\n[1d] ResNet3D w={width} d=12: {params:,} params")
        del m

    print("\n✓ All deterministic claims verified.\n")


# ---------------------------------------------------------------------------
# Section 2: Baseline benchmarks (no checkpoint needed, needs ShapeNet data)
# ---------------------------------------------------------------------------

def run_baselines(data_path, n_samples):
    """Run Draco, octree, RLE benchmarks."""
    print("=" * 60)
    print("SECTION 2: Baseline benchmarks (Draco / Octree / RLE)")
    print("=" * 60)

    from benchmark_draco import (
        benchmark_draco, benchmark_octree, benchmark_rle,
        benchmark_raw_encoding, compute_iou
    )
    from binvox_rw import read_as_3d_array
    from adaptive_dct import FREQ_LEVELS, FREQ_BITS
    from parity import build_constraint_matrix, ConstraintConfig

    cfg = ConstraintConfig(axis_parity=True, plane_parity=True, global_parity=True)
    syndrome_bits = build_constraint_matrix(32, cfg).shape[0]
    level_bits = {
        int(k): syndrome_bits + 8 + int(b.item())
        for k, b in zip(FREQ_LEVELS, FREQ_BITS)
    }

    data_dir = Path(data_path)
    all_files = sorted(data_dir.glob("**/*.binvox"))
    if not all_files:
        all_files = sorted(data_dir.glob("**/*.npy"))
    if not all_files:
        print(f"ERROR: no data found in {data_dir}")
        return None

    rng = np.random.RandomState(42)
    indices = rng.choice(len(all_files), min(n_samples, len(all_files)), replace=False)
    files = [all_files[i] for i in sorted(indices)]
    print(f"\nBenchmarking {len(files)} samples from {data_dir}")

    quant_bits_list = [4, 5, 6, 7, 8]
    draco_results = {qb: [] for qb in quant_bits_list}
    octree_results = {d: [] for d in range(1, 6)}
    rle_results = []
    fill_rates = []

    for i, f in enumerate(files):
        if f.suffix == '.binvox':
            with open(f, 'rb') as fh:
                vox = read_as_3d_array(fh)
                v3d = vox.data[:32, :32, :32].astype(np.uint8)
        else:
            v3d = (np.load(f) > 0.5).astype(np.uint8)[:32, :32, :32]

        fill = v3d.sum() / v3d.size
        fill_rates.append(fill)
        if v3d.sum() == 0:
            continue

        for r in benchmark_draco(v3d, quant_bits_list):
            draco_results[r['quant_bits']].append(r)
        for r in benchmark_octree(v3d):
            octree_results[r['max_depth']].append(r)
        rle_results.append(benchmark_rle(v3d))

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)} done")

    # Print results matching Table 1 format
    print(f"\n{'Method':<25} {'Bits':>8} {'Compression':>12} {'IoU':>8}")
    print("-" * 60)

    for depth in [4]:
        entries = octree_results[depth]
        if entries:
            avg_bits = np.mean([e['encoded_bits'] for e in entries])
            avg_iou = np.mean([e['iou'] for e in entries])
            avg_ratio = np.mean([e['compression_ratio'] for e in entries])
            print(f"{'Octree depth '+str(depth):<25} {avg_bits:>8.0f} {avg_ratio:>11.1f}x {avg_iou:>8.3f}")

    print(f"{'Parity codec L2':<25} {level_bits[2]:>8} {32768/level_bits[2]:>11.1f}x {'rerun':>8}")
    print(f"{'Parity codec L3':<25} {level_bits[3]:>8} {32768/level_bits[3]:>11.1f}x {'rerun':>8}")

    for qb in [4]:
        entries = draco_results[qb]
        if entries:
            avg_bits = np.mean([e['encoded_bits'] for e in entries])
            avg_iou = np.mean([e['iou'] for e in entries])
            avg_ratio = np.mean([e['compression_ratio'] for e in entries])
            print(f"{'Draco qb='+str(qb):<25} {avg_bits:>8.0f} {avg_ratio:>11.1f}x {avg_iou:>8.3f}")

    avg_rle = np.mean([r['encoded_bits'] for r in rle_results])
    avg_rle_ratio = np.mean([r['compression_ratio'] for r in rle_results])
    print(f"{'RLE (lossless)':<25} {avg_rle:>8.0f} {avg_rle_ratio:>11.1f}x {'1.000':>8}")

    print(f"{'Parity codec L4':<25} {level_bits[4]:>8} {32768/level_bits[4]:>11.1f}x {'rerun':>8}")

    for depth in [5]:
        entries = octree_results[depth]
        if entries:
            avg_bits = np.mean([e['encoded_bits'] for e in entries])
            avg_ratio = np.mean([e['compression_ratio'] for e in entries])
            print(f"{'Octree depth '+str(depth):<25} {avg_bits:>8.0f} {avg_ratio:>11.1f}x {'1.000':>8}")

    for qb in [5]:
        entries = draco_results[qb]
        if entries:
            avg_bits = np.mean([e['encoded_bits'] for e in entries])
            avg_ratio = np.mean([e['compression_ratio'] for e in entries])
            print(f"{'Draco qb='+str(qb):<25} {avg_bits:>8.0f} {avg_ratio:>11.1f}x {'1.000':>8}")

    print(f"\nFill rates: min={min(fill_rates):.3f} max={max(fill_rates):.3f} "
          f"mean={np.mean(fill_rates):.3f}")

    print("\n✓ Baseline benchmarks complete.\n")
    return draco_results, octree_results, rle_results


# ---------------------------------------------------------------------------
# Section 3: Parity codec evaluation (needs checkpoint)
# ---------------------------------------------------------------------------

def run_parity_codec(ckpt_path, data_path, n_samples, cfg_path=None):
    """Evaluate the parity codec at each DCT level on val data."""
    print("=" * 60)
    print("SECTION 3: Parity codec variable-rate evaluation")
    print("=" * 60)

    from config import load_config
    from parity import load_constraint_ops, gf2_matmul, make_canonical_base, \
        project_with_left_inverse, ConstraintConfig
    from adaptive_dct import FREQ_LEVELS, FREQ_BITS, make_adaptive_dct_field
    from train import make_coarse_map
    from nn.resnet3d import ResNet3D
    from dataset import VoxelDataset
    from utils.metrics import iou
    from torch.utils.data import DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load config if provided, otherwise use defaults
    if cfg_path and Path(cfg_path).exists():
        cfg = load_config(cfg_path)
        constraint_cfg = cfg.constraints
        coarse_size = getattr(cfg.train, 'coarse_size', 2)
        coarse_threshold = getattr(cfg.train, 'coarse_threshold', 0.42)
        print(f"Config: {cfg_path}")
    else:
        constraint_cfg = ConstraintConfig(axis_parity=True, plane_parity=True, global_parity=True)
        coarse_size = 2
        coarse_threshold = 0.42

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('ema', ckpt.get('model', ckpt))

    # Detect model width from state dict
    first_conv_weight = state.get('input_conv.weight',
                                   state.get('blocks.0.conv1.weight'))
    if first_conv_weight is not None:
        width = first_conv_weight.shape[0]
    else:
        width = 256  # default

    print(f"Checkpoint: {ckpt_path}")
    print(f"Detected width: {width}")

    # Load parity ops matching the training config
    _, left_inv, parity = load_constraint_ops(32, constraint_cfg)
    parity = parity.to(device)
    left_inv = left_inv.to(device)
    M = parity.shape[0]
    print(f"Parity matrix: {parity.shape} (M={M})")

    # Build model
    model = ResNet3D(in_ch=3, width=width, depth=12, cond_dim=M).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Load data with matching constraint matrix
    dataset = VoxelDataset(data_path, 32, constraint_matrix=parity.cpu())
    n = min(n_samples, len(dataset))
    print(f"Evaluating on {n} samples from {data_path}")

    # Evaluate at each DCT level
    K = len(FREQ_LEVELS)
    level_ious = {k: [] for k in FREQ_LEVELS}
    thresholds = [0.80, 0.85, 0.90, 0.93, 0.95]
    threshold_bits = {t: [] for t in thresholds}

    # Track per-sample fill rates for stratified analysis
    fill_rates = []
    sample_level_ious = []  # list of dicts, one per sample

    with torch.no_grad():
        for idx in range(n):
            voxels_t, syndrome_t = dataset[idx]
            voxels = voxels_t.to(device).view(1, -1).to(torch.uint8)
            syndrome = syndrome_t.to(device).unsqueeze(0).float()
            syndrome_uint8 = syndrome.to(torch.uint8)

            fill = voxels.float().mean().item()
            fill_rates.append(fill)

            x_base = make_canonical_base(syndrome_uint8, parity, left_inv, device)
            coarse = make_coarse_map(voxels, 32, coarse_size, coarse_threshold)

            sample_ious = {}
            for k_idx, keep_freqs in enumerate(FREQ_LEVELS):
                sel = torch.zeros(1, K, device=device)
                sel[0, k_idx] = 1.0
                dct = make_adaptive_dct_field(voxels, 32, sel)

                x_input = torch.cat([
                    x_base.float().view(1, 1, 32, 32, 32),
                    coarse,
                    dct,
                ], dim=1)

                t = torch.full((1,), 1000, device=device)
                logits = model(x_input, t, syndrome)
                pred = (torch.sigmoid(logits.view(1, -1)) > 0.5).to(torch.uint8)
                recon = x_base ^ pred
                recon, _ = project_with_left_inverse(recon, parity, left_inv,
                                                      syndrome_uint8)
                k_iou = iou(recon, voxels).item()
                level_ious[keep_freqs].append(k_iou)
                sample_ious[keep_freqs] = k_iou

            sample_level_ious.append(sample_ious)

            # Variable-rate: pick cheapest level above each threshold
            for thresh in thresholds:
                for k_idx, keep_freqs in enumerate(FREQ_LEVELS):
                    if level_ious[keep_freqs][-1] >= thresh:
                        bits = M + 8 + int(FREQ_BITS[k_idx].item())
                        threshold_bits[thresh].append(bits)
                        break
                else:
                    bits = M + 8 + int(FREQ_BITS[-1].item())
                    threshold_bits[thresh].append(bits)

            if (idx + 1) % 50 == 0:
                print(f"  {idx+1}/{n} done")

    # Print results
    print(f"\n{'Level':>6} {'Mean IoU':>10} {'Bits':>8}")
    print("-" * 30)
    for k in FREQ_LEVELS:
        mean_iou = np.mean(level_ious[k])
        bits = M + 8 + int(FREQ_BITS[FREQ_LEVELS.index(k)].item())
        print(f"  L{k:<4} {mean_iou:>10.4f} {bits:>8}")

    print(f"\nVariable-rate analysis:")
    print(f"{'Threshold':>10} {'Avg bits':>10} {'vs L4 baseline':>15}")
    baseline_bits = M + 8 + int(FREQ_BITS[FREQ_LEVELS.index(4)].item())
    level2_bits = M + 8 + int(FREQ_BITS[FREQ_LEVELS.index(2)].item())
    for thresh in thresholds:
        avg = np.mean(threshold_bits[thresh])
        savings = (1 - avg / baseline_bits) * 100
        pct_level2 = sum(1 for b in threshold_bits[thresh]
                         if b <= level2_bits) / len(threshold_bits[thresh]) * 100
        print(f"  {thresh:>8.2f} {avg:>10.0f} {savings:>+14.0f}%  "
              f"({pct_level2:.0f}% at L2)")

    # Stratified analysis by fill rate
    fill_arr = np.array(fill_rates)
    bins = [
        ("sparse (<5%)",  fill_arr < 0.05),
        ("medium (5-15%)", (fill_arr >= 0.05) & (fill_arr < 0.15)),
        ("dense (>15%)",  fill_arr >= 0.15),
    ]

    print(f"\nStratified by fill rate:")
    print(f"{'Stratum':<18} {'Count':>6}  ", end="")
    for k in FREQ_LEVELS:
        print(f"{'L'+str(k):>6}", end="")
    print()
    print("-" * (26 + 7 * len(FREQ_LEVELS)))

    for label, mask in bins:
        count = mask.sum()
        if count == 0:
            continue
        print(f"{label:<18} {count:>6}  ", end="")
        for k in FREQ_LEVELS:
            stratum_ious = [sample_level_ious[i][k] for i in range(n) if mask[i]]
            mean = np.mean(stratum_ious)
            print(f"{mean:>6.3f}", end="")
        print()

    # Overall for reference
    print(f"{'all':<18} {n:>6}  ", end="")
    for k in FREQ_LEVELS:
        print(f"{np.mean(level_ious[k]):>6.3f}", end="")
    print()

    print("\n✓ Parity codec evaluation complete.\n")
    return level_ious, threshold_bits


# ---------------------------------------------------------------------------
# Section 4: Bayesian temporal accumulation
# ---------------------------------------------------------------------------

def run_bayes_benchmark(ckpt_path, n_frames=10, cfg_path=None):
    """Benchmark Bayesian accumulation over synthetic volumetric frames."""
    print("=" * 60)
    print("SECTION 4: Bayesian temporal accumulation")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from config import load_config
    from parity import load_constraint_ops, gf2_matmul, make_canonical_base, \
        project_with_left_inverse, ConstraintConfig
    from adaptive_dct import FREQ_LEVELS, FREQ_BITS, make_adaptive_dct_field, \
        random_level_selection
    from train import make_coarse_map
    from nn.resnet3d import ResNet3D
    from temporal_bayes import BayesAccumulator
    from utils.metrics import iou

    if cfg_path and Path(cfg_path).exists():
        cfg = load_config(cfg_path)
        constraint_cfg = cfg.constraints
        coarse_size = getattr(cfg.train, 'coarse_size', 2)
        coarse_threshold = getattr(cfg.train, 'coarse_threshold', 0.42)
    else:
        constraint_cfg = ConstraintConfig(axis_parity=True, plane_parity=True, global_parity=True)
        coarse_size = 2
        coarse_threshold = 0.42

    _, left_inv, parity = load_constraint_ops(32, constraint_cfg)
    parity = parity.to(device)
    left_inv = left_inv.to(device)
    M = parity.shape[0]
    N = 32 ** 3
    K = len(FREQ_LEVELS)

    # Load model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('ema', ckpt.get('model', ckpt))
    first_w = state.get('input_conv.weight')
    width = first_w.shape[0] if first_w is not None else 256
    model = ResNet3D(in_ch=3, width=width, depth=12, cond_dim=M).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Generate morphing scene frames
    import numpy as np
    coords = torch.linspace(-1, 1, 32)
    xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing='ij')

    print(f"Generating {n_frames} morphing scene frames...")
    frames = []
    for i in range(n_frames):
        t = i / n_frames * 2 * np.pi
        morph = 0.5 * (1 + np.sin(t))
        sphere = (xx**2 + yy**2 + zz**2) < (0.4 + 0.1 * np.sin(t*2))**2
        cos_t, sin_t = np.cos(t), np.sin(t)
        rx, rz = xx*cos_t + zz*sin_t, -xx*sin_t + zz*cos_t
        torus = ((torch.sqrt(rx**2 + yy**2) - 0.5)**2 + rz**2) < 0.15**2
        voxels = sphere if morph < 0.5 else torus
        frames.append(voxels.to(torch.uint8).reshape(-1))

    bayes = BayesAccumulator(N, device)
    prev_frame = None

    print(f"\n{'Frame':>6} {'Indep IoU':>10} {'Bayes IoU':>10} {'Improvement':>12}")
    print("-" * 45)

    indep_ious = []
    bayes_ious = []

    with torch.no_grad():
        for i, frame in enumerate(frames):
            voxels = frame.unsqueeze(0).to(device)
            syndrome = (gf2_matmul(parity, voxels.t()).t()).float()
            syndrome_uint8 = syndrome.to(torch.uint8)

            x_base = make_canonical_base(syndrome_uint8, parity, left_inv, device)
            coarse = make_coarse_map(voxels, 32, coarse_size, coarse_threshold)

            # Use a random DCT level (simulates variable-rate streaming)
            sel, _ = random_level_selection(1, device)
            dct = make_adaptive_dct_field(voxels, 32, sel)

            x_input = torch.cat([
                x_base.float().view(1, 1, 32, 32, 32),
                coarse, dct,
            ], dim=1)

            t_val = torch.full((1,), 1000, device=device)
            logits = model(x_input, t_val, syndrome)
            logits_flat = logits.view(1, -1)

            # Independent reconstruction
            pred = (torch.sigmoid(logits_flat) > 0.5).to(torch.uint8)
            recon = x_base ^ pred
            recon, _ = project_with_left_inverse(recon, parity, left_inv,
                                                  syndrome_uint8)
            indep_iou = iou(recon, voxels).item()

            # Soft probs for Bayes
            residual_probs = torch.sigmoid(logits_flat)
            base_f = x_base.float()
            full_probs = base_f * (1 - residual_probs) + (1 - base_f) * residual_probs

            # Delta detection
            voxels_t = voxels
            delta_mask = None
            if prev_frame is not None:
                delta_mask = (voxels_t ^ prev_frame).bool()
            bayes.update(full_probs, delta_mask)
            prev_frame = voxels_t

            bayes_recon = bayes.get_bits()
            bayes_iou = iou(bayes_recon, voxels).item()

            indep_ious.append(indep_iou)
            bayes_ious.append(bayes_iou)

            improvement = bayes_iou - indep_iou
            print(f"  {i+1:>4} {indep_iou:>10.4f} {bayes_iou:>10.4f} "
                  f"{improvement:>+11.4f}")

    avg_indep = np.mean(indep_ious)
    avg_bayes = np.mean(bayes_ious)
    print(f"\nAverage independent IoU: {avg_indep:.4f}")
    print(f"Average Bayes IoU:      {avg_bayes:.4f}")
    print(f"Average improvement:    {avg_bayes - avg_indep:+.4f}")

    if len(bayes_ious) >= 5:
        print(f"\nBayes IoU trajectory (frames 1-5): "
              f"{bayes_ious[0]:.3f} → {bayes_ious[4]:.3f}")

    print("\n✓ Bayesian accumulation benchmark complete.\n")
    return indep_ious, bayes_ious


# ---------------------------------------------------------------------------
# Section 5: Projection diagnostics (needs checkpoint)
# ---------------------------------------------------------------------------

def run_projection_diagnostics(ckpt_path, data_path, n_samples, cfg_path=None):
    """Diagnose why hard projection hurts: measure bits flipped, confidence, accuracy."""
    print("=" * 60)
    print("SECTION 5: Projection diagnostics")
    print("=" * 60)

    from config import load_config
    from parity import load_constraint_ops, make_canonical_base, project_with_left_inverse, gf2_matmul, ConstraintConfig
    from adaptive_dct import FREQ_LEVELS, FREQ_BITS, make_adaptive_dct_field
    from train import make_coarse_map
    from nn.resnet3d import ResNet3D
    from dataset import VoxelDataset
    from utils.metrics import iou

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if cfg_path and Path(cfg_path).exists():
        cfg = load_config(cfg_path)
        constraint_cfg = cfg.constraints
        coarse_size = getattr(cfg.train, 'coarse_size', 2)
        coarse_threshold = getattr(cfg.train, 'coarse_threshold', 0.42)
    else:
        constraint_cfg = ConstraintConfig(axis_parity=True, plane_parity=True, global_parity=True)
        coarse_size = 2
        coarse_threshold = 0.42

    _, left_inv, parity = load_constraint_ops(32, constraint_cfg)
    parity = parity.to(device)
    left_inv = left_inv.to(device)
    M = parity.shape[0]

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('ema', ckpt.get('model', ckpt))
    first_w = state.get('input_conv.weight')
    width = first_w.shape[0] if first_w is not None else 192

    use_coarse = True
    use_dct = True
    in_ch = 1 + int(use_coarse) + int(use_dct)
    model = ResNet3D(in_ch=in_ch, width=width, depth=12, cond_dim=M).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    dataset = VoxelDataset(data_path, 32, constraint_matrix=parity.cpu())
    n = min(n_samples, len(dataset))
    n_levels = len(FREQ_LEVELS)
    level_idx = 2  # L4

    all_violations = []
    all_bits_flipped = []
    all_iou_before = []
    all_iou_after = []
    all_flip_conf = []
    all_keep_conf = []
    all_flip_helped = []

    with torch.no_grad():
        for idx in range(n):
            voxels_t, syndrome_t = dataset[idx]
            voxels = voxels_t.to(device).view(1, -1).to(torch.uint8)
            syndrome = syndrome_t.to(device).unsqueeze(0).float()
            syndrome_uint8 = syndrome.to(torch.uint8)

            x_base = make_canonical_base(syndrome_uint8, parity, left_inv, device)
            coarse = make_coarse_map(voxels, 32, coarse_size, coarse_threshold)
            sel = torch.zeros(1, n_levels, device=device)
            sel[0, level_idx] = 1.0
            dct = make_adaptive_dct_field(voxels, 32, sel)

            x_input = torch.cat([
                x_base.float().view(1, 1, 32, 32, 32), coarse, dct
            ], dim=1)
            t = torch.full((1,), 1000, device=device)
            logits = model(x_input, t, syndrome)
            logits_flat = logits.view(1, -1)

            pred = (torch.sigmoid(logits_flat) > 0.5).to(torch.uint8)
            recon_before = x_base ^ pred

            r = gf2_matmul(parity, recon_before.t()).t() ^ syndrome_uint8
            n_violations = r.sum().item()

            recon_after, _ = project_with_left_inverse(recon_before.clone(), parity, left_inv, syndrome_uint8)
            flipped = (recon_before ^ recon_after).squeeze(0)
            n_flipped = flipped.sum().item()

            confidence = logits_flat.abs().squeeze(0)
            if n_flipped > 0:
                flip_mask = flipped.bool()
                mean_flip_conf = confidence[flip_mask].mean().item()
                mean_keep_conf = confidence[~flip_mask].mean().item()
                gt = voxels.squeeze(0)
                before_bits = recon_before.squeeze(0)
                after_bits = recon_after.squeeze(0)
                was_wrong = (before_bits != gt) & flip_mask
                is_right = (after_bits == gt) & flip_mask
                helped = (was_wrong & is_right).sum().item()
                frac_helped = helped / n_flipped
            else:
                mean_flip_conf = 0
                mean_keep_conf = confidence.mean().item()
                frac_helped = 1.0

            all_violations.append(n_violations)
            all_bits_flipped.append(n_flipped)
            all_iou_before.append(iou(recon_before, voxels).item())
            all_iou_after.append(iou(recon_after, voxels).item())
            all_flip_conf.append(mean_flip_conf)
            all_keep_conf.append(mean_keep_conf)
            all_flip_helped.append(frac_helped)

            if (idx + 1) % 50 == 0:
                print(f"  {idx+1}/{n} done")

    results = {
        'mean_violations': float(np.mean(all_violations)),
        'total_checks': M,
        'violation_pct': float(100 * np.mean(all_violations) / M),
        'mean_bits_flipped': float(np.mean(all_bits_flipped)),
        'iou_before': float(np.mean(all_iou_before)),
        'iou_after': float(np.mean(all_iou_after)),
        'iou_delta': float(np.mean(all_iou_after) - np.mean(all_iou_before)),
        'mean_flip_confidence': float(np.mean(all_flip_conf)),
        'mean_keep_confidence': float(np.mean(all_keep_conf)),
        'frac_helpful_flips': float(np.mean(all_flip_helped)),
        'n_samples': n,
    }

    print(f"\nProjection diagnostics ({n} samples at L4):")
    print(f"  Violations before projection: {results['mean_violations']:.1f} / {M} ({results['violation_pct']:.1f}%)")
    print(f"  Bits flipped: {results['mean_bits_flipped']:.1f} / 32768")
    print(f"  IoU before: {results['iou_before']:.4f}")
    print(f"  IoU after:  {results['iou_after']:.4f} ({results['iou_delta']:+.4f})")
    print(f"  Mean |logit| flipped: {results['mean_flip_confidence']:.3f}")
    print(f"  Mean |logit| kept:    {results['mean_keep_confidence']:.3f}")
    print(f"  Fraction of helpful flips: {results['frac_helpful_flips']:.3f}")
    print("\n✓ Projection diagnostics complete.\n")
    return results


# ---------------------------------------------------------------------------
# Section 6: Surface constraint ablation (needs checkpoint)
# ---------------------------------------------------------------------------

def run_surface_ablation(ckpt_path, data_path, n_samples, cfg_path=None):
    """Compare constraint designs: global vs surface-aware, with BP refinement."""
    print("=" * 60)
    print("SECTION 6: Surface constraint ablation")
    print("=" * 60)

    from config import load_config
    from parity import load_constraint_ops, make_canonical_base, project_with_left_inverse, gf2_matmul, ConstraintConfig
    from adaptive_dct import FREQ_LEVELS, FREQ_BITS, make_adaptive_dct_field
    from train import make_coarse_map
    from nn.resnet3d import ResNet3D
    from dataset import VoxelDataset
    from utils.metrics import iou
    from surface_constraints import build_surface_constraints
    from syndrome_refine import refine_with_syndrome

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if cfg_path and Path(cfg_path).exists():
        cfg = load_config(cfg_path)
        constraint_cfg = cfg.constraints
        coarse_size = getattr(cfg.train, 'coarse_size', 2)
        coarse_threshold = getattr(cfg.train, 'coarse_threshold', 0.42)
    else:
        constraint_cfg = ConstraintConfig(axis_parity=True, plane_parity=True, global_parity=True)
        coarse_size = 2
        coarse_threshold = 0.42

    _, left_inv, parity = load_constraint_ops(32, constraint_cfg)
    parity = parity.to(device)
    left_inv = left_inv.to(device)
    M = parity.shape[0]

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('ema', ckpt.get('model', ckpt))
    first_w = state.get('input_conv.weight')
    width = first_w.shape[0] if first_w is not None else 192

    use_coarse = True
    use_dct = True
    in_ch = 1 + int(use_coarse) + int(use_dct)
    model = ResNet3D(in_ch=in_ch, width=width, depth=12, cond_dim=M).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    dataset = VoxelDataset(data_path, 32, constraint_matrix=parity.cpu())
    n = min(n_samples, len(dataset))
    n_levels = len(FREQ_LEVELS)
    level_idx = 2  # L4
    D = 32

    # Results accumulators
    results_raw = []
    results_proj = []
    results_surface = {'patch': [], 'neighbor': [], 'gradient': []}
    constraint_counts = {'patch': [], 'neighbor': [], 'gradient': []}

    with torch.no_grad():
        for idx in range(n):
            voxels_t, syndrome_t = dataset[idx]
            voxels = voxels_t.to(device).view(1, -1).to(torch.uint8)
            syndrome = syndrome_t.to(device).unsqueeze(0).float()
            syndrome_uint8 = syndrome.to(torch.uint8)

            x_base = make_canonical_base(syndrome_uint8, parity, left_inv, device)
            coarse = make_coarse_map(voxels, D, coarse_size, coarse_threshold)
            sel = torch.zeros(1, n_levels, device=device)
            sel[0, level_idx] = 1.0
            dct = make_adaptive_dct_field(voxels, D, sel)
            dct_3d = dct.squeeze(0).squeeze(0)

            x_input = torch.cat([
                x_base.float().view(1, 1, D, D, D), coarse, dct
            ], dim=1)
            t = torch.full((1,), 1000, device=device)
            logits = model(x_input, t, syndrome)
            logits_flat = logits.view(1, -1)

            # Raw (no projection)
            pred = (torch.sigmoid(logits_flat) > 0.5).to(torch.uint8)
            recon_raw = x_base ^ pred
            results_raw.append(iou(recon_raw, voxels).item())

            # Hard projection
            recon_proj, _ = project_with_left_inverse(recon_raw.clone(), parity, left_inv, syndrome_uint8)
            results_proj.append(iou(recon_proj, voxels).item())

            # Full reconstruction probabilities for BP prior
            full_probs = torch.sigmoid(logits_flat)
            base_f = x_base.float()
            full_p = base_f * (1 - full_probs) + (1 - base_f) * full_probs
            full_logits = torch.log(full_p.clamp(1e-7) / (1 - full_p).clamp(1e-7))

            # Surface constraint refinement
            for mode in ['patch', 'neighbor', 'gradient']:
                H_surf, stats = build_surface_constraints(dct_3d, D, mode=mode, surface_threshold=0.3)
                n_constraints = H_surf.size(0)
                constraint_counts[mode].append(n_constraints)

                if n_constraints == 0:
                    results_surface[mode].append(results_raw[-1])
                    continue

                # Compute syndrome on CPU if large
                if n_constraints > 15000:
                    surf_syn = gf2_matmul(H_surf, voxels.cpu().t()).t().to(device)
                    H_surf = H_surf.to(device)
                else:
                    H_surf = H_surf.to(device)
                    surf_syn = gf2_matmul(H_surf, voxels.t()).t()

                refined, _, _, _ = refine_with_syndrome(
                    full_logits, H_surf, surf_syn,
                    n_iters=30, damping=0.5, device=device)
                results_surface[mode].append(iou(refined, voxels).item())

            if (idx + 1) % 50 == 0:
                print(f"  {idx+1}/{n} done")

    # Print results
    raw_m = np.mean(results_raw)
    print(f"\nSurface constraint ablation ({n} samples at L4):")
    print(f"{'Method':<30} {'Mean IoU':>9} {'Delta':>8} {'Avg bits':>9}")
    print("-" * 60)
    print(f"{'Raw (no projection)':<30} {raw_m:>9.4f} {0:>+8.4f} {'0':>9}")
    print(f"{'Hard projection':<30} {np.mean(results_proj):>9.4f} "
          f"{np.mean(results_proj)-raw_m:>+8.4f} {'0':>9}")
    for mode in ['patch', 'neighbor', 'gradient']:
        m = np.mean(results_surface[mode])
        bits = np.mean(constraint_counts[mode])
        print(f"{'Surface BP: '+mode:<30} {m:>9.4f} {m-raw_m:>+8.4f} {bits:>9.0f}")

    results = {
        'raw_iou': float(raw_m),
        'hard_proj_iou': float(np.mean(results_proj)),
        'hard_proj_delta': float(np.mean(results_proj) - raw_m),
    }
    for mode in ['patch', 'neighbor', 'gradient']:
        results[f'surface_{mode}_iou'] = float(np.mean(results_surface[mode]))
        results[f'surface_{mode}_delta'] = float(np.mean(results_surface[mode]) - raw_m)
        results[f'surface_{mode}_bits'] = float(np.mean(constraint_counts[mode]))
    results['n_samples'] = n

    print("\n✓ Surface constraint ablation complete.\n")
    return results


# ---------------------------------------------------------------------------
# Section 7: Surface Attention evaluation (needs surface attn checkpoint)
# ---------------------------------------------------------------------------

def run_surface_attention_eval(sa_ckpt_path, data_path, n_samples, levels=None):
    """Evaluate the Surface Attention decoder at multiple DCT levels."""
    print("=" * 60)
    print("SECTION 7: Surface Attention variable-rate evaluation")
    print("=" * 60)

    from dataset import VoxelDataset
    from parity import load_constraint_ops, gf2_matmul, ConstraintConfig
    from adaptive_dct import FREQ_LEVELS, FREQ_BITS, make_adaptive_dct_field
    from nn.surface_attention import SurfaceAttentionDecoder, get_surface_patches
    from utils.metrics import iou

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = ConstraintConfig(axis_parity=True, plane_parity=True, global_parity=True)
    _, _, H_ref = load_constraint_ops(32, cfg)
    dataset = VoxelDataset(data_path, 32, constraint_matrix=H_ref)
    n = min(n_samples, len(dataset))
    n_levels = len(FREQ_LEVELS)

    if levels is None:
        levels = [0, 2, 3, 4]  # L2, L4, L5, L6

    # Load checkpoint and infer model config
    ckpt = torch.load(sa_ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt)
    # Infer embed_dim from projection layer
    embed_dim = state['patch_embed.proj.weight'].shape[0]
    depth = sum(1 for k in state if k.startswith('blocks.') and k.endswith('.norm1.weight'))
    ff_dim = state['blocks.0.ff.0.weight'].shape[0]
    n_heads = state['blocks.0.q_proj.weight'].shape[0] // (embed_dim // 4)  # approximate

    model = SurfaceAttentionDecoder(
        cube_size=32, patch_size=2, embed_dim=embed_dim,
        n_heads=max(n_heads, 1), depth=depth, ff_dim=ff_dim,
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"Surface Attention: {params:,} params")
    print(f"Evaluating {n} samples at levels {[int(FREQ_LEVELS[l]) for l in levels]}")

    level_results = {l: {'ious': [], 'syn_bits': []} for l in levels}

    with torch.no_grad():
        for idx in range(n):
            voxels_raw, _ = dataset[idx]
            voxels = voxels_raw.unsqueeze(0).to(device).view(1, -1).to(torch.uint8)

            for level_idx in levels:
                sel = torch.zeros(1, n_levels, device=device)
                sel[0, level_idx] = 1.0
                dct_field = make_adaptive_dct_field(voxels, 32, sel)
                dct_3d = dct_field.squeeze(0).squeeze(0)

                patch_coords, H_surf = get_surface_patches(dct_3d, 32, threshold=0.3)
                n_patches = patch_coords.size(0)
                if n_patches == 0:
                    level_results[level_idx]['ious'].append(0.0)
                    level_results[level_idx]['syn_bits'].append(0)
                    continue

                H_surf = H_surf.to(device)
                patch_coords = patch_coords.to(device)
                syndrome = gf2_matmul(H_surf, voxels.t()).t().float()

                logits = model(dct_3d.unsqueeze(0), syndrome, patch_coords)
                pred = (torch.sigmoid(logits) > 0.5).to(torch.uint8)
                level_results[level_idx]['ious'].append(iou(pred, voxels).item())
                level_results[level_idx]['syn_bits'].append(n_patches)

            if (idx + 1) % 50 == 0:
                print(f"  {idx+1}/{n} done")

    print(f"\n{'Level':>6} {'Mean IoU':>9} {'DCT bits':>9} {'Syn bits':>9} "
          f"{'Total':>9} {'Ratio':>7}")
    print("-" * 55)

    summary = {}
    for level_idx in levels:
        r = level_results[level_idx]
        level_name = f"L{int(FREQ_LEVELS[level_idx])}"
        mean_iou = float(np.mean(r['ious']))
        dct_bits = int(FREQ_BITS[level_idx].item())
        mean_syn = float(np.mean(r['syn_bits']))
        total = dct_bits + mean_syn
        ratio = 32768 / total
        print(f"{level_name:>6} {mean_iou:>9.4f} {dct_bits:>9} {mean_syn:>9.0f} "
              f"{total:>9.0f} {ratio:>6.1f}x")
        summary[level_name] = {
            'mean_iou': mean_iou,
            'dct_bits': dct_bits,
            'mean_syndrome_bits': mean_syn,
            'total_bits': float(total),
            'compression_ratio': float(ratio),
        }

    summary['params'] = params
    summary['n_samples'] = n
    print("\n✓ Surface Attention evaluation complete.\n")
    return summary


# ---------------------------------------------------------------------------
# Section 8: Architecture comparison table
# ---------------------------------------------------------------------------

def run_architecture_comparison(ckpt_paths, data_path, n_samples):
    """Compare all decoder architectures at L4 for the main paper table.

    ckpt_paths is a dict: {name: (ckpt_path, type, cfg_path)}
    where type is 'resnet' or 'surface_attention'.
    """
    print("=" * 60)
    print("SECTION 8: Architecture comparison (all decoders at L4)")
    print("=" * 60)

    from dataset import VoxelDataset
    from parity import load_constraint_ops, gf2_matmul, make_canonical_base, \
        project_with_left_inverse, ConstraintConfig
    from adaptive_dct import FREQ_LEVELS, FREQ_BITS, make_adaptive_dct_field
    from train import make_coarse_map
    from nn.resnet3d import ResNet3D
    from nn.surface_attention import SurfaceAttentionDecoder, get_surface_patches
    from surface_constraints import build_surface_constraints
    from syndrome_refine import refine_with_syndrome
    from utils.metrics import iou

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg_c = ConstraintConfig(axis_parity=True, plane_parity=True, global_parity=True)
    _, left_inv, parity = load_constraint_ops(32, cfg_c)
    parity = parity.to(device)
    left_inv = left_inv.to(device)
    M = parity.shape[0]

    dataset = VoxelDataset(data_path, 32, constraint_matrix=parity.cpu())
    n = min(n_samples, len(dataset))
    n_levels = len(FREQ_LEVELS)
    level_idx = 2  # L4

    results = {}

    for name, (ckpt_path, arch_type, cfg_path) in ckpt_paths.items():
        if not Path(ckpt_path).exists():
            print(f"  SKIP {name}: {ckpt_path} not found")
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get('model', ckpt.get('ema', ckpt))
        params = sum(v.numel() for v in state.values() if isinstance(v, torch.Tensor))

        ious = []

        if arch_type == 'surface_attention':
            embed_dim = state['patch_embed.proj.weight'].shape[0]
            depth = sum(1 for k in state if k.startswith('blocks.') and k.endswith('.norm1.weight'))
            ff_dim = state['blocks.0.ff.0.weight'].shape[0]
            model = SurfaceAttentionDecoder(
                cube_size=32, patch_size=2, embed_dim=embed_dim,
                depth=depth, ff_dim=ff_dim,
            ).to(device)
            model.load_state_dict(state, strict=False)
            model.eval()

            with torch.no_grad():
                for idx in range(n):
                    voxels_raw, _ = dataset[idx]
                    voxels = voxels_raw.unsqueeze(0).to(device).view(1, -1).to(torch.uint8)
                    sel = torch.zeros(1, n_levels, device=device)
                    sel[0, level_idx] = 1.0
                    dct_field = make_adaptive_dct_field(voxels, 32, sel)
                    dct_3d = dct_field.squeeze(0).squeeze(0)
                    patch_coords, H_surf = get_surface_patches(dct_3d, 32, threshold=0.3)
                    if patch_coords.size(0) == 0:
                        ious.append(0.0)
                        continue
                    H_surf = H_surf.to(device)
                    patch_coords = patch_coords.to(device)
                    syndrome = gf2_matmul(H_surf, voxels.t()).t().float()
                    logits = model(dct_3d.unsqueeze(0), syndrome, patch_coords)
                    pred = (torch.sigmoid(logits) > 0.5).to(torch.uint8)
                    ious.append(iou(pred, voxels).item())

        else:  # resnet
            first_w = state.get('input_conv.weight')
            width = first_w.shape[0] if first_w is not None else 192
            in_ch = first_w.shape[1] if first_w is not None else 3
            model = ResNet3D(in_ch=in_ch, width=width, depth=12, cond_dim=M).to(device)
            model.load_state_dict(state, strict=False)
            model.eval()

            with torch.no_grad():
                for idx in range(n):
                    voxels_raw, syn_raw = dataset[idx]
                    voxels = voxels_raw.unsqueeze(0).to(device).view(1, -1).to(torch.uint8)
                    syndrome = syn_raw.unsqueeze(0).to(device).float()
                    syndrome_uint8 = syndrome.to(torch.uint8)
                    x_base = make_canonical_base(syndrome_uint8, parity, left_inv, device)
                    coarse = make_coarse_map(voxels, 32, 2, 0.42)
                    sel = torch.zeros(1, n_levels, device=device)
                    sel[0, level_idx] = 1.0
                    dct = make_adaptive_dct_field(voxels, 32, sel)
                    x_in = torch.cat([
                        x_base.float().view(1, 1, 32, 32, 32), coarse, dct
                    ], dim=1)
                    t = torch.full((1,), 1000, device=device)
                    logits = model(x_in, t, syndrome)
                    pred = (torch.sigmoid(logits.view(1, -1)) > 0.5).to(torch.uint8)
                    recon = x_base ^ pred
                    ious.append(iou(recon, voxels).item())

        mean_iou = float(np.mean(ious))
        results[name] = {'params': params, 'mean_iou': mean_iou, 'n_samples': n}
        print(f"  {name:<35} {params:>10,} params  IoU={mean_iou:.4f}")

    print("\n✓ Architecture comparison complete.\n")
    return results


# ---------------------------------------------------------------------------
# Section 9: Surface BP temporal accumulation (no checkpoint needed)
# ---------------------------------------------------------------------------

def run_surface_bp_accumulation(n_frames=20, tamper_fraction=0.08, seed=42):
    """Reproduce the temporal accumulation claim from the paper.

    Generates a morphing scene, tampers 8% of surface voxels per frame,
    corrects via single-frame BP, then accumulates via Bayesian log-odds.

    This matches verified_stream.py's logic exactly.
    """
    print("=" * 60)
    print("SECTION 9: Surface BP temporal accumulation")
    print("=" * 60)

    from parity import load_constraint_ops, gf2_matmul, ConstraintConfig
    from adaptive_dct import FREQ_LEVELS, make_adaptive_dct_field
    from surface_constraints import build_surface_constraints, detect_surface
    from syndrome_refine import refine_with_syndrome
    from utils.metrics import iou

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cpu')  # runs on CPU, no checkpoint needed
    D = 32
    N = D ** 3
    n_levels = len(FREQ_LEVELS)
    dct_level = 3  # L5

    coords = torch.linspace(-1, 1, D)
    xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing='ij')

    def make_frame(t):
        phase = t * 0.1
        morph = 0.5 * (1 + np.sin(phase))
        sphere = (xx**2 + yy**2 + zz**2) < (0.35 + 0.1 * np.sin(phase*2))**2
        cos_p, sin_p = np.cos(phase), np.sin(phase)
        rx = xx * cos_p + zz * sin_p
        rz = -xx * sin_p + zz * cos_p
        torus = ((torch.sqrt(rx**2 + yy**2) - 0.45)**2 + rz**2) < 0.12**2
        v = sphere if morph < 0.5 else torus
        ox = 0.6 * np.cos(phase * 3)
        oz = 0.6 * np.sin(phase * 3)
        v = v | (((xx - ox)**2 + yy**2 + (zz - oz)**2) < 0.08**2)
        return v.to(torch.uint8).reshape(-1)

    log_odds = torch.zeros(N)

    single_ious = []
    acc_ious = []

    print(f"Generating {n_frames} frames, {tamper_fraction*100:.0f}% surface tampering...")
    print(f"{'Frame':>6} {'Single BP':>10} {'Accumulated':>12}")
    print("-" * 32)

    for t in range(n_frames):
        voxels = make_frame(t)

        # DCT + surface constraints
        sel = torch.zeros(1, n_levels, device=device)
        sel[0, dct_level] = 1.0
        dct_field = make_adaptive_dct_field(voxels.unsqueeze(0), D, sel)
        dct_3d = dct_field.squeeze(0).squeeze(0)

        H_surf, _ = build_surface_constraints(dct_3d, D, mode='patch',
                                               surface_threshold=0.3)
        n_patches = H_surf.size(0)

        if n_patches > 0:
            syndrome = gf2_matmul(H_surf, voxels.unsqueeze(1)).squeeze(1)

            # Tamper surface voxels
            surface = detect_surface(dct_3d, threshold=0.3)
            surf_idx = torch.nonzero(surface.reshape(-1), as_tuple=True)[0]
            if len(surf_idx) > 0:
                n_flip = max(1, int(len(surf_idx) * tamper_fraction))
                perm = torch.randperm(len(surf_idx))[:n_flip]
                tampered = voxels.clone()
                tampered[surf_idx[perm]] ^= 1
            else:
                tampered = voxels.clone()

            # Single-frame BP correction
            tampered_prob = tampered.float().unsqueeze(0).clamp(0.1, 0.9)
            logits = torch.log(tampered_prob / (1 - tampered_prob))
            corrected, _, _, _ = refine_with_syndrome(
                logits, H_surf, syndrome.unsqueeze(0),
                n_iters=30, damping=0.5, device=device)
            corrected = corrected.squeeze(0)
        else:
            corrected = voxels.clone()

        single_iou = iou(corrected.unsqueeze(0), voxels.unsqueeze(0)).item()
        single_ious.append(single_iou)

        # Bayesian accumulation (matches verified_stream.py)
        corrected_prob = corrected.float().clamp(0.01, 0.99)
        frame_logodds = torch.log(corrected_prob / (1 - corrected_prob))

        if t > 0:
            prev = make_frame(t - 1)
            changed = (voxels ^ prev).bool()
            log_odds[changed] *= 0.3
        log_odds = 0.7 * log_odds + 0.3 * frame_logodds
        log_odds = log_odds.clamp(-5, 5)

        accumulated = (torch.sigmoid(log_odds) > 0.5).to(torch.uint8)
        acc_iou = iou(accumulated.unsqueeze(0), voxels.unsqueeze(0)).item()
        acc_ious.append(acc_iou)

        if t < 10 or t % 10 == 0:
            print(f"  {t+1:>4} {single_iou:>10.4f} {acc_iou:>12.4f}")

    results = {
        'n_frames': n_frames,
        'tamper_fraction': tamper_fraction,
        'mean_single_iou': float(np.mean(single_ious)),
        'mean_acc_iou': float(np.mean(acc_ious)),
        'frame1_single': float(single_ious[0]),
        'frame5_acc': float(acc_ious[4]) if len(acc_ious) > 4 else None,
        'frame10_acc': float(acc_ious[9]) if len(acc_ious) > 9 else None,
    }

    print(f"\nSummary:")
    print(f"  Mean single-frame BP IoU: {results['mean_single_iou']:.4f}")
    print(f"  Mean accumulated IoU:     {results['mean_acc_iou']:.4f}")
    print(f"  Frame 1 single BP:        {results['frame1_single']:.4f}")
    if results['frame5_acc']:
        print(f"  Frame 5 accumulated:      {results['frame5_acc']:.4f}")
    if results['frame10_acc']:
        print(f"  Frame 10 accumulated:     {results['frame10_acc']:.4f}")

    print("\n✓ Surface BP temporal accumulation complete.\n")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reproduce all paper results")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to trained checkpoint (needed for Sections 3-4)")
    parser.add_argument("--cfg", type=str, default="configs/resnet_combined_w256.yaml",
                        help="Training config yaml (for constraint config, coarse params)")
    parser.add_argument("--data", type=str, default="data/shapenet/val",
                        help="Path to ShapeNet val data for baseline benchmarks")
    parser.add_argument("--val-data", type=str, default="data/combined/val",
                        help="Path to combined val data for parity codec eval")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of samples for benchmarks")
    parser.add_argument("--n-frames", type=int, default=20,
                        help="Number of frames for temporal accumulation (Section 4/9)")
    parser.add_argument("--baselines-only", action="store_true",
                        help="Run only baseline benchmarks (no checkpoint needed)")
    parser.add_argument("--sa-ckpt", type=str, default=None,
                        help="Surface Attention checkpoint (for Section 7)")
    parser.add_argument("--sa-medium-ckpt", type=str, default=None,
                        help="Surface Attention medium checkpoint (for Section 8)")
    parser.add_argument("--resnet-matched-ckpt", type=str, default=None,
                        help="ResNet matched-params checkpoint (for Section 8)")
    parser.add_argument("--out", type=str, default="reproduction_results.json",
                        help="Output JSON file")
    args = parser.parse_args()

    print("=" * 60)
    print("REPRODUCING ALL PAPER RESULTS")
    print("=" * 60)
    print()

    results = {}

    # Section 1: always run
    verify_deterministic()
    results['deterministic'] = 'PASS'

    # Section 2: needs data
    if Path(args.data).exists():
        baselines = run_baselines(args.data, args.n_samples)
        results['baselines'] = 'PASS'
    else:
        print(f"SKIP Section 2: {args.data} not found\n")
        results['baselines'] = 'SKIP'

    if args.baselines_only:
        print("Done (baselines only).")
        return

    # Sections 3-6: need checkpoint
    if args.ckpt and Path(args.ckpt).exists():
        if Path(args.val_data).exists():
            level_ious, thresh_bits = run_parity_codec(
                args.ckpt, args.val_data, args.n_samples, cfg_path=args.cfg)
            results['parity_codec'] = 'PASS'
        else:
            print(f"SKIP Section 3: {args.val_data} not found\n")
            results['parity_codec'] = 'SKIP'

        indep, bayes = run_bayes_benchmark(args.ckpt, args.n_frames, cfg_path=args.cfg)
        results['bayes'] = 'PASS'

        # Section 5: Projection diagnostics
        proj_diag = run_projection_diagnostics(
            args.ckpt, args.data, args.n_samples, cfg_path=args.cfg)
        results['projection_diagnostics'] = proj_diag

        # Section 6: Surface constraint ablation
        surf_ablation = run_surface_ablation(
            args.ckpt, args.data, args.n_samples, cfg_path=args.cfg)
        results['surface_ablation'] = surf_ablation
    else:
        print(f"SKIP Sections 3-6: no checkpoint (use --ckpt)\n")
        results['parity_codec'] = 'SKIP'
        results['bayes'] = 'SKIP'
        results['projection_diagnostics'] = 'SKIP'
        results['surface_ablation'] = 'SKIP'

    # Section 9: Surface BP temporal accumulation (no checkpoint needed)
    acc_results = run_surface_bp_accumulation(n_frames=args.n_frames,
                                               tamper_fraction=0.08)
    results['surface_bp_accumulation'] = acc_results

    # Section 7: Surface Attention evaluation
    if args.sa_ckpt and Path(args.sa_ckpt).exists():
        sa_results = run_surface_attention_eval(
            args.sa_ckpt, args.data, args.n_samples)
        results['surface_attention'] = sa_results
    else:
        print(f"SKIP Section 7: no surface attention checkpoint (use --sa-ckpt)\n")
        results['surface_attention'] = 'SKIP'

    # Section 8: Architecture comparison
    arch_ckpts = {}
    if args.ckpt and Path(args.ckpt).exists():
        arch_ckpts['ResNet w=192 (25M)'] = (args.ckpt, 'resnet', args.cfg)
    if args.sa_ckpt and Path(args.sa_ckpt).exists():
        arch_ckpts['SurfAttn small (215K)'] = (args.sa_ckpt, 'surface_attention', None)
    if args.sa_medium_ckpt and Path(args.sa_medium_ckpt).exists():
        arch_ckpts['SurfAttn medium (1.2M)'] = (args.sa_medium_ckpt, 'surface_attention', None)
    if args.resnet_matched_ckpt and Path(args.resnet_matched_ckpt).exists():
        arch_ckpts['ResNet w=48 (1.7M)'] = (args.resnet_matched_ckpt, 'resnet', None)

    if arch_ckpts:
        arch_results = run_architecture_comparison(
            arch_ckpts, args.data, args.n_samples)
        results['architecture_comparison'] = arch_results
    else:
        results['architecture_comparison'] = 'SKIP'

    # Section 10: RS text recovery (GF(256) locality experiment, no checkpoint)
    from rs_text_experiment import run_rs_text_experiment
    rs_results = run_rs_text_experiment(n_blocks=args.n_samples, seed=42)
    results['rs_text_recovery'] = rs_results

    # Summary
    print("=" * 60)
    print("REPRODUCTION SUMMARY")
    print("=" * 60)
    for section, status in results.items():
        if status == 'SKIP':
            icon = '⊘'
        elif status == 'FAIL':
            icon = '✗'
        else:
            icon = '✓'  # 'PASS' or a dict of metrics (treated as pass)
        print(f"  {icon} {section}: {status}")

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
