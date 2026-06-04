"""Clean test-set evaluation for QEC neural decoder checkpoints.

Implements the corrected methodology from qec_review_response.md:

1. Generate a fresh TEST set with a disjoint Stim sampler seed
   (seed * 1000 + 800), separate from both train and val seeds.
2. For each saved training run (distance, seed):
   a. Load all saved epoch checkpoints
   b. Evaluate each on the VAL set
   c. Select ONE checkpoint via a single predeclared scalar rule
      (mean logical error rate across eval rates, equally weighted)
   d. Evaluate that single chosen checkpoint on the TEST set
   e. Record per-p test-set logical error rates
3. Compute MWPM on the same test shots (paired comparison).
4. Report McNemar's test p-values (paired significance on same shots).
5. Report 95% CIs on each rate.

Usage:
    # Evaluate a single run from local checkpoints:
    python eval_test_split.py --ckpt-dir ckpts/qec_d5 --distance 5 --seed 42

    # Evaluate a run from an external checkpoint directory on the cluster:
    bin/gpurun --repo briangu/galareon -- bash -c '
      pip install -q stim pymatching
      cd parity_diffusion
      python eval_test_split.py --ckpt-dir /path/to/qec_runs/d5_s42_w128_dep12/ckpts --distance 5 --seed 42
    '

    # Higher shots at p=0.001:
    python eval_test_split.py --ckpt-dir ckpts/qec_d5 --distance 5 --seed 42 --n-test 10000000
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import pymatching
import stim

from surface_code_resnet import (
    ResNetDecoder, get_detector_grid_mapping, detectors_to_volume,
    generate_data,
)


def detect_arch_from_state(state):
    """Infer (block_type, zero_init_residual) from a checkpoint state dict.

    LC0 blocks have se_fc1/se_fc2 (dual-output SE) that classic blocks lack.
    ReZero-initialized blocks have a 'rezero' scalar parameter per block.
    """
    block_type = 'lc0' if 'blocks.0.se_fc1.weight' in state else 'classic'
    zero_init = any(k.endswith('.rezero') for k in state.keys())
    return block_type, zero_init


def mcnemar_test(neural_correct, mwpm_correct):
    """McNemar's test for paired binary classifiers on the same shots.

    Returns (chi2, p_value, b, c). The test checks whether the two decoders
    disagree in a systematic direction.
    """
    # b = shots where neural is right and MWPM is wrong
    # c = shots where neural is wrong and MWPM is right
    b = int(((neural_correct) & (~mwpm_correct)).sum())
    c = int(((~neural_correct) & (mwpm_correct)).sum())
    if b + c == 0:
        return 0.0, 1.0, b, c
    # McNemar's chi-squared with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    # One-sided p-value (we care whether neural is better)
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2, df=1)
    # Direction: if b > c, neural is better
    if b < c:
        p_value = 1 - p_value  # flip for one-sided
    return float(chi2), float(p_value), int(b), int(c)


def binomial_ci_95(p_hat, n):
    """Wilson 95% confidence interval half-width for a binomial proportion."""
    if n <= 0:
        return 0.0
    z = 1.96
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = z * math.sqrt((p_hat * (1 - p_hat) / n) + (z * z / (4 * n * n))) / denom
    # Call sites print `p_hat ± ci`; return a conservative half-width around
    # p_hat that covers the Wilson interval even when it is asymmetric.
    return float(max(abs(center - half - p_hat), abs(center + half - p_hat)))


def evaluate_checkpoint(model, grid_size, det_to_grid, det_events, obs_flips,
                        eval_rates, n_shots, device, batch_size=4096):
    """Evaluate a model on detector events, return per-rate error rates
    and per-shot correctness arrays (for paired tests)."""
    model.eval()
    results = {}
    for i, p in enumerate(eval_rates):
        start = i * n_shots
        end = start + n_shots
        det = det_events[start:end]
        obs = obs_flips[start:end]

        vol = detectors_to_volume(
            torch.from_numpy(det.astype(np.float32)), grid_size, det_to_grid)

        all_preds = []
        with torch.no_grad():
            for j in range(0, len(vol), batch_size):
                batch_vol = vol[j:j+batch_size].to(device)
                batch_p = torch.full((len(batch_vol),), p, device=device)
                logits = model(batch_vol, batch_p)
                all_preds.append((logits > 0).cpu().numpy().astype(np.uint8))

        pred = np.concatenate(all_preds)
        obs_flat = obs.squeeze(-1) if obs.ndim > 1 else obs
        neural_correct = (pred == obs_flat)
        neural_err = (~neural_correct).sum() / n_shots

        results[p] = {
            'neural_err': float(neural_err),
            'neural_correct': neural_correct,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, required=True,
                        help="Directory containing epoch checkpoint .pt files")
    parser.add_argument("--distance", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--noise", type=str, default="circuit")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--n-val", type=int, default=1000000,
                        help="Shots per rate for val (checkpoint selection)")
    parser.add_argument("--n-test", type=int, default=1000000,
                        help="Shots per rate for test (final reporting)")
    parser.add_argument("--n-test-low-p", type=int, default=10000000,
                        help="Extra shots at p=0.001 for tighter CI")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_rates = [0.001, 0.003, 0.005, 0.007]

    print(f"=== Clean Test-Split Evaluation ===", flush=True)
    print(f"distance={args.distance}, seed={args.seed}, noise={args.noise}", flush=True)
    print(f"device={device}", flush=True)

    # Get grid mapping
    _, _, ref_circuit = generate_data(args.distance, 0.005, 1, args.noise)
    grid_size, det_to_grid = get_detector_grid_mapping(ref_circuit)
    print(f"grid={grid_size}, detectors={ref_circuit.num_detectors}", flush=True)

    # Find all checkpoints
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_files = sorted(ckpt_dir.glob("*.pt"))
    if not ckpt_files:
        print(f"ERROR: no checkpoints found in {ckpt_dir}")
        return
    print(f"Checkpoints: {[f.name for f in ckpt_files]}", flush=True)

    # --- PHASE 1: Generate VAL data (same seed offset as training) ---
    print(f"\nGenerating val data ({args.n_val} shots/rate)...", flush=True)
    val_det_all = []
    val_obs_all = []
    val_mwpm_correct = {}
    val_mwpm_rates = {}
    for i, p in enumerate(eval_rates):
        det, obs, circuit = generate_data(
            args.distance, p, args.n_val, args.noise,
            sampler_seed=args.seed * 1000 + 500 + i)
        val_det_all.append(det)
        val_obs_all.append(obs)
        # MWPM on val
        dem = circuit.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(dem)
        mwpm_pred = matcher.decode_batch(det)
        obs_flat = obs.squeeze(-1) if obs.ndim > 1 else obs
        val_mwpm_rates[p] = float((mwpm_pred.flatten() != obs_flat).sum() / args.n_val)
        print(f"  p={p}: MWPM={val_mwpm_rates[p]:.6f}", flush=True)
    val_det_all = np.concatenate(val_det_all)
    val_obs_all = np.concatenate(val_obs_all)

    # --- PHASE 2: Evaluate all checkpoints on VAL, select best ---
    print(f"\nEvaluating checkpoints on val set...", flush=True)
    best_ckpt = None
    best_mean_val_err = float('inf')
    ckpt_val_results = {}

    for ckpt_path in ckpt_files:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        block_type, zero_init = detect_arch_from_state(ckpt['model'])
        model = ResNetDecoder(grid_size, width=args.width, depth=args.depth,
                              block_type=block_type,
                              zero_init_residual=zero_init).to(device)
        model.load_state_dict(ckpt['model'])

        results = evaluate_checkpoint(
            model, grid_size, det_to_grid,
            val_det_all, val_obs_all,
            eval_rates, args.n_val, device, args.batch_size)

        mean_err = np.mean([results[p]['neural_err'] for p in eval_rates])
        ckpt_val_results[ckpt_path.name] = {
            'mean_err': float(mean_err),
            'per_rate': {str(p): results[p]['neural_err'] for p in eval_rates},
        }
        tag = ""
        if mean_err < best_mean_val_err:
            best_mean_val_err = mean_err
            best_ckpt = ckpt_path
            tag = " <-- BEST"
        rates_str = "  ".join(f"p={p}:{results[p]['neural_err']:.5f}" for p in eval_rates)
        print(f"  {ckpt_path.name}: mean={mean_err:.6f}  {rates_str}{tag}", flush=True)

        del model
        torch.cuda.empty_cache()

    print(f"\nSelected checkpoint: {best_ckpt.name} (mean val err={best_mean_val_err:.6f})",
          flush=True)

    # --- PHASE 3+4: Per-rate streaming generate -> MWPM -> ResNet eval.
    # Old code accumulated all 4 rates' detectors in CPU RAM, then built a
    # single dense (n, 1, Dx, Dy, Dz) volume per rate. For d11 grid 12^3 with
    # n_test_low_p=10M that volume alone is ~70 GB and OOMKilled the pod.
    # Stream per-rate, and build the volume in batch_size chunks.
    print(f"\nLoading checkpoint for test eval...", flush=True)
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    block_type, zero_init = detect_arch_from_state(ckpt['model'])
    model = ResNetDecoder(grid_size, width=args.width, depth=args.depth,
                          block_type=block_type,
                          zero_init_residual=zero_init).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"  block_type={block_type}, zero_init_residual={zero_init}", flush=True)

    print(f"\n{'p':>8} {'n':>10} {'MWPM':>10} {'±95%':>8} {'ResNet':>10} {'±95%':>8} {'Diff':>8} {'McNemar_p':>10} {'b':>6} {'c':>6}")
    print("-" * 100)

    import gc

    final_results = {}
    for i, p in enumerate(eval_rates):
        n = args.n_test_low_p if p == 0.001 else args.n_test

        # Generate this rate's test data
        det, obs, circuit = generate_data(
            args.distance, p, n, args.noise,
            sampler_seed=args.seed * 1000 + 800 + i)

        # MWPM on test (paired comparison shots)
        dem = circuit.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(dem)
        mwpm_pred = matcher.decode_batch(det)
        obs_flat = obs.squeeze(-1) if obs.ndim > 1 else obs
        mwpm_corr = (mwpm_pred.flatten() == obs_flat)
        mwpm_err = float((~mwpm_corr).sum() / n)
        del mwpm_pred, dem, matcher, circuit

        # ResNet inference: build volume in batch chunks so the dense
        # (chunk, 1, Dx, Dy, Dz) tensor never balloons past ~chunk*grid bytes.
        all_preds = []
        with torch.no_grad():
            for j in range(0, n, args.batch_size):
                j_end = min(j + args.batch_size, n)
                chunk_det = det[j:j_end]
                chunk_vol = detectors_to_volume(
                    torch.from_numpy(chunk_det.astype(np.float32)),
                    grid_size, det_to_grid).to(device)
                batch_p = torch.full((len(chunk_vol),), p, device=device)
                logits = model(chunk_vol, batch_p)
                all_preds.append((logits > 0).cpu().numpy().astype(np.uint8))
                del chunk_vol, batch_p, logits

        pred = np.concatenate(all_preds)
        neural_correct = (pred == obs_flat)
        neural_err = float((~neural_correct).sum() / n)

        neural_ci = binomial_ci_95(neural_err, n)
        mwpm_ci = binomial_ci_95(mwpm_err, n)
        chi2, p_val, b, c = mcnemar_test(neural_correct, mwpm_corr)
        diff_pct = (mwpm_err - neural_err) / mwpm_err * 100 if mwpm_err > 0 else 0

        print(f"{p:>8.4f} {n:>10,} {mwpm_err:>10.6f} {mwpm_ci:>8.6f} "
              f"{neural_err:>10.6f} {neural_ci:>8.6f} {diff_pct:>+7.1f}% "
              f"{p_val:>10.4f} {b:>6} {c:>6}", flush=True)

        final_results[str(p)] = {
            'n_shots': n,
            'mwpm_err': mwpm_err,
            'mwpm_ci_95': float(mwpm_ci),
            'neural_err': neural_err,
            'neural_ci_95': float(neural_ci),
            'improvement_pct': float(diff_pct),
            'mcnemar_chi2': chi2,
            'mcnemar_p': p_val,
            'mcnemar_b': b,
            'mcnemar_c': c,
        }

        # Free this rate's data before generating the next
        del det, obs, obs_flat, mwpm_corr, pred, neural_correct, all_preds
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

    # Save results
    out_path = args.out or f"qec_test_d{args.distance}_s{args.seed}.json"
    output = {
        'distance': args.distance,
        'seed': args.seed,
        'noise': args.noise,
        'width': args.width,
        'depth': args.depth,
        'block_type': block_type,
        'zero_init_residual': zero_init,
        'selected_checkpoint': best_ckpt.name,
        'selection_criterion': 'min mean val error across eval rates (equal weight)',
        'val_results': ckpt_val_results,
        'test_results': final_results,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
