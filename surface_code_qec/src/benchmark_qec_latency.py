"""Benchmark inference latency of the QEC neural decoder vs distance.

Builds the same ResNet3D architecture used in training (width=128, depth=12)
at distances d ∈ {3, 5, 7, 9, 11}, and measures both batched throughput
(amortized cost per shot at large batch) and single-shot latency
(cost per individual decode, dominated by kernel launch overhead).

Run on the cluster via gpurun:

    bin/gpurun --node q9 --repo briangu/galareon -- bash -c '
      pip install -q stim pymatching
      cd parity_diffusion
      python benchmark_qec_latency.py
    '

Or run on a specific GPU type for an apples-to-apples comparison:

    bin/gpurun --node alpha --repo briangu/galareon -- ...   # 4090
    bin/gpurun --node moles --repo briangu/galareon -- ...   # 4090
    bin/gpurun --node q9    --repo briangu/galareon -- ...   # 4070 Ti
    bin/gpurun --node wopr  --repo briangu/galareon -- ...   # 3090

Results from the paper were measured on the q9 node (RTX 4070 Ti).
"""

import argparse
import json
import time
import torch
import stim

from surface_code_resnet import ResNetDecoder, get_detector_grid_mapping


def build_for(d, p=0.005):
    """Construct a Stim circuit at distance d and return the detector grid."""
    c = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=d, distance=d,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )
    grid, _ = get_detector_grid_mapping(c)
    return grid, c.num_detectors


def benchmark_distance(d, width=128, depth=12, batch_size=1024,
                       n_warmup=3, n_iters=20, n_iters_bs1=100):
    """Time forward pass at distance d for both batched and single-shot."""
    grid, n_det = build_for(d)
    model = ResNetDecoder(grid, width=width, depth=depth).cuda().eval()
    n_params = sum(p.numel() for p in model.parameters())

    # Batched: amortized per-shot cost at large batch
    x = torch.zeros(batch_size, 1, *grid).cuda()
    p_t = torch.full((batch_size,), 0.005).cuda()
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(x, p_t)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = model(x, p_t)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    per_shot_us = (elapsed / (n_iters * batch_size)) * 1e6
    shots_per_sec = (n_iters * batch_size) / elapsed

    # Single-shot: per-call latency, dominated by kernel launch overhead
    x1 = torch.zeros(1, 1, *grid).cuda()
    p1 = torch.full((1,), 0.005).cuda()
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(x1, p1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters_bs1):
        with torch.no_grad():
            _ = model(x1, p1)
    torch.cuda.synchronize()
    bs1_us = (time.perf_counter() - t0) / n_iters_bs1 * 1e6

    del model, x, p_t, x1, p1
    torch.cuda.empty_cache()

    return {
        "distance": d,
        "grid": list(grid),
        "n_detectors": n_det,
        "n_params": n_params,
        "batched_per_shot_us": float(per_shot_us),
        "batched_shots_per_sec": float(shots_per_sec),
        "single_shot_us": float(bs1_us),
        "batch_size": batch_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", type=int, nargs="+",
                        default=[3, 5, 7, 9, 11])
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--out", type=str, default="qec_latency_results.json")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Architecture: ResNet3D w={args.width} d={args.depth}")
    print(f"Batch size: {args.batch_size}")
    print()
    print(f"{'d':>3} | {'grid':<14} | {'detectors':>9} | "
          f"{'batched (us/shot)':>17} | {'shots/sec':>12} | {'bs=1 (us)':>10}")
    print("-" * 90)

    results = []
    for d in args.distances:
        r = benchmark_distance(d, width=args.width, depth=args.depth,
                               batch_size=args.batch_size)
        results.append(r)
        print(f"{r['distance']:>3} | {str(tuple(r['grid'])):<14} | "
              f"{r['n_detectors']:>9} | {r['batched_per_shot_us']:>15.1f}   | "
              f"{r['batched_shots_per_sec']:>12,.0f} | {r['single_shot_us']:>10.0f}")

    # Scaling analysis
    print()
    print("Scaling analysis (vs d=3 baseline):")
    base = results[0]["batched_per_shot_us"]
    base_d = results[0]["distance"]
    for r in results:
        d_ratio = r["distance"] / base_d
        latency_ratio = r["batched_per_shot_us"] / base
        # Theoretical d^k: latency_ratio = d_ratio^k -> k = log(latency_ratio)/log(d_ratio)
        if d_ratio > 1:
            import math
            k = math.log(latency_ratio) / math.log(d_ratio)
            print(f"  d={r['distance']:2d}: {latency_ratio:>5.1f}x latency, "
                  f"d_ratio={d_ratio:.2f}, effective scaling exponent k={k:.2f}")
        else:
            print(f"  d={r['distance']:2d}: 1.0x (baseline)")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
