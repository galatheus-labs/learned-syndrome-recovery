"""Benchmark: Parity Syndrome Codec vs Draco on ShapeNet 32^3 voxels.

Draco compresses point clouds, so we convert voxels → occupied coordinates,
compress with Draco at various quantization levels, decompress, and convert
back to voxels. This gives a rate-quality curve to compare against our codec.

Usage:
    python benchmark_draco.py [--n-samples 200] [--out benchmark_results.json]
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path

import DracoPy

from binvox_rw import read_as_3d_array


def voxels_to_points(voxels_3d):
    """Convert (D, D, D) binary voxels to (N, 3) float32 point cloud."""
    coords = np.argwhere(voxels_3d > 0).astype(np.float32)
    return coords


def points_to_voxels(points, cube_size):
    """Convert (N, 3) points back to (D, D, D) binary voxels."""
    grid = np.zeros((cube_size, cube_size, cube_size), dtype=np.uint8)
    if len(points) == 0:
        return grid
    # Round to nearest integer, clip to valid range
    coords = np.round(points).astype(int)
    coords = np.clip(coords, 0, cube_size - 1)
    # Remove duplicates
    coords = np.unique(coords, axis=0)
    grid[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return grid


def compute_iou(a, b):
    """IoU between two binary grids."""
    intersection = np.sum(a & b)
    union = np.sum(a | b)
    if union == 0:
        return 1.0
    return intersection / union


def compute_hamming(a, b):
    """Hamming distance between two binary grids."""
    return np.sum(a != b)


def benchmark_draco(voxels_3d, quant_bits_list):
    """Compress voxels via Draco at various quantization levels.

    Returns list of {quant_bits, encoded_bytes, encoded_bits, iou, hamming, n_points}.
    """
    cube_size = voxels_3d.shape[0]
    points = voxels_to_points(voxels_3d)
    n_points = len(points)
    original_bits = cube_size ** 3  # raw voxel grid

    if n_points == 0:
        return []

    results = []
    for qb in quant_bits_list:
        try:
            encoded = DracoPy.encode(points, quantization_bits=qb,
                                     compression_level=10)
            encoded_bits = len(encoded) * 8

            decoded = DracoPy.decode(encoded)
            recon_points = np.array(decoded.points)
            recon_voxels = points_to_voxels(recon_points, cube_size)

            iou_val = compute_iou(voxels_3d, recon_voxels)
            ham = compute_hamming(voxels_3d, recon_voxels)

            results.append({
                'quant_bits': qb,
                'encoded_bytes': len(encoded),
                'encoded_bits': encoded_bits,
                'compression_ratio': original_bits / encoded_bits if encoded_bits > 0 else 0,
                'iou': float(iou_val),
                'hamming': int(ham),
                'n_points': n_points,
                'original_bits': original_bits,
            })
        except Exception as e:
            print(f"  Draco qb={qb} failed: {e}")

    return results


def benchmark_raw_encoding(voxels_3d):
    """Baseline: raw coordinate encoding (no compression).

    Each occupied voxel needs 3 * ceil(log2(32)) = 15 bits for coordinates.
    """
    n_points = int(voxels_3d.sum())
    cube_size = voxels_3d.shape[0]
    bits_per_coord = int(np.ceil(np.log2(cube_size)))
    raw_bits = n_points * 3 * bits_per_coord
    return {
        'method': 'raw_coords',
        'encoded_bits': raw_bits,
        'compression_ratio': cube_size ** 3 / raw_bits if raw_bits > 0 else 0,
        'iou': 1.0,
        'hamming': 0,
        'n_points': n_points,
    }


def benchmark_octree(voxels_3d, max_depth_list=None):
    """Octree encoding at various depth limits.

    Full depth = lossless. Truncated depth = lossy (fills truncated nodes
    based on majority vote of children).

    Octree encoding: each internal node is 1 bit (has children) + 8 bits
    for child occupancy mask. Leaf nodes are 1 bit each.
    """
    cube_size = voxels_3d.shape[0]
    full_depth = int(np.log2(cube_size))  # 5 for 32^3
    if max_depth_list is None:
        max_depth_list = list(range(1, full_depth + 1))

    results = []
    for max_depth in max_depth_list:
        bits, recon = _octree_encode_decode(voxels_3d, cube_size, max_depth, full_depth)
        iou_val = compute_iou(voxels_3d, recon)
        ham = compute_hamming(voxels_3d, recon)
        results.append({
            'max_depth': max_depth,
            'encoded_bits': bits,
            'compression_ratio': cube_size ** 3 / bits if bits > 0 else 0,
            'iou': float(iou_val),
            'hamming': int(ham),
        })
    return results


def _octree_encode_decode(voxels_3d, cube_size, max_depth, full_depth):
    """Encode voxels as octree, truncate at max_depth, decode back.

    Returns (total_bits, reconstructed_voxels).
    """
    total_bits = 0
    recon = np.zeros_like(voxels_3d)

    def encode_node(x, y, z, size, depth):
        nonlocal total_bits
        region = voxels_3d[x:x+size, y:y+size, z:z+size]
        occupancy = region.sum()

        if occupancy == 0:
            # Empty node: 1 bit (empty flag)
            total_bits += 1
            return
        if occupancy == size ** 3:
            # Full node: 1 bit (full flag)
            total_bits += 1
            recon[x:x+size, y:y+size, z:z+size] = 1
            return

        if depth >= max_depth or size == 1:
            # Leaf at truncation: 1 bit for majority vote
            total_bits += 1
            if occupancy > size ** 3 / 2:
                recon[x:x+size, y:y+size, z:z+size] = 1
            return

        # Internal node: 8-bit child mask + recurse
        half = size // 2
        total_bits += 8  # child occupancy mask
        for dx in range(2):
            for dy in range(2):
                for dz in range(2):
                    encode_node(x + dx*half, y + dy*half, z + dz*half,
                                half, depth + 1)

    encode_node(0, 0, 0, cube_size, 0)
    return total_bits, recon


def benchmark_rle(voxels_3d):
    """Run-length encoding of the voxel grid (flattened).

    Simple baseline: encode runs of 0s and 1s. Each run is (value: 1 bit,
    length: variable-length integer).
    """
    flat = voxels_3d.flatten()
    cube_size = voxels_3d.shape[0]
    total_bits = 0
    i = 0
    while i < len(flat):
        val = flat[i]
        run_len = 1
        while i + run_len < len(flat) and flat[i + run_len] == val:
            run_len += 1
        # 1 bit for value + variable-length encoding of run length
        # Use Elias gamma coding: 2*floor(log2(n))+1 bits
        total_bits += 1 + (2 * int(np.floor(np.log2(max(run_len, 1)))) + 1)
        i += run_len

    return {
        'method': 'rle',
        'encoded_bits': total_bits,
        'compression_ratio': cube_size ** 3 / total_bits if total_bits > 0 else 0,
        'iou': 1.0,
        'hamming': 0,
    }


def benchmark_parity_theoretical(voxels_3d, syndrome_bits=None, coarse_bits=8):
    """Our codec's theoretical bit budget (syndrome + coarse + DCT).

    Actual IoU depends on the trained model, so we just report the bit budget
    at each DCT level for fair comparison on the rate axis.
    """
    from adaptive_dct import FREQ_LEVELS, FREQ_BITS
    if syndrome_bits is None:
        from parity import build_constraint_matrix, ConstraintConfig
        cfg = ConstraintConfig(axis_parity=True, plane_parity=True, global_parity=True)
        syndrome_bits = build_constraint_matrix(voxels_3d.shape[0], cfg).shape[0]
    cube_size = voxels_3d.shape[0]
    original_bits = cube_size ** 3
    results = []
    for i, (level, dct_bits) in enumerate(zip(FREQ_LEVELS, FREQ_BITS)):
        total = syndrome_bits + coarse_bits + int(dct_bits.item())
        results.append({
            'dct_level': level,
            'syndrome_bits': syndrome_bits,
            'coarse_bits': coarse_bits,
            'dct_bits': int(dct_bits.item()),
            'total_bits': total,
            'compression_ratio': original_bits / total,
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--out", type=str, default="benchmark_results.json")
    parser.add_argument("--data", type=str, default="data/shapenet/val")
    args = parser.parse_args()

    quant_bits_list = [4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16]

    data_dir = Path(args.data)
    all_files = sorted(data_dir.glob("**/*.binvox"))
    if not all_files:
        all_files = sorted(data_dir.glob("**/*.npy"))
    if not all_files:
        print(f"No data found in {data_dir}")
        return

    # Sample evenly across categories
    rng = np.random.RandomState(42)
    indices = rng.choice(len(all_files), min(args.n_samples, len(all_files)), replace=False)
    files = [all_files[i] for i in sorted(indices)]
    print(f"Benchmarking {len(files)} samples from {data_dir}")

    # Aggregate results
    draco_results = {qb: [] for qb in quant_bits_list}
    octree_results = {d: [] for d in range(1, 6)}
    rle_results = []
    raw_results = []
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

        # Draco benchmark
        results = benchmark_draco(v3d, quant_bits_list)
        for r in results:
            draco_results[r['quant_bits']].append(r)

        # Octree benchmark
        oct_results = benchmark_octree(v3d)
        for r in oct_results:
            octree_results[r['max_depth']].append(r)

        # RLE benchmark
        rle_results.append(benchmark_rle(v3d))

        # Raw coords baseline
        raw_results.append(benchmark_raw_encoding(v3d))

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)} done")

    # Aggregate
    print("\n=== Draco Results (averaged) ===")
    print(f"{'QB':>4} {'Bits':>8} {'Ratio':>8} {'IoU':>8} {'Hamming':>8}")
    draco_summary = []
    for qb in quant_bits_list:
        entries = draco_results[qb]
        if not entries:
            continue
        avg_bits = np.mean([e['encoded_bits'] for e in entries])
        avg_ratio = np.mean([e['compression_ratio'] for e in entries])
        avg_iou = np.mean([e['iou'] for e in entries])
        avg_ham = np.mean([e['hamming'] for e in entries])
        print(f"{qb:>4} {avg_bits:>8.0f} {avg_ratio:>8.1f}x {avg_iou:>8.4f} {avg_ham:>8.0f}")
        draco_summary.append({
            'quant_bits': qb,
            'avg_bits': float(avg_bits),
            'avg_compression_ratio': float(avg_ratio),
            'avg_iou': float(avg_iou),
            'avg_hamming': float(avg_ham),
            'n_samples': len(entries),
        })

    # Octree results
    print("\n=== Octree Results (averaged) ===")
    print(f"{'Depth':>6} {'Bits':>8} {'Ratio':>8} {'IoU':>8} {'Hamming':>8}")
    octree_summary = []
    for depth in range(1, 6):
        entries = octree_results[depth]
        if not entries:
            continue
        avg_bits = np.mean([e['encoded_bits'] for e in entries])
        avg_ratio = np.mean([e['compression_ratio'] for e in entries])
        avg_iou = np.mean([e['iou'] for e in entries])
        avg_ham = np.mean([e['hamming'] for e in entries])
        print(f"{depth:>6} {avg_bits:>8.0f} {avg_ratio:>8.1f}x {avg_iou:>8.4f} {avg_ham:>8.0f}")
        octree_summary.append({
            'max_depth': depth,
            'avg_bits': float(avg_bits),
            'avg_compression_ratio': float(avg_ratio),
            'avg_iou': float(avg_iou),
            'avg_hamming': float(avg_ham),
            'n_samples': len(entries),
        })

    # RLE results
    avg_rle_bits = np.mean([r['encoded_bits'] for r in rle_results])
    avg_rle_ratio = np.mean([r['compression_ratio'] for r in rle_results])
    print(f"\nRLE (lossless): {avg_rle_bits:.0f} bits avg ({avg_rle_ratio:.1f}x compression)")

    # Raw coords
    avg_raw_bits = np.mean([r['encoded_bits'] for r in raw_results])
    avg_raw_ratio = np.mean([r['compression_ratio'] for r in raw_results])
    print(f"Raw coordinates: {avg_raw_bits:.0f} bits avg ({avg_raw_ratio:.1f}x compression)")

    # Our codec theoretical bit budget
    print("\n=== Parity Codec Bit Budget ===")
    parity_rates = benchmark_parity_theoretical(np.zeros((32, 32, 32), dtype=np.uint8))
    for r in parity_rates:
        print(f"  DCT level {r['dct_level']}: {r['total_bits']} bits ({r['compression_ratio']:.1f}x)")

    print(f"\nFill rates: min={min(fill_rates):.3f} max={max(fill_rates):.3f} "
          f"mean={np.mean(fill_rates):.3f} median={np.median(fill_rates):.3f}")

    # Save full results
    output = {
        'draco': draco_summary,
        'octree': octree_summary,
        'rle': {
            'avg_bits': float(avg_rle_bits),
            'avg_compression_ratio': float(avg_rle_ratio),
        },
        'raw_coords': {
            'avg_bits': float(avg_raw_bits),
            'avg_compression_ratio': float(avg_raw_ratio),
        },
        'parity_codec': parity_rates,
        'fill_rates': {
            'min': float(min(fill_rates)),
            'max': float(max(fill_rates)),
            'mean': float(np.mean(fill_rates)),
            'median': float(np.median(fill_rates)),
        },
        'n_samples': len(files),
    }
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
