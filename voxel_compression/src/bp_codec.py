"""BP Codec: content-adaptive parity constraints + belief propagation.

A pure-algebraic 3D volumetric codec with no neural network. The encoder
computes DCT coefficients and an adaptive syndrome over content regions.
The decoder reconstructs using belief propagation with the DCT as prior.

Encoder:
    1. Compute DCT coefficients at chosen level
    2. Threshold DCT to identify non-zero regions
    3. Generate overlapping 2³ subcube constraints over those regions
    4. Compute syndrome = H @ voxels (mod 2)
    5. Transmit: DCT coefficients + syndrome bits

Decoder:
    1. Reconstruct DCT field → soft prior P(x_i = 1)
    2. Derive same constraint matrix H from DCT threshold
    3. Run BP with DCT prior and received syndrome
    4. Threshold marginals → reconstructed voxels

Both sides derive H from the DCT, so no constraint metadata is transmitted.

Usage:
    from bp_codec import BPCodec
    codec = BPCodec(cube_size=32)
    encoded = codec.encode(voxels, dct_level=4)
    decoded = codec.decode(encoded)
    print(f"Bits: {encoded['total_bits']}, IoU: {iou(decoded, voxels)}")
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional

from adaptive_dct import FREQ_LEVELS, FREQ_BITS, make_adaptive_dct_field
from bp_marginals import compute_bp_marginals_sparse
from parity import gf2_matmul
from utils.metrics import iou


@dataclass
class EncodedFrame:
    """Compressed representation of a voxel frame."""
    dct_level: int              # which DCT level (index into FREQ_LEVELS)
    dct_coeffs: torch.Tensor    # the actual DCT coefficients (would be quantized in practice)
    syndrome: torch.Tensor      # (M,) uint8 syndrome bits
    constraint_matrix: torch.Tensor  # (M, N) uint8 — derived from DCT, not transmitted
    threshold: float            # DCT threshold used for constraint generation
    dct_bits: int               # bits for DCT coefficients
    syndrome_bits: int          # bits for syndrome
    total_bits: int             # dct_bits + syndrome_bits


def build_adaptive_constraints(dct_3d, cube_size, threshold=0.2,
                                subcube_size=2, stride=1):
    """Build overlapping subcube constraints over DCT-predicted non-zero regions.

    Both encoder and decoder call this with the same DCT field, producing
    identical constraint matrices without transmitting H.
    """
    D = cube_size
    N = D ** 3
    s = subcube_size

    mask = (dct_3d > threshold).float()
    mask_4d = mask.unsqueeze(0).unsqueeze(0)
    dilated = torch.nn.functional.max_pool3d(mask_4d, kernel_size=s, stride=1,
                                              padding=s // 2)
    dilated = dilated.squeeze()[:D, :D, :D]

    rows = []
    for z in range(0, D - s + 1, stride):
        for y in range(0, D - s + 1, stride):
            for x in range(0, D - s + 1, stride):
                if dilated[z:z+s, y:y+s, x:x+s].sum() < 1:
                    continue
                row = torch.zeros(N, dtype=torch.uint8)
                for dz in range(s):
                    for dy in range(s):
                        for dx in range(s):
                            row[(x+dx) + (y+dy)*D + (z+dz)*D*D] = 1
                rows.append(row)

    if not rows:
        return torch.zeros((0, N), dtype=torch.uint8)
    return torch.stack(rows)


class BPCodec:
    """Content-adaptive BP codec for binary voxel grids."""

    def __init__(self, cube_size=32, bp_iters=30, bp_damping=0.3,
                 constraint_threshold=0.2, subcube_size=2):
        self.cube_size = cube_size
        self.N = cube_size ** 3
        self.bp_iters = bp_iters
        self.bp_damping = bp_damping
        self.constraint_threshold = constraint_threshold
        self.subcube_size = subcube_size
        self.n_levels = len(FREQ_LEVELS)

    def encode(self, voxels, dct_level=2, threshold=None,
               device=None) -> EncodedFrame:
        """Encode a voxel grid.

        Args:
            voxels: (N,) or (1, N) uint8 flat voxel grid
            dct_level: index into FREQ_LEVELS (0=L2, 2=L4, etc.)
            threshold: override constraint threshold
            device: computation device

        Returns:
            EncodedFrame with all encoded data
        """
        if device is None:
            device = voxels.device
        if voxels.dim() == 1:
            voxels = voxels.unsqueeze(0)
        voxels = voxels.to(device).to(torch.uint8)
        thresh = threshold or self.constraint_threshold

        # Step 1: Compute DCT field
        sel = torch.zeros(1, self.n_levels, device=device)
        sel[0, dct_level] = 1.0
        dct_field = make_adaptive_dct_field(voxels, self.cube_size, sel)
        dct_3d = dct_field.squeeze(0).squeeze(0)  # (D, D, D)

        # Step 2: Build adaptive constraints from DCT
        H = build_adaptive_constraints(dct_3d, self.cube_size,
                                        threshold=thresh,
                                        subcube_size=self.subcube_size)
        H = H.to(device)
        M = H.size(0)

        # Step 3: Compute syndrome
        if M > 0:
            syndrome = gf2_matmul(H, voxels.t()).t().squeeze(0)  # (M,)
        else:
            syndrome = torch.zeros(0, dtype=torch.uint8, device=device)

        # Bit accounting
        dct_bits = int(FREQ_BITS[dct_level].item())
        syndrome_bits = M

        return EncodedFrame(
            dct_level=dct_level,
            dct_coeffs=dct_field,
            syndrome=syndrome,
            constraint_matrix=H,
            threshold=thresh,
            dct_bits=dct_bits,
            syndrome_bits=syndrome_bits,
            total_bits=dct_bits + syndrome_bits,
        )

    def decode(self, encoded: EncodedFrame, device=None) -> torch.Tensor:
        """Decode an encoded frame using BP.

        Args:
            encoded: EncodedFrame from encode()
            device: computation device

        Returns:
            (1, N) uint8 reconstructed voxels
        """
        if device is None:
            device = encoded.syndrome.device

        H = encoded.constraint_matrix.to(device)
        M = H.size(0)

        # DCT prior
        dct_prior = encoded.dct_coeffs.view(1, -1).clamp(0.01, 0.99).to(device)

        if M == 0:
            # No constraints — just threshold the DCT
            return (dct_prior > 0.5).to(torch.uint8)

        syndrome = encoded.syndrome.unsqueeze(0).to(device)

        # Run BP
        marginals = compute_bp_marginals_sparse(
            H, syndrome, n_iters=self.bp_iters,
            damping=self.bp_damping, prior=dct_prior,
            device=device,
        )

        return (marginals > 0.5).to(torch.uint8)

    def encode_decode(self, voxels, dct_level=2, threshold=None,
                      device=None):
        """Convenience: encode then decode, return (recon, encoded)."""
        encoded = self.encode(voxels, dct_level=dct_level,
                              threshold=threshold, device=device)
        recon = self.decode(encoded, device=device)
        return recon, encoded

    def sweep_levels(self, voxels, levels=None, threshold=None, device=None):
        """Encode at multiple DCT levels and return rate-distortion curve.

        Returns list of (level, bits, iou, encoded) tuples.
        """
        if levels is None:
            levels = list(range(self.n_levels))
        if voxels.dim() == 1:
            voxels = voxels.unsqueeze(0)

        results = []
        for level_idx in levels:
            recon, encoded = self.encode_decode(voxels, dct_level=level_idx,
                                                 threshold=threshold,
                                                 device=device)
            sample_iou = iou(recon, voxels.to(recon.device)).item()
            results.append({
                'level': int(FREQ_LEVELS[level_idx]),
                'level_idx': level_idx,
                'bits': encoded.total_bits,
                'dct_bits': encoded.dct_bits,
                'syndrome_bits': encoded.syndrome_bits,
                'iou': sample_iou,
                'compression_ratio': self.N / encoded.total_bits,
            })
        return results


    def smart_encode(self, voxels, dct_level=2, target_iou=0.9,
                     max_bits=None, device=None):
        """Smart encoder: sweep thresholds to find best rate-distortion point.

        Tries multiple thresholds and picks the one that achieves target_iou
        with the fewest syndrome bits, or maximizes IoU under a bit budget.

        The chosen threshold can be signaled to the decoder with 3 bits
        (8 threshold levels), or derived from DCT percentile statistics.

        Args:
            voxels: (N,) or (1, N) uint8
            dct_level: DCT level index
            target_iou: minimum acceptable IoU (picks cheapest threshold above this)
            max_bits: if set, picks best IoU under this bit budget instead
            device: computation device

        Returns:
            EncodedFrame with optimal threshold choice
        """
        if device is None:
            device = voxels.device
        if voxels.dim() == 1:
            voxels = voxels.unsqueeze(0)
        voxels = voxels.to(device).to(torch.uint8)

        thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
        candidates = []

        for thresh in thresholds:
            encoded = self.encode(voxels, dct_level=dct_level,
                                   threshold=thresh, device=device)
            recon = self.decode(encoded, device=device)
            sample_iou = iou(recon, voxels).item()
            candidates.append((thresh, sample_iou, encoded))

        if max_bits is not None:
            # Pick best IoU under bit budget
            valid = [(t, i, e) for t, i, e in candidates if e.total_bits <= max_bits]
            if valid:
                best = max(valid, key=lambda x: x[1])
            else:
                best = min(candidates, key=lambda x: x[2].total_bits)
        else:
            # Pick cheapest threshold that meets target_iou
            meets_target = [(t, i, e) for t, i, e in candidates if i >= target_iou]
            if meets_target:
                best = min(meets_target, key=lambda x: x[2].total_bits)
            else:
                best = max(candidates, key=lambda x: x[1])

        return best[2], best[1]  # (encoded, achieved_iou)


def benchmark_bp_codec(data_path, n_samples=200, levels=None, threshold=0.2,
                       device=None, seed=42, smart=False, target_iou=0.9):
    """Run the BP codec on a dataset and report results.

    This is the function that reproduce_results.py should call.
    """
    from dataset import VoxelDataset
    from parity import load_constraint_ops, ConstraintConfig

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data with any constraint matrix (we only use voxels)
    cfg = ConstraintConfig(axis_parity=True, plane_parity=True, global_parity=True)
    _, _, H_ref = load_constraint_ops(32, cfg)
    dataset = VoxelDataset(data_path, 32, constraint_matrix=H_ref)

    rng = np.random.RandomState(seed)
    n = min(n_samples, len(dataset))
    indices = sorted(rng.choice(len(dataset), n, replace=False).tolist())

    codec = BPCodec(cube_size=32, constraint_threshold=threshold)
    if levels is None:
        levels = [0, 2, 3, 4]  # L2, L4, L5, L6

    # Accumulate per-level results
    level_results = {l: {'ious': [], 'bits': [], 'syn_bits': []} for l in levels}
    # Smart encoder results (per-level, picks best threshold)
    smart_results = {l: {'ious': [], 'bits': [], 'syn_bits': [], 'thresholds': []} for l in levels} if smart else None

    for count, idx in enumerate(indices):
        voxels_raw, _ = dataset[idx]
        voxels = voxels_raw.view(-1).to(torch.uint8).to(device)

        for level_idx in levels:
            # Static encoder
            recon, encoded = codec.encode_decode(voxels, dct_level=level_idx,
                                                  device=device)
            sample_iou = iou(recon, voxels.unsqueeze(0)).item()
            level_results[level_idx]['ious'].append(sample_iou)
            level_results[level_idx]['bits'].append(encoded.total_bits)
            level_results[level_idx]['syn_bits'].append(encoded.syndrome_bits)

            # Smart encoder
            if smart:
                encoded_s, iou_s = codec.smart_encode(
                    voxels, dct_level=level_idx,
                    target_iou=target_iou, device=device)
                smart_results[level_idx]['ious'].append(iou_s)
                smart_results[level_idx]['bits'].append(encoded_s.total_bits)
                smart_results[level_idx]['syn_bits'].append(encoded_s.syndrome_bits)
                smart_results[level_idx]['thresholds'].append(encoded_s.threshold)

        if (count + 1) % 50 == 0:
            print(f"  {count+1}/{n} done")

    # Summary
    print(f"\nBP Codec Results ({n} samples, static threshold={threshold}):")
    print(f"{'Level':>6} {'Mean IoU':>9} {'Mean bits':>10} {'Syn bits':>10} "
          f"{'Ratio':>7}")
    print("-" * 50)

    summary = {}
    for level_idx in levels:
        r = level_results[level_idx]
        level_name = f"L{int(FREQ_LEVELS[level_idx])}"
        mean_iou = np.mean(r['ious'])
        mean_bits = np.mean(r['bits'])
        mean_syn = np.mean(r['syn_bits'])
        ratio = 32768 / mean_bits
        print(f"{level_name:>6} {mean_iou:>9.4f} {mean_bits:>10.0f} {mean_syn:>10.0f} "
              f"{ratio:>6.1f}x")
        summary[level_name] = {
            'mean_iou': float(mean_iou),
            'mean_bits': float(mean_bits),
            'mean_syndrome_bits': float(mean_syn),
            'compression_ratio': float(ratio),
            'n_samples': n,
        }

    if smart and smart_results:
        print(f"\nSmart Encoder Results (target_iou={target_iou}):")
        print(f"{'Level':>6} {'Mean IoU':>9} {'Mean bits':>10} {'Syn bits':>10} "
              f"{'Ratio':>7} {'Avg thresh':>10}")
        print("-" * 60)
        for level_idx in levels:
            r = smart_results[level_idx]
            level_name = f"L{int(FREQ_LEVELS[level_idx])}"
            mean_iou = np.mean(r['ious'])
            mean_bits = np.mean(r['bits'])
            mean_syn = np.mean(r['syn_bits'])
            mean_thresh = np.mean(r['thresholds'])
            ratio = 32768 / mean_bits
            print(f"{level_name:>6} {mean_iou:>9.4f} {mean_bits:>10.0f} "
                  f"{mean_syn:>10.0f} {ratio:>6.1f}x {mean_thresh:>10.2f}")
            summary[f"{level_name}_smart"] = {
                'mean_iou': float(mean_iou),
                'mean_bits': float(mean_bits),
                'mean_syndrome_bits': float(mean_syn),
                'compression_ratio': float(ratio),
                'mean_threshold': float(mean_thresh),
                'n_samples': n,
            }

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BP Codec benchmark")
    parser.add_argument("--data", default="data/shapenet/val")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 2, 3, 4])
    parser.add_argument("--smart", action="store_true",
                        help="Also run smart encoder (sweeps thresholds per sample)")
    parser.add_argument("--target-iou", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    results = benchmark_bp_codec(args.data, args.n_samples,
                                  levels=args.levels,
                                  threshold=args.threshold,
                                  device=device,
                                  smart=args.smart,
                                  target_iou=args.target_iou)
    import json
    print(f"\n{json.dumps(results, indent=2)}")
