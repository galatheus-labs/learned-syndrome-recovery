"""Temporal Bayesian accumulation for holographic streaming.

Each frame's decoder outputs soft probabilities per voxel. Instead of
hard-thresholding independently, accumulate evidence across frames
using Bayes' rule. Voxels that don't change between frames get
multiple independent "observations" — the consensus converges toward
ground truth.

Works in log-odds space for numerical stability:
  log_odds += new_log_odds  (accumulate evidence)
  prob = sigmoid(log_odds)

When a voxel changes (detected via delta), reset its prior to 0.5
(log_odds = 0) so it can be re-estimated.

No training needed — pure inference-time improvement on top of
any decoder that outputs probabilities.
"""

import torch
import numpy as np
from parity import gf2_matmul, project_with_left_inverse, make_canonical_base
from train import make_coarse_map
from adaptive_dct import FREQ_LEVELS, FREQ_BITS, _apply_freq_cutoff_batch
from utils.metrics import hamming_distance, iou


class BayesAccumulator:
    """Accumulates voxel probabilities across frames in log-odds space."""

    def __init__(self, n_voxels, device='cuda'):
        self.n_voxels = n_voxels
        self.device = device
        # Log-odds: 0 = uninformative prior (prob 0.5)
        self.log_odds = torch.zeros(1, n_voxels, device=device)
        self.frame_count = 0

    def update(self, new_probs, delta_mask=None):
        """Incorporate new decoder probabilities.

        Args:
            new_probs: (1, N) float in (0, 1) — decoder's sigmoid output
            delta_mask: (1, N) bool — True where voxels changed since last frame.
                       These voxels get their prior reset before update.
        """
        eps = 1e-6
        new_probs = new_probs.clamp(eps, 1 - eps)
        new_log_odds = torch.log(new_probs / (1 - new_probs))

        if delta_mask is not None and self.frame_count > 0:
            # Reset prior for changed voxels
            self.log_odds[delta_mask] = 0.0

        # Accumulate evidence
        self.log_odds = self.log_odds + new_log_odds
        self.frame_count += 1

    def get_probs(self):
        """Current posterior probabilities."""
        return torch.sigmoid(self.log_odds)

    def get_bits(self):
        """Hard threshold the posterior."""
        return (self.get_probs() > 0.5).to(torch.uint8)

    def reset(self):
        """Full reset (for keyframes)."""
        self.log_odds.zero_()
        self.frame_count = 0


@torch.no_grad()
def decode_frame_soft(model, voxels, syndrome, parity, left_inv, cube_size,
                      device, keep_freqs=4):
    """Decode a frame and return soft probabilities (not hard bits).

    Returns:
        recon_probs: (1, N) float — soft voxel probabilities
        x_base: (1, N) uint8 — canonical base
    """
    syndrome_uint8 = syndrome.to(torch.uint8)
    x_base = make_canonical_base(syndrome_uint8, parity, left_inv, device)
    coarse = make_coarse_map(voxels, cube_size)

    # DCT field at specified level
    D = cube_size
    v3d = voxels.float().view(1, D, D, D)
    fft_3d = torch.fft.rfftn(v3d, dim=(-3, -2, -1))
    field = _apply_freq_cutoff_batch(fft_3d, D, keep_freqs).squeeze(0)
    lo = field.amin()
    hi = field.amax()
    field = ((field - lo) / (hi - lo + 1e-8)).unsqueeze(0).unsqueeze(0)

    x_input = torch.cat([
        x_base.float().view(1, 1, cube_size, cube_size, cube_size),
        coarse,
        field,
    ], dim=1)

    t = torch.full((1,), 1000, device=device)
    logits = model(x_input, t, syndrome.float())
    residual_probs = torch.sigmoid(logits.view(1, -1))

    # Convert residual probs to full voxel probs
    # XOR in probability space: p(base XOR residual = 1)
    base_f = x_base.float()
    full_probs = base_f * (1 - residual_probs) + (1 - base_f) * residual_probs

    return full_probs, x_base


@torch.no_grad()
def eval_temporal_bayes(model, frames, parity, left_inv, device,
                        keep_freqs=4, keyframe_interval=10):
    """Evaluate Bayesian temporal accumulation on a frame sequence.

    Compares three strategies:
    1. Independent: each frame decoded independently (baseline)
    2. Bayes: accumulate probabilities across frames
    3. Bayes + project: accumulate then project to parity

    Args:
        model: trained decoder
        frames: list of (N,) numpy arrays
        parity, left_inv: parity constraint ops
        device: torch device
        keep_freqs: DCT level to use
        keyframe_interval: reset Bayes accumulator every N frames
    """
    cube_size = 32
    N = cube_size ** 3
    M = parity.size(0)

    accum = BayesAccumulator(N, device)
    prev_frame = None

    results = []

    for frame_idx, frame_np in enumerate(frames):
        frame = torch.from_numpy(frame_np).to(torch.uint8).unsqueeze(0).to(device)
        syndrome = gf2_matmul(parity, frame.t()).t().float()

        # Detect delta
        delta_mask = None
        if prev_frame is not None:
            delta = frame ^ prev_frame
            delta_mask = delta.bool()

        # Keyframe reset
        is_keyframe = (frame_idx % keyframe_interval == 0)
        if is_keyframe:
            accum.reset()

        # Get soft probabilities from decoder
        full_probs, x_base = decode_frame_soft(
            model, frame, syndrome, parity, left_inv, cube_size, device, keep_freqs)

        # Strategy 1: Independent (hard threshold, project)
        independent_bits = (full_probs > 0.5).to(torch.uint8)
        independent_bits, _ = project_with_left_inverse(
            independent_bits, parity, left_inv, syndrome.to(torch.uint8))
        iou_independent = iou(independent_bits, frame).item()
        ham_independent = hamming_distance(independent_bits, frame).item()

        # Strategy 2: Bayesian accumulation
        accum.update(full_probs, delta_mask)
        bayes_bits = accum.get_bits()
        iou_bayes_raw = iou(bayes_bits, frame).item()
        ham_bayes_raw = hamming_distance(bayes_bits, frame).item()

        # Strategy 3: Bayes + project to parity
        bayes_projected, _ = project_with_left_inverse(
            bayes_bits.clone(), parity, left_inv, syndrome.to(torch.uint8))
        iou_bayes_proj = iou(bayes_projected, frame).item()
        ham_bayes_proj = hamming_distance(bayes_projected, frame).item()

        delta_count = int(delta_mask.sum().item()) if delta_mask is not None else N
        kf_marker = "KEY" if is_keyframe else "   "

        results.append({
            'frame': frame_idx,
            'keyframe': is_keyframe,
            'delta_voxels': delta_count,
            'iou_independent': iou_independent,
            'iou_bayes': iou_bayes_raw,
            'iou_bayes_proj': iou_bayes_proj,
            'ham_independent': ham_independent,
            'ham_bayes': ham_bayes_raw,
            'ham_bayes_proj': ham_bayes_proj,
            'accum_frames': accum.frame_count,
        })

        prev_frame = frame

    return results


if __name__ == "__main__":
    import argparse
    from nn.resnet3d import ResNet3D
    from parity import load_constraint_ops, ConstraintConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="ckpts/adaptive_dct_embedder_v1/ckpt_28000.pt")
    parser.add_argument("--n-frames", type=int, default=30)
    parser.add_argument("--keep-freqs", type=int, default=4)
    parser.add_argument("--keyframe-interval", type=int, default=30)
    args = parser.parse_args()

    device = 'cuda'
    cfg = ConstraintConfig(axis_parity=True, plane_parity=True, global_parity=True)
    _, left_inv, parity = load_constraint_ops(32, cfg)
    parity = parity.to(device)
    left_inv = left_inv.to(device)
    M = parity.size(0)

    decoder = ResNet3D(in_ch=3, width=128, depth=12, cond_dim=M).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    decoder.load_state_dict(ckpt.get('ema', ckpt.get('model', ckpt)), strict=False)
    decoder.eval()

    # Generate orbit frames
    n = 32
    coords = torch.linspace(-1, 1, n)
    xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing='ij')

    frames = []
    for i in range(args.n_frames):
        t = i / args.n_frames * 2 * np.pi
        voxels = torch.zeros_like(xx, dtype=torch.bool)
        for k in range(4):
            angle = t + k * np.pi / 2
            cx = 0.4 * np.cos(angle)
            cz = 0.4 * np.sin(angle)
            cy = 0.2 * np.sin(angle * 2)
            r = 0.15 + 0.03 * np.sin(t * 3 + k)
            voxels = voxels | (((xx-cx)**2 + (yy-cy)**2 + (zz-cz)**2) < r**2)
        voxels = voxels | ((xx**2 + yy**2 + zz**2) < 0.2**2)
        frames.append(voxels.to(torch.uint8).reshape(-1).numpy())

    print(f"Evaluating {len(frames)} frames, keep_freqs={args.keep_freqs}, "
          f"keyframe_interval={args.keyframe_interval}")

    results = eval_temporal_bayes(decoder, frames, parity, left_inv, device,
                                  keep_freqs=args.keep_freqs,
                                  keyframe_interval=args.keyframe_interval)

    print(f"\n{'Frame':>5} {'KF':>3} {'Delta':>6} | {'Indep IoU':>9} {'Ham':>5} | "
          f"{'Bayes IoU':>9} {'Ham':>5} | {'B+Proj IoU':>10} {'Ham':>5} | {'Accum':>5}")
    print("-" * 85)
    for r in results:
        kf = "KEY" if r['keyframe'] else ""
        print(f"{r['frame']:>5} {kf:>3} {r['delta_voxels']:>6} | "
              f"{r['iou_independent']:>9.4f} {r['ham_independent']:>5.0f} | "
              f"{r['iou_bayes']:>9.4f} {r['ham_bayes']:>5.0f} | "
              f"{r['iou_bayes_proj']:>10.4f} {r['ham_bayes_proj']:>5.0f} | "
              f"{r['accum_frames']:>5}")

    print("-" * 85)
    avg_indep = np.mean([r['iou_independent'] for r in results])
    avg_bayes = np.mean([r['iou_bayes'] for r in results])
    avg_bayes_proj = np.mean([r['iou_bayes_proj'] for r in results])
    print(f"Average IoU — Independent: {avg_indep:.4f}  "
          f"Bayes: {avg_bayes:.4f}  "
          f"Bayes+Proj: {avg_bayes_proj:.4f}")
    print(f"Bayes improvement: {avg_bayes_proj - avg_indep:+.4f} IoU")
