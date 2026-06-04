"""Adaptive variable-rate DCT: random level training + brute-force inference.

Training: each sample gets a randomly chosen frequency level (uniform).
The decoder learns to handle any bandwidth equally well.

Inference: the sender tries levels cheapest-first and picks the cheapest
one where reconstruction quality meets a threshold.

No router network, no Gumbel-Softmax, no routing collapse.

Bits per level:
  keep_freqs=2:   216 bits (27 coefficients)
  keep_freqs=3:  1000 bits
  keep_freqs=4:  2744 bits  ← previous fixed baseline
  keep_freqs=5:  5832 bits
  keep_freqs=6: 10648 bits
  keep_freqs=7: 17576 bits
  keep_freqs=8: 27000 bits
"""

import torch

FREQ_LEVELS = [2, 3, 4, 5, 6, 7, 8]
FREQ_BITS = torch.tensor([(2 * k - 1) ** 3 * 8 for k in FREQ_LEVELS], dtype=torch.float)


def random_level_selection(batch_size, device, levels=None):
    """Sample a random level per sample (uniform). Returns one-hot (B, K)."""
    levels = levels or FREQ_LEVELS
    K = len(levels)
    idx = torch.randint(0, K, (batch_size,), device=device)
    return torch.nn.functional.one_hot(idx, K).float(), idx


def rate_bits(selection):
    """Expected bits given selection weights. (B,)"""
    bits = FREQ_BITS.to(selection.device)
    return (selection * bits).sum(dim=-1)


def selected_level(selection):
    """Which level index was picked. (B,) int."""
    return selection.argmax(dim=-1)


def make_adaptive_dct_field(voxels, cube_size, selection, levels=None):
    """Build DCT field weighted by categorical selection.

    With one-hot selection, only one level is active per sample.

    Args:
        voxels: (B, N) uint8
        cube_size: grid size
        selection: (B, K) one-hot selection weights

    Returns:
        (B, 1, D, D, D) spatial field normalized to [0, 1]
    """
    levels = levels or FREQ_LEVELS
    B = voxels.size(0)
    D = cube_size
    v3d = voxels.float().view(B, D, D, D)

    fft_3d = torch.fft.rfftn(v3d, dim=(-3, -2, -1))

    result = torch.zeros(B, D, D, D, device=voxels.device)

    for k_idx, keep_freqs in enumerate(levels):
        weight = selection[:, k_idx]
        if weight.sum() < 1e-6:
            continue

        field = _apply_freq_cutoff_batch(fft_3d, D, keep_freqs)
        result = result + weight.view(B, 1, 1, 1) * field

    lo = result.amin(dim=(-3, -2, -1), keepdim=True)
    hi = result.amax(dim=(-3, -2, -1), keepdim=True)
    result = (result - lo) / (hi - lo + 1e-8)

    return result.unsqueeze(1)


def _apply_freq_cutoff_batch(fft_3d, D, keep_freqs):
    """Apply frequency cutoff to a batch of FFTs."""
    k = keep_freqs
    mask = torch.zeros(D, D, D // 2 + 1, dtype=torch.bool, device=fft_3d.device)
    mask[:k, :k, :k] = True
    if k > 1:
        mask[-k+1:, :k, :k] = True
        mask[:k, -k+1:, :k] = True
        mask[-k+1:, -k+1:, :k] = True

    filtered = fft_3d * mask.unsqueeze(0)
    return torch.fft.irfftn(filtered, s=(D, D, D), dim=(-3, -2, -1))
