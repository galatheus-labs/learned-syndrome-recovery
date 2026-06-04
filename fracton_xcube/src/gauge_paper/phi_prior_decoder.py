r"""
phi-prior decoder: ResNet3D backbone whose final output is interpreted as
a scalar potential phi on the dual lattice, then mapped to a residual
voxel field via a fixed (non-learned) F_2 third-derivative output operator.

By the gauge theorem (see PAPER_DRAFT.md), the codespace of axial
line/plane/global parity on the 3-torus equals the image of
\partial_x \partial_y \partial_z. So every output of this network is a
codeword by construction; no capacity is spent enforcing the parity
constraints.

Interface matches the existing ResNet3D so we can A/B at matched params.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# Allow running this file from either internal/gauge_paper/ or the parity_diffusion root.
import os as _os
_DEFAULT_PARITY_ROOT = Path(__file__).resolve().parent.parent.parent
_PARITY_ROOT = Path(_os.environ.get("GALAREON_PD_ROOT", str(_DEFAULT_PARITY_ROOT)))
if not (_PARITY_ROOT / "parity.py").exists():
    _alt = Path("/app/parity_diffusion")
    if (_alt / "parity.py").exists():
        _PARITY_ROOT = _alt
if str(_PARITY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PARITY_ROOT))

from nn.resnet3d import ResNet3D  # noqa: E402


def soft_xor_8corners(phi_logits: torch.Tensor) -> torch.Tensor:
    """Soft (differentiable) F_2 XOR of phi over the 8 corners of each unit cube.

    Uses the bilinear formulation: XOR(a, b) = a + b - 2ab. This is exact for
    binary {0, 1} inputs and gives a clean continuous extension to [0, 1].
    Composing 7 bilinear steps over 8 corners. Gradient |∂XOR/∂a| = |1 - 2b|
    is bounded in [0, 1], no multiplicative attenuation across the chain.
    """
    B, C, n1, n2, n3 = phi_logits.shape
    assert C == 1, "phi_prior expects single-channel scalar potential"

    p = torch.sigmoid(phi_logits)
    p_pad = F.pad(p, (0, 1, 0, 1, 0, 1), mode="circular")

    # Combine 8 corners pairwise via bilinear XOR.
    out = None
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                pc = p_pad[:, :, a:a + n1, b:b + n2, c:c + n3]
                if out is None:
                    out = pc
                else:
                    out = out + pc - 2.0 * out * pc

    # `out` is P(x_v = 1) in [0, 1]; convert to logit for BCEWithLogits.
    out = out.clamp(min=1e-6, max=1.0 - 1e-6)
    x_logits = torch.log(out / (1.0 - out))
    return x_logits


def hard_xor_8corners(phi_binary: torch.Tensor) -> torch.Tensor:
    """Exact F_2 XOR over the 8 corners of each unit cube.

    For inference / evaluation when phi is binary (0/1).

    Args:
      phi_binary: (B, 1, n, n, n) with values in {0, 1} (any dtype).
    Returns:
      x_binary: same shape, values in {0, 1}, dtype torch.uint8.
    """
    B, C, n1, n2, n3 = phi_binary.shape
    p = phi_binary.to(torch.uint8)
    p_pad = F.pad(p, (0, 1, 0, 1, 0, 1), mode="circular")
    out = torch.zeros_like(p)
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                out = out ^ p_pad[:, :, a:a + n1, b:b + n2, c:c + n3]
    return out


class PhiPriorResNet3D(nn.Module):
    """Wrapper: ResNet3D backbone -> phi logits -> soft XOR -> x logits.

    Identical params and forward signature as the bare ResNet3D, so the
    training script can swap one for the other transparently.
    """

    def __init__(self, output_bias_init: float = -5.0, **resnet_kwargs):
        super().__init__()
        # Backbone ends with a 1x1 conv to a single output channel — we
        # treat that single-channel output as phi_logits at every voxel.
        self.backbone = ResNet3D(**resnet_kwargs)
        # Bias the final 1x1 conv so phi starts very negative.  At init,
        # sigmoid(phi) ~= 0, so the bilinear XOR over 8 corners is also ~= 0,
        # i.e. the network predicts "no flip" everywhere — matching the
        # sparse residual prior.  Breaks the all-zero-vs-all-one symmetry
        # of the XOR and gives gradient descent a clear direction to push
        # individual voxels' phi up to flip them.
        with torch.no_grad():
            self.backbone.output_conv.bias.fill_(output_bias_init)

    @property
    def block_type(self):
        return self.backbone.block_type

    @property
    def cond_dim(self):
        return self.backbone.cond_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        phi_logits = self.backbone(x, t, cond)
        x_logits = soft_xor_8corners(phi_logits)
        return x_logits

    @torch.no_grad()
    def predict_codeword(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Inference helper: threshold phi at zero, apply hard XOR, return binary x."""
        phi_logits = self.backbone(x, t, cond)
        phi_binary = (phi_logits > 0).to(torch.uint8)
        return hard_xor_8corners(phi_binary)


class PhiPriorResNet3DEscape(nn.Module):
    """phi-prior with a residual escape hatch.

    Architecture: same backbone, but the final 1x1 conv produces 2 output
    channels — phi_logits and residual_extra. The training-time output is
    soft_xor(phi_logits) + alpha * residual_extra, giving the network a
    direct, easy gradient path while phi_prior catches up on the harder
    XOR-induced credit assignment. At eval time, residual_extra is dropped
    so the prediction is purely soft_xor(phi_logits) — a guaranteed
    codeword, which is the hypothesis being tested.

    The escape hatch is a training aid only; the eval-time IoU still
    measures the phi_prior's standalone performance.
    """

    def __init__(self, output_bias_init: float = -5.0, alpha: float = 1.0,
                 **resnet_kwargs):
        super().__init__()
        self.alpha = alpha
        self.backbone = ResNet3D(**resnet_kwargs)
        # Replace the 1-channel output_conv with a 2-channel one.
        old = self.backbone.output_conv
        new = nn.Conv3d(old.in_channels, 2, kernel_size=1)
        with torch.no_grad():
            # Channel 0 (phi): zero weights, biased to predict near-zero residual.
            # Channel 1 (residual_extra): zero weights, zero bias, network grows it.
            new.weight.zero_()
            new.bias[0] = output_bias_init
            new.bias[1] = 0.0
        self.backbone.output_conv = new

    @property
    def block_type(self):
        return self.backbone.block_type

    @property
    def cond_dim(self):
        return self.backbone.cond_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        out = self.backbone(x, t, cond)  # (B, 2, n, n, n)
        phi_logits = out[:, :1]
        x_logits_phi = soft_xor_8corners(phi_logits)
        if self.training:
            residual_extra = out[:, 1:]
            return x_logits_phi + self.alpha * residual_extra
        return x_logits_phi  # eval: phi-only, codeword guaranteed

    @torch.no_grad()
    def predict_codeword(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        out = self.backbone(x, t, cond)
        phi_logits = out[:, :1]
        phi_binary = (phi_logits > 0).to(torch.uint8)
        return hard_xor_8corners(phi_binary)


def quick_self_test():
    """Sanity check: random binary phi -> hard XOR output is in the codespace."""
    import numpy as np
    from parity import build_parity_matrix

    n = 8
    H = build_parity_matrix(n).numpy().astype(np.int64)  # (M, n^3)
    # Sample 5 random binary phi fields and check their hard-XOR images.
    rng = np.random.default_rng(42)
    for trial in range(5):
        phi = torch.tensor(rng.integers(0, 2, size=(1, 1, n, n, n)), dtype=torch.uint8)
        x = hard_xor_8corners(phi).reshape(-1).numpy().astype(np.int64)
        s = (H @ x) % 2
        violated = int(s.sum())
        print(f"trial {trial}: phi sum={int(phi.sum())}, x sum={int(x.sum())}, violated checks={violated}/{H.shape[0]}")


if __name__ == "__main__":
    quick_self_test()

    # Param count sanity for matched config.
    cond_dim = 3 * 32 * 32 - 3 * 32 + 1  # = 2977
    for w, d in [(48, 8), (64, 8), (96, 12)]:
        x_model = ResNet3D(in_ch=3, width=w, depth=d, cond_dim=cond_dim)
        phi_model = PhiPriorResNet3D(in_ch=3, width=w, depth=d, cond_dim=cond_dim)
        nx = sum(p.numel() for p in x_model.parameters())
        nphi = sum(p.numel() for p in phi_model.parameters())
        print(f"w={w} d={d}: x-prior {nx:,} params,  phi-prior {nphi:,} params,  match={nx == nphi}")
