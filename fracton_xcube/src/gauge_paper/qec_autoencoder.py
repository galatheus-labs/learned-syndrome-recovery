"""
QEC autoencoder backbone — multi-scale latent encoder + pooled scalar head.

Mirrors our compression-side LatentAutoencoder but adapted for the QEC
syndrome-to-logical-observable task:
  - Input:  (B, in_ch, Dx, Dy, Dz)  (e.g. ~11x11x12 for d=5 circuit noise)
  - Output: (B,) scalar logit
  - Cond:   error_rate scalar via FiLM-style projection per stage

Hypothesis: the multi-scale latent + smoothness prior that wins for
sparse-binary voxel reconstruction also helps QEC syndrome decoding.
This is the unification test before deciding whether to merge the
compression / QEC / gauge papers.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, max_groups: int = 8) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class CondConvBlock3D(nn.Module):
    """Conv3d + GroupNorm + SiLU with optional FiLM-style conditioning."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.gn = nn.GroupNorm(_group_count(out_ch), out_ch)
        self.cond_proj = nn.Linear(cond_dim, out_ch)

    def forward(self, x, cond):
        h = self.conv(x)
        h = self.gn(h)
        if cond is not None:
            h = h + self.cond_proj(cond).view(cond.size(0), -1, 1, 1, 1)
        return F.silu(h)


class CondResBlock3D(nn.Module):
    """Residual block with optional conditioning, matching the QEC ResNet style."""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(_group_count(channels), channels)
        self.gn2 = nn.GroupNorm(_group_count(channels), channels)
        self.cond_proj = nn.Linear(cond_dim, channels)

    def forward(self, x, cond=None):
        h = F.silu(self.gn1(self.conv1(x)))
        if cond is not None:
            h = h + self.cond_proj(cond).view(cond.size(0), -1, 1, 1, 1)
        h = self.gn2(self.conv2(h))
        return F.silu(x + h)


class LatentEncoderQEC(nn.Module):
    """Multi-scale latent encoder + pooled scalar head for QEC decoding.

    Drop-in replacement for surface_code_resnet.ResNetDecoder. Same
    forward signature: forward(x, error_rate) -> (B,) scalar logit.

    Architecture:
      - input_conv (Conv3d 3x3x3) -> width channels
      - stage 0: ResBlock x 2 at full resolution
      - stage 1 (if down_stages >= 1): ConvBlock stride=2 -> width*2 channels, ResBlock x 2
      - stage 2 (if down_stages >= 2): ConvBlock stride=2 -> width*4 channels, ResBlock x 2
      - AdaptiveAvgPool3d -> MLP head -> scalar
    """

    def __init__(self, grid_size, width: int = 64, cond_dim: int = 32, in_channels: int = 1,
                 down_stages: int = 2, blocks_per_stage: int = 2,
                 # ignored kwargs to allow drop-in compatibility with ResNetDecoder
                 depth: Optional[int] = None, **_unused):
        super().__init__()
        self.cond_dim = cond_dim
        self.grid_size = grid_size

        self.rate_mlp = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.input_conv = nn.Conv3d(in_channels, width, 3, padding=1)

        # Multi-scale tower
        self.stages = nn.ModuleList()
        ch = width
        for s in range(down_stages + 1):
            stage = nn.ModuleList()
            if s > 0:
                # downsample at stage entry
                stage.append(CondConvBlock3D(ch, ch * 2, cond_dim, stride=2))
                ch = ch * 2
            for _ in range(blocks_per_stage):
                stage.append(CondResBlock3D(ch, cond_dim))
            self.stages.append(stage)
        self.final_ch = ch

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Linear(ch, ch),
            nn.SiLU(),
            nn.Linear(ch, 1),
        )

    def forward(self, x, error_rate=None):
        h = F.silu(self.input_conv(x))
        if error_rate is not None:
            cond = self.rate_mlp(error_rate.view(-1, 1))
        else:
            cond = None
        for stage in self.stages:
            for block in stage:
                if isinstance(block, CondConvBlock3D):
                    h = block(h, cond)
                else:
                    h = block(h, cond)
        h = self.pool(h).flatten(1)
        return self.head(h).squeeze(-1)


if __name__ == "__main__":
    # Param count sanity check at d=5 grid (~11x11x12).
    cond_dim = 32
    for w, ds in [(48, 1), (64, 2), (96, 2)]:
        for in_ch in (1,):
            m = LatentEncoderQEC((11, 11, 12), width=w, in_channels=in_ch, down_stages=ds, cond_dim=cond_dim)
            n = sum(p.numel() for p in m.parameters())
            x = torch.randn(2, in_ch, 11, 11, 12)
            y = m(x, torch.tensor([0.003, 0.005]))
            print(f"w={w} ds={ds} in_ch={in_ch}: {n:,} params, output={y.shape}")
