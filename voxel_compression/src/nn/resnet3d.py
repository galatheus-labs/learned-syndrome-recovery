"""Flat 3D ResNet — chess-stack architecture.

No downsampling/upsampling. Constant channel width throughout.
Like AlphaZero's residual tower but for 3D voxels.

Memory efficient: peak memory proportional to width, not width^2.
A 20-block tower at 128 channels uses ~8M params vs 50M for UNet.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=t.device) * -emb)
    emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


class SqueezeExcite3D(nn.Module):
    """Squeeze-and-Excitation block for 3D."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        # Squeeze: global average pool over spatial dims
        s = x.mean(dim=(-3, -2, -1))  # (B, C)
        # Excite: FC → ReLU → FC → Sigmoid
        s = torch.sigmoid(self.fc2(F.silu(self.fc1(s))))
        return x * s.view(s.size(0), -1, 1, 1, 1)


class ResBlock3D(nn.Module):
    """Residual block with squeeze-excite attention."""

    def __init__(self, channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, channels)
        self.gn2 = nn.GroupNorm(8, channels)
        self.time_mlp = nn.Linear(time_dim, channels)
        self.se = SqueezeExcite3D(channels)

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.gn1(h)
        h = F.silu(h)
        # Add time embedding
        h = h + self.time_mlp(t_emb).view(t_emb.size(0), -1, 1, 1, 1)
        h = self.conv2(h)
        h = self.gn2(h)
        h = F.silu(h)
        h = self.se(h)  # channel attention
        return x + h


class ResBlockLC0(nn.Module):
    """LC0-style residual block: pre-activation, dual-output SE, FiLM time cond.

    Ported from surface_code_resnet.CondResBlockLC0 (QEC decoder).
    Differences from ResBlock3D:
      - Pre-activation (GN -> SiLU -> conv) for clean identity skip
      - SE has dual output: sigmoid-gated scale + additive bias per channel
      - Time embedding injected as per-channel FiLM (scale + bias), not additive
      - Optional ReZero scalar for true identity-at-init
    """

    def __init__(self, channels, time_dim, se_ratio=8, zero_init=False):
        super().__init__()
        self.gn1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)

        mid = max(channels // se_ratio, 8)
        self.se_fc1 = nn.Linear(channels, mid)
        self.se_fc2 = nn.Linear(mid, 2 * channels)

        self.time_mlp = nn.Linear(time_dim, 2 * channels)

        self.rezero = nn.Parameter(torch.zeros(1)) if zero_init else None

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.gn1(x)))

        # FiLM from time embedding (between convolutions, like the original)
        t = self.time_mlp(t_emb)
        t_scale, t_bias = t.chunk(2, dim=1)
        h = h * (1 + t_scale).view(*t_scale.shape, 1, 1, 1) \
              + t_bias.view(*t_bias.shape, 1, 1, 1)

        h = self.conv2(F.silu(self.gn2(h)))

        # LC0 SE: per-channel scale (sigmoid) + additive bias
        s = h.mean(dim=(-3, -2, -1))
        s = self.se_fc2(F.silu(self.se_fc1(s)))
        se_scale, se_bias = s.chunk(2, dim=1)
        h = h * torch.sigmoid(se_scale).view(*se_scale.shape, 1, 1, 1) \
              + se_bias.view(*se_bias.shape, 1, 1, 1)

        if self.rezero is not None:
            h = h * self.rezero

        return x + h


class ResNet3D(nn.Module):
    """Flat 3D ResNet for voxel prediction.

    Args:
        in_ch: input channels (1 for base, +1 for coarse, +1 for DCT, etc.)
        width: channel width throughout (constant)
        depth: number of residual blocks
        time_dim: timestep/conditioning embedding dimension
        cond_dim: conditioning vector size (syndrome length)
    """

    def __init__(self, in_ch=1, width=128, depth=20, time_dim=128, cond_dim=None,
                 block_type="classic", zero_init_residual=False):
        super().__init__()
        self.time_dim = time_dim
        self.cond_dim = cond_dim
        self.block_type = block_type

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Conditioning embedding (syndrome → time_dim)
        if cond_dim is not None:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, time_dim * 2),
                nn.SiLU(),
                nn.Linear(time_dim * 2, time_dim),
            )

        # Input projection
        self.input_conv = nn.Conv3d(in_ch, width, 3, padding=1)

        # Residual tower
        if block_type == "lc0":
            self.blocks = nn.ModuleList([
                ResBlockLC0(width, time_dim, zero_init=zero_init_residual)
                for _ in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                ResBlock3D(width, time_dim) for _ in range(depth)
            ])
            if zero_init_residual:
                for block in self.blocks:
                    nn.init.zeros_(block.gn2.weight)

        # Output projection
        self.output_norm = nn.GroupNorm(8, width)
        self.output_conv = nn.Conv3d(width, 1, 1)

    def forward(self, x, t, cond=None):
        # Embed timestep + conditioning
        t_emb = self.time_mlp(timestep_embedding(t, self.time_dim))
        if self.cond_dim is not None and cond is not None:
            t_emb = t_emb + self.cond_mlp(cond)

        # Input
        h = self.input_conv(x)

        # Residual tower
        for block in self.blocks:
            h = block(h, t_emb)

        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)
        return h


if __name__ == "__main__":
    """Quick test and param count."""
    cond_dim = 2977
    for btype in ["classic", "lc0"]:
        print(f"\n--- {btype} ---")
        for width, depth in [(64, 10), (96, 15), (128, 20), (64, 20)]:
            model = ResNet3D(in_ch=3, width=width, depth=depth, cond_dim=cond_dim,
                             block_type=btype, zero_init_residual=(btype == "lc0"))
            params = sum(p.numel() for p in model.parameters())
            mem_mb = params * 4 * 2 / 1024**2
            x = torch.randn(1, 3, 32, 32, 32)
            t = torch.tensor([500])
            cond = torch.randn(1, cond_dim)
            y = model(x, t, cond)
            print(f"width={width:3d} depth={depth:2d}: {params:>10,} params, ~{mem_mb:.0f}MB optimizer, output={y.shape}")
