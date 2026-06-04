"""
Gauge-inspired voxel transporter — soft U(1) latent gauge variant.

Sender encodes 32^3 voxels into:
  1. one latent seed voxel,
  2. a low-resolution U(1) gauge/link field on an 8^3 latent grid,
  3. shared decoder weights.

Receiver reconstructs the latent grid by iterative parallel transport,
then upsamples and sharpens to predict the residual_target.

This is a *soft* gauge prior — curvature and local-transport are added
as auxiliary regularisers, not enforced architecturally. Contrasts with
the phi-prior decoder (hard F_2 fracton gauge that empirically plateaus
at ~0.29 IoU).

Pipeline adaptation:
- Wrapper exposes a (x, t, cond) -> logits forward matching ResNet3D.
- Cond (syndrome, dim M=2977) is projected via MLP and added as a bias
  to the encoder's output features.
- forward returns logits only; the trainer computes auxiliary losses
  via the wrapper's `extra_losses_for_last_forward()` accessor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def _group_count(channels: int, max_groups: int = 8) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ResBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(_group_count(channels), channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(_group_count(channels), channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return self.act(x + h)


# -----------------------------------------------------------------------------
# U(1) link rotations + lattice shifts
# -----------------------------------------------------------------------------


def rotate_pairs(x: Tensor, theta: Tensor) -> Tensor:
    b, c, d, h, w = x.shape
    if c % 2 != 0:
        raise ValueError(f"rotate_pairs requires even channels, got C={c}")
    p = c // 2
    if theta.shape != (b, p, d, h, w):
        raise ValueError(f"theta shape must be {(b, p, d, h, w)}, got {tuple(theta.shape)}")
    x2 = x.view(b, p, 2, d, h, w)
    x0, x1 = x2[:, :, 0], x2[:, :, 1]
    ct, st = torch.cos(theta), torch.sin(theta)
    y0 = ct * x0 - st * x1
    y1 = st * x0 + ct * x1
    return torch.stack([y0, y1], dim=2).view(b, c, d, h, w)


def shift_plus_x(x: Tensor) -> Tensor:
    out = torch.zeros_like(x)
    out[..., 1:] = x[..., :-1]
    return out


def shift_minus_x(x: Tensor) -> Tensor:
    out = torch.zeros_like(x)
    out[..., :-1] = x[..., 1:]
    return out


def shift_plus_y(x: Tensor) -> Tensor:
    out = torch.zeros_like(x)
    out[..., 1:, :] = x[..., :-1, :]
    return out


def shift_minus_y(x: Tensor) -> Tensor:
    out = torch.zeros_like(x)
    out[..., :-1, :] = x[..., 1:, :]
    return out


def shift_plus_z(x: Tensor) -> Tensor:
    out = torch.zeros_like(x)
    out[..., 1:, :, :] = x[..., :-1, :, :]
    return out


def shift_minus_z(x: Tensor) -> Tensor:
    out = torch.zeros_like(x)
    out[..., :-1, :, :] = x[..., 1:, :, :]
    return out


# -----------------------------------------------------------------------------
# Encoder, link head, seed head, transporter, SharpNet decoder
# -----------------------------------------------------------------------------


class VoxelEncoder(nn.Module):
    """Configurable encoder.

    down_stages controls how many factor-2 downsamples after the input
    block: 2 (default) gives 32 -> 8, 3 gives 32 -> 4, 1 gives 32 -> 16.
    """

    def __init__(self, in_ch: int = 1, latent_dim: int = 16, base_ch: int = 32, down_stages: int = 2):
        super().__init__()
        layers = [ConvBlock3D(in_ch, base_ch), ResBlock3D(base_ch)]
        ch = base_ch
        for k in range(down_stages):
            next_ch = ch * 2 if k < down_stages - 1 else latent_dim
            layers.append(ConvBlock3D(ch, next_ch, stride=2))
            layers.append(ResBlock3D(next_ch))
            ch = next_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GaugeLinkHeadU1(nn.Module):
    def __init__(self, latent_dim: int = 16, hidden_ch: int = 32, max_angle: float = math.pi):
        super().__init__()
        if latent_dim % 2 != 0:
            raise ValueError("latent_dim must be even for channel-pair U(1) rotations")
        self.num_pairs = latent_dim // 2
        self.max_angle = max_angle
        self.net = nn.Sequential(
            ConvBlock3D(latent_dim, hidden_ch),
            ResBlock3D(hidden_ch),
            nn.Conv3d(hidden_ch, 3 * self.num_pairs, kernel_size=1),
        )

    def forward(self, z: Tensor) -> Tensor:
        b, _, d, h, w = z.shape
        raw = self.net(z)
        theta = self.max_angle * torch.tanh(raw)
        return theta.view(b, 3, self.num_pairs, d, h, w)


class SeedHead(nn.Module):
    def __init__(self, latent_dim: int = 16, mode: Literal["global", "center"] = "global"):
        super().__init__()
        self.mode = mode
        if mode == "global":
            self.seed = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(latent_dim, latent_dim, 1),
                nn.SiLU(inplace=True),
                nn.Conv3d(latent_dim, latent_dim, 1),
            )
        elif mode == "center":
            self.seed = nn.Identity()
        else:
            raise ValueError("mode must be 'global' or 'center'")

    def forward(self, z: Tensor) -> Tuple[Tensor, Tuple[int, int, int]]:
        _, _, d, h, w = z.shape
        seed_index = (d // 2, h // 2, w // 2)
        if self.mode == "global":
            return self.seed(z), seed_index
        sd, sh, sw = seed_index
        return z[:, :, sd : sd + 1, sh : sh + 1, sw : sw + 1], seed_index


class GaugeTransportDecoderU1(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        if latent_dim % 2 != 0:
            raise ValueError("latent_dim must be even")
        self.latent_dim = latent_dim

    def forward(self, seed, theta, steps=None, seed_index=None):
        b, three, p, d, h, w = theta.shape
        c = p * 2
        if steps is None:
            steps = d + h + w
        if seed_index is None:
            seed_index = (d // 2, h // 2, w // 2)
        sd, sh, sw = seed_index
        state = torch.zeros((b, c, d, h, w), dtype=theta.dtype, device=theta.device)
        mask = torch.zeros((b, 1, d, h, w), dtype=theta.dtype, device=theta.device)
        state[:, :, sd : sd + 1, sh : sh + 1, sw : sw + 1] = seed
        mask[:, :, sd : sd + 1, sh : sh + 1, sw : sw + 1] = 1.0
        tz, ty, tx = theta[:, 0], theta[:, 1], theta[:, 2]
        for _ in range(steps):
            vals = [state]
            weights = [mask]
            vals.append(shift_plus_x(rotate_pairs(state, tx)))
            weights.append(shift_plus_x(mask))
            vals.append(rotate_pairs(shift_minus_x(state), -tx))
            weights.append(shift_minus_x(mask))
            vals.append(shift_plus_y(rotate_pairs(state, ty)))
            weights.append(shift_plus_y(mask))
            vals.append(rotate_pairs(shift_minus_y(state), -ty))
            weights.append(shift_minus_y(mask))
            vals.append(shift_plus_z(rotate_pairs(state, tz)))
            weights.append(shift_plus_z(mask))
            vals.append(rotate_pairs(shift_minus_z(state), -tz))
            weights.append(shift_minus_z(mask))
            denom = torch.stack(weights, dim=0).sum(dim=0).clamp_min(1e-6)
            numer = sum(v * wgt for v, wgt in zip(vals, weights))
            state = numer / denom
            mask = torch.clamp(denom, 0.0, 1.0)
        return state


class VoxelSharpNet3D(nn.Module):
    """Decoder mirror of VoxelEncoder.

    up_stages controls how many factor-2 upsamples (after which the
    sharpening conv produces output_size outputs via interpolation if
    needed). Use up_stages = down_stages of the encoder for an
    auto-symmetric autoencoder.
    """

    def __init__(self, latent_dim: int = 16, out_ch: int = 1, base_ch: int = 32, up_stages: int = 2):
        super().__init__()
        ups = []
        in_ch = latent_dim
        for k in range(up_stages):
            stage_out_ch = base_ch * (2 ** (up_stages - 1 - k))
            block = [
                nn.ConvTranspose3d(in_ch, stage_out_ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(_group_count(stage_out_ch), stage_out_ch),
                nn.SiLU(inplace=True),
                ResBlock3D(stage_out_ch),
            ]
            if k == up_stages - 1:
                block.append(ResBlock3D(stage_out_ch))  # extra residual at finest scale
            ups.append(nn.Sequential(*block))
            in_ch = stage_out_ch
        self.ups = nn.ModuleList(ups)
        self.sharp = nn.Sequential(
            ConvBlock3D(in_ch, in_ch),
            ResBlock3D(in_ch),
            nn.Conv3d(in_ch, out_ch, kernel_size=1),
        )

    def forward(self, z, output_size=None):
        h = z
        for up in self.ups:
            h = up(h)
        logits = self.sharp(h)
        if output_size is not None and tuple(logits.shape[-3:]) != tuple(output_size):
            logits = F.interpolate(logits, size=output_size, mode="trilinear", align_corners=False)
        return logits


# -----------------------------------------------------------------------------
# Auxiliary gauge losses
# -----------------------------------------------------------------------------


def plaquette_curvature_loss(theta: Tensor) -> Tensor:
    tz, ty, tx = theta[:, 0], theta[:, 1], theta[:, 2]
    f_xy = tx[:, :, :, :-1, :-1] + ty[:, :, :, :-1, 1:] - tx[:, :, :, 1:, :-1] - ty[:, :, :, :-1, :-1]
    f_yz = ty[:, :, :-1, :-1, :] + tz[:, :, :-1, 1:, :] - ty[:, :, 1:, :-1, :] - tz[:, :, :-1, :-1, :]
    f_zx = tx[:, :, :-1, :, :-1] + tz[:, :, :-1, :, 1:] - tx[:, :, 1:, :, :-1] - tz[:, :, :-1, :, :-1]
    return ((1.0 - torch.cos(f_xy)).mean() + (1.0 - torch.cos(f_yz)).mean() + (1.0 - torch.cos(f_zx)).mean()) / 3.0


def local_parallel_transport_loss(z: Tensor, theta: Tensor) -> Tensor:
    tz, ty, tx = theta[:, 0], theta[:, 1], theta[:, 2]
    pred_x = rotate_pairs(z[..., :-1], tx[..., :-1])
    target_x = z[..., 1:]
    pred_y = rotate_pairs(z[..., :-1, :], ty[..., :-1, :])
    target_y = z[..., 1:, :]
    pred_z = rotate_pairs(z[..., :-1, :, :], tz[..., :-1, :, :])
    target_z = z[..., 1:, :, :]
    return (F.mse_loss(pred_x, target_x) + F.mse_loss(pred_y, target_y) + F.mse_loss(pred_z, target_z)) / 3.0


# -----------------------------------------------------------------------------
# ResNet3D-shaped wrapper
# -----------------------------------------------------------------------------


class GaugeTransporter(nn.Module):
    """Wraps the soft-gauge transporter to match ResNet3D's forward signature.

    Forward signature:  (x, t, cond) -> x_logits  (B, 1, D, H, W).

    Cond (syndrome of shape (B, M)) is projected by an MLP and added as
    a bias to the encoder's latent output (FiLM-style channel bias).
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        latent_dim: int = 16,
        base_ch: int = 32,
        cond_dim: Optional[int] = None,
        seed_mode: Literal["global", "center"] = "global",
        # ignored kwargs so the train script can pass {width, depth, block_type, ...}
        **_unused,
    ):
        super().__init__()
        if latent_dim % 2 != 0:
            raise ValueError("latent_dim must be even")
        self.cond_dim = cond_dim
        self.encoder = VoxelEncoder(in_ch, latent_dim, base_ch)
        if cond_dim is not None:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, latent_dim * 2),
                nn.SiLU(inplace=True),
                nn.Linear(latent_dim * 2, latent_dim),
            )
        self.link_head = GaugeLinkHeadU1(latent_dim, hidden_ch=base_ch)
        self.seed_head = SeedHead(latent_dim, mode=seed_mode)
        self.transport = GaugeTransportDecoderU1(latent_dim)
        self.sharpnet = VoxelSharpNet3D(latent_dim, out_ch, base_ch)
        # Holders for auxiliary loss terms at the most recent forward pass.
        self._last_aux = None
        # Knob exposed to keep the train loop interface uniform.
        self.block_type = "gauge_transporter"

    def forward(self, x, t, cond=None):
        z = self.encoder(x)
        if self.cond_dim is not None and cond is not None:
            c_emb = self.cond_mlp(cond)  # (B, latent_dim)
            z = z + c_emb.view(c_emb.size(0), c_emb.size(1), 1, 1, 1)
        theta = self.link_head(z)
        seed, seed_index = self.seed_head(z)
        z_hat = self.transport(seed, theta, seed_index=seed_index)
        logits = self.sharpnet(z_hat, output_size=tuple(x.shape[-3:]))
        # Stash aux signals for the trainer to pick up.
        self._last_aux = {"z": z, "z_hat": z_hat, "theta": theta}
        return logits

    def extra_losses_for_last_forward(self) -> Dict[str, Tensor]:
        if self._last_aux is None:
            return {}
        z = self._last_aux["z"]
        z_hat = self._last_aux["z_hat"]
        theta = self._last_aux["theta"]
        return {
            "curvature": plaquette_curvature_loss(theta),
            "local_transport": local_parallel_transport_loss(z, theta),
            "latent_recon": F.mse_loss(z_hat, z),
        }


class LatentAutoencoder(nn.Module):
    """Ablation: encoder -> 8^3 latent -> SharpNet, NO parallel transport.

    Tests whether the gauge transporter's win comes from parallel transport
    (the gauge primitive) or just from the latent bottleneck + upsampling.
    Same parameter count region as the transporter, same data flow except
    the link-head + transport pipeline is removed.

    Forward signature matches GaugeTransporter for plug-in swap.
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        latent_dim: int = 16,
        base_ch: int = 32,
        cond_dim: Optional[int] = None,
        down_stages: int = 2,
        **_unused,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.encoder = VoxelEncoder(in_ch, latent_dim, base_ch, down_stages=down_stages)
        if cond_dim is not None:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, latent_dim * 2),
                nn.SiLU(inplace=True),
                nn.Linear(latent_dim * 2, latent_dim),
            )
        self.sharpnet = VoxelSharpNet3D(latent_dim, out_ch, base_ch, up_stages=down_stages)
        self.block_type = "latent_autoencoder"

    def forward(self, x, t, cond=None):
        z = self.encoder(x)
        if self.cond_dim is not None and cond is not None:
            c = self.cond_mlp(cond)
            z = z + c.view(c.size(0), c.size(1), 1, 1, 1)
        logits = self.sharpnet(z, output_size=tuple(x.shape[-3:]))
        return logits


if __name__ == "__main__":
    cond_dim = 2977
    for base_ch, latent_dim in [(32, 16), (48, 24), (64, 32)]:
        model = GaugeTransporter(in_ch=1, out_ch=1, latent_dim=latent_dim, base_ch=base_ch, cond_dim=cond_dim)
        n = sum(p.numel() for p in model.parameters())
        print(f"base_ch={base_ch} latent_dim={latent_dim}: {n:,} params")
