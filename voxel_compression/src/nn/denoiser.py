"""Voxel DLSS — lightweight denoiser for post-reconstruction cleanup.

Takes a noisy 32³ voxel reconstruction and outputs a clean version.
Trained on (decoder_output, ground_truth) pairs.

Tiny model: 4 conv layers, ~50K params. Runs in <1ms per frame.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxelDenoiser(nn.Module):
    """Small 3D CNN that cleans up reconstruction artifacts.

    Input: (B, 1, 32, 32, 32) noisy binary voxels (as float)
    Output: (B, 1, 32, 32, 32) logits for clean voxels
    """

    def __init__(self, width=32, depth=6):
        super().__init__()
        layers = [nn.Conv3d(1, width, 3, padding=1), nn.SiLU()]
        for _ in range(depth - 2):
            layers += [nn.Conv3d(width, width, 3, padding=1), nn.GroupNorm(4, width), nn.SiLU()]
        layers += [nn.Conv3d(width, 1, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    m = VoxelDenoiser(width=32, depth=6)
    p = sum(p.numel() for p in m.parameters())
    x = torch.randn(1, 1, 32, 32, 32)
    y = m(x)
    print(f"{p:,} params, output {y.shape}")
