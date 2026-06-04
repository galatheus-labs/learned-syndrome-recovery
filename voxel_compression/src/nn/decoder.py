import torch
import torch.nn as nn


class SyndromeDecoder(nn.Module):
    """CNN decoder: syndrome vector → voxel grid.

    Architecture:
        syndrome (M,) → linear → (C, 2, 2, 2)
        → 4x ConvTranspose3d (stride 2) → (1, 32, 32, 32)
    """

    def __init__(self, syndrome_dim: int, base: int = 64):
        super().__init__()
        c = base
        self.project = nn.Linear(syndrome_dim, c * 8 * 2 * 2 * 2)
        self.init_channels = c * 8
        self.decoder = nn.Sequential(
            # (C*8, 2, 2, 2) → (C*4, 4, 4, 4)
            nn.ConvTranspose3d(c * 8, c * 4, 4, stride=2, padding=1),
            nn.BatchNorm3d(c * 4),
            nn.ReLU(inplace=True),
            # (C*4, 4, 4, 4) → (C*2, 8, 8, 8)
            nn.ConvTranspose3d(c * 4, c * 2, 4, stride=2, padding=1),
            nn.BatchNorm3d(c * 2),
            nn.ReLU(inplace=True),
            # (C*2, 8, 8, 8) → (C, 16, 16, 16)
            nn.ConvTranspose3d(c * 2, c, 4, stride=2, padding=1),
            nn.BatchNorm3d(c),
            nn.ReLU(inplace=True),
            # (C, 16, 16, 16) → (C//2, 32, 32, 32)
            nn.ConvTranspose3d(c, c // 2, 4, stride=2, padding=1),
            nn.BatchNorm3d(c // 2),
            nn.ReLU(inplace=True),
            # (C//2, 32, 32, 32) → (1, 32, 32, 32)
            nn.Conv3d(c // 2, 1, 3, padding=1),
        )

    def forward(self, syndrome: torch.Tensor) -> torch.Tensor:
        """
        Args:
            syndrome: (B, M) float syndrome vector
        Returns:
            logits: (B, 1, 32, 32, 32) voxel logits
        """
        B = syndrome.size(0)
        x = self.project(syndrome)
        x = x.view(B, self.init_channels, 2, 2, 2)
        return self.decoder(x)
