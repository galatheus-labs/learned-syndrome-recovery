import math
import torch
from torch import nn
from torch.nn import functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=t.device) * -emb)
    emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        h = self.gn1(h)
        h = F.silu(h)
        time_emb = self.time_mlp(t).view(t.size(0), -1, 1, 1, 1)
        h = h + time_emb
        h = self.conv2(h)
        h = self.gn2(h)
        h = F.silu(h)
        return h + self.skip(x)


class UNet3D(nn.Module):
    """Flexible 3-D U-Net.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    base : int
        Base number of feature channels for the first layer. Each deeper
        layer doubles this value.
    num_layers : int
        Depth of the encoder / decoder (i.e. number of down-sampling steps).
    time_dim : int
        Dimensionality of the diffusion timestep embedding.
    cond_dim : int | None
        Optional conditioning vector dimensionality.
    """

    def __init__(
        self,
        in_ch: int = 1,
        base: int = 32,
        num_layers: int = 2,
        time_dim: int = 128,
        cond_dim: int | None = None,
    ) -> None:
        super().__init__()
        assert num_layers >= 1, "UNet3D requires at least one layer"
        self.time_dim = time_dim
        self.cond_dim = cond_dim
        self.num_layers = num_layers

        # ------------------------------------------------------------------
        # Embeddings for timestep (and optional conditioning)  -------------
        # ------------------------------------------------------------------
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        if cond_dim is not None:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, time_dim * 4),
                nn.SiLU(),
                nn.Linear(time_dim * 4, time_dim),
            )

        # ------------------------------------------------------------------
        # Build encoder (downs)  -------------------------------------------
        # ------------------------------------------------------------------
        self.downs: list[nn.Module] = []
        self.pools: list[nn.Module] = []
        channels: list[int] = [base * (2 ** i) for i in range(num_layers)]
        for idx in range(num_layers):
            in_c = in_ch if idx == 0 else channels[idx - 1]
            out_c = channels[idx]
            self.downs.append(ResidualBlock(in_c, out_c, time_dim))
            if idx < num_layers - 1:
                self.pools.append(nn.AvgPool3d(2))
        self.downs = nn.ModuleList(self.downs)
        self.pools = nn.ModuleList(self.pools)

        # ------------------------------------------------------------------
        # Bottleneck  ------------------------------------------------------
        # ------------------------------------------------------------------
        self.mid = ResidualBlock(channels[-1], channels[-1], time_dim)

        # ------------------------------------------------------------------
        # Build decoder (ups)  ---------------------------------------------
        # ------------------------------------------------------------------
        self.ups: list[nn.Module] = []
        prev_ch = channels[-1]
        for skip_ch in reversed(channels[:-1]):
            # Combine up-sampled feature map with its corresponding skip.
            if prev_ch == skip_ch:
                in_c = prev_ch  # we will *add* the skip connection later
            else:
                in_c = prev_ch + skip_ch  # we will *concat* later
            self.ups.append(ResidualBlock(in_c, skip_ch, time_dim))
            prev_ch = skip_ch
        self.ups = nn.ModuleList(self.ups)

        # Final 1×1×1 convolution to get single-channel output
        self.out_conv = nn.Conv3d(base, 1, 1)

    # ----------------------------------------------------------------------
    # Forward pass  ---------------------------------------------------------
    # ----------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Embed diffusion timestep (+ optional conditioning)
        t_emb = self.time_mlp(timestep_embedding(t, self.time_dim))
        if self.cond_dim is not None and cond is not None:
            t_emb = t_emb + self.cond_mlp(cond)

        # Encoder ----------------------------------------------------------
        skips: list[torch.Tensor] = []
        for idx, down in enumerate(self.downs):
            x = down(x, t_emb)
            if idx < self.num_layers - 1:
                skips.append(x)
                x = self.pools[idx](x)

        # Bottleneck -------------------------------------------------------
        x = self.mid(x, t_emb)

        # Decoder ----------------------------------------------------------
        for up in self.ups:
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
            skip = skips.pop()
            if x.shape[1] == skip.shape[1]:
                x = x + skip  # channels match -> use additive skip
            else:
                x = torch.cat([x, skip], dim=1)  # concat otherwise
            x = up(x, t_emb)

        return self.out_conv(x)
