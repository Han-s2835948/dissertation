from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype)
            * (-math.log(10_000.0) / max(half - 1, 1))
        )
        emb = torch.cat([torch.sin(t[:, None] * freq[None] * 1000),
                         torch.cos(t[:, None] * freq[None] * 1000)], dim=1)
        return self.net(emb)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ScoreUNet(nn.Module):
    """Small time-conditioned U-Net returning an image-shaped score."""

    def __init__(self, base_channels: int = 32, time_dim: int = 64):
        super().__init__()
        c = base_channels
        self.temb = TimeEmbedding(time_dim)
        self.stem = nn.Conv2d(1, c, 3, padding=1)
        self.enc1 = ResBlock(c, c, time_dim)
        self.down1 = nn.Conv2d(c, 2 * c, 4, stride=2, padding=1)
        self.enc2 = ResBlock(2 * c, 2 * c, time_dim)
        self.down2 = nn.Conv2d(2 * c, 4 * c, 4, stride=2, padding=1)
        self.mid = ResBlock(4 * c, 4 * c, time_dim)
        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, 4, stride=2, padding=1)
        self.dec2 = ResBlock(4 * c, 2 * c, time_dim)
        self.up1 = nn.ConvTranspose2d(2 * c, c, 4, stride=2, padding=1)
        self.dec1 = ResBlock(2 * c, c, time_dim)
        self.out = nn.Sequential(nn.GroupNorm(min(8, c), c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.temb(t)
        e1 = self.enc1(self.stem(x), temb)
        e2 = self.enc2(self.down1(e1), temb)
        h = self.mid(self.down2(e2), temb)
        h = self.dec2(torch.cat([self.up2(h), e2], dim=1), temb)
        h = self.dec1(torch.cat([self.up1(h), e1], dim=1), temb)
        return self.out(h)


class QNet(ScoreUNet):
    """Image-shaped estimator q_psi(t,y) of the spatial gradient of h."""

    pass


class MNISTClassifier(nn.Module):
    """Ten-class CNN used to define or independently evaluate digit identity."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class HNet(nn.Module):
    """Approximates h(tau, y)=P(Y_1 is target class | Y_tau=y)."""

    def __init__(self, base_channels: int = 32, time_dim: int = 64):
        super().__init__()
        c = base_channels
        self.temb = TimeEmbedding(time_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(c, 2 * c, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(2 * c, 4 * c, 3, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Linear(4 * c + time_dim, 2 * c), nn.SiLU(), nn.Linear(2 * c, 1))

    def logits(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x).flatten(1)
        return self.head(torch.cat([feat, self.temb(tau)], dim=1)).squeeze(1)

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(x, tau))
