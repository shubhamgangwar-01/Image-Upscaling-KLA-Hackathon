from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class BicubicResidualSR(nn.Module):
    """A small CPU-friendly x2 super-resolution + denoising baseline."""

    def __init__(
        self,
        in_channels: int = 1,
        features: int = 64,
        blocks: int = 8,
        upscale: int = 2,
    ) -> None:
        super().__init__()
        if upscale != 2:
            raise ValueError("This baseline is configured for x2 upscaling only.")

        self.upscale = upscale
        self.head = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        self.body = nn.Sequential(*(ResidualBlock(features) for _ in range(blocks)))
        self.body_tail = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.pre_shuffle = nn.Conv2d(
            features,
            features * upscale * upscale,
            kernel_size=3,
            padding=1,
        )
        self.shuffle = nn.PixelShuffle(upscale)
        self.refine = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(features, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(
            x,
            scale_factor=self.upscale,
            mode="bicubic",
            align_corners=False,
        )
        features = self.head(x)
        residual_features = self.body_tail(self.body(features))
        features = features + residual_features
        upsampled_features = self.shuffle(self.pre_shuffle(features))
        residual = self.refine(upsampled_features)
        return base + residual
