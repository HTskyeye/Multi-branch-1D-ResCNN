from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """1D residual block: Conv->BN->ReLU->Conv->BN + shortcut."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block(x)
        out = out + self.shortcut(x)
        return self.relu(out)


class ChannelSpectralCNN(nn.Module):
    """Single-channel spectral regression network."""

    def __init__(self, in_len: int = 320, out_len: int = 120):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            ResidualBlock1D(64, 64),
            nn.MaxPool1d(4),  # 320 -> 80

            ResidualBlock1D(64, 128),
            nn.MaxPool1d(4),  # 80 -> 20

            ResidualBlock1D(128, 256),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(256, out_len),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.regressor(x)
        return x


class SpectralCNNIndependent(nn.Module):
    """Three independent networks for 3-channel prediction (no parameter sharing)."""

    def __init__(self, n_channels: int = 3, in_len: int = 320, out_len: int = 120):
        super().__init__()
        self.nets = nn.ModuleList([ChannelSpectralCNN(in_len=in_len, out_len=out_len) for _ in range(n_channels)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, F)
        x = x.unsqueeze(1)  # (batch,1,F)
        outs = [net(x) for net in self.nets]  # each (batch,out_len)
        return torch.stack(outs, dim=1)  # (batch,3,out_len)
