from __future__ import annotations

import torch
from torch import nn


class TemporalConvHead(nn.Module):
    """
    Lightweight temporal modeling head for per-segment classification.

    Input:  x [B, T, D]
    Output: y [B, T, C]
    """

    def __init__(self, *, in_dim: int, num_classes: int, hidden_dim: int = 256, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd to preserve length with symmetric padding")
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        y = self.net(x.transpose(1, 2))  # [B, C, T]
        return y.transpose(1, 2).contiguous()  # [B, T, C]

