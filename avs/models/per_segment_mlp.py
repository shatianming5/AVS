from __future__ import annotations

import torch
from torch import nn


class PerSegmentMLP(nn.Module):
    def __init__(self, *, in_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        returns logits: [B, T, C]
        """
        b, t, d = x.shape
        y = self.net(x.view(b * t, d))
        return y.view(b, t, -1)

