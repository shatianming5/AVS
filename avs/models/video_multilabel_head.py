from __future__ import annotations

import torch
from torch import nn


class VideoMultiLabelHead(nn.Module):
    """
    Simple video-level multi-label classifier on top of per-second visual features.

    Inputs:
      - x: [B, T, D]
      - mask (optional): [B, T] (1 for valid steps, 0 for padding)

    Outputs:
      - logits: [B, C] (use BCEWithLogitsLoss for multi-label targets)
    """

    def __init__(
        self,
        *,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def forward(self, x: torch.Tensor, *, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected x to be [B,T,D], got shape={tuple(x.shape)}")

        if mask is None:
            pooled = x.mean(dim=1)
        else:
            if mask.ndim != 2:
                raise ValueError(f"expected mask to be [B,T], got shape={tuple(mask.shape)}")
            if mask.shape[0] != x.shape[0] or mask.shape[1] != x.shape[1]:
                raise ValueError(f"mask shape {tuple(mask.shape)} must match x[:2] {tuple(x.shape[:2])}")
            m = mask.to(dtype=x.dtype)
            denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (x * m.unsqueeze(-1)).sum(dim=1) / denom

        return self.net(pooled)

