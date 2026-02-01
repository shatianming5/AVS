from __future__ import annotations

from dataclasses import dataclass

import torch

from avs.sampling.plans import SamplingPlan


@dataclass(frozen=True)
class SyntheticAVEConfig:
    num_samples: int = 256
    num_segments: int = 10
    num_classes: int = 29  # background + 28 events
    feat_dim: int = 64
    seed: int = 0


def _noise_scale_for_resolution(resolution: int) -> float:
    # Higher resolution -> less noise.
    if resolution >= 448:
        return 0.2
    if resolution >= 224:
        return 0.9
    return 1.8


def make_synthetic_ave(
    cfg: SyntheticAVEConfig,
    *,
    plan: SamplingPlan,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      x: [N, T, D]
      y: [N, T]
    """
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    prototypes = torch.randn(cfg.num_classes, cfg.feat_dim, generator=g)
    prototypes[0].zero_()  # background prototype: easy to separate

    # Fixed event segments to emulate "anchors" at seconds 6 and 7.
    event_segments = [6, 7]

    y = torch.zeros(cfg.num_samples, cfg.num_segments, dtype=torch.long)
    for i in range(cfg.num_samples):
        for t in event_segments:
            y[i, t] = int(torch.randint(1, cfg.num_classes, (1,), generator=g).item())

    x = torch.empty(cfg.num_samples, cfg.num_segments, cfg.feat_dim, dtype=torch.float32)
    for t in range(cfg.num_segments):
        # Background segments are intentionally easy even at low resolution;
        # event segments get sharper with higher resolution.
        sigma_event = _noise_scale_for_resolution(plan.resolutions[t])
        sigma_bg = 0.15

        noise = torch.randn(cfg.num_samples, cfg.feat_dim, generator=g)
        sigma = torch.full((cfg.num_samples, 1), sigma_bg, dtype=torch.float32)
        sigma[y[:, t] != 0] = sigma_event
        x[:, t, :] = prototypes[y[:, t]] + noise * sigma

    return x.to(device), y.to(device)
