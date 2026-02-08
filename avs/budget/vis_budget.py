from __future__ import annotations

import math
from dataclasses import dataclass

from avs.metrics.time_windows import TimeWindow
from avs.sampling.token_budget import TokenBudget


@dataclass(frozen=True)
class VisualConfig:
    """
    Discrete visual sampling configuration.

    - fps: frames per second processed in this window (budget accounting only; actual sampling is handled elsewhere)
    - resolution: square input resolution (e.g. 112/224/448)
    - r_keep: token keep ratio in (0, 1]; for "no token reduction", use 1.0
    """

    name: str
    fps: float
    resolution: int
    r_keep: float = 1.0

    def __post_init__(self) -> None:
        if float(self.fps) <= 0.0:
            raise ValueError(f"fps must be > 0, got {self.fps}")
        if int(self.resolution) <= 0:
            raise ValueError(f"resolution must be > 0, got {self.resolution}")
        rk = float(self.r_keep)
        if not (0.0 < rk <= 1.0):
            raise ValueError(f"r_keep must be in (0,1], got {self.r_keep}")


def token_cost(*, cfg: VisualConfig, duration_s: float, patch_size: int = 16) -> float:
    """
    Visual token cost following the paper definition:

      Tok(c; L) = (fps · L) · (res / p)^2 · r_keep

    We compute (res / p)^2 via `TokenBudget.tokens_for_resolution`.
    """
    d = float(duration_s)
    if d < 0.0:
        raise ValueError(f"duration_s must be >= 0, got {duration_s}")
    budget = TokenBudget(patch_size=int(patch_size))
    tokens_per_frame = float(budget.tokens_for_resolution(int(cfg.resolution)))
    return float(float(cfg.fps) * d * tokens_per_frame * float(cfg.r_keep))


def window_duration_s(w: TimeWindow) -> float:
    return float(max(0.0, float(w.end_s) - float(w.start_s)))


def token_cost_window(*, cfg: VisualConfig, window: TimeWindow, patch_size: int = 16) -> float:
    return token_cost(cfg=cfg, duration_s=window_duration_s(window), patch_size=int(patch_size))


def split_anchor_budget(*, b_vis: float, alpha: float) -> tuple[float, float]:
    """
    Split total visual budget into (anchor_budget, background_budget) where:
      B_anchor = (1 - alpha) * B_vis.
    """
    b = float(b_vis)
    if b < 0.0:
        raise ValueError(f"b_vis must be >= 0, got {b_vis}")
    a = float(alpha)
    if a < 0.0 or a > 1.0:
        raise ValueError(f"alpha must be in [0,1], got {alpha}")
    b_anchor = float((1.0 - a) * b)
    b_back = float(a * b)
    return b_anchor, b_back


def default_h(cfg: VisualConfig) -> float:
    """
    Pre-registered monotonic "utility prior" for configs.

    We use a weakly sub-linear proxy to avoid canceling the cost definition:
      h(c) = log(1 + Tok(c; L=1s))

    This stays monotonic in fps/res/r_keep while not being trivially proportional to Tok.
    """
    c = token_cost(cfg=cfg, duration_s=1.0, patch_size=16)
    return float(math.log1p(max(0.0, c)))

