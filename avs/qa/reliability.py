from __future__ import annotations

from dataclasses import dataclass

from avs.utils.scores import minmax_01


@dataclass(frozen=True)
class ReliabilityReport:
    metric: str
    q_bar: float
    alpha: float
    alpha_min: float
    alpha_max: float


def confidence_top3_bottom3_gap_norm(scores: list[float]) -> float:
    """
    Scale-invariant "separation" confidence:

      q = mean(top-3(scores_norm)) - mean(bottom-3(scores_norm))

    Returns in [0, 1] (best-effort; clamps for safety).
    """
    s01 = minmax_01(scores)
    if not s01:
        return 0.0
    xs = sorted(float(x) for x in s01)
    k = min(3, len(xs))
    bot = float(sum(xs[:k]) / float(k))
    top = float(sum(xs[-k:]) / float(k))
    q = float(top - bot)
    return float(max(0.0, min(1.0, q)))


def alpha_from_qbar(
    q_bar: float,
    *,
    alpha_min: float = 0.10,
    alpha_max: float = 0.60,
) -> float:
    """
    Reliability-gated alpha for background fallback.
    """
    q = float(max(0.0, min(1.0, float(q_bar))))
    a0 = float(alpha_min)
    a1 = float(alpha_max)
    if not (0.0 <= a0 <= a1 <= 1.0):
        raise ValueError(f"invalid alpha range: alpha_min={alpha_min}, alpha_max={alpha_max}")
    alpha = a0 + (1.0 - q) * (a1 - a0)
    return float(max(a0, min(a1, alpha)))


def reliability_and_alpha(
    scores: list[float],
    *,
    metric: str = "top3_bottom3_gap_norm",
    alpha_min: float = 0.10,
    alpha_max: float = 0.60,
) -> ReliabilityReport:
    m = str(metric)
    if m != "top3_bottom3_gap_norm":
        raise ValueError(f"unsupported reliability metric: {metric!r}")
    q_bar = confidence_top3_bottom3_gap_norm(scores)
    alpha = alpha_from_qbar(q_bar, alpha_min=float(alpha_min), alpha_max=float(alpha_max))
    return ReliabilityReport(metric=m, q_bar=float(q_bar), alpha=float(alpha), alpha_min=float(alpha_min), alpha_max=float(alpha_max))

