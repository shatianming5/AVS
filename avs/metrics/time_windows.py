from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeWindow:
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if float(self.end_s) < float(self.start_s):
            raise ValueError(f"invalid TimeWindow: end_s({self.end_s}) < start_s({self.start_s})")

    @property
    def duration_s(self) -> float:
        return float(self.end_s) - float(self.start_s)


def iou_1d(a: TimeWindow, b: TimeWindow) -> float:
    """
    1D interval IoU in seconds.
    """
    inter = max(0.0, min(float(a.end_s), float(b.end_s)) - max(float(a.start_s), float(b.start_s)))
    union = max(float(a.duration_s) + float(b.duration_s) - float(inter), 0.0)
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def max_iou(windows: list[TimeWindow], evidence: TimeWindow) -> float:
    if not windows:
        return 0.0
    return float(max(iou_1d(w, evidence) for w in windows))


def coverage_at_tau(*, windows: list[TimeWindow], evidence: list[TimeWindow], tau: float) -> float:
    """
    Coverage@τ: fraction of evidence intervals that are covered by at least one predicted window
    with IoU >= τ.
    """
    if not evidence:
        return 0.0
    t = float(tau)
    if t < 0.0 or t > 1.0:
        raise ValueError(f"tau must be in [0,1], got {tau}")
    hit = 0
    for e in evidence:
        if max_iou(windows, e) >= t:
            hit += 1
    return float(hit / len(evidence))

