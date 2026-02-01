from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnchorMetrics:
    recall: float
    covered: list[int]


def dilate_anchors(anchors: list[int], *, num_segments: int, delta: int = 0) -> set[int]:
    covered: set[int] = set()
    for a in anchors:
        for t in range(a - delta, a + delta + 1):
            if 0 <= t < num_segments:
                covered.add(t)
    return covered


def recall_at_k(gt_segments: list[int], anchors: list[int], *, num_segments: int, delta: int = 0) -> AnchorMetrics:
    gt_set = set(int(x) for x in gt_segments)
    if not gt_set:
        return AnchorMetrics(recall=0.0, covered=[])

    covered = dilate_anchors(anchors, num_segments=num_segments, delta=delta)
    hit = len(gt_set.intersection(covered))
    return AnchorMetrics(recall=hit / len(gt_set), covered=sorted(covered))

