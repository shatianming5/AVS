from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from avs.sampling.token_budget import TokenBudget


@dataclass(frozen=True)
class SamplingPlan:
    resolutions: list[int]
    patch_size: int = 16

    def total_tokens(self) -> int:
        budget = TokenBudget(patch_size=self.patch_size)
        return sum(budget.tokens_for_resolution(r) for r in self.resolutions)

    def to_jsonable(self) -> dict:
        return {"patch_size": self.patch_size, "resolutions": self.resolutions, "total_tokens": self.total_tokens()}

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_jsonable(), indent=2, sort_keys=True) + "\n")


def uniform_plan(*, num_segments: int, resolution: int, patch_size: int = 16) -> SamplingPlan:
    return SamplingPlan(resolutions=[resolution] * num_segments, patch_size=patch_size)


def equal_token_budget_anchored_plan(
    *,
    num_segments: int,
    anchors: list[int],
    low_res: int = 112,
    base_res: int = 224,
    high_res: int = 448,
    max_high_anchors: int | None = None,
    patch_size: int = 16,
) -> SamplingPlan:
    """
    Create a per-segment resolution plan that:
      - assigns `high_res` to up to `K` anchor segments
      - assigns `base_res` to `B` segments closest to anchors
      - assigns `low_res` to remaining segments
    while matching the exact token budget of `num_segments * base_res`.

    For the default (112/224/448, patch=16, num_segments=10), the Diophantine constraint implies K∈{0,1,2}.
    """
    budget = TokenBudget(patch_size=patch_size)
    c_low = budget.tokens_for_resolution(low_res)
    c_base = budget.tokens_for_resolution(base_res)
    c_high = budget.tokens_for_resolution(high_res)

    inc_base = c_base - c_low
    inc_high = c_high - c_low
    if inc_base <= 0:
        raise ValueError("expected base_res to cost more than low_res")
    if inc_high <= 0:
        raise ValueError("expected high_res to cost more than low_res")
    if inc_high % inc_base != 0:
        raise ValueError(
            f"cannot exactly match budget with given resolutions: (high-low)={inc_high} not divisible by (base-low)={inc_base}"
        )

    ratio = inc_high // inc_base
    max_high = num_segments // ratio

    uniq_anchors: list[int] = []
    for a in anchors:
        a = int(a)
        if 0 <= a < num_segments and a not in uniq_anchors:
            uniq_anchors.append(a)
    k_high = min(len(uniq_anchors), max_high)
    if max_high_anchors is not None:
        k_high = min(k_high, int(max_high_anchors))
    high_set = set(uniq_anchors[:k_high])
    anchor_set = set(uniq_anchors)

    num_base = num_segments - ratio * k_high
    if not (0 <= num_base <= num_segments):
        raise ValueError(f"no feasible equal-budget plan: num_base={num_base}, k_high={k_high}, ratio={ratio}")

    # Default everything to low.
    resolutions = [low_res] * num_segments
    for a in high_set:
        resolutions[a] = high_res

    if num_base:
        candidates = [i for i in range(num_segments) if i not in high_set]
        if anchor_set:
            # Prefer segments closest to anchors for "base" allocation.
            candidates.sort(key=lambda i: (min(abs(i - a) for a in anchor_set), i))
        resolutions_base = candidates[:num_base]
        for i in resolutions_base:
            resolutions[i] = base_res

    plan = SamplingPlan(resolutions=resolutions, patch_size=patch_size)
    target = num_segments * c_base
    got = plan.total_tokens()
    if got != target:
        raise AssertionError(f"bug: expected exact budget {target}, got {got}")
    return plan


def equal_token_budget_anchored_plan_scored(
    *,
    num_segments: int,
    anchors: list[int],
    scores: list[float] | None = None,
    base_alloc: str = "distance",
    low_res: int = 112,
    base_res: int = 224,
    high_res: int = 448,
    max_high_anchors: int | None = None,
    patch_size: int = 16,
) -> SamplingPlan:
    """
    Score-aware variant of `equal_token_budget_anchored_plan` with configurable `base_res` allocation.

    Same equal-budget constraints, but allocates `base_res` segments using `scores` (if provided)
    instead of only distance-to-anchor. This helps when eventness is multi-peak or anchors are imperfect:
    we still spend the limited `base_res` budget on high-confidence seconds.
    """
    budget = TokenBudget(patch_size=patch_size)
    c_low = budget.tokens_for_resolution(low_res)
    c_base = budget.tokens_for_resolution(base_res)
    c_high = budget.tokens_for_resolution(high_res)

    inc_base = c_base - c_low
    inc_high = c_high - c_low
    if inc_base <= 0:
        raise ValueError("expected base_res to cost more than low_res")
    if inc_high <= 0:
        raise ValueError("expected high_res to cost more than low_res")
    if inc_high % inc_base != 0:
        raise ValueError(
            f"cannot exactly match budget with given resolutions: (high-low)={inc_high} not divisible by (base-low)={inc_base}"
        )

    ratio = inc_high // inc_base
    max_high = num_segments // ratio

    uniq_anchors: list[int] = []
    for a in anchors:
        a = int(a)
        if 0 <= a < num_segments and a not in uniq_anchors:
            uniq_anchors.append(a)
    k_high = min(len(uniq_anchors), max_high)
    if max_high_anchors is not None:
        k_high = min(k_high, int(max_high_anchors))
    high_set = set(uniq_anchors[:k_high])
    anchor_set = set(uniq_anchors)

    num_base = num_segments - ratio * k_high
    if not (0 <= num_base <= num_segments):
        raise ValueError(f"no feasible equal-budget plan: num_base={num_base}, k_high={k_high}, ratio={ratio}")

    alloc = str(base_alloc)
    if alloc not in ("distance", "score", "farthest", "mixed"):
        raise ValueError(f"unknown base_alloc={alloc!r}; expected 'distance', 'score', 'farthest', or 'mixed'")

    if alloc == "score":
        if scores is None:
            raise ValueError("base_alloc='score' requires scores to be provided")
        if len(scores) < int(num_segments):
            raise ValueError(f"scores length {len(scores)} < num_segments {num_segments}")

    # Default everything to low.
    resolutions = [low_res] * num_segments
    for a in high_set:
        resolutions[a] = high_res

    if num_base:
        candidates = [i for i in range(num_segments) if i not in high_set]

        if alloc == "score":
            # Prefer high-score segments; break ties by distance-to-anchor then index for determinism.
            if anchor_set:
                candidates.sort(
                    key=lambda i: (
                        -float(scores[i]),
                        min(abs(i - a) for a in anchor_set),
                        i,
                    )
                )
            else:
                candidates.sort(key=lambda i: (-float(scores[i]), i))
        elif alloc == "distance":
            # Backward-compatible behavior: closest to anchors (or left-to-right if no anchors).
            if anchor_set:
                candidates.sort(key=lambda i: (min(abs(i - a) for a in anchor_set), i))
            else:
                candidates.sort(key=lambda i: i)
        elif alloc == "farthest":
            # alloc == "farthest": spend the limited base budget away from anchors to preserve context.
            if anchor_set:
                candidates.sort(key=lambda i: (-min(abs(i - a) for a in anchor_set), i))
            else:
                candidates.sort(key=lambda i: i)
        else:
            # alloc == "mixed": allocate half near anchors (evidence) and half far (context).
            if anchor_set:
                near_sorted = sorted(candidates, key=lambda i: (min(abs(i - a) for a in anchor_set), i))
                far_sorted = sorted(candidates, key=lambda i: (-min(abs(i - a) for a in anchor_set), i))
                n_near = max(0, int(num_base) // 2)
                selected: list[int] = []
                for i in near_sorted:
                    selected.append(int(i))
                    if len(selected) >= n_near:
                        break
                for i in far_sorted:
                    if int(i) in selected:
                        continue
                    selected.append(int(i))
                    if len(selected) >= int(num_base):
                        break
                candidates = selected
            else:
                candidates.sort(key=lambda i: i)

        for i in candidates[:num_base]:
            resolutions[i] = base_res

    plan = SamplingPlan(resolutions=resolutions, patch_size=patch_size)
    target = num_segments * c_base
    got = plan.total_tokens()
    if got != target:
        raise AssertionError(f"bug: expected exact budget {target}, got {got}")
    return plan
