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


def budget_band_anchored_plan_scored(
    *,
    num_segments: int,
    anchors: list[int],
    scores: list[float] | None = None,
    base_alloc: str = "distance",
    low_res: int = 112,
    base_res: int = 224,
    high_res: int = 448,
    extra_resolutions: list[int] | None = None,
    max_high_anchors: int | None = None,
    patch_size: int = 16,
    epsilon_frac: float = 0.01,
) -> SamplingPlan:
    """
    Budget-band variant of `equal_token_budget_anchored_plan_scored`.

    Motivation:
      - The exact equal-budget plan requires a divisibility constraint on token deltas which can severely
        restrict the feasible triads (low/base/high).
      - On AVE/P0, this constraint can force overly-extreme plans in the 2-high regime (context loss),
        hurting transfer.

    This planner:
      - Never exceeds the uniform `base_res` token budget (same target as the exact planner).
      - Allows a small under-budget band: `target - total_tokens <= epsilon_frac * target`.
      - Can use additional resolutions (`extra_resolutions`) to better fit the budget while preserving
        more `base_res` context.

    The allocation is deterministic and score-aware only through the candidate ordering determined by
    `base_alloc` (same semantics as the exact planner).
    """
    budget = TokenBudget(patch_size=patch_size)
    c_base = budget.tokens_for_resolution(base_res)
    c_high = budget.tokens_for_resolution(high_res)

    # Build the non-high resolution option set (includes base_res; excludes high_res).
    opts = [int(low_res), int(base_res)]
    if extra_resolutions:
        opts.extend(int(r) for r in extra_resolutions)
    opts = sorted({int(r) for r in opts if int(r) != int(high_res)})
    if int(base_res) not in opts:
        opts.append(int(base_res))
        opts = sorted(set(opts))

    if not opts:
        raise ValueError("no non-high resolution options provided")

    opt_costs = {int(r): int(budget.tokens_for_resolution(int(r))) for r in opts}

    # Resolve unique anchors and a feasible high-set under the banded budget.
    uniq_anchors: list[int] = []
    for a in anchors:
        a = int(a)
        if 0 <= a < int(num_segments) and a not in uniq_anchors:
            uniq_anchors.append(a)

    target_total = int(num_segments) * int(c_base)
    eps = float(epsilon_frac)
    if eps < 0.0:
        raise ValueError("epsilon_frac must be non-negative")
    min_total = float(target_total) * (1.0 - eps)

    min_opt_cost = min(opt_costs.values())
    k_high = len(uniq_anchors)
    if max_high_anchors is not None:
        k_high = min(int(k_high), int(max_high_anchors))
    while k_high > 0:
        high_cost = int(k_high) * int(c_high)
        min_possible = high_cost + (int(num_segments) - int(k_high)) * int(min_opt_cost)
        if min_possible <= int(target_total):
            break
        k_high -= 1

    high_set = set(int(a) for a in uniq_anchors[: int(k_high)])
    anchor_set = set(int(a) for a in uniq_anchors)

    high_cost = int(len(high_set)) * int(c_high)
    remaining_max = int(target_total) - int(high_cost)
    # Use ceil to ensure the final plan satisfies the *float* epsilon bound.
    import math

    remaining_min = max(0, int(math.ceil(float(min_total) - float(high_cost) - 1e-9)))

    if remaining_min > remaining_max:
        raise ValueError(
            f"no feasible band-budget plan: remaining_min={remaining_min} > remaining_max={remaining_max} "
            f"(target={target_total}, high_cost={high_cost})"
        )

    alloc_raw = str(base_alloc).strip()
    base_anchor_highonly = False
    if alloc_raw.endswith("_high"):
        base_anchor_highonly = True
        alloc = alloc_raw[: -len("_high")]
    else:
        alloc = alloc_raw

    if alloc not in ("distance", "balanced", "bridge", "score", "farthest", "mixed"):
        raise ValueError(
            f"unknown base_alloc={alloc_raw!r}; expected "
            f"'distance', 'balanced', 'bridge', 'score', 'farthest', or 'mixed' "
            f"(optionally with suffix '_high')"
        )

    if alloc == "score":
        if scores is None:
            raise ValueError("base_alloc='score' requires scores to be provided")
        if len(scores) < int(num_segments):
            raise ValueError(f"scores length {len(scores)} < num_segments {num_segments}")

    candidates = [i for i in range(int(num_segments)) if int(i) not in high_set]
    base_anchor_set = set(int(a) for a in (high_set if base_anchor_highonly and high_set else anchor_set))

    if alloc == "score":
        if anchor_set:
            candidates.sort(
                key=lambda i: (
                    -float(scores[i]),
                    min(abs(int(i) - int(a)) for a in base_anchor_set),
                    int(i),
                )
            )
        else:
            candidates.sort(key=lambda i: (-float(scores[i]), int(i)))
    elif alloc == "distance":
        if base_anchor_set:
            candidates.sort(key=lambda i: (min(abs(int(i) - int(a)) for a in base_anchor_set), int(i)))
        else:
            candidates.sort(key=lambda i: int(i))
    elif alloc == "balanced":
        if base_anchor_set:
            anchors_sorted = sorted(int(a) for a in base_anchor_set)
            selected: list[int] = []
            selected_set = set(int(a) for a in high_set)

            def _try_add(i: int) -> None:
                if int(i) < 0 or int(i) >= int(num_segments):
                    return
                if int(i) in selected_set:
                    return
                selected_set.add(int(i))
                selected.append(int(i))

            for d in range(1, int(num_segments) + 1):
                for a in anchors_sorted:
                    _try_add(int(a) - int(d))
                for a in anchors_sorted:
                    _try_add(int(a) + int(d))
                for i in candidates:
                    _try_add(int(i))
            candidates = selected
        else:
            candidates.sort(key=lambda i: int(i))
    elif alloc == "bridge":
        if len(high_set) >= 2:
            hs = sorted(int(a) for a in high_set)
            left, right = int(hs[0]), int(hs[1])
            mid = 0.5 * (float(left) + float(right))

            between = [int(i) for i in candidates if int(left) < int(i) < int(right)]
            between.sort(key=lambda i: (abs(float(i) - mid), int(i)))

            rest = [int(i) for i in candidates if int(i) not in set(between)]
            if base_anchor_set:
                rest.sort(key=lambda i: (min(abs(int(i) - int(a)) for a in base_anchor_set), int(i)))
            else:
                rest.sort(key=lambda i: int(i))

            candidates = [*between, *rest]
        else:
            if base_anchor_set:
                candidates.sort(key=lambda i: (min(abs(int(i) - int(a)) for a in base_anchor_set), int(i)))
            else:
                candidates.sort(key=lambda i: int(i))
    elif alloc == "farthest":
        if base_anchor_set:
            candidates.sort(key=lambda i: (-min(abs(int(i) - int(a)) for a in base_anchor_set), int(i)))
        else:
            candidates.sort(key=lambda i: int(i))
    else:
        # alloc == "mixed": interleave near and far ordering deterministically.
        if base_anchor_set:
            near_sorted = sorted(candidates, key=lambda i: (min(abs(int(i) - int(a)) for a in base_anchor_set), int(i)))
            far_sorted = sorted(
                candidates, key=lambda i: (-min(abs(int(i) - int(a)) for a in base_anchor_set), int(i))
            )
            seen = set()
            merged: list[int] = []
            for i in near_sorted + far_sorted:
                if int(i) in seen:
                    continue
                seen.add(int(i))
                merged.append(int(i))
            candidates = merged
        else:
            candidates.sort(key=lambda i: int(i))

    # DP: assign non-high resolutions to candidates to maximize a simple priority-weighted utility,
    # while meeting the banded budget constraint.
    #
    # We use a rank-based weight (higher priority => larger weight) and an ordinal utility per resolution
    # (higher resolution => larger level). This keeps the planner transparent and reproducible.
    opts_by_cost = sorted(opts, key=lambda r: (int(opt_costs[int(r)]), int(r)))
    levels = {int(r): int(i) for i, r in enumerate(opts_by_cost)}
    costs = {int(r): int(opt_costs[int(r)]) for r in opts_by_cost}

    n = len(candidates)
    if n == 0:
        plan = SamplingPlan(resolutions=[high_res if i in high_set else base_res for i in range(int(num_segments))], patch_size=patch_size)
        got = plan.total_tokens()
        if got > int(target_total) or (float(int(target_total) - int(got)) > float(target_total) * float(eps) + 1e-9):
            raise AssertionError("bug: empty-candidate band plan violates budget constraints")
        return plan

    weights = [int(n - i) for i in range(n)]

    neg_inf = -10**18
    dp = [neg_inf] * (int(remaining_max) + 1)
    dp[0] = 0
    prev_cost: list[list[int]] = [[-1] * (int(remaining_max) + 1) for _ in range(n)]
    prev_choice: list[list[int]] = [[-1] * (int(remaining_max) + 1) for _ in range(n)]

    opt_list = [int(r) for r in opts_by_cost]
    for i in range(n):
        new_dp = [neg_inf] * (int(remaining_max) + 1)
        w = int(weights[i])
        for b in range(int(remaining_max) + 1):
            if dp[b] == neg_inf:
                continue
            base_util = int(dp[b])
            for r in opt_list:
                cb = int(costs[int(r)])
                nb = int(b) + int(cb)
                if nb > int(remaining_max):
                    continue
                util = int(base_util) + int(w) * int(levels[int(r)])
                if util > int(new_dp[nb]):
                    new_dp[nb] = int(util)
                    prev_cost[i][nb] = int(b)
                    prev_choice[i][nb] = int(r)
        dp = new_dp

    best_b = None
    best_u = neg_inf
    for b in range(int(remaining_min), int(remaining_max) + 1):
        u = int(dp[b])
        if u == neg_inf:
            continue
        if u > int(best_u) or (u == int(best_u) and (best_b is None or int(b) > int(best_b))):
            best_u = int(u)
            best_b = int(b)

    if best_b is None:
        raise ValueError(
            f"no feasible plan in budget band: remaining_min={remaining_min}, remaining_max={remaining_max}, opts={opts_by_cost}"
        )

    chosen_res_by_idx: dict[int, int] = {}
    b = int(best_b)
    for i in range(n - 1, -1, -1):
        r = int(prev_choice[i][b])
        if r < 0:
            raise AssertionError("bug: DP backpointer missing")
        chosen_res_by_idx[int(candidates[i])] = int(r)
        b = int(prev_cost[i][b])
        if b < 0:
            raise AssertionError("bug: DP prev_cost missing")

    resolutions = [int(opts_by_cost[0])] * int(num_segments)
    for i in range(int(num_segments)):
        if int(i) in high_set:
            resolutions[int(i)] = int(high_res)
        else:
            resolutions[int(i)] = int(chosen_res_by_idx[int(i)])

    plan = SamplingPlan(resolutions=resolutions, patch_size=patch_size)
    got = int(plan.total_tokens())
    if got > int(target_total):
        raise AssertionError(f"bug: band plan exceeds budget: got={got} > target={target_total}")
    if float(int(target_total) - int(got)) > float(target_total) * float(eps) + 1e-9:
        raise AssertionError(
            f"bug: band plan under-uses budget beyond epsilon: got={got}, target={target_total}, epsilon_frac={epsilon_frac}"
        )
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

    alloc_raw = str(base_alloc).strip()
    base_anchor_highonly = False
    if alloc_raw.endswith("_high"):
        base_anchor_highonly = True
        alloc = alloc_raw[: -len("_high")]
    else:
        alloc = alloc_raw

    if alloc not in ("distance", "balanced", "bridge", "score", "farthest", "mixed"):
        raise ValueError(
            f"unknown base_alloc={alloc_raw!r}; expected "
            f"'distance', 'balanced', 'bridge', 'score', 'farthest', or 'mixed' "
            f"(optionally with suffix '_high')"
        )

    if alloc == "score":
        if scores is None:
            raise ValueError("base_alloc='score' requires scores to be provided")
        if len(scores) < int(num_segments):
            raise ValueError(f"scores length {len(scores)} < num_segments {num_segments}")

    # Default everything to low.
    resolutions = [low_res] * num_segments
    for a in high_set:
        resolutions[a] = high_res
    base_anchor_set = set(int(a) for a in (high_set if base_anchor_highonly and high_set else anchor_set))

    if num_base:
        candidates = [i for i in range(num_segments) if i not in high_set]

        if alloc == "score":
            # Prefer high-score segments; break ties by distance-to-anchor then index for determinism.
            if anchor_set:
                candidates.sort(
                    key=lambda i: (
                        -float(scores[i]),
                        min(abs(i - a) for a in base_anchor_set),
                        i,
                    )
                )
            else:
                candidates.sort(key=lambda i: (-float(scores[i]), i))
        elif alloc == "distance":
            # Backward-compatible behavior: closest to anchors (or left-to-right if no anchors).
            if base_anchor_set:
                candidates.sort(key=lambda i: (min(abs(i - a) for a in base_anchor_set), i))
            else:
                candidates.sort(key=lambda i: i)
        elif alloc == "balanced":
            # Distance-to-anchor, but avoid pathological tie-breaking that can allocate all base slots around
            # a single anchor when multiple anchors have equally-close neighbors.
            #
            # We expand outwards from anchors by distance and pick candidates in a round-robin fashion:
            #   for d=1..: left neighbors for all anchors, then right neighbors for all anchors.
            if base_anchor_set:
                anchors_sorted = sorted(int(a) for a in base_anchor_set)
                selected: list[int] = []
                selected_set = set(high_set)

                def _try_add(i: int) -> None:
                    if i < 0 or i >= int(num_segments):
                        return
                    if i in selected_set:
                        return
                    selected_set.add(int(i))
                    selected.append(int(i))

                for d in range(1, int(num_segments) + 1):
                    for a in anchors_sorted:
                        _try_add(int(a) - d)
                        if len(selected) >= int(num_base):
                            break
                    if len(selected) >= int(num_base):
                        break
                    for a in anchors_sorted:
                        _try_add(int(a) + d)
                        if len(selected) >= int(num_base):
                            break
                    if len(selected) >= int(num_base):
                        break

                # Fill any remaining slots (unlikely) deterministically.
                if len(selected) < int(num_base):
                    for i in candidates:
                        _try_add(int(i))
                        if len(selected) >= int(num_base):
                            break
                candidates = selected
            else:
                candidates.sort(key=lambda i: i)
        elif alloc == "bridge":
            # alloc == "bridge": for the 2-high regime, spend base budget between the two high-res anchors
            # to preserve mid-context (often where evidence lies) instead of over-focusing near each peak.
            #
            # Behavior:
            #   - If we don't have two high anchors (k_high < 2), fall back to 'distance'.
            #   - Otherwise, try to select base seconds strictly inside the [left,right] interval between the
            #     two high anchors, closest to the midpoint. If the interval is too small, fill remaining
            #     slots using the 'distance' heuristic for determinism.
            if len(high_set) >= 2:
                hs = sorted(int(a) for a in high_set)
                left, right = int(hs[0]), int(hs[1])
                mid = 0.5 * (float(left) + float(right))

                between = [i for i in candidates if int(left) < int(i) < int(right)]
                between.sort(key=lambda i: (abs(float(i) - mid), i))

                selected = [int(i) for i in between[: int(num_base)]]
                selected_set = set(selected) | set(high_set)

                if len(selected) < int(num_base):
                    rest = [i for i in candidates if int(i) not in selected_set]
                    if base_anchor_set:
                        rest.sort(key=lambda i: (min(abs(int(i) - int(a)) for a in base_anchor_set), i))
                    else:
                        rest.sort(key=lambda i: i)
                    for i in rest:
                        selected.append(int(i))
                        if len(selected) >= int(num_base):
                            break

                candidates = selected
            else:
                if base_anchor_set:
                    candidates.sort(key=lambda i: (min(abs(i - a) for a in base_anchor_set), i))
                else:
                    candidates.sort(key=lambda i: i)
        elif alloc == "farthest":
            # alloc == "farthest": spend the limited base budget away from anchors to preserve context.
            if base_anchor_set:
                candidates.sort(key=lambda i: (-min(abs(i - a) for a in base_anchor_set), i))
            else:
                candidates.sort(key=lambda i: i)
        else:
            # alloc == "mixed": allocate half near anchors (evidence) and half far (context).
            if base_anchor_set:
                near_sorted = sorted(candidates, key=lambda i: (min(abs(i - a) for a in base_anchor_set), i))
                far_sorted = sorted(candidates, key=lambda i: (-min(abs(i - a) for a in base_anchor_set), i))
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
