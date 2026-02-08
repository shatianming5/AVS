from __future__ import annotations

from dataclasses import dataclass

from avs.budget.vis_budget import VisualConfig, default_h, token_cost_window
from avs.metrics.time_windows import TimeWindow


@dataclass(frozen=True)
class WindowAllocation:
    window: TimeWindow
    weight: float
    cfg: VisualConfig | None
    cost: float
    dropped: bool

    def to_jsonable(self) -> dict:
        return {
            "window": {"start_s": float(self.window.start_s), "end_s": float(self.window.end_s)},
            "weight": float(self.weight),
            "cfg": None
            if self.cfg is None
            else {"name": str(self.cfg.name), "fps": float(self.cfg.fps), "resolution": int(self.cfg.resolution), "r_keep": float(self.cfg.r_keep)},
            "cost": float(self.cost),
            "dropped": bool(self.dropped),
        }


@dataclass(frozen=True)
class AllocationReport:
    budget: float
    patch_size: int
    allocations: list[WindowAllocation]
    total_cost: float
    ok: bool

    def to_jsonable(self) -> dict:
        return {
            "budget": float(self.budget),
            "patch_size": int(self.patch_size),
            "total_cost": float(self.total_cost),
            "ok": bool(self.ok),
            "allocations": [a.to_jsonable() for a in self.allocations],
        }


def allocate_budgeted_windows(
    *,
    windows: list[TimeWindow],
    weights: list[float],
    configs: list[VisualConfig],
    budget: float,
    patch_size: int = 16,
    drop_if_needed: bool = True,
) -> AllocationReport:
    """
    Deterministic Listen-then-Look allocator (baseline):

    - Start from the most expensive config for all windows.
    - If over budget, downgrade windows from low weight to high weight (tie-break by index),
      stepping down the config ladder by cost-per-second.
    - If still over budget and `drop_if_needed`, drop windows from low weight upward.
    """
    if len(windows) != len(weights):
        raise ValueError(f"windows/weights length mismatch: {len(windows)} vs {len(weights)}")
    if not configs:
        raise ValueError("configs must be non-empty")

    b = float(budget)
    if b < 0.0:
        raise ValueError(f"budget must be >= 0, got {budget}")

    # Order configs by increasing cost-per-second (ties by name for stability).
    def _cps(cfg: VisualConfig) -> float:
        # cost for 1s; patch_size affects scaling but not ordering when fixed.
        return float(token_cost_window(cfg=cfg, window=TimeWindow(0.0, 1.0), patch_size=int(patch_size)))

    cfgs = sorted(list(configs), key=lambda c: (_cps(c), str(c.name)))
    cheapest = cfgs[0]
    richest = cfgs[-1]

    # Initialize with the richest config for all windows.
    chosen_idx = [len(cfgs) - 1] * len(windows)
    dropped = [False] * len(windows)

    def _cost(i: int) -> float:
        if dropped[i]:
            return 0.0
        return float(token_cost_window(cfg=cfgs[chosen_idx[i]], window=windows[i], patch_size=int(patch_size)))

    total = float(sum(_cost(i) for i in range(len(windows))))

    # Downgrade until within budget.
    order = sorted(range(len(windows)), key=lambda i: (float(weights[i]), i))
    for i in order:
        if total <= b:
            break
        while total > b and (not dropped[i]) and chosen_idx[i] > 0:
            prev = float(_cost(i))
            chosen_idx[i] -= 1
            total = float(total - prev + float(_cost(i)))

    # If still over budget, drop windows from low weight upward.
    if total > b and drop_if_needed:
        for i in order:
            if total <= b:
                break
            if dropped[i]:
                continue
            prev = float(_cost(i))
            dropped[i] = True
            total = float(total - prev)

    ok = total <= b + 1e-6
    allocs: list[WindowAllocation] = []
    for i, w in enumerate(windows):
        if dropped[i]:
            allocs.append(WindowAllocation(window=w, weight=float(weights[i]), cfg=None, cost=0.0, dropped=True))
        else:
            cfg = cfgs[chosen_idx[i]]
            allocs.append(WindowAllocation(window=w, weight=float(weights[i]), cfg=cfg, cost=float(_cost(i)), dropped=False))

    # Attach a small "audit hint": include the ladder endpoints in the report for readability.
    # (We keep the report schema stable and machine-friendly; callers can print cfgs separately.)
    _ = (cheapest, richest, default_h)  # referenced for future extensions

    return AllocationReport(budget=b, patch_size=int(patch_size), allocations=allocs, total_cost=float(total), ok=bool(ok))


def allocate_budgeted_windows_knapsack_lagrangian(
    *,
    windows: list[TimeWindow],
    weights: list[float],
    configs: list[VisualConfig],
    budget: float,
    patch_size: int = 16,
    include_drop: bool = True,
    max_iter: int = 60,
) -> AllocationReport:
    """
    Multiple-choice knapsack (one config per window) via Lagrangian relaxation + deterministic repair.

    Objective:
      maximize sum_i weight[i] * h(cfg_i)
      subject to sum_i Tok(cfg_i; window_i) <= budget

    where h(cfg) is `avs.budget.vis_budget.default_h`.

    Notes:
      - Deterministic given inputs.
      - If `include_drop`, an additional "drop" option is available per window (cfg=None, cost=0, util=0),
        so the problem is always feasible even under tiny budgets.
    """
    if len(windows) != len(weights):
        raise ValueError(f"windows/weights length mismatch: {len(windows)} vs {len(weights)}")
    if not configs:
        raise ValueError("configs must be non-empty")

    b = float(budget)
    if b < 0.0:
        raise ValueError(f"budget must be >= 0, got {budget}")

    # Stable config order (by cost-per-second, then name).
    def _cps(cfg: VisualConfig) -> float:
        return float(token_cost_window(cfg=cfg, window=TimeWindow(0.0, 1.0), patch_size=int(patch_size)))

    cfgs = sorted(list(configs), key=lambda c: (_cps(c), str(c.name)))

    # Precompute costs and base utilities.
    n = int(len(windows))
    m = int(len(cfgs))
    costs = [[0.0 for _ in range(m)] for _ in range(n)]
    utils = [[0.0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        w = float(weights[i])
        for j in range(m):
            c = cfgs[j]
            cost = float(token_cost_window(cfg=c, window=windows[i], patch_size=int(patch_size)))
            costs[i][j] = float(cost)
            utils[i][j] = float(w * float(default_h(c)))

    def _choose_for_lambda(lam: float) -> tuple[list[int | None], float, float]:
        chosen: list[int | None] = []
        total_cost = 0.0
        total_util = 0.0
        for i in range(n):
            best_j: int | None = None
            best_v = float("-inf")

            # Optional drop option.
            if include_drop:
                best_j = None
                best_v = 0.0

            for j in range(m):
                v = float(utils[i][j]) - float(lam) * float(costs[i][j])
                if v > best_v + 1e-12:
                    best_v = float(v)
                    best_j = int(j)
                    continue
                if abs(float(v) - float(best_v)) <= 1e-12:
                    # Tie-break: prefer lower cost, then cheaper cfg index.
                    if best_j is None:
                        continue
                    if float(costs[i][j]) < float(costs[i][best_j]) - 1e-9:
                        best_j = int(j)
                    elif abs(float(costs[i][j]) - float(costs[i][best_j])) <= 1e-9 and int(j) < int(best_j):
                        best_j = int(j)

            chosen.append(best_j)
            if best_j is not None:
                total_cost += float(costs[i][int(best_j)])
                total_util += float(utils[i][int(best_j)])
        return chosen, float(total_cost), float(total_util)

    # Find an upper lambda that satisfies the budget.
    lam_lo = 0.0
    lam_hi = 1.0
    chosen_hi, cost_hi, _ = _choose_for_lambda(lam_hi)
    tries = 0
    while cost_hi > b + 1e-6 and tries < 40:
        lam_hi *= 2.0
        chosen_hi, cost_hi, _ = _choose_for_lambda(lam_hi)
        tries += 1

    # Binary search for the smallest lambda with cost <= budget.
    chosen_best = chosen_hi
    cost_best = cost_hi
    util_best = float("-inf")
    if cost_best <= b + 1e-6:
        # Track util at feasible points.
        _, _, util0 = _choose_for_lambda(lam_hi)
        util_best = float(util0)

    for _ in range(int(max_iter)):
        mid = 0.5 * (lam_lo + lam_hi)
        chosen_mid, cost_mid, util_mid = _choose_for_lambda(mid)
        if cost_mid > b + 1e-6:
            lam_lo = float(mid)
            continue
        # feasible: move hi down, keep best (prefer higher util, then higher cost as tie-break)
        lam_hi = float(mid)
        if util_mid > util_best + 1e-9 or (abs(util_mid - util_best) <= 1e-9 and cost_mid > cost_best + 1e-6):
            chosen_best, cost_best, util_best = chosen_mid, float(cost_mid), float(util_mid)

    # Deterministic repair: use remaining budget to upgrade windows greedily by best gain/cost.
    remaining = float(b - cost_best)
    chosen = list(chosen_best)

    def _current_j(i: int) -> int | None:
        return chosen[i]

    upgrades: list[tuple[float, float, int, int]] = []
    for i in range(n):
        cur = _current_j(i)
        cur_cost = 0.0 if cur is None else float(costs[i][int(cur)])
        cur_util = 0.0 if cur is None else float(utils[i][int(cur)])
        best_ratio = float("-inf")
        best = None
        for j in range(m):
            if cur is not None and int(j) == int(cur):
                continue
            dc = float(costs[i][j] - cur_cost)
            du = float(utils[i][j] - cur_util)
            if dc <= 1e-9 or du <= 1e-12:
                continue
            ratio = du / dc
            if ratio > best_ratio + 1e-12 or (abs(ratio - best_ratio) <= 1e-12 and (best is None or int(j) < int(best))):
                best_ratio = float(ratio)
                best = int(j)
        if best is not None:
            upgrades.append((best_ratio, float(costs[i][best] - cur_cost), int(i), int(best)))

    upgrades.sort(key=lambda x: (-float(x[0]), float(x[1]), int(x[2]), int(x[3])))
    for _ratio, dc, i, j in upgrades:
        if float(dc) <= remaining + 1e-9:
            cur = _current_j(i)
            cur_cost = 0.0 if cur is None else float(costs[i][int(cur)])
            # Re-check because earlier upgrades may have changed state.
            new_cost = float(costs[i][int(j)])
            add = float(new_cost - cur_cost)
            if add <= remaining + 1e-9 and add > 1e-9:
                chosen[int(i)] = int(j)
                remaining -= float(add)

    # Build report.
    allocs: list[WindowAllocation] = []
    total = 0.0
    for i, w in enumerate(windows):
        j = chosen[int(i)]
        if j is None:
            allocs.append(WindowAllocation(window=w, weight=float(weights[i]), cfg=None, cost=0.0, dropped=True))
        else:
            cfg = cfgs[int(j)]
            c = float(token_cost_window(cfg=cfg, window=w, patch_size=int(patch_size)))
            total += float(c)
            allocs.append(WindowAllocation(window=w, weight=float(weights[i]), cfg=cfg, cost=float(c), dropped=False))
    ok = total <= b + 1e-6
    return AllocationReport(budget=float(b), patch_size=int(patch_size), allocations=allocs, total_cost=float(total), ok=bool(ok))
