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
