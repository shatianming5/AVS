from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from avs.budget.vis_budget import VisualConfig, default_h, token_cost_window
from avs.metrics.time_windows import TimeWindow
from avs.sampling.allocator import allocate_budgeted_windows, allocate_budgeted_windows_knapsack_lagrangian


def _utility(weights: list[float], allocs) -> float:
    u = 0.0
    for w, a in zip(weights, allocs, strict=True):
        if a.cfg is None:
            continue
        u += float(w) * float(default_h(a.cfg))
    return float(u)


def _make_default_configs() -> list[VisualConfig]:
    # A small, monotonic ladder for ablation.
    return [
        VisualConfig(name="fps0.5_r224", fps=0.5, resolution=224, r_keep=1.0),
        VisualConfig(name="fps1_r224", fps=1.0, resolution=224, r_keep=1.0),
        VisualConfig(name="fps1_r336", fps=1.0, resolution=336, r_keep=1.0),
        VisualConfig(name="fps2_r336", fps=2.0, resolution=336, r_keep=1.0),
    ]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ablate Stage-2 allocator solvers (greedy vs Lagrangian MCKP).")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-windows", type=int, default=12)
    p.add_argument("--min-dur", type=float, default=1.0)
    p.add_argument("--max-dur", type=float, default=3.0)
    p.add_argument("--budget", type=float, default=20000.0)
    p.add_argument("--patch-size", type=int, default=16)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    n = int(args.num_windows)
    if n <= 0:
        raise SystemExit("--num-windows must be > 0")

    # Generate non-overlapping windows on a timeline.
    windows: list[TimeWindow] = []
    t = 0.0
    for _ in range(n):
        dur = float(rng.uniform(float(args.min_dur), float(args.max_dur)))
        windows.append(TimeWindow(start_s=float(t), end_s=float(t + dur)))
        t += dur

    weights = [float(x) for x in rng.uniform(0.0, 1.0, size=n).tolist()]
    cfgs = _make_default_configs()
    budget = float(args.budget)
    patch_size = int(args.patch_size)

    # Run solvers.
    t0 = time.time()
    rep_greedy = allocate_budgeted_windows(windows=windows, weights=weights, configs=cfgs, budget=budget, patch_size=patch_size, drop_if_needed=True)
    t1 = time.time()
    rep_lagr = allocate_budgeted_windows_knapsack_lagrangian(windows=windows, weights=weights, configs=cfgs, budget=budget, patch_size=patch_size, include_drop=True)
    t2 = time.time()

    # Utility/cost.
    u_g = _utility(weights, rep_greedy.allocations)
    u_l = _utility(weights, rep_lagr.allocations)

    payload = {
        "ok": True,
        "seed": int(args.seed),
        "num_windows": int(n),
        "budget": float(budget),
        "patch_size": int(patch_size),
        "configs": [{"name": c.name, "fps": c.fps, "resolution": c.resolution, "r_keep": c.r_keep} for c in cfgs],
        "windows": [{"start_s": w.start_s, "end_s": w.end_s, "duration_s": float(w.end_s - w.start_s)} for w in windows],
        "weights": [float(x) for x in weights],
        "solvers": {
            "greedy": {
                "report": rep_greedy.to_jsonable(),
                "utility": float(u_g),
                "elapsed_s": float(t1 - t0),
            },
            "lagrangian": {
                "report": rep_lagr.to_jsonable(),
                "utility": float(u_l),
                "elapsed_s": float(t2 - t1),
            },
        },
        "delta": {"utility": float(u_l - u_g), "cost": float(rep_lagr.total_cost - rep_greedy.total_cost)},
    }

    out_json = out_dir / "allocator_ablation.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

