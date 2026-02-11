from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _bootstrap_ci_mean(xs: np.ndarray, *, seed: int, n_boot: int = 5000, alpha: float = 0.05) -> dict[str, float | int]:
    rng = np.random.default_rng(int(seed))
    n = int(xs.size)
    if n <= 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0, "n_boot": int(n_boot)}
    means = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        means.append(float(xs[idx].mean()))
    arr = np.asarray(means, dtype=np.float64)
    lo = float(np.quantile(arr, float(alpha / 2.0)))
    hi = float(np.quantile(arr, float(1.0 - alpha / 2.0)))
    return {"mean": float(xs.mean()), "lo": lo, "hi": hi, "n": n, "n_boot": int(n_boot)}


def _stable_offset(s: str, mod: int = 10000) -> int:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h[:12], 16) % int(mod)


def _collect_metrics_paths(run_metrics: list[str], run_globs: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in run_metrics:
        pp = Path(p)
        if not pp.exists():
            raise FileNotFoundError(pp)
        out.append(pp)
    for pat in run_globs:
        for s in sorted(glob.glob(pat)):
            pp = Path(s)
            if pp.exists():
                out.append(pp)
    uniq: list[Path] = []
    seen: set[str] = set()
    for p in out:
        sp = str(p.resolve())
        if sp in seen:
            continue
        seen.add(sp)
        uniq.append(p)
    if not uniq:
        raise ValueError("no metrics paths found")
    return uniq


def _parse_summary_method_acc(metrics_obj: dict[str, Any]) -> dict[str, tuple[float, int]]:
    sm = metrics_obj.get("summary")
    if not isinstance(sm, list):
        raise ValueError("metrics.json missing list field: summary")
    out: dict[str, tuple[float, int]] = {}
    for row in sm:
        if not isinstance(row, dict):
            continue
        m = str(row.get("method", ""))
        if not m:
            continue
        acc = row.get("acc")
        if acc is None:
            continue
        out[m] = (float(acc), int(row.get("n", 0)))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Build QA budget curve (accuracy vs budget_frames) from metrics.json files.")
    p.add_argument("--task", type=str, default="")
    p.add_argument("--run-metrics", type=str, action="append", default=[])
    p.add_argument("--run-glob", type=str, action="append", default=[])
    p.add_argument("--methods", type=str, default="")
    p.add_argument("--uniform-method", type=str, default="uniform")
    p.add_argument("--n-boot", type=int, default=3000)
    p.add_argument("--bootstrap-seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    paths = _collect_metrics_paths(args.run_metrics, args.run_glob)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Merge by (budget, seed): this allows methods to come from different runs.
    by_budget_seed_method: dict[tuple[int, int], dict[str, tuple[float, int]]] = {}
    by_budget_seed_sources: dict[tuple[int, int], list[str]] = {}
    all_methods: set[str] = set()
    budgets: set[int] = set()

    for mp in paths:
        obj = json.loads(mp.read_text(encoding="utf-8"))
        if "seed" not in obj or "budget_frames" not in obj:
            raise ValueError(f"metrics missing seed/budget_frames: {mp}")
        seed = int(obj["seed"])
        budget = int(obj["budget_frames"])
        key = (budget, seed)
        by_budget_seed_method.setdefault(key, {})
        meth = _parse_summary_method_acc(obj)
        all_methods.update(meth.keys())
        for m, (acc, n) in meth.items():
            if m in by_budget_seed_method[key]:
                old_acc, old_n = by_budget_seed_method[key][m]
                if abs(old_acc - acc) > 1e-12 or old_n != n:
                    raise ValueError(
                        f"conflicting metrics for budget={budget} seed={seed} method={m}: "
                        f"{old_acc}/{old_n} vs {acc}/{n} ({mp})"
                    )
            by_budget_seed_method[key][m] = (acc, n)
        by_budget_seed_sources.setdefault(key, []).append(str(mp))
        budgets.add(budget)

    methods = sorted(all_methods)
    if str(args.methods).strip():
        methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]

    uniform_method = str(args.uniform_method)
    if uniform_method not in methods:
        methods = [uniform_method] + methods
        methods = [m for i, m in enumerate(methods) if m and m not in methods[:i]]

    budget_rows: list[dict[str, Any]] = []
    for budget in sorted(budgets):
        row: dict[str, Any] = {"budget_frames": int(budget), "methods": {}}
        seeds_here = sorted([seed for (b, seed) in by_budget_seed_method.keys() if b == budget])
        row["seeds"] = seeds_here
        for m in methods:
            vals: list[float] = []
            ns: list[int] = []
            seeds: list[int] = []
            for seed in seeds_here:
                rec = by_budget_seed_method.get((budget, seed), {}).get(m)
                if rec is None:
                    continue
                acc, n = rec
                vals.append(float(acc))
                ns.append(int(n))
                seeds.append(int(seed))
            arr = np.asarray(vals, dtype=np.float64)
            ci = _bootstrap_ci_mean(
                arr,
                seed=int(args.bootstrap_seed) + (budget * 1009) + _stable_offset(f"acc:{m}", mod=997),
                n_boot=int(args.n_boot),
                alpha=0.05,
            )
            row["methods"][m] = {
                "n_seeds": int(arr.size),
                "seeds": seeds,
                "acc_values": vals,
                "n_values": ns,
                "acc_mean": float(arr.mean()) if arr.size else 0.0,
                "acc_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "acc_ci95": {"lo": float(ci["lo"]), "hi": float(ci["hi"])},
            }
        # delta vs uniform at this budget.
        dvu: dict[str, Any] = {}
        for m in methods:
            if m == uniform_method:
                continue
            diffs: list[float] = []
            seeds: list[int] = []
            for seed in seeds_here:
                ru = by_budget_seed_method.get((budget, seed), {}).get(uniform_method)
                rm = by_budget_seed_method.get((budget, seed), {}).get(m)
                if ru is None or rm is None:
                    continue
                diffs.append(float(rm[0] - ru[0]))
                seeds.append(int(seed))
            arr = np.asarray(diffs, dtype=np.float64)
            ci = _bootstrap_ci_mean(
                arr,
                seed=int(args.bootstrap_seed) + (budget * 1013) + _stable_offset(f"delta:{m}", mod=101),
                n_boot=int(args.n_boot),
                alpha=0.05,
            )
            dvu[m] = {
                "n_seeds": int(arr.size),
                "seeds": seeds,
                "delta_values": diffs,
                "delta_mean": float(arr.mean()) if arr.size else 0.0,
                "delta_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "delta_ci95": {"lo": float(ci["lo"]), "hi": float(ci["hi"])},
            }
        row["delta_vs_uniform"] = dvu
        budget_rows.append(row)

    payload = {
        "ok": True,
        "task": str(args.task),
        "uniform_method": uniform_method,
        "methods": methods,
        "budgets": sorted([int(b) for b in budgets]),
        "n_boot": int(args.n_boot),
        "bootstrap_seed": int(args.bootstrap_seed),
        "run_metrics": [str(p) for p in paths],
        "rows": budget_rows,
        "artifacts": {
            "budget_curve_json": str(out_dir / "budget_curve.json"),
            "budget_curve_md": str(out_dir / "budget_curve.md"),
            "budget_curve_png": str(out_dir / "budget_curve.png"),
        },
    }

    # Markdown table.
    md_lines = [
        "# QA Budget Curve",
        "",
        f"- task: `{str(args.task)}`",
        f"- budgets: `{', '.join(str(b) for b in payload['budgets'])}`",
        f"- methods: `{', '.join(methods)}`",
        "",
        "## Accuracy",
        "",
        "| budget_frames | method | n_seeds | acc_mean | ci95 |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in budget_rows:
        b = int(row["budget_frames"])
        for m in methods:
            r = row["methods"].get(m, {})
            ci = r.get("acc_ci95", {"lo": 0.0, "hi": 0.0})
            md_lines.append(
                f"| {b} | `{m}` | {r.get('n_seeds', 0)} | {r.get('acc_mean', 0.0):.4f} | "
                f"[{ci.get('lo', 0.0):.4f}, {ci.get('hi', 0.0):.4f}] |"
            )
    md_lines += [
        "",
        "## Delta Vs Uniform",
        "",
        "| budget_frames | method | n_seeds | delta_mean | ci95 |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in budget_rows:
        b = int(row["budget_frames"])
        for m in methods:
            if m == uniform_method:
                continue
            r = row["delta_vs_uniform"].get(m, {})
            ci = r.get("delta_ci95", {"lo": 0.0, "hi": 0.0})
            md_lines.append(
                f"| {b} | `{m}` | {r.get('n_seeds', 0)} | {r.get('delta_mean', 0.0):+.4f} | "
                f"[{ci.get('lo', 0.0):+.4f}, {ci.get('hi', 0.0):+.4f}] |"
            )
    md_lines.append("")

    (out_dir / "budget_curve.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "budget_curve.md").write_text("\n".join(md_lines), encoding="utf-8")

    # Plot (best-effort).
    try:
        import matplotlib.pyplot as plt

        xs = sorted([int(b) for b in budgets])
        fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
        for m in methods:
            ys: list[float] = []
            yerr_lo: list[float] = []
            yerr_hi: list[float] = []
            for b in xs:
                row = next(r for r in budget_rows if int(r["budget_frames"]) == b)
                r = row["methods"].get(m, {})
                y = float(r.get("acc_mean", 0.0))
                ci = r.get("acc_ci95", {"lo": y, "hi": y})
                ys.append(y)
                yerr_lo.append(max(0.0, y - float(ci.get("lo", y))))
                yerr_hi.append(max(0.0, float(ci.get("hi", y)) - y))
            ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], marker="o", linewidth=1.8, capsize=3, label=m)
        ax.set_xlabel("B_FRAMES")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"QA Budget Curve ({str(args.task) or 'task'})")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "budget_curve.png")
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        # Keep json/md artifacts valid even if matplotlib is unavailable.
        (out_dir / "budget_curve.png").write_text(f"plot_failed: {repr(e)}\n", encoding="utf-8")

    print(out_dir / "budget_curve.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
