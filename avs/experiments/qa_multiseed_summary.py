from __future__ import annotations

import argparse
import glob
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SeedMethod:
    seed: int
    method: str
    acc: float
    n: int


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
    # stable de-dup
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
    p = argparse.ArgumentParser(description="Aggregate QA eval metrics across seeds (possibly split across runs).")
    p.add_argument("--task", type=str, default="", help="Optional task label for reporting only.")
    p.add_argument("--run-metrics", type=str, action="append", default=[], help="Path to metrics.json (repeatable).")
    p.add_argument("--run-glob", type=str, action="append", default=[], help="Glob for metrics.json files.")
    p.add_argument("--methods", type=str, default="", help="Comma-separated methods to report; default=all found.")
    p.add_argument("--uniform-method", type=str, default="uniform")
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--bootstrap-seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    paths = _collect_metrics_paths(args.run_metrics, args.run_glob)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Merge by seed (a seed may be spread across multiple runs, e.g., text_only separated).
    by_seed_method: dict[int, dict[str, tuple[float, int]]] = {}
    seed_sources: dict[int, list[str]] = {}
    all_methods: set[str] = set()
    n_items_by_seed: dict[int, int] = {}

    for mp in paths:
        obj = json.loads(mp.read_text(encoding="utf-8"))
        if "seed" not in obj:
            raise ValueError(f"metrics missing seed: {mp}")
        seed = int(obj["seed"])
        meth = _parse_summary_method_acc(obj)
        all_methods.update(meth.keys())
        by_seed_method.setdefault(seed, {})
        for m, (acc, n) in meth.items():
            if m in by_seed_method[seed]:
                old_acc, old_n = by_seed_method[seed][m]
                if abs(old_acc - acc) > 1e-12 or old_n != n:
                    raise ValueError(
                        f"conflicting method metrics for seed={seed} method={m}: "
                        f"{old_acc}/{old_n} vs {acc}/{n} ({mp})"
                    )
            by_seed_method[seed][m] = (acc, n)
        n_items = int(obj.get("n_items", 0))
        if n_items > 0:
            if seed in n_items_by_seed and n_items_by_seed[seed] != n_items:
                # keep max to stay conservative if partial runs had different filters.
                n_items_by_seed[seed] = max(n_items_by_seed[seed], n_items)
            else:
                n_items_by_seed[seed] = n_items
        seed_sources.setdefault(seed, []).append(str(mp))

    methods = sorted(all_methods)
    if str(args.methods).strip():
        methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]

    uniform_method = str(args.uniform_method)
    if uniform_method not in methods:
        methods = [uniform_method] + methods
        methods = [m for i, m in enumerate(methods) if m and m not in methods[:i]]

    per_method: dict[str, Any] = {}
    for m in methods:
        vals: list[float] = []
        seeds: list[int] = []
        ns: list[int] = []
        for s in sorted(by_seed_method.keys()):
            rec = by_seed_method[s].get(m)
            if rec is None:
                continue
            acc, n = rec
            vals.append(float(acc))
            seeds.append(int(s))
            ns.append(int(n))
        arr = np.asarray(vals, dtype=np.float64)
        ci = _bootstrap_ci_mean(
            arr,
            seed=int(args.bootstrap_seed) + _stable_offset(f"acc:{m}", mod=9973),
            n_boot=int(args.n_boot),
            alpha=0.05,
        )
        per_method[m] = {
            "n_seeds": int(arr.size),
            "seeds": seeds,
            "acc_values": vals,
            "n_values": ns,
            "acc_mean": float(arr.mean()) if arr.size else 0.0,
            "acc_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "acc_ci95": {"lo": float(ci["lo"]), "hi": float(ci["hi"])},
        }

    # Paired deltas vs uniform over common seeds.
    deltas_vs_uniform: dict[str, Any] = {}
    for m in methods:
        if m == uniform_method:
            continue
        diffs: list[float] = []
        seeds: list[int] = []
        for s in sorted(by_seed_method.keys()):
            if uniform_method not in by_seed_method[s] or m not in by_seed_method[s]:
                continue
            acc_u = by_seed_method[s][uniform_method][0]
            acc_m = by_seed_method[s][m][0]
            diffs.append(float(acc_m - acc_u))
            seeds.append(int(s))
        arr = np.asarray(diffs, dtype=np.float64)
        ci = _bootstrap_ci_mean(
            arr,
            seed=int(args.bootstrap_seed) + _stable_offset(f"delta:{m}", mod=9973),
            n_boot=int(args.n_boot),
            alpha=0.05,
        )
        if arr.size:
            rng = np.random.default_rng(int(args.bootstrap_seed) + _stable_offset(f"p:{m}", mod=7919))
            means = []
            for _ in range(int(args.n_boot)):
                idx = rng.integers(0, int(arr.size), size=int(arr.size))
                means.append(float(arr[idx].mean()))
            means_arr = np.asarray(means, dtype=np.float64)
            p_two = float(2.0 * min(np.mean(means_arr <= 0.0), np.mean(means_arr >= 0.0)))
        else:
            p_two = 1.0
        deltas_vs_uniform[m] = {
            "n_seeds": int(arr.size),
            "seeds": seeds,
            "delta_values": diffs,
            "delta_mean": float(arr.mean()) if arr.size else 0.0,
            "delta_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "delta_ci95": {"lo": float(ci["lo"]), "hi": float(ci["hi"])},
            "p_bootstrap_two_sided": float(p_two),
        }

    coverage = {
        str(seed): {
            "methods_present": sorted(by_seed_method[int(seed)].keys()),
            "methods_missing": sorted([m for m in methods if m not in by_seed_method[int(seed)]]),
            "n_items": int(n_items_by_seed.get(int(seed), 0)),
            "sources": seed_sources.get(int(seed), []),
        }
        for seed in sorted(by_seed_method.keys())
    }

    payload = {
        "ok": True,
        "task": str(args.task),
        "uniform_method": uniform_method,
        "methods": methods,
        "n_boot": int(args.n_boot),
        "bootstrap_seed": int(args.bootstrap_seed),
        "run_metrics": [str(p) for p in paths],
        "seeds": sorted([int(s) for s in by_seed_method.keys()]),
        "per_method": per_method,
        "delta_vs_uniform": deltas_vs_uniform,
        "coverage": coverage,
        "artifacts": {
            "summary_json": str(out_dir / "metrics_summary.json"),
            "summary_md": str(out_dir / "metrics_summary.md"),
        },
    }

    # Short markdown.
    md_lines = [
        "# QA Multi-Seed Summary",
        "",
        f"- task: `{str(args.task)}`",
        f"- seeds: `{', '.join(str(s) for s in payload['seeds'])}`",
        f"- uniform_method: `{uniform_method}`",
        "",
        "## Accuracy By Method",
        "",
        "| method | n_seeds | acc_mean | acc_std | ci95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in methods:
        pm = per_method.get(m, {})
        ci = pm.get("acc_ci95", {"lo": 0.0, "hi": 0.0})
        md_lines.append(
            f"| `{m}` | {pm.get('n_seeds', 0)} | "
            f"{pm.get('acc_mean', 0.0):.4f} | {pm.get('acc_std', 0.0):.4f} | "
            f"[{ci.get('lo', 0.0):.4f}, {ci.get('hi', 0.0):.4f}] |"
        )
    md_lines += [
        "",
        "## Delta Vs Uniform",
        "",
        "| method | n_seeds | delta_mean | delta_std | ci95 | p_boot(two-sided) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in methods:
        if m == uniform_method:
            continue
        d = deltas_vs_uniform.get(m, {})
        ci = d.get("delta_ci95", {"lo": 0.0, "hi": 0.0})
        md_lines.append(
            f"| `{m}` | {d.get('n_seeds', 0)} | "
            f"{d.get('delta_mean', 0.0):+.4f} | {d.get('delta_std', 0.0):.4f} | "
            f"[{ci.get('lo', 0.0):+.4f}, {ci.get('hi', 0.0):+.4f}] | "
            f"{d.get('p_bootstrap_two_sided', 1.0):.4f} |"
        )
    md_lines.append("")

    (out_dir / "metrics_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "metrics_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(out_dir / "metrics_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
