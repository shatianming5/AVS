from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _extract_metrics(obj: dict) -> dict:
    # Accept either raw `ave_p0.py` metrics.json or the nested `ave_p0_end2end/metrics.json`.
    if isinstance(obj.get("metrics"), dict):
        return obj["metrics"]
    return obj


def _points_from_p0_metrics(metrics: dict) -> list[dict]:
    baselines = metrics.get("baselines") or []
    summary = metrics.get("summary") or {}
    results_by_seed = metrics.get("results_by_seed") or []

    tokens_by_baseline: dict[str, int] = {}
    if results_by_seed:
        seed0 = results_by_seed[0].get("baselines") or {}
        for b in baselines:
            if b in seed0 and isinstance(seed0[b], dict) and "token_budget" in seed0[b]:
                tokens_by_baseline[str(b)] = int(seed0[b]["token_budget"])

    points: list[dict] = []
    for b in baselines:
        b = str(b)
        if b not in summary:
            continue
        tokens = tokens_by_baseline.get(b, int(metrics.get("token_budget", 0)))
        row = summary[b]
        points.append(
            {
                "baseline": b,
                "token_budget": int(tokens),
                "acc_mean": float(row.get("mean", 0.0)),
                "acc_std": float(row.get("std", 0.0)),
            }
        )
    return points


def plot_efficiency_curve(*, points: list[dict], out_png: Path, title: str = "AVE-P0: Accuracy vs Token Budget") -> Path:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Jitter baselines that share identical token budgets so they are visible on a scatter plot.
    by_tokens: dict[int, list[dict]] = {}
    for p in points:
        by_tokens.setdefault(int(p["token_budget"]), []).append(p)

    xs: list[float] = []
    ys: list[float] = []
    yerrs: list[float] = []
    labels: list[str] = []

    for tokens in sorted(by_tokens.keys()):
        group = sorted(by_tokens[tokens], key=lambda d: d["baseline"])
        n = len(group)
        if n == 1:
            offsets = [0.0]
        else:
            span = 0.03 * float(tokens)  # ±3%
            offsets = np.linspace(-span, span, n).tolist()
        for off, p in zip(offsets, group, strict=True):
            xs.append(float(tokens) + float(off))
            ys.append(float(p["acc_mean"]))
            yerrs.append(float(p["acc_std"]))
            labels.append(str(p["baseline"]))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(xs, ys, yerr=yerrs, fmt="o", capsize=3, linewidth=1.2)

    for x, y, lab in zip(xs, ys, labels, strict=True):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    ax.set_xlabel("token budget (patch tokens; per clip)")
    ax.set_ylabel("segment accuracy (mean)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot Accuracy–Token efficiency curve from AVE-P0 metrics.json.")
    p.add_argument("--in-metrics", type=Path, required=True, help="Path to ave_p0.py metrics.json (or ave_p0_end2end/metrics.json)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--title", type=str, default="AVE-P0: Accuracy vs Token Budget")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    obj = json.loads(args.in_metrics.read_text(encoding="utf-8"))
    metrics = _extract_metrics(obj)
    points = _points_from_p0_metrics(metrics)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "efficiency_curve.json"
    out_png = args.out_dir / "efficiency_curve.png"

    payload = {"in_metrics": str(args.in_metrics), "points": points}
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot_efficiency_curve(points=points, out_png=out_png, title=str(args.title))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

