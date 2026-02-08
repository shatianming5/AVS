from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_pareto_report(*, points: list[dict], out_png: Path, title: str = "Acc–Tok Pareto") -> Path:
    """
    Plot an accuracy-vs-token scatter with error bars.

    `points` schema (minimum):
      - method: str
      - token_budget: number
      - acc_mean: number
      - acc_std: number (optional; defaults to 0)
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Group by token budget so multiple methods at same budget are visible.
    by_tokens: dict[float, list[dict]] = {}
    for p in points:
        by_tokens.setdefault(float(p["token_budget"]), []).append(p)

    xs: list[float] = []
    ys: list[float] = []
    yerrs: list[float] = []
    labels: list[str] = []
    colors: list[str] = []

    palette = {
        "uniform": "#1f77b4",
        "random": "#7f7f7f",
        "predicted": "#d62728",
        "cheap_visual": "#ff7f0e",
        "anchored": "#d62728",
        "oracle": "#2ca02c",
    }

    for tokens in sorted(by_tokens.keys()):
        group = sorted(by_tokens[tokens], key=lambda d: str(d.get("method", "")))
        n = len(group)
        if n == 1:
            offsets = [0.0]
        else:
            span = 0.03 * float(tokens)
            offsets = np.linspace(-span, span, n).tolist()
        for off, p in zip(offsets, group, strict=True):
            xs.append(float(tokens) + float(off))
            ys.append(float(p.get("acc_mean", 0.0)))
            yerrs.append(float(p.get("acc_std", 0.0)))
            m = str(p.get("method", "method"))
            labels.append(m)
            colors.append(palette.get(m, "#000000"))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(xs, ys, yerr=yerrs, fmt="o", capsize=3, linewidth=1.2, color="black", ecolor="black")

    for x, y, lab, c in zip(xs, ys, labels, colors, strict=True):
        ax.scatter([x], [y], color=c)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    ax.set_xlabel("token budget")
    ax.set_ylabel("accuracy (mean)")
    ax.set_title(str(title))
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png
