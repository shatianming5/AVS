#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _bar_labels(vals: list[float], ns: list[int]) -> list[str]:
    out = []
    for v, n in zip(vals, ns):
        out.append(f"{v:+.3f}\n(n={n})")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Make the 'Why +2% is hard' C0003 decomposition figure.")
    p.add_argument("--energy-metrics", type=Path, required=True, help="Energy baseline metrics.json (has oracle/uniform).")
    p.add_argument("--best-metrics", type=Path, required=True, help="Best C0003 candidate metrics.json (anchored/uniform).")
    p.add_argument("--best-diagnose", type=Path, required=True, help="Diagnose JSON for the best candidate.")
    p.add_argument("--best-evidence-alignment", type=Path, required=False, default=None, help="Optional evidence_alignment.json.")
    p.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    p.add_argument("--title", type=str, default="C0003 Decomposition (Why +2% Is Hard)")
    p.add_argument("--best-label", type=str, default="Best", help="Label for the best deployable method in the bar chart.")
    args = p.parse_args()

    energy = _read_json(args.energy_metrics)
    best = _read_json(args.best_metrics)
    diag = _read_json(args.best_diagnose)
    ea = _read_json(args.best_evidence_alignment) if args.best_evidence_alignment else None

    # Energy oracle/uniform live in the nested format used by official verify runs.
    oracle = _as_float(energy.get("metrics", {}).get("summary", {}).get("oracle_top2", {}).get("mean"))
    uniform = _as_float(energy.get("metrics", {}).get("summary", {}).get("uniform", {}).get("mean"))

    anchored_best = _as_float(best.get("summary", {}).get("anchored_top2", {}).get("mean"))
    uniform_best = _as_float(best.get("summary", {}).get("uniform", {}).get("mean"))
    delta_best = anchored_best - uniform_best
    p_best = _as_float(best.get("paired_ttest", {}).get("anchored_vs_uniform", {}).get("p"))

    # Diagnose buckets.
    plan_stats = (diag.get("anchor_plan_stats") or {}) if isinstance(diag, dict) else {}
    high = plan_stats.get("delta_by_high_count") or {}
    dist = plan_stats.get("delta_by_anchor_dist") or {}
    fallback_frac = _as_float(plan_stats.get("anchors_len_fallback_frac"))

    high_keys = sorted((int(k) for k in high.keys()), key=int)
    high_vals = [_as_float(high[str(k)]["mean_delta"]) for k in high_keys]
    high_ns = [int(high[str(k)]["n"]) for k in high_keys]

    dist_keys = sorted((int(k) for k in dist.keys()), key=int)
    dist_vals = [_as_float(dist[str(k)]["mean_delta"]) for k in dist_keys]
    dist_ns = [int(dist[str(k)]["n"]) for k in dist_keys]

    # Evidence Alignment: just surface one representative correlation value.
    ea_text = None
    if isinstance(ea, dict):
        corr = (ea.get("corr_by_tau") or {}).get("0.300") or {}
        pearson = corr.get("pearson")
        spearman = corr.get("spearman")
        if pearson is not None and spearman is not None:
            ea_text = f"Evidence Alignment corr@tau=0.3: pearson={pearson:.3f}, spearman={spearman:.3f}"

    # Plot.
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(args.title)

    # (A) Uniform → Best → Oracle (absolute acc, with gap annotations)
    ax0 = fig.add_subplot(2, 2, 1)
    labels0 = ["Uniform", str(args.best_label), "Oracle"]
    vals0 = [uniform, anchored_best, oracle]
    ax0.bar(labels0, vals0, color=["#9aa0a6", "#1f77b4", "#2ca02c"])
    ax0.set_ylabel("Accuracy (test402)")
    ax0.set_ylim(min(vals0) - 0.02, max(vals0) + 0.02)
    ax0.set_title("Ceiling vs Best Deployable")
    ax0.text(
        0.02,
        0.02,
        f"Best Δ=+{delta_best:.3f} (p={p_best:.3g})\nOracle-Uniform=+{(oracle - uniform):.3f}\nOracle-Best=+{(oracle - anchored_best):.3f}",
        transform=ax0.transAxes,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#dddddd"},
    )

    # (B) High-count bucket deltas
    ax1 = fig.add_subplot(2, 2, 2)
    ax1.set_title("Mean Δ by High-Res Anchor Count")
    xs1 = [str(k) for k in high_keys]
    bars1 = ax1.bar(xs1, high_vals, color=["#2ca02c" if v > 0 else "#d62728" for v in high_vals])
    ax1.axhline(0.0, color="#444444", lw=1)
    ax1.set_xlabel("high_count")
    ax1.set_ylabel("anchored - uniform (mean)")
    for b, lab in zip(bars1, _bar_labels(high_vals, high_ns)):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height(), lab, ha="center", va="bottom", fontsize=8)

    # (C) Anchor-distance bucket deltas
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_title("Mean Δ by Anchor Distance (dist=|a1-a2| for 2-anchor clips)")
    xs2 = [str(k) for k in dist_keys]
    bars2 = ax2.bar(xs2, dist_vals, color=["#2ca02c" if v > 0 else "#d62728" for v in dist_vals])
    ax2.axhline(0.0, color="#444444", lw=1)
    ax2.set_xlabel("anchor_dist")
    ax2.set_ylabel("anchored - uniform (mean)")
    for b, lab in zip(bars2, _bar_labels(dist_vals, dist_ns)):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height(), lab, ha="center", va="bottom", fontsize=8)

    # Global annotations.
    footer = f"fallback_frac≈{fallback_frac:.3f} (fraction of clips falling back to uniform)"
    if ea_text:
        footer = f"{footer}\n{ea_text}"
    fig.text(0.01, 0.01, footer, ha="left", va="bottom", fontsize=9, color="#333333")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(args.out, bbox_inches="tight")
    print(str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
