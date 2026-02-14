#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _summary_by_method(m: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in (m.get("summary") or []):
        method = row.get("method")
        if not method:
            continue
        out[str(method)] = dict(row)
    return out


def _delta_ci(m: dict, method: str) -> tuple[float | None, float | None, float | None]:
    d = (m.get("delta_vs_uniform") or {}).get(method) or {}
    mean = d.get("mean")
    lo = d.get("lo")
    hi = d.get("hi")
    try:
        mean = float(mean) if mean is not None else None
    except Exception:
        mean = None
    try:
        lo = float(lo) if lo is not None else None
    except Exception:
        lo = None
    try:
        hi = float(hi) if hi is not None else None
    except Exception:
        hi = None
    return mean, lo, hi


def main() -> int:
    p = argparse.ArgumentParser(description="Make the Video-MME priors-controls / controlled-transfer slide figure.")
    p.add_argument("--metrics-json", type=Path, required=True, help="Video-MME metrics.json (E1101 output).")
    p.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    p.add_argument("--title", type=str, default="Video-MME Controlled Transfer (Priors Controls)")
    args = p.parse_args()

    m = _read_json(args.metrics_json)
    summ = _summary_by_method(m)
    if "uniform" not in summ:
        raise SystemExit("metrics.json missing summary['uniform']")
    uniform_acc = float(summ["uniform"]["acc"])

    # Fixed canonical order. (Some controls are intentionally not equal-budget.)
    groups: list[tuple[str, list[str]]] = [
        ("priors_controls", ["text_only", "random_frame1"]),
        ("equal_budget_controls", ["random", "uniform"]),
        ("selectors", ["audio", "fused", "ql2l_clip", "qframe_gumbel_clip", "maxinfo_maxvol_clip", "mdp3_dpp_clip"]),
    ]
    methods: list[tuple[str, str]] = []
    for g, ms in groups:
        for meth in ms:
            if meth in summ:
                methods.append((g, meth))

    # Display labels.
    label = {
        "text_only": "text_only\n(0 frame)",
        "random_frame1": "random_frame1\n(1 frame)",
        "random": "random\n(16 frames)",
        "uniform": "uniform\n(16 frames)",
        "audio": "audio\n(16 frames)",
        "fused": "fused\n(16 frames)",
        "ql2l_clip": "ql2l_clip\n(16 frames)",
        "qframe_gumbel_clip": "Q-Frame (gumbel)\n(16 frames)",
        "maxinfo_maxvol_clip": "MaxInfo (MaxVol)\n(16 frames)",
        "mdp3_dpp_clip": "DPP (MAP)\n(16 frames)",
    }

    xs = list(range(len(methods)))
    ys = [float(summ[meth]["acc"]) for _, meth in methods]

    # Convert delta bootstrap CIs into accuracy CIs by anchoring at uniform.
    yerr_lo: list[float] = []
    yerr_hi: list[float] = []
    for _, meth in methods:
        if meth == "uniform":
            yerr_lo.append(0.0)
            yerr_hi.append(0.0)
            continue
        mean, lo, hi = _delta_ci(m, meth)
        if lo is None or hi is None:
            yerr_lo.append(0.0)
            yerr_hi.append(0.0)
            continue
        acc_lo = uniform_acc + lo
        acc_hi = uniform_acc + hi
        acc = float(summ[meth]["acc"])
        yerr_lo.append(max(0.0, acc - acc_lo))
        yerr_hi.append(max(0.0, acc_hi - acc))

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

    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(args.title)

    colors = {
        "priors_controls": "#d62728",
        "equal_budget_controls": "#9aa0a6",
        "selectors": "#1f77b4",
    }

    bars = []
    for i, (g, meth) in enumerate(methods):
        c = colors.get(g, "#1f77b4")
        hatch = "//" if g == "priors_controls" else None
        b = ax.bar(
            [xs[i]],
            [ys[i]],
            yerr=[[yerr_lo[i]], [yerr_hi[i]]],
            capsize=3,
            color=c,
            edgecolor="#333333",
            linewidth=0.7,
            hatch=hatch,
            label=g if i == 0 or (i > 0 and methods[i - 1][0] != g) else None,
        )
        bars.append(b[0])

        # Annotate with acc and delta.
        mean, _, _ = _delta_ci(m, meth)
        delta_txt = ""
        if meth != "uniform" and mean is not None:
            delta_txt = f"\n({mean:+.3f})"
        ax.text(xs[i], ys[i] + 0.01, f"{ys[i]:.3f}{delta_txt}", ha="center", va="bottom", fontsize=8)

    # Uniform reference line.
    ax.axhline(uniform_acc, color="#444444", lw=1, linestyle="--")
    ax.text(0.01, uniform_acc + 0.005, f"uniform={uniform_acc:.3f}", transform=ax.get_yaxis_transform(), fontsize=9)

    ax.set_xticks(xs)
    ax.set_xticklabels([label.get(meth, meth) for _, meth in methods])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, max(ys) + 0.10)

    # Footer protocol string (keep short; full protocol lives in docs/oral_narrative.md).
    n = int(summ["uniform"].get("n") or 0)
    max_seconds = m.get("max_seconds")
    budget_frames = m.get("budget_frames")
    proto = f"split={m.get('split','?')}, n={n}, MAX_SECONDS={max_seconds}, B_FRAMES={budget_frames}, base=Qwen2-VL-2B"
    fig.text(0.01, 0.01, proto, ha="left", va="bottom", fontsize=9, color="#333333")

    ax.legend(loc="lower right", frameon=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig(args.out, bbox_inches="tight")
    print(str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

