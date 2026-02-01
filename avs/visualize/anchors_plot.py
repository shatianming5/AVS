from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_anchor_plan(
    *,
    scores: list[float],
    anchors: list[int],
    gt_segments: list[int] | None,
    resolutions: list[int] | None,
    out_path: Path,
    title: str = "AVS: anchors vs GT vs sampling plan",
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t = list(range(len(scores)))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, scores, marker="o", linewidth=1.5, label="s(t)")

    if anchors:
        ax.scatter(anchors, [scores[i] for i in anchors], color="red", zorder=3, label="anchors")

    if gt_segments:
        for seg in gt_segments:
            ax.axvspan(seg - 0.5, seg + 0.5, color="green", alpha=0.15)
        ax.plot([], [], color="green", alpha=0.15, linewidth=8, label="GT segments")

    ax.set_xlabel("second (t)")
    ax.set_ylabel("eventness")
    ax.set_xticks(t)
    ax.set_title(title)

    if resolutions:
        ax2 = ax.twinx()
        ax2.step(t, resolutions, where="mid", color="black", alpha=0.6, label="resolution")
        ax2.set_ylabel("resolution")
        ax2.set_ylim(0, max(resolutions) * 1.2)

    lines, labels = ax.get_legend_handles_labels()
    if resolutions:
        l2, lb2 = ax2.get_legend_handles_labels()
        lines += l2
        labels += lb2
    ax.legend(lines, labels, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot s(t), anchors, GT segments, and sampling plan resolutions.")
    p.add_argument("--in-json", type=Path, required=True, help="JSON with scores, anchors, gt_segments, resolutions")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", type=str, default="AVS: anchors vs GT vs sampling plan")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    obj = json.loads(args.in_json.read_text(encoding="utf-8"))
    plot_anchor_plan(
        scores=[float(x) for x in obj["scores"]],
        anchors=[int(x) for x in obj.get("anchors", [])],
        gt_segments=[int(x) for x in obj.get("gt_segments", [])] or None,
        resolutions=[int(x) for x in obj.get("resolutions", [])] or None,
        out_path=args.out,
        title=args.title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

