from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected dict json: {path}")
    return obj


def _summary_map(metrics_obj: dict[str, Any]) -> dict[str, float]:
    sm = metrics_obj.get("summary")
    out: dict[str, float] = {}
    if isinstance(sm, list):
        for row in sm:
            if not isinstance(row, dict):
                continue
            m = str(row.get("method", ""))
            a = row.get("acc")
            if m and a is not None:
                out[m] = float(a)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Compare AVQA metrics before/after coverage expansion.")
    p.add_argument("--baseline-metrics", type=Path, required=True)
    p.add_argument("--expanded-metrics", type=Path, required=True)
    p.add_argument("--download-json", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    base = _load_json(Path(args.baseline_metrics))
    exp = _load_json(Path(args.expanded_metrics))
    dl = _load_json(Path(args.download_json)) if args.download_json else None

    base_map = _summary_map(base)
    exp_map = _summary_map(exp)
    methods = sorted(set(base_map.keys()) | set(exp_map.keys()))
    deltas = {m: float(exp_map.get(m, 0.0) - base_map.get(m, 0.0)) for m in methods}

    payload = {
        "ok": True,
        "baseline_metrics": str(args.baseline_metrics),
        "expanded_metrics": str(args.expanded_metrics),
        "download_json": str(args.download_json) if args.download_json else None,
        "coverage": {
            "baseline_n_items": int(base.get("n_items", 0)),
            "expanded_n_items": int(exp.get("n_items", 0)),
            "baseline_skipped_videos": int(len(base.get("skipped_videos", []) or [])),
            "expanded_skipped_videos": int(len(exp.get("skipped_videos", []) or [])),
        },
        "download_stats": dl,
        "acc_baseline": base_map,
        "acc_expanded": exp_map,
        "delta_expanded_minus_baseline": deltas,
        "artifacts": {
            "coverage_sensitivity_json": str(Path(args.out_dir) / "coverage_sensitivity.json"),
            "coverage_sensitivity_md": str(Path(args.out_dir) / "coverage_sensitivity.md"),
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = [
        "# QA Coverage Sensitivity",
        "",
        f"- baseline_n_items: `{payload['coverage']['baseline_n_items']}`",
        f"- expanded_n_items: `{payload['coverage']['expanded_n_items']}`",
        f"- baseline_skipped_videos: `{payload['coverage']['baseline_skipped_videos']}`",
        f"- expanded_skipped_videos: `{payload['coverage']['expanded_skipped_videos']}`",
        "",
        "| method | baseline_acc | expanded_acc | delta |",
        "|---|---:|---:|---:|",
    ]
    for m in methods:
        md.append(
            f"| `{m}` | {base_map.get(m, 0.0):.4f} | {exp_map.get(m, 0.0):.4f} | {deltas.get(m, 0.0):+.4f} |"
        )
    md.append("")

    (out_dir / "coverage_sensitivity.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "coverage_sensitivity.md").write_text("\n".join(md), encoding="utf-8")
    print(out_dir / "coverage_sensitivity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

