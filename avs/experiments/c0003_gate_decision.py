from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected dict json: {path}")
    return obj


def _extract_delta_and_p(metrics_obj: dict[str, Any]) -> tuple[float, float]:
    sm = metrics_obj.get("summary") or {}
    pt = metrics_obj.get("paired_ttest") or {}
    a = float((sm.get("anchored_top2") or {}).get("mean", 0.0))
    u = float((sm.get("uniform") or {}).get("mean", 0.0))
    p = float((pt.get("anchored_vs_uniform") or {}).get("p", 1.0))
    return float(a - u), float(p)


def _extract_val_best(val_obj: dict[str, Any]) -> tuple[float, float]:
    best = val_obj.get("best") or {}
    d = float(best.get("anchored_minus_uniform_mean", 0.0))
    p = float(best.get("anchored_vs_uniform_p", 1.0))
    return d, p


def main() -> int:
    p = argparse.ArgumentParser(description="D0701 gate decision helper for C0003.")
    p.add_argument("--full-metrics", type=Path, required=True, help="Full test402 metrics.json")
    p.add_argument("--quick-metrics", type=Path, default=None, help="Quick test402 metrics.json")
    p.add_argument("--val-summary", type=Path, default=None, help="val402 sweep_summary.json")
    p.add_argument("--target-delta", type=float, default=0.02)
    p.add_argument("--target-p", type=float, default=0.05)
    p.add_argument("--quick-min-delta", type=float, default=0.01)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_obj = _read_json(Path(args.full_metrics))
    full_delta, full_p = _extract_delta_and_p(full_obj)

    quick_delta = None
    quick_p = None
    if args.quick_metrics is not None:
        quick_obj = _read_json(Path(args.quick_metrics))
        quick_delta, quick_p = _extract_delta_and_p(quick_obj)

    val_delta = None
    val_p = None
    if args.val_summary is not None:
        val_obj = _read_json(Path(args.val_summary))
        val_delta, val_p = _extract_val_best(val_obj)

    gate_quick_pass = True
    if quick_delta is not None:
        gate_quick_pass = bool(float(quick_delta) >= float(args.quick_min_delta))

    gate_full_pass = bool((float(full_delta) >= float(args.target_delta)) and (float(full_p) < float(args.target_p)))
    decision = "proven" if gate_full_pass else "revised_claim"

    payload = {
        "ok": True,
        "target": {"delta": float(args.target_delta), "p": float(args.target_p)},
        "quick_gate": {"min_delta": float(args.quick_min_delta), "pass": bool(gate_quick_pass)},
        "inputs": {
            "val_summary": str(args.val_summary) if args.val_summary else None,
            "quick_metrics": str(args.quick_metrics) if args.quick_metrics else None,
            "full_metrics": str(args.full_metrics),
        },
        "observed": {
            "val_delta": float(val_delta) if val_delta is not None else None,
            "val_p": float(val_p) if val_p is not None else None,
            "quick_delta": float(quick_delta) if quick_delta is not None else None,
            "quick_p": float(quick_p) if quick_p is not None else None,
            "full_delta": float(full_delta),
            "full_p": float(full_p),
        },
        "gate_results": {
            "quick_pass": bool(gate_quick_pass),
            "full_pass": bool(gate_full_pass),
            "decision": decision,
            "c0003_proven": bool(gate_full_pass),
        },
        "artifacts": {
            "decision_json": str(out_dir / "decision.json"),
            "decision_md": str(out_dir / "decision.md"),
        },
    }

    md = [
        "# D0701 Gate Decision",
        "",
        f"- target: `delta >= {float(args.target_delta):.4f}` and `p < {float(args.target_p):.4f}`",
        f"- quick gate: `delta >= {float(args.quick_min_delta):.4f}`",
        "",
        "## Observed",
        "",
        f"- val402 best: delta={payload['observed']['val_delta']}, p={payload['observed']['val_p']}",
        f"- quick test402: delta={payload['observed']['quick_delta']}, p={payload['observed']['quick_p']}",
        f"- full test402: delta={payload['observed']['full_delta']:+.5f}, p={payload['observed']['full_p']:.5f}",
        "",
        "## Decision",
        "",
        f"- quick_pass: `{str(bool(gate_quick_pass)).lower()}`",
        f"- full_pass: `{str(bool(gate_full_pass)).lower()}`",
        f"- decision: `{decision}`",
        "",
    ]

    (out_dir / "decision.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "decision.md").write_text("\n".join(md), encoding="utf-8")
    print(out_dir / "decision.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

