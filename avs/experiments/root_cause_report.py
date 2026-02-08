from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


def _read_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _metrics_view(metrics: dict | None) -> dict:
    if not isinstance(metrics, dict):
        return {
            "anchored_mean": float("nan"),
            "uniform_mean": float("nan"),
            "delta": float("nan"),
            "p_anchored_vs_uniform": float("nan"),
            "fallback_used_frac": float("nan"),
            "token_budget": float("nan"),
        }

    summary = metrics.get("summary") or {}
    anchored = _safe_float((summary.get("anchored_top2") or {}).get("mean"))
    uniform = _safe_float((summary.get("uniform") or {}).get("mean"))
    p_val = _safe_float(((metrics.get("paired_ttest") or {}).get("anchored_vs_uniform") or {}).get("p"))
    token_budget = _safe_float(metrics.get("token_budget"))

    fallback = float("nan")
    dbg = (metrics.get("debug_eval") or {}).get("anchored_top2")
    if isinstance(dbg, dict) and len(dbg) > 0:
        used = sum(1 for row in dbg.values() if isinstance(row, dict) and bool(row.get("fallback_used")))
        fallback = float(used) / float(len(dbg))

    return {
        "anchored_mean": anchored,
        "uniform_mean": uniform,
        "delta": anchored - uniform if not (math.isnan(anchored) or math.isnan(uniform)) else float("nan"),
        "p_anchored_vs_uniform": p_val,
        "fallback_used_frac": fallback,
        "token_budget": token_budget,
    }


def _oracle_view(oracle_vs_predicted: dict | None) -> dict:
    if not isinstance(oracle_vs_predicted, dict):
        return {
            "oracle_minus_predicted": float("nan"),
            "predicted_delta": float("nan"),
            "predicted_p": float("nan"),
        }

    gap = _safe_float(oracle_vs_predicted.get("oracle_minus_predicted"))
    predicted = (oracle_vs_predicted.get("predicted") or {}).get("summary") or {}
    p_mean = _safe_float((predicted.get("anchored_top2") or {}).get("mean"))
    u_mean = _safe_float((predicted.get("uniform") or {}).get("mean"))
    pred_delta = p_mean - u_mean if not (math.isnan(p_mean) or math.isnan(u_mean)) else float("nan")
    pred_p = _safe_float((oracle_vs_predicted.get("predicted") or {}).get("p_anchored_vs_uniform"))
    return {
        "oracle_minus_predicted": gap,
        "predicted_delta": pred_delta,
        "predicted_p": pred_p,
    }


def _evidence_view(evidence_alignment: dict | None) -> dict:
    if not isinstance(evidence_alignment, dict):
        return {"cov_mean": float("nan"), "corr_abs_mean": float("nan")}

    cov = evidence_alignment.get("cov_by_tau") or {}
    cov_vals = [_safe_float(v) for v in cov.values()]
    cov_vals = [x for x in cov_vals if not math.isnan(x)]
    cov_mean = float(sum(cov_vals) / len(cov_vals)) if cov_vals else float("nan")

    corr = evidence_alignment.get("corr_by_tau") or {}
    corr_vals: list[float] = []
    for row in corr.values():
        if not isinstance(row, dict):
            continue
        pearson = _safe_float(row.get("pearson"))
        if not math.isnan(pearson):
            corr_vals.append(abs(pearson))
    corr_abs_mean = float(sum(corr_vals) / len(corr_vals)) if corr_vals else float("nan")
    return {"cov_mean": cov_mean, "corr_abs_mean": corr_abs_mean}


def _degradation_view(degradation_accuracy: dict | None) -> dict:
    if not isinstance(degradation_accuracy, dict):
        return {"alpha_num_fail": float("nan"), "worst_delta_anchored_minus_uniform": float("nan")}

    floor = degradation_accuracy.get("alpha_floor_checks") or {}
    num_fail = _safe_float(floor.get("num_fail"))

    worst = float("nan")
    rows = degradation_accuracy.get("rows") or []
    if isinstance(rows, list):
        vals = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            delta = _safe_float(((row.get("acc_delta") or {}).get("anchored_minus_uniform")))
            if not math.isnan(delta):
                vals.append(delta)
        if vals:
            worst = float(min(vals))

    return {"alpha_num_fail": num_fail, "worst_delta_anchored_minus_uniform": worst}


def _epic_view(epic_compare: dict | None) -> dict:
    if not isinstance(epic_compare, dict):
        return {"epic_delta_map": float("nan"), "epic_delta_macro_f1": float("nan")}
    return {
        "epic_delta_map": _safe_float(epic_compare.get("delta_mAP")),
        "epic_delta_macro_f1": _safe_float(epic_compare.get("delta_macro_f1@0.5")),
    }


def _solution_pool() -> list[dict]:
    return [
        {
            "bucket": "stage1_coverage",
            "items": [
                "Use denser stride eventness (`energy_stride_max`) and keep `window_topk+nms` fixed.",
                "Add per-clip autoshift from audio↔cheap-visual correlation before anchor selection.",
                "Switch confidence metric to `top1_med` or `gini`; tune only on val, freeze on test.",
                "Increase anchor diversity (`anchor_nms_radius`, adjacent-veto for far second anchor).",
                "Fuse cheap visual signals (`clipdiff` / `flow`) as Stage-1 fallback, not as main baseline.",
                "Train lightweight eventness calibrator (LR/MLP/TCN) with strict no-test-tuning protocol.",
            ],
        },
        {
            "bucket": "stage2_allocator",
            "items": [
                "Compare `anchor_base_alloc={distance,score,balanced,mixed}` under identical budget.",
                "Constrain high-res anchors (`max_high_anchors=1`) when confidence is ambiguous.",
                "Use budget band mode only when exact mode impossible; log epsilon and chosen resolutions.",
                "Add allocation diagnostics by anchor distance and high_count bucket.",
            ],
        },
        {
            "bucket": "robustness_and_data",
            "items": [
                "Run decode integrity audit before training and blacklist irrecoverable corrupted clips.",
                "Separate dataset-missing block from method-failure block in conclusions.",
                "Re-run degradation grid with fixed seeds and report alpha floor checks.",
                "Keep one canonical score cache path per experiment to avoid silent drift.",
            ],
        },
        {
            "bucket": "statistics_and_reporting",
            "items": [
                "Use seeds 0..9 for all promotion runs; no promotion from p-filtered quick runs alone.",
                "Report paired tests for all comparisons (`anchored_vs_uniform`, `oracle_vs_uniform`).",
                "Publish Oracle→Pred gap across all budget triads; avoid single-budget claims.",
                "Add fail-fast root-cause report artifact per promoted experiment.",
            ],
        },
    ]


def build_root_cause_report(
    *,
    metrics_json: Path,
    oracle_vs_predicted_json: Path | None,
    evidence_alignment_json: Path | None,
    degradation_accuracy_json: Path | None,
    epic_compare_json: Path | None,
    out_dir: Path,
    target_delta: float,
    target_p: float,
    oracle_gap_threshold: float,
    fallback_threshold: float,
    coverage_threshold: float,
    corr_threshold: float,
) -> dict:
    metrics = _read_json(metrics_json)
    ovp = _read_json(oracle_vs_predicted_json)
    evidence = _read_json(evidence_alignment_json)
    degradation = _read_json(degradation_accuracy_json)
    epic = _read_json(epic_compare_json)

    m = _metrics_view(metrics)
    o = _oracle_view(ovp)
    e = _evidence_view(evidence)
    d = _degradation_view(degradation)
    ep = _epic_view(epic)

    reasons: list[dict] = []
    if not math.isnan(m["delta"]) and float(m["delta"]) < float(target_delta):
        reasons.append(
            {
                "id": "R_TARGET_DELTA",
                "title": "Primary gain target not met",
                "evidence": {"delta": float(m["delta"]), "target_delta": float(target_delta)},
                "candidate_solutions": [
                    "Prioritize Stage-1 reliability (coverage first) before further head tuning.",
                    "Run val-only gate sweep and freeze gate for test.",
                    "Use Oracle→Pred gap as promotion gate, not val delta alone.",
                ],
            }
        )

    if not math.isnan(m["p_anchored_vs_uniform"]) and float(m["p_anchored_vs_uniform"]) >= float(target_p):
        reasons.append(
            {
                "id": "R_SIGNIFICANCE",
                "title": "Paired significance not met",
                "evidence": {"p_anchored_vs_uniform": float(m["p_anchored_vs_uniform"]), "target_p": float(target_p)},
                "candidate_solutions": [
                    "Increase seeds to 0..9 for decision runs.",
                    "Avoid quick-run promotion without full-seed confirmation.",
                    "Use fixed score cache and fixed split IDs for reproducibility.",
                ],
            }
        )

    if not math.isnan(o["oracle_minus_predicted"]) and float(o["oracle_minus_predicted"]) > float(oracle_gap_threshold):
        reasons.append(
            {
                "id": "R_ORACLE_GAP",
                "title": "Oracle→Predicted gap remains large",
                "evidence": {"oracle_minus_predicted": float(o["oracle_minus_predicted"]), "threshold": float(oracle_gap_threshold)},
                "candidate_solutions": [
                    "Upgrade Stage-1 eventness (denser stride or AV fused fallback).",
                    "Tune anchor selection diversity (NMS radius, adjacent/far controls).",
                    "Re-evaluate per-clip autoshift with strict no-test tuning.",
                ],
            }
        )

    if not math.isnan(m["fallback_used_frac"]) and float(m["fallback_used_frac"]) > float(fallback_threshold):
        reasons.append(
            {
                "id": "R_FALLBACK_HIGH",
                "title": "Fallback to uniform is too frequent",
                "evidence": {"fallback_used_frac": float(m["fallback_used_frac"]), "threshold": float(fallback_threshold)},
                "candidate_solutions": [
                    "Recalibrate confidence metric/threshold by val gate sweep.",
                    "Normalize score scale before gating (method-dependent).",
                    "Log fallback reasons by clip bucket and remove pathological thresholds.",
                ],
            }
        )

    if not math.isnan(e["cov_mean"]) and float(e["cov_mean"]) < float(coverage_threshold):
        reasons.append(
            {
                "id": "R_COVERAGE_LOW",
                "title": "Evidence coverage is low",
                "evidence": {"cov_mean": float(e["cov_mean"]), "threshold": float(coverage_threshold)},
                "candidate_solutions": [
                    "Increase Stage-1 recall with stride/window config and mild dilation.",
                    "Use cheap visual fallback around low-confidence anchors.",
                    "Audit labels/temporal granularity mismatch for evidence windows.",
                ],
            }
        )

    if not math.isnan(e["corr_abs_mean"]) and float(e["corr_abs_mean"]) < float(corr_threshold):
        reasons.append(
            {
                "id": "R_ALIGNMENT_WEAK",
                "title": "Coverage-to-accuracy linkage is weak",
                "evidence": {"corr_abs_mean": float(e["corr_abs_mean"]), "threshold": float(corr_threshold)},
                "candidate_solutions": [
                    "Add bucket-level diagnosis (distance, high_count, fallback reason).",
                    "Constrain high-res allocation for ambiguous two-anchor cases.",
                    "Verify eventness quality independent of downstream head capacity.",
                ],
            }
        )

    if not math.isnan(d["alpha_num_fail"]) and float(d["alpha_num_fail"]) > 0.0:
        reasons.append(
            {
                "id": "R_ALPHA_FLOOR_FAIL",
                "title": "Robustness alpha floor violated",
                "evidence": {"alpha_num_fail": float(d["alpha_num_fail"]), "worst_delta_anchored_minus_uniform": float(d["worst_delta_anchored_minus_uniform"])},
                "candidate_solutions": [
                    "Raise alpha fallback budget and enforce lower-bound-safe policy.",
                    "Retune gate for high-noise/high-silence regimes.",
                    "Use mixed or balanced base allocation for stronger context retention.",
                ],
            }
        )

    if not math.isnan(ep["epic_delta_map"]) and float(ep["epic_delta_map"]) <= 0.0:
        reasons.append(
            {
                "id": "R_EPIC_DOWNSTREAM",
                "title": "EPIC downstream transfer is not positive",
                "evidence": {"epic_delta_map": float(ep["epic_delta_map"]), "epic_delta_macro_f1": float(ep["epic_delta_macro_f1"])},
                "candidate_solutions": [
                    "Audit long-video decode quality and missing-video bias first.",
                    "Use the same Stage-1 gate protocol as AVE before EPIC full run.",
                    "Rebalance train/val video sampling to reduce domain skew.",
                ],
            }
        )

    priority = [r["id"] for r in reasons]
    payload = {
        "ok": True,
        "inputs": {
            "metrics_json": str(metrics_json),
            "oracle_vs_predicted_json": str(oracle_vs_predicted_json) if oracle_vs_predicted_json is not None else None,
            "evidence_alignment_json": str(evidence_alignment_json) if evidence_alignment_json is not None else None,
            "degradation_accuracy_json": str(degradation_accuracy_json) if degradation_accuracy_json is not None else None,
            "epic_compare_json": str(epic_compare_json) if epic_compare_json is not None else None,
        },
        "targets": {
            "delta": float(target_delta),
            "p_value": float(target_p),
            "oracle_gap_threshold": float(oracle_gap_threshold),
            "fallback_threshold": float(fallback_threshold),
            "coverage_threshold": float(coverage_threshold),
            "corr_threshold": float(corr_threshold),
        },
        "observed": {
            "metrics": m,
            "oracle_vs_predicted": o,
            "evidence_alignment": e,
            "degradation_accuracy": d,
            "epic_downstream": ep,
        },
        "reasons": reasons,
        "priority_queue": priority,
        "solution_pool": _solution_pool(),
        "proven": bool(
            (not math.isnan(m["delta"]))
            and (not math.isnan(m["p_anchored_vs_uniform"]))
            and float(m["delta"]) >= float(target_delta)
            and float(m["p_anchored_vs_uniform"]) < float(target_p)
        ),
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "root_cause_report.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"out_json": str(out_json), "summary": {"num_reasons": int(len(reasons)), "priority_queue": priority, "proven": bool(payload["proven"])}}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate a machine-readable root-cause report for unproven anchored gains.")
    p.add_argument("--metrics-json", type=Path, required=True)
    p.add_argument("--oracle-vs-predicted-json", type=Path, default=None)
    p.add_argument("--evidence-alignment-json", type=Path, default=None)
    p.add_argument("--degradation-accuracy-json", type=Path, default=None)
    p.add_argument("--epic-compare-json", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("runs") / f"E0502_root_cause_report_{time.strftime('%Y%m%d-%H%M%S')}")
    p.add_argument("--target-delta", type=float, default=0.02)
    p.add_argument("--target-p", type=float, default=0.05)
    p.add_argument("--oracle-gap-threshold", type=float, default=0.015)
    p.add_argument("--fallback-threshold", type=float, default=0.60)
    p.add_argument("--coverage-threshold", type=float, default=0.20)
    p.add_argument("--corr-threshold", type=float, default=0.15)
    p.add_argument("--fail-if-unproven", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rep = build_root_cause_report(
        metrics_json=Path(args.metrics_json),
        oracle_vs_predicted_json=Path(args.oracle_vs_predicted_json) if args.oracle_vs_predicted_json is not None else None,
        evidence_alignment_json=Path(args.evidence_alignment_json) if args.evidence_alignment_json is not None else None,
        degradation_accuracy_json=Path(args.degradation_accuracy_json) if args.degradation_accuracy_json is not None else None,
        epic_compare_json=Path(args.epic_compare_json) if args.epic_compare_json is not None else None,
        out_dir=Path(args.out_dir),
        target_delta=float(args.target_delta),
        target_p=float(args.target_p),
        oracle_gap_threshold=float(args.oracle_gap_threshold),
        fallback_threshold=float(args.fallback_threshold),
        coverage_threshold=float(args.coverage_threshold),
        corr_threshold=float(args.corr_threshold),
    )
    print(rep["out_json"])
    if bool(args.fail_if_unproven) and (not bool(rep["summary"]["proven"])):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
