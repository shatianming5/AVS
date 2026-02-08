from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SmokeResult:
    name: str
    ok: bool
    details: dict


def _runs_dir() -> Path:
    base = os.environ.get("AVS_RUNS_DIR", "runs")
    return Path(base)


def _new_run_dir(prefix: str = "smoke") -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = _runs_dir() / f"{prefix}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_result(run_dir: Path, results: list[SmokeResult]) -> None:
    payload = {
        "ok": all(r.ok for r in results),
        "results": [{"name": r.name, "ok": r.ok, "details": r.details} for r in results],
    }
    (run_dir / "smoke.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _check_basic(run_dir: Path) -> SmokeResult:
    from avs import __version__

    return SmokeResult(name="basic", ok=True, details={"version": __version__, "run_dir": str(run_dir)})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="avs-smoke", description="AVS smoke checks")
    sub = parser.add_subparsers(dest="check", required=False)

    sub.add_parser("basic", help="Basic import + run-dir write check")
    sub.add_parser("ave_meta", help="Smoke: AVE metadata parsing")
    sub.add_parser("epic_sounds_meta", help="Smoke: EPIC-SOUNDS annotations parsing")
    sub.add_parser("avqa_meta", help="Smoke: AVQA annotations + prompt template")
    sub.add_parser("vggsound_meta", help="Smoke: VGGSound metadata parsing")
    sub.add_parser("epic_sounds_audio", help="Smoke: EPIC-SOUNDS untrimmed audio extraction")
    sub.add_parser("epic_sounds_frames", help="Smoke: EPIC-SOUNDS untrimmed frame extraction")
    sub.add_parser("epic_sounds_long_pack", help="Smoke: EPIC-SOUNDS long-video pack (audio+frames→plan→manifest/cache)")
    sub.add_parser("epic_sounds_video_cls_synth", help="Smoke: EPIC-SOUNDS video-level multi-label classification (synthetic)")
    sub.add_parser("ave_download", help="Smoke: AVE raw-video acquisition helper")
    sub.add_parser("preprocess_one", help="Smoke: preprocess one short video")
    sub.add_parser("preprocess_dataset", help="Smoke: preprocess a tiny dataset directory")
    sub.add_parser("anchors", help="Smoke: anchors + Recall@K")
    sub.add_parser("anchors_dataset", help="Smoke: dataset-wide anchor evaluation")
    sub.add_parser("epic_sounds_anchor_eval", help="Smoke: EPIC-SOUNDS-style long-audio anchor evaluation")
    sub.add_parser("evidence_windows", help="Smoke: time-window IoU + Coverage@τ (Evidence Alignment)")
    sub.add_parser("ltl_budget_allocator", help="Smoke: Listen-then-Look budget allocator (toy)")
    sub.add_parser("ltl_anchor_windows", help="Smoke: stride-based eventness + anchor windows (toy)")
    sub.add_parser("ltl_eventness_stride_max", help="Smoke: stride-max per-second eventness (Listen-then-Look Stage-1)")
    sub.add_parser("ltl_eventness_av_fused", help="Smoke: AV-fused eventness (audio silent, visual change)")
    sub.add_parser("ltl_eventness_autoshift", help="Smoke: auto-shift (audio↔visual alignment) helper")
    sub.add_parser("cheap_visual_eventness", help="Smoke: cheap visual eventness (frame diff)")
    sub.add_parser("mde_pareto_toy", help="Smoke: MDE Pareto report (toy)")
    sub.add_parser("ltl_degradation_suite_toy", help="Smoke: Listen-then-Look degradation suite (toy)")
    sub.add_parser("dataset_integrity_audit", help="Smoke: video integrity audit (ffprobe + decode)")
    sub.add_parser("root_cause_report", help="Smoke: root-cause report aggregation")
    sub.add_parser("sampling_plan", help="Smoke: token-budgeted sampling plan")
    sub.add_parser("plan_jsonl", help="Smoke: generate plan.jsonl from wav(s)")
    sub.add_parser("plan_jsonl_long", help="Smoke: generate plan.jsonl for long wavs (infer segments)")
    sub.add_parser("ast_eventness", help="Smoke: AST-based audio eventness probe (random weights)")
    sub.add_parser("energy_delta_eventness", help="Smoke: energy-delta (novelty/change-point) eventness")
    sub.add_parser("panns_eventness", help="Smoke: PANNs-based audio eventness probe (random weights)")
    sub.add_parser("audiomae_eventness", help="Smoke: AudioMAE(-style) audio eventness probe (random weights)")
    sub.add_parser("anchor_knobs", help="Smoke: anchor shift + fallback knobs")
    sub.add_parser("anchor_window_select", help="Smoke: windowed anchor selection (window_topk)")
    sub.add_parser("anchor_confidence_gate", help="Smoke: anchor confidence gating (fallback reasons)")
    sub.add_parser("temporal_head", help="Smoke: temporal head option (1D conv)")
    sub.add_parser("feature_cache", help="Smoke: multi-resolution feature cache builder")
    sub.add_parser("vision_encoder", help="Smoke: variable-resolution vision encoder")
    sub.add_parser("vision_efficiency", help="Smoke: vision wall-clock efficiency benchmark")
    sub.add_parser("train_smoke", help="Smoke: minimal train/eval loop")
    sub.add_parser("ave_p0_toy", help="Smoke: AVE-P0 runner on synthetic cached features")
    sub.add_parser("ave_p0_uniform_low", help="Smoke: AVE-P0 Uniform-112 (uniform_low) token-budget check")
    sub.add_parser("efficiency_curve", help="Smoke: Accuracy–Token efficiency curve plot")
    sub.add_parser("viz", help="Smoke: qualitative visualization (plot)")
    sub.add_parser("ave_p0_end2end", help="Smoke: end-to-end AVE-P0 subset pipeline")
    sub.add_parser("all", help="Run all smoke checks")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir = _new_run_dir()

    checks = [args.check] if args.check else ["basic"]
    if checks == ["all"]:
        checks = [
            "basic",
            "ave_meta",
            "epic_sounds_meta",
            "avqa_meta",
            "vggsound_meta",
            "epic_sounds_audio",
            "epic_sounds_frames",
            "epic_sounds_long_pack",
            "epic_sounds_video_cls_synth",
            "ave_download",
            "preprocess_one",
            "preprocess_dataset",
            "anchors",
            "anchors_dataset",
            "epic_sounds_anchor_eval",
            "evidence_windows",
            "ltl_budget_allocator",
            "ltl_anchor_windows",
            "ltl_eventness_stride_max",
            "ltl_eventness_av_fused",
            "ltl_eventness_autoshift",
            "cheap_visual_eventness",
            "mde_pareto_toy",
            "ltl_degradation_suite_toy",
            "dataset_integrity_audit",
            "root_cause_report",
            "sampling_plan",
            "plan_jsonl",
            "plan_jsonl_long",
            "ast_eventness",
            "energy_delta_eventness",
            "panns_eventness",
            "audiomae_eventness",
            "anchor_knobs",
            "anchor_window_select",
            "anchor_confidence_gate",
            "temporal_head",
            "feature_cache",
            "vision_encoder",
            "vision_efficiency",
            "train_smoke",
            "ave_p0_toy",
            "ave_p0_uniform_low",
            "efficiency_curve",
            "viz",
            "ave_p0_end2end",
        ]

    results: list[SmokeResult] = []
    for check in checks:
        if check == "basic":
            results.append(_check_basic(run_dir))
            continue

        # Defer imports so the scaffold can land before all modules exist.
        try:
            handler = getattr(__import__("avs.smoke_checks", fromlist=["handlers"]), "handlers")[check]
        except Exception as e:  # noqa: BLE001 - smoke UX: show root cause
            results.append(SmokeResult(name=check, ok=False, details={"error": repr(e)}))
            continue

        try:
            results.append(handler(run_dir))
        except Exception as e:  # noqa: BLE001 - smoke UX: show root cause
            results.append(SmokeResult(name=check, ok=False, details={"error": repr(e)}))

    _write_result(run_dir, results)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
