from __future__ import annotations

import argparse
import json
from dataclasses import replace
import math
import random
import time
from pathlib import Path

from avs.audio.eventness import (
    compute_eventness_wav_energy,
    compute_eventness_wav_energy_delta,
    compute_eventness_wav_energy_stride_max,
)
from avs.datasets.ave import AVEIndex, ensure_ave_meta
from avs.datasets.layout import ave_paths
from avs.experiments.ave_p0 import P0Config, run_p0_from_caches
from avs.train.train_loop import TrainConfig
from avs.vision.cheap_eventness import frame_diff_eventness, list_frames
from avs.visualize.pareto_report import plot_pareto_report
from avs.utils.scores import AV_FUSED_SCORE_SCALE, fuse_max, fuse_prod, minmax_01, scale


def write_toy_pareto_report(*, out_dir: Path) -> dict:
    """
    Toy-only Pareto report used by smoke tests.

    This is intentionally small and deterministic; full AVE/EPIC modes will be added under P0048/P0049.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Two budgets, multiple methods each.
    points = [
        {"method": "uniform", "token_budget": 500, "acc_mean": 0.62, "acc_std": 0.01},
        {"method": "random", "token_budget": 500, "acc_mean": 0.60, "acc_std": 0.01},
        {"method": "anchored", "token_budget": 500, "acc_mean": 0.66, "acc_std": 0.01},
        {"method": "oracle", "token_budget": 500, "acc_mean": 0.70, "acc_std": 0.01},
        {"method": "uniform", "token_budget": 1000, "acc_mean": 0.70, "acc_std": 0.01},
        {"method": "random", "token_budget": 1000, "acc_mean": 0.68, "acc_std": 0.01},
        {"method": "anchored", "token_budget": 1000, "acc_mean": 0.75, "acc_std": 0.01},
        {"method": "oracle", "token_budget": 1000, "acc_mean": 0.79, "acc_std": 0.01},
    ]

    payload = {
        "ok": True,
        "mode": "toy",
        "points": points,
    }

    out_json = out_dir / "pareto_report.json"
    out_png = out_dir / "pareto.png"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot_pareto_report(points=points, out_png=out_png, title="Toy Pareto: Acc vs Tok")
    return {"out_dir": str(out_dir), "out_json": str(out_json), "out_png": str(out_png), "points": points}


def _read_ids_file(path: Path, limit: int | None) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = str(line).strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        ids.append(s)
        if limit is not None and len(ids) >= int(limit):
            break
    return ids


def _split_ids(index: AVEIndex, split: str, limit: int | None) -> list[str]:
    ids = index.splits[str(split)]
    if limit is not None:
        ids = ids[: int(limit)]
    return [index.clips[int(i)].video_id for i in ids]


def _load_or_select_ids(index: AVEIndex, ids_file: Path | None, split: str, limit: int | None) -> list[str]:
    if ids_file is not None:
        return _read_ids_file(ids_file, limit)
    return _split_ids(index, split, limit)


def _filter_missing(*, ids: list[str], caches_dir: Path) -> list[str]:
    cached = {p.stem for p in caches_dir.glob("*.npz")}
    return [cid for cid in ids if cid in cached]


def _labels_for_ids(index: AVEIndex, ids: list[str]) -> dict[str, list[int]]:
    clip_by_id = {c.video_id: c for c in index.clips}
    out: dict[str, list[int]] = {}
    for cid in ids:
        clip = clip_by_id.get(cid)
        if clip is None:
            continue
        out[cid] = [int(x) for x in index.segment_labels(clip)]
    return out


def _scores_predicted(
    *,
    clip_ids: list[str],
    processed_dir: Path,
    num_segments: int,
    eventness_method: str,
) -> dict[str, list[float]]:
    method = str(eventness_method)
    out: dict[str, list[float]] = {}
    for cid in clip_ids:
        wav_path = processed_dir / cid / "audio.wav"
        if method == "energy":
            ev = compute_eventness_wav_energy(wav_path, num_segments=int(num_segments))
            out[cid] = [float(x) for x in ev.scores]
        elif method == "energy_delta":
            ev = compute_eventness_wav_energy_delta(wav_path, num_segments=int(num_segments))
            out[cid] = [float(x) for x in ev.scores]
        elif method == "energy_stride_max":
            ev = compute_eventness_wav_energy_stride_max(wav_path, num_segments=int(num_segments), stride_s=0.2, win_s=0.4)
            out[cid] = [float(x) for x in ev.scores]
        elif method == "av_fused":
            ev = compute_eventness_wav_energy_stride_max(wav_path, num_segments=int(num_segments), stride_s=0.2, win_s=0.4)
            a = minmax_01([float(x) for x in ev.scores])

            frames_dir = processed_dir / cid / "frames"
            frames = list_frames(frames_dir) if frames_dir.exists() else []
            v = frame_diff_eventness(frames, size=32) if frames else []
            v = minmax_01([float(x) for x in v])
            out[cid] = scale(fuse_max(a, v, num_segments=int(num_segments)), AV_FUSED_SCORE_SCALE)
        elif method == "av_fused_prod":
            ev = compute_eventness_wav_energy_stride_max(wav_path, num_segments=int(num_segments), stride_s=0.2, win_s=0.4)
            a = minmax_01([float(x) for x in ev.scores])

            frames_dir = processed_dir / cid / "frames"
            frames = list_frames(frames_dir) if frames_dir.exists() else []
            v = frame_diff_eventness(frames, size=32) if frames else []
            v = minmax_01([float(x) for x in v])
            out[cid] = scale(fuse_prod(a, v, num_segments=int(num_segments)), AV_FUSED_SCORE_SCALE)
        else:
            raise ValueError(f"unsupported eventness_method for ave_official MDE: {method}")
    return out


def _scores_cheap_visual(*, clip_ids: list[str], processed_dir: Path, num_segments: int) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for cid in clip_ids:
        frames_dir = processed_dir / cid / "frames"
        frames = list_frames(frames_dir)
        scores = frame_diff_eventness(frames)
        if len(scores) < int(num_segments):
            scores = list(scores) + [0.0] * (int(num_segments) - len(scores))
        out[cid] = [float(x) for x in scores[: int(num_segments)]]
    return out


def _toy_oracle_vs_predicted(*, out_dir: Path) -> dict:
    """
    Toy-only Oracle vs Predicted report (smoke-like).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    budgets = [500, 1000]
    rows = [
        {"token_budget": 500, "uniform": 0.62, "predicted": 0.66, "oracle": 0.70},
        {"token_budget": 1000, "uniform": 0.70, "predicted": 0.75, "oracle": 0.79},
    ]
    out = {"ok": True, "mode": "toy", "budgets": budgets, "rows": rows}
    out_json = out_dir / "oracle_vs_predicted.json"
    out_json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"out_dir": str(out_dir), "out_json": str(out_json)}


def _parse_triads(value: str) -> list[tuple[int, int, int]]:
    """
    Parse `--triads` like:
      "112,160,224;160,224,352;224,352,448"
    """
    triads: list[tuple[int, int, int]] = []
    for chunk in str(value).split(";"):
        s = chunk.strip()
        if not s:
            continue
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError(f"invalid triad chunk {chunk!r}; expected 'low,base,high'")
        low, base, high = (int(parts[0]), int(parts[1]), int(parts[2]))
        triads.append((low, base, high))
    if not triads:
        raise ValueError("triads is empty")
    return triads


def _parse_csv_ints(value: str) -> list[int]:
    out: list[int] = []
    for s in str(value).split(","):
        s = s.strip()
        if not s:
            continue
        out.append(int(s))
    return out


def _load_scores_json(path: Path) -> dict[str, list[float]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    scores = obj.get("scores")
    if not isinstance(scores, dict):
        raise ValueError(f"invalid scores_json (missing dict 'scores'): {path}")
    out: dict[str, list[float]] = {}
    for k, v in scores.items():
        if not isinstance(v, list):
            continue
        out[str(k)] = [float(x) for x in v]
    return out


def _write_scores_json(*, path: Path, eventness_method: str, num_segments: int, scores_by_clip: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": True,
        "eventness_method": str(eventness_method),
        "num_segments": int(num_segments),
        "scores": {str(k): [float(x) for x in v] for k, v in scores_by_clip.items()},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _p0_config_from_json(obj: dict) -> P0Config:
    """
    Best-effort loader for `best_config.json`-style dicts produced by `ave_p0_sweep`.

    Unknown keys are ignored; missing keys fall back to `P0Config` defaults.
    """
    fields = set(P0Config.__dataclass_fields__.keys())  # type: ignore[attr-defined]

    int_fields = {
        "k",
        "low_res",
        "base_res",
        "high_res",
        "patch_size",
        "anchor_shift",
        "anchor_nms_radius",
        "anchor_window",
        "anchor_smooth_window",
        "anchor_high_adjacent_dist",
        "temporal_kernel_size",
    }
    float_fields = {
        "anchor_std_threshold",
        "anchor_nms_strong_gap",
        "anchor_conf_threshold",
        "anchor_high_conf_threshold",
        "anchor_high_gap_threshold",
        "head_dropout",
        "budget_epsilon_frac",
    }
    opt_int_fields = {"max_high_anchors", "triad_alt_max_high_anchors"}
    opt_str_fields = {"anchor_conf_metric", "anchor_high_conf_metric"}
    tuple_int_fields = {"budget_extra_resolutions"}

    kwargs: dict[str, object] = {}
    for k, v in obj.items():
        if k not in fields:
            continue
        if k in int_fields:
            if v is None:
                continue
            kwargs[k] = int(v)
        elif k in float_fields:
            if v is None:
                continue
            kwargs[k] = float(v)
        elif k in opt_int_fields:
            kwargs[k] = int(v) if v is not None else None
        elif k in opt_str_fields:
            kwargs[k] = str(v) if v is not None else None
        elif k in tuple_int_fields:
            kwargs[k] = tuple(int(x) for x in (v or []))
        else:
            kwargs[k] = v

    return P0Config(**kwargs)


def run_pareto_grid_ave_official(
    *,
    out_dir: Path,
    meta_dir: Path,
    processed_dir: Path,
    caches_dir: Path,
    train_ids_file: Path | None,
    eval_ids_file: Path | None,
    split_train: str,
    split_eval: str,
    limit_train: int | None,
    limit_eval: int | None,
    allow_missing: bool,
    seeds: list[int],
    train_cfg: TrainConfig,
    train_device: str,
    audio_device: str,
    ast_pretrained: bool,
    eventness_method: str,
    cfg_base: P0Config,
    triads: list[tuple[int, int, int]],
    budget_mode: str,
    budget_epsilon_frac: float,
    budget_extra_resolutions: list[int],
    include_cheap_visual: bool,
    scores_json: Path | None,
) -> dict:
    """
    MDE Pareto grid runner: evaluate multiple token budgets (triads) and emit a single Pareto report
    (JSON + PNG) for Oracle vs Predicted (+ controls).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_ave_meta(meta_dir)
    index = AVEIndex.from_meta_dir(meta_dir)

    train_ids = _load_or_select_ids(index, train_ids_file, split_train, limit_train)
    eval_ids = _load_or_select_ids(index, eval_ids_file, split_eval, limit_eval)

    if allow_missing:
        train_ids = _filter_missing(ids=train_ids, caches_dir=caches_dir)
        eval_ids = _filter_missing(ids=eval_ids, caches_dir=caches_dir)
    if not train_ids or not eval_ids:
        raise ValueError(f"no usable ids after filtering (train={len(train_ids)} eval={len(eval_ids)})")

    labels_by_clip = {**_labels_for_ids(index, train_ids), **_labels_for_ids(index, eval_ids)}
    all_ids = sorted(set(train_ids + eval_ids))

    # Stage-1 scores cache (so Pareto budgets do not re-train/re-score Stage-1).
    scores_by_clip_override: dict[str, list[float]] | None = None
    if scores_json is not None and scores_json.exists():
        scores_by_clip_override = _load_scores_json(scores_json)
    if scores_by_clip_override is None:
        scores_by_clip_override = {}

    missing_scores = [cid for cid in all_ids if cid not in scores_by_clip_override]
    if missing_scores:
        from avs.experiments.ave_p0_sweep import _compute_scores_by_clip  # local import: heavy

        computed = _compute_scores_by_clip(
            clip_ids=missing_scores,
            processed_dir=processed_dir,
            caches_dir=caches_dir,
            num_segments=10,
            eventness_method=str(eventness_method),
            audio_device=str(audio_device),
            ast_pretrained=bool(ast_pretrained),
            panns_random=False,
            panns_checkpoint=None,
            audiomae_random=False,
            audiomae_checkpoint=None,
            train_ids=train_ids,
            labels_by_clip=labels_by_clip,
        )
        scores_by_clip_override.update({str(k): [float(x) for x in v] for k, v in computed.items()})
        if scores_json is not None:
            _write_scores_json(
                path=scores_json,
                eventness_method=str(eventness_method),
                num_segments=10,
                scores_by_clip=scores_by_clip_override,
            )

    cheap_scores: dict[str, list[float]] | None = None
    if include_cheap_visual:
        cheap_scores = _scores_cheap_visual(clip_ids=all_ids, processed_dir=processed_dir, num_segments=10)

    points: list[dict] = []
    runs: list[dict] = []

    for low, base, high in triads:
        requested_mode = str(budget_mode)

        def _cfg_for_mode(mode: str) -> P0Config:
            return replace(
                cfg_base,
                low_res=int(low),
                base_res=int(base),
                high_res=int(high),
                budget_mode=str(mode),
                budget_epsilon_frac=float(budget_epsilon_frac),
                budget_extra_resolutions=tuple(int(r) for r in budget_extra_resolutions),
            )

        def _run_p0(cfg: P0Config, *, method: str, scores_override: dict[str, list[float]] | None) -> dict:
            return run_p0_from_caches(
                clip_ids_train=train_ids,
                clip_ids_eval=eval_ids,
                labels_by_clip=labels_by_clip,
                caches_dir=caches_dir,
                audio_dir=processed_dir,
                cfg=cfg,
                baselines=["uniform", "random_top2", "anchored_top2", "oracle_top2"],
                seeds=seeds,
                train_cfg=train_cfg,
                train_device=str(train_device),
                num_classes=index.num_classes,
                class_names=[str(index.idx_to_label[i]) for i in range(int(index.num_classes))],
                num_segments=10,
                eventness_method=str(method),
                audio_device=str(audio_device),
                ast_pretrained=bool(ast_pretrained),
                scores_by_clip_override=scores_override,
            )

        used_mode = requested_mode
        if requested_mode == "auto":
            used_mode = "exact"

        cfg = _cfg_for_mode(used_mode)
        try:
            metrics_pred = _run_p0(cfg, method=str(eventness_method), scores_override=scores_by_clip_override)
        except ValueError as e:
            msg = str(e)
            if requested_mode == "auto" and "cannot exactly match budget" in msg:
                used_mode = "band"
                cfg = _cfg_for_mode(used_mode)
                metrics_pred = _run_p0(cfg, method=str(eventness_method), scores_override=scores_by_clip_override)
            else:
                raise
        out_metrics_pred = out_dir / f"metrics_predicted_{low}_{base}_{high}.json"
        out_metrics_pred.write_text(json.dumps(metrics_pred, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        def _acc(m: dict, baseline: str) -> tuple[float, float]:
            s = (m.get("summary") or {}).get(baseline) or {}
            return float(s.get("mean", 0.0)), float(s.get("std", 0.0))

        def _tok(m: dict, baseline: str) -> float:
            try:
                return float(m.get("token_usage", {})[baseline]["eval"]["mean"])
            except Exception:
                return float(m.get("token_budget", cfg.token_budget(num_segments=10)))

        # One budget => 4 core points.
        u_mean, u_std = _acc(metrics_pred, "uniform")
        r_mean, r_std = _acc(metrics_pred, "random_top2")
        p_mean, p_std = _acc(metrics_pred, "anchored_top2")
        o_mean, o_std = _acc(metrics_pred, "oracle_top2")

        budget_tag = f"{low}_{base}_{high}"
        points.extend(
            [
                {
                    "method": "uniform",
                    "token_budget": _tok(metrics_pred, "uniform"),
                    "acc_mean": u_mean,
                    "acc_std": u_std,
                    "triad": budget_tag,
                    "raw_metrics_path": str(out_metrics_pred),
                },
                {
                    "method": "random",
                    "token_budget": _tok(metrics_pred, "random_top2"),
                    "acc_mean": r_mean,
                    "acc_std": r_std,
                    "triad": budget_tag,
                    "raw_metrics_path": str(out_metrics_pred),
                },
                {
                    "method": "predicted",
                    "token_budget": _tok(metrics_pred, "anchored_top2"),
                    "acc_mean": p_mean,
                    "acc_std": p_std,
                    "triad": budget_tag,
                    "raw_metrics_path": str(out_metrics_pred),
                    "anchor_source": str(eventness_method),
                },
                {
                    "method": "oracle",
                    "token_budget": _tok(metrics_pred, "oracle_top2"),
                    "acc_mean": o_mean,
                    "acc_std": o_std,
                    "triad": budget_tag,
                    "raw_metrics_path": str(out_metrics_pred),
                },
            ]
        )

        out_metrics_cheap = None
        if cheap_scores is not None:
            metrics_cheap = _run_p0(cfg, method="energy", scores_override=cheap_scores)
            out_metrics_cheap = out_dir / f"metrics_cheap_visual_{low}_{base}_{high}.json"
            out_metrics_cheap.write_text(json.dumps(metrics_cheap, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            c_mean, c_std = _acc(metrics_cheap, "anchored_top2")
            points.append(
                {
                    "method": "cheap_visual",
                    "token_budget": _tok(metrics_cheap, "anchored_top2"),
                    "acc_mean": c_mean,
                    "acc_std": c_std,
                    "triad": budget_tag,
                    "raw_metrics_path": str(out_metrics_cheap),
                }
            )

        runs.append(
            {
                "triad": {"low_res": int(low), "base_res": int(base), "high_res": int(high)},
                "budget_mode_used": str(used_mode),
                "predicted_metrics": str(out_metrics_pred),
                "cheap_visual_metrics": str(out_metrics_cheap) if out_metrics_cheap is not None else None,
            }
        )

    report = {
        "ok": True,
        "mode": "ave_official",
        "meta_dir": str(meta_dir),
        "processed_dir": str(processed_dir),
        "caches_dir": str(caches_dir),
        "split_train": str(split_train),
        "split_eval": str(split_eval),
        "num_train_ids": int(len(train_ids)),
        "num_eval_ids": int(len(eval_ids)),
        "eventness_method": str(eventness_method),
        "include_cheap_visual": bool(include_cheap_visual),
        "seeds": [int(x) for x in seeds],
        "train_cfg": {"epochs": int(train_cfg.epochs), "batch_size": int(train_cfg.batch_size), "lr": float(train_cfg.lr), "weight_decay": float(train_cfg.weight_decay)},
        "p0_cfg_base": {
            "k": int(cfg_base.k),
            "anchor_shift": int(cfg_base.anchor_shift),
            "anchor_select": str(cfg_base.anchor_select),
            "anchor_nms_radius": int(cfg_base.anchor_nms_radius),
            "anchor_nms_strong_gap": float(cfg_base.anchor_nms_strong_gap),
            "anchor_window": int(cfg_base.anchor_window),
            "anchor_conf_metric": str(cfg_base.anchor_conf_metric) if cfg_base.anchor_conf_metric is not None else None,
            "anchor_conf_threshold": float(cfg_base.anchor_conf_threshold) if cfg_base.anchor_conf_threshold is not None else None,
            "anchor_base_alloc": str(cfg_base.anchor_base_alloc),
            "anchor_high_policy": str(cfg_base.anchor_high_policy),
            "head": str(cfg_base.head),
        },
        "budget": {
            "triads": [{"low_res": int(a), "base_res": int(b), "high_res": int(c)} for a, b, c in triads],
            "budget_mode": str(budget_mode),
            "budget_epsilon_frac": float(budget_epsilon_frac),
            "budget_extra_resolutions": [int(r) for r in budget_extra_resolutions],
        },
        "points": points,
        "runs": runs,
    }

    out_json = out_dir / "pareto_report.json"
    out_png = out_dir / "pareto.png"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot_pareto_report(points=points, out_png=out_png, title="AVE Pareto: Acc vs Tok")
    return {"out_dir": str(out_dir), "out_json": str(out_json), "out_png": str(out_png)}


def run_oracle_vs_predicted_ave_official(
    *,
    out_dir: Path,
    meta_dir: Path,
    processed_dir: Path,
    caches_dir: Path,
    train_ids_file: Path | None,
    eval_ids_file: Path | None,
    split_train: str,
    split_eval: str,
    limit_train: int | None,
    limit_eval: int | None,
    allow_missing: bool,
    seeds: list[int],
    train_cfg: TrainConfig,
    train_device: str,
    audio_device: str,
    ast_pretrained: bool,
    eventness_method: str,
    cfg: P0Config,
    include_cheap_visual: bool,
    scores_json: Path | None,
) -> dict:
    """
    Minimal decision experiment for MDE-2 on AVE: compare Oracle vs Predicted (and controls)
    under a fixed token budget, with audit-friendly artifacts.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_ave_meta(meta_dir)
    index = AVEIndex.from_meta_dir(meta_dir)

    train_ids = _load_or_select_ids(index, train_ids_file, split_train, limit_train)
    eval_ids = _load_or_select_ids(index, eval_ids_file, split_eval, limit_eval)

    if allow_missing:
        train_ids = _filter_missing(ids=train_ids, caches_dir=caches_dir)
        eval_ids = _filter_missing(ids=eval_ids, caches_dir=caches_dir)
    if not train_ids or not eval_ids:
        raise ValueError(f"no usable ids after filtering (train={len(train_ids)} eval={len(eval_ids)})")

    labels_by_clip = {**_labels_for_ids(index, train_ids), **_labels_for_ids(index, eval_ids)}
    all_ids = sorted(set(train_ids + eval_ids))

    # Predicted anchors:
    # - Either from an explicit Stage-1 score cache (scores_json), OR
    # - from a direct score function (energy/stride/av_fused), OR
    # - computed inside P0 (e.g., supervised audio_basic_* methods).
    scores_by_clip_override: dict[str, list[float]] | None = None
    if scores_json is not None:
        scores_json = Path(scores_json)
        if scores_json.exists():
            scores_by_clip_override = _load_scores_json(scores_json)
        else:
            scores_by_clip_override = {}

        missing_scores = [cid for cid in all_ids if cid not in scores_by_clip_override]
        if missing_scores:
            from avs.experiments.ave_p0_sweep import _compute_scores_by_clip  # local import: heavy

            computed = _compute_scores_by_clip(
                clip_ids=missing_scores,
                processed_dir=processed_dir,
                caches_dir=caches_dir,
                num_segments=10,
                eventness_method=str(eventness_method),
                audio_device=str(audio_device),
                ast_pretrained=bool(ast_pretrained),
                panns_random=False,
                panns_checkpoint=None,
                audiomae_random=False,
                audiomae_checkpoint=None,
                train_ids=train_ids,
                labels_by_clip=labels_by_clip,
            )
            scores_by_clip_override.update({str(k): [float(x) for x in v] for k, v in computed.items()})
            _write_scores_json(path=scores_json, eventness_method=str(eventness_method), num_segments=10, scores_by_clip=scores_by_clip_override)
    elif str(eventness_method) in ("energy", "energy_delta", "energy_stride_max", "av_fused", "av_fused_prod"):
        scores_by_clip_override = _scores_predicted(
            clip_ids=all_ids,
            processed_dir=processed_dir,
            num_segments=10,
            eventness_method=str(eventness_method),
        )

    if str(eventness_method) in ("energy_autoshift_clipdiff", "energy_autoshift_clipdiff_pos"):
        # This Stage-1 method estimates per-clip shifts and bakes them into the score sequence.
        # Keep the report consistent and avoid accidental double-shifting.
        cfg = replace(cfg, anchor_shift=0)

    metrics_pred = run_p0_from_caches(
        clip_ids_train=train_ids,
        clip_ids_eval=eval_ids,
        labels_by_clip=labels_by_clip,
        caches_dir=caches_dir,
        audio_dir=processed_dir,
        cfg=cfg,
        baselines=["uniform", "random_top2", "anchored_top2", "oracle_top2"],
        seeds=seeds,
        train_cfg=train_cfg,
        train_device=str(train_device),
        num_classes=index.num_classes,
        class_names=[str(index.idx_to_label[i]) for i in range(int(index.num_classes))],
        num_segments=10,
        eventness_method=str(eventness_method),
        audio_device=str(audio_device),
        ast_pretrained=bool(ast_pretrained),
        scores_by_clip_override=scores_by_clip_override,
    )

    cheap = None
    if include_cheap_visual:
        cheap_scores = _scores_cheap_visual(clip_ids=all_ids, processed_dir=processed_dir, num_segments=10)
        cheap_metrics = run_p0_from_caches(
            clip_ids_train=train_ids,
            clip_ids_eval=eval_ids,
            labels_by_clip=labels_by_clip,
            caches_dir=caches_dir,
            audio_dir=processed_dir,
            cfg=cfg,
            baselines=["uniform", "random_top2", "anchored_top2", "oracle_top2"],
            seeds=seeds,
            train_cfg=train_cfg,
            train_device=str(train_device),
            num_classes=index.num_classes,
            class_names=[str(index.idx_to_label[i]) for i in range(int(index.num_classes))],
            num_segments=10,
            eventness_method="energy",
            audio_device=str(audio_device),
            scores_by_clip_override=cheap_scores,
        )
        cheap = {
            "anchor_source": "cheap_visual",
            "metrics": cheap_metrics,
        }

    # Summaries (keep report small and machine-friendly; full per-seed arrays remain in the raw metrics.json).
    def _summarize(m: dict) -> dict:
        s = m.get("summary") or {}
        return {k: {"mean": float(v["mean"]), "std": float(v["std"])} for k, v in s.items() if isinstance(v, dict) and "mean" in v}

    def _p(m: dict, key: str) -> float | None:
        try:
            return float(m.get("paired_ttest", {})[key]["p"])
        except Exception:
            return None

    summary_pred = _summarize(metrics_pred)
    token_budget = int(metrics_pred.get("token_budget", cfg.token_budget(num_segments=10)))

    report = {
        "ok": True,
        "mode": "ave_official",
        "meta_dir": str(meta_dir),
        "processed_dir": str(processed_dir),
        "caches_dir": str(caches_dir),
        "split_train": str(split_train),
        "split_eval": str(split_eval),
        "num_train_ids": int(len(train_ids)),
        "num_eval_ids": int(len(eval_ids)),
        "seeds": [int(x) for x in seeds],
        "train_cfg": {"epochs": int(train_cfg.epochs), "batch_size": int(train_cfg.batch_size), "lr": float(train_cfg.lr), "weight_decay": float(train_cfg.weight_decay)},
        "token_budget": int(token_budget),
        "p0_cfg": {
            "k": int(cfg.k),
            "low_res": int(cfg.low_res),
            "base_res": int(cfg.base_res),
            "high_res": int(cfg.high_res),
            "anchor_shift": int(cfg.anchor_shift),
            "anchor_select": str(cfg.anchor_select),
            "anchor_nms_radius": int(cfg.anchor_nms_radius),
            "anchor_nms_strong_gap": float(cfg.anchor_nms_strong_gap),
            "anchor_std_threshold": float(cfg.anchor_std_threshold),
            "anchor_conf_metric": cfg.anchor_conf_metric,
            "anchor_conf_threshold": cfg.anchor_conf_threshold,
            "anchor_high_policy": str(cfg.anchor_high_policy),
            "anchor_high_adjacent_dist": int(cfg.anchor_high_adjacent_dist),
            "anchor_high_gap_threshold": float(cfg.anchor_high_gap_threshold),
            "anchor_base_alloc": str(cfg.anchor_base_alloc),
            "head": str(cfg.head),
            "temporal_kernel_size": int(cfg.temporal_kernel_size),
        },
        "predicted": {
            "anchor_source": str(eventness_method),
            "summary": summary_pred,
            "p_anchored_vs_uniform": _p(metrics_pred, "anchored_vs_uniform"),
            "p_oracle_vs_uniform": _p(metrics_pred, "oracle_vs_uniform"),
            "p_anchored_vs_random": _p(metrics_pred, "anchored_vs_random"),
            "raw_metrics_path": str(out_dir / "metrics_predicted.json"),
        },
        "cheap_visual": None,
        "oracle_minus_predicted": None,
    }

    if "oracle_top2" in summary_pred and "anchored_top2" in summary_pred:
        report["oracle_minus_predicted"] = float(summary_pred["oracle_top2"]["mean"] - summary_pred["anchored_top2"]["mean"])

    # Persist raw metrics for reproducibility.
    (out_dir / "metrics_predicted.json").write_text(json.dumps(metrics_pred, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if cheap is not None:
        report["cheap_visual"] = {
            "anchor_source": "cheap_visual",
            "summary": _summarize(cheap["metrics"]),
            "p_anchored_vs_uniform": _p(cheap["metrics"], "anchored_vs_uniform"),
            "p_oracle_vs_uniform": _p(cheap["metrics"], "oracle_vs_uniform"),
            "p_anchored_vs_random": _p(cheap["metrics"], "anchored_vs_random"),
            "raw_metrics_path": str(out_dir / "metrics_cheap_visual.json"),
        }
        (out_dir / "metrics_cheap_visual.json").write_text(
            json.dumps(cheap["metrics"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    out_json = out_dir / "oracle_vs_predicted.json"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"out_dir": str(out_dir), "out_json": str(out_json)}


def run_gate_sweep_ave_official(
    *,
    out_dir: Path,
    meta_dir: Path,
    processed_dir: Path,
    caches_dir: Path,
    train_ids_file: Path | None,
    eval_ids_file: Path | None,
    split_train: str,
    split_eval: str,
    limit_train: int | None,
    limit_eval: int | None,
    allow_missing: bool,
    seeds: list[int],
    train_cfg: TrainConfig,
    train_device: str,
    audio_device: str,
    ast_pretrained: bool,
    eventness_method: str,
    cfg: P0Config,
    gate_metric: str,
    gate_thresholds: list[float],
) -> dict:
    """
    Select a confidence gate (metric+threshold) on a validation split, then reuse it for test runs.

    This is intentionally small and pre-registered (default: gini thresholds), to avoid tuning-on-test.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_ave_meta(meta_dir)
    index = AVEIndex.from_meta_dir(meta_dir)

    train_ids = _load_or_select_ids(index, train_ids_file, split_train, limit_train)
    eval_ids = _load_or_select_ids(index, eval_ids_file, split_eval, limit_eval)
    if allow_missing:
        train_ids = _filter_missing(ids=train_ids, caches_dir=caches_dir)
        eval_ids = _filter_missing(ids=eval_ids, caches_dir=caches_dir)
    if not train_ids or not eval_ids:
        raise ValueError(f"no usable ids after filtering (train={len(train_ids)} eval={len(eval_ids)})")

    labels_by_clip = {**_labels_for_ids(index, train_ids), **_labels_for_ids(index, eval_ids)}
    all_ids = sorted(set(train_ids + eval_ids))

    # Keep predicted scoring outside P0 for cheap methods (matches E0201 behavior).
    scores_by_clip_override = None
    if str(eventness_method) in ("energy", "energy_delta", "energy_stride_max", "av_fused", "av_fused_prod"):
        scores_by_clip_override = _scores_predicted(
            clip_ids=all_ids,
            processed_dir=processed_dir,
            num_segments=10,
            eventness_method=str(eventness_method),
        )

    if str(eventness_method) in ("energy_autoshift_clipdiff", "energy_autoshift_clipdiff_pos"):
        cfg = replace(cfg, anchor_shift=0)

    if str(eventness_method) == "ast_lr":
        # Precompute once so the sweep does not re-run AST inference and re-train the probe per threshold.
        from avs.audio.ast_probe import ASTEventnessProbe, ASTProbeConfig
        from avs.experiments.ave_p0 import _train_ast_lr_eventness

        import torch

        ast_probe = ASTEventnessProbe(ASTProbeConfig(pretrained=bool(ast_pretrained), device=str(audio_device)))
        ast_lr_model, train_logits_by_clip = _train_ast_lr_eventness(
            clip_ids_train=train_ids,
            labels_by_clip=labels_by_clip,
            audio_dir=processed_dir,
            ast_probe=ast_probe,
            num_segments=10,
            device="cpu",
        )
        ast_lr_model_cpu = ast_lr_model.to(torch.device("cpu"))
        ast_lr_model_cpu.eval()

        scores_by_clip_override = {}
        for i, cid in enumerate(all_ids):
            feats = train_logits_by_clip.get(cid)
            if feats is None:
                wav_path = processed_dir / cid / "audio.wav"
                feats = ast_probe.logits_per_second(wav_path, num_segments=10)

            feats_t = torch.from_numpy(feats).float()
            with torch.no_grad():
                s = ast_lr_model_cpu(feats_t).squeeze(-1)
            scores_by_clip_override[cid] = [float(x) for x in s.detach().cpu().numpy().astype("float32").tolist()]

            if (i + 1) % 200 == 0 or (i + 1) == len(all_ids):
                print(f"[gate_sweep ast_lr] precomputed {i+1}/{len(all_ids)} clips", flush=True)

        # Free the large AST model / logits cache before running multiple P0 training loops.
        del train_logits_by_clip
        del ast_probe

    if str(eventness_method) == "ast_evt_mlp":
        # Precompute once so the sweep does not re-run AST inference and re-train the MLP per threshold.
        from avs.audio.ast_probe import ASTEventnessProbe, ASTProbeConfig
        from avs.experiments.ave_p0 import _train_ast_evt_mlp_eventness

        import torch

        ast_probe = ASTEventnessProbe(ASTProbeConfig(pretrained=bool(ast_pretrained), device=str(audio_device)))
        ast_evt_mlp_model, train_logits_by_clip = _train_ast_evt_mlp_eventness(
            clip_ids_train=train_ids,
            labels_by_clip=labels_by_clip,
            audio_dir=processed_dir,
            ast_probe=ast_probe,
            num_segments=10,
            device="cpu",
            hidden_dim=128,
        )
        ast_evt_mlp_cpu = ast_evt_mlp_model.to(torch.device("cpu"))
        ast_evt_mlp_cpu.eval()

        scores_by_clip_override = {}
        for i, cid in enumerate(all_ids):
            feats = train_logits_by_clip.get(cid)
            if feats is None:
                wav_path = processed_dir / cid / "audio.wav"
                feats = ast_probe.logits_per_second(wav_path, num_segments=10)

            feats_t = torch.from_numpy(feats).float()
            with torch.no_grad():
                s = ast_evt_mlp_cpu(feats_t).squeeze(-1)
            scores_by_clip_override[cid] = [float(x) for x in s.detach().cpu().numpy().astype("float32").tolist()]

            if (i + 1) % 200 == 0 or (i + 1) == len(all_ids):
                print(f"[gate_sweep ast_evt_mlp] precomputed {i+1}/{len(all_ids)} clips", flush=True)

        del train_logits_by_clip
        del ast_probe

    if str(eventness_method) in ("ast_mlp_cls", "ast_mlp_cls_target"):
        # Precompute once so the sweep does not re-run AST inference and re-train the classifier per threshold.
        from avs.audio.ast_probe import ASTEventnessProbe, ASTProbeConfig
        from avs.experiments.ave_p0 import _train_ast_mlp_cls_eventness

        import torch

        ast_probe = ASTEventnessProbe(ASTProbeConfig(pretrained=bool(ast_pretrained), device=str(audio_device)))
        ast_mlp_model, train_logits_by_clip = _train_ast_mlp_cls_eventness(
            clip_ids_train=train_ids,
            labels_by_clip=labels_by_clip,
            audio_dir=processed_dir,
            ast_probe=ast_probe,
            num_classes=int(index.num_classes),
            num_segments=10,
            device="cpu",
            hidden_dim=128,
        )
        ast_mlp_cpu = ast_mlp_model.to(torch.device("cpu"))
        ast_mlp_cpu.eval()

        scores_by_clip_override = {}
        for i, cid in enumerate(all_ids):
            feats = train_logits_by_clip.get(cid)
            if feats is None:
                wav_path = processed_dir / cid / "audio.wav"
                feats = ast_probe.logits_per_second(wav_path, num_segments=10)

            feats_t = torch.from_numpy(feats).float()
            with torch.no_grad():
                logits = ast_mlp_cpu(feats_t)
                bg = logits[:, 0]
                if str(eventness_method) == "ast_mlp_cls":
                    mx = logits[:, 1:].max(dim=-1).values
                    scores = mx - bg
                else:
                    clip_logits = logits.mean(dim=0)
                    clip_logits = clip_logits.clone()
                    clip_logits[0] = float("-inf")
                    cls = int(torch.argmax(clip_logits).item())
                    scores = logits[:, cls] - bg

            scores_by_clip_override[cid] = [float(x) for x in scores.detach().cpu().numpy().astype("float32").tolist()]
            if (i + 1) % 200 == 0 or (i + 1) == len(all_ids):
                print(f"[gate_sweep ast_mlp_cls] precomputed {i+1}/{len(all_ids)} clips", flush=True)

        del train_logits_by_clip
        del ast_probe

    if str(eventness_method) in ("av_basic_lr", "av_basic_mlp"):
        # Precompute once so the sweep does not re-extract (audio + frame-diff) features and re-train
        # the supervised AV eventness model per threshold.
        #
        # NOTE: This uses cheap-visual frame differences; treat as an A/V fusion baseline (not the core audio-only method).
        from avs.audio.features import audio_features_per_second
        from avs.experiments.ave_p0 import _train_audio_basic_lr_eventness, _train_audio_basic_mlp_eventness
        from avs.vision.cheap_eventness import frame_diff_eventness, list_frames

        import numpy as np
        import torch

        feats_by_clip: dict[str, np.ndarray] = {}
        for i, cid in enumerate(all_ids):
            wav_path = processed_dir / cid / "audio.wav"
            a = audio_features_per_second(wav_path, num_segments=10, feature_set="basic")  # [10, F]

            frames_dir = processed_dir / cid / "frames"
            frames = list_frames(frames_dir) if frames_dir.exists() else []
            vis = frame_diff_eventness(frames, size=32) if frames else []

            v = np.zeros((10, 1), dtype=np.float32)
            for t, s in enumerate(vis[:10]):
                v[int(t), 0] = float(s)

            feats_by_clip[cid] = np.concatenate([a, v], axis=1).astype(np.float32, copy=False)
            if (i + 1) % 200 == 0 or (i + 1) == len(all_ids):
                print(f"[gate_sweep {eventness_method}] extracted {i+1}/{len(all_ids)} clips", flush=True)

        if str(eventness_method) == "av_basic_lr":
            av_model = _train_audio_basic_lr_eventness(
                clip_ids_train=train_ids,
                labels_by_clip=labels_by_clip,
                audio_feats_by_clip=feats_by_clip,
                device="cpu",
            )
        else:
            av_model = _train_audio_basic_mlp_eventness(
                clip_ids_train=train_ids,
                labels_by_clip=labels_by_clip,
                audio_feats_by_clip=feats_by_clip,
                device="cpu",
            )

        av_model_cpu = av_model.to(torch.device("cpu"))
        av_model_cpu.eval()

        scores_by_clip_override = {}
        for i, cid in enumerate(all_ids):
            xt = torch.from_numpy(feats_by_clip[cid]).to(dtype=torch.float32, device=torch.device("cpu"))
            with torch.no_grad():
                logits = av_model_cpu(xt).squeeze(-1)
            scores_by_clip_override[cid] = [float(x) for x in logits.detach().cpu().numpy().astype("float32").tolist()]
            if (i + 1) % 200 == 0 or (i + 1) == len(all_ids):
                print(f"[gate_sweep {eventness_method}] scored {i+1}/{len(all_ids)} clips", flush=True)

    if str(eventness_method) in ("audio_basic_tcn", "audio_fbank_tcn"):
        # Precompute once so the sweep does not re-extract audio features and re-train the TCN per threshold.
        from avs.audio.features import audio_features_per_second
        from avs.experiments.ave_p0 import _train_audio_tcn_eventness

        import numpy as np
        import torch

        feature_set = "basic" if str(eventness_method) == "audio_basic_tcn" else "fbank_stats"

        feats_by_clip: dict[str, np.ndarray] = {}
        for i, cid in enumerate(all_ids):
            wav_path = processed_dir / cid / "audio.wav"
            feats_by_clip[cid] = audio_features_per_second(wav_path, num_segments=10, feature_set=str(feature_set))
            if (i + 1) % 200 == 0 or (i + 1) == len(all_ids):
                print(f"[gate_sweep {eventness_method}] extracted {i+1}/{len(all_ids)} clips", flush=True)

        audio_model = _train_audio_tcn_eventness(
            clip_ids_train=train_ids,
            labels_by_clip=labels_by_clip,
            audio_feats_by_clip=feats_by_clip,
            device="cpu",
            epochs=50,
            batch_size=128,
            lr=1e-3,
            hidden_channels=64,
            kernel_size=3,
            dropout=0.1,
        )

        audio_model_cpu = audio_model.to(torch.device("cpu"))
        audio_model_cpu.eval()

        scores_by_clip_override = {}
        for i, cid in enumerate(all_ids):
            xt = torch.from_numpy(feats_by_clip[cid]).to(dtype=torch.float32, device=torch.device("cpu"))
            with torch.no_grad():
                logits = audio_model_cpu(xt).squeeze(-1)
            scores_by_clip_override[cid] = [float(x) for x in logits.detach().cpu().numpy().astype("float32").tolist()]
            if (i + 1) % 200 == 0 or (i + 1) == len(all_ids):
                print(f"[gate_sweep {eventness_method}] scored {i+1}/{len(all_ids)} clips", flush=True)

    def _fmt_thr(x: float) -> str:
        s = f"{float(x):.3f}"
        s = s.replace("-", "m").replace(".", "p")
        return s

    rows: list[dict] = []
    best: dict | None = None
    best_key: tuple[float, float] | None = None  # (delta_mean, -p)

    for thr in gate_thresholds:
        cfg_gate = P0Config(
            k=int(cfg.k),
            low_res=int(cfg.low_res),
            base_res=int(cfg.base_res),
            high_res=int(cfg.high_res),
            patch_size=int(cfg.patch_size),
            max_high_anchors=int(cfg.max_high_anchors) if cfg.max_high_anchors is not None else None,
            anchor_shift=int(cfg.anchor_shift),
            anchor_std_threshold=float(cfg.anchor_std_threshold),
            anchor_select=str(cfg.anchor_select),
            anchor_nms_radius=int(cfg.anchor_nms_radius),
            anchor_nms_strong_gap=float(cfg.anchor_nms_strong_gap),
            anchor_window=int(cfg.anchor_window),
            anchor_smooth_window=int(cfg.anchor_smooth_window),
            anchor_smooth_mode=str(cfg.anchor_smooth_mode),
            anchor_conf_metric=str(gate_metric),
            anchor_conf_threshold=float(thr),
            anchor_base_alloc=str(cfg.anchor_base_alloc),
            anchor_high_policy=str(cfg.anchor_high_policy),
            anchor_high_adjacent_dist=int(cfg.anchor_high_adjacent_dist),
            anchor_high_gap_threshold=float(cfg.anchor_high_gap_threshold),
            head=str(cfg.head),
            temporal_kernel_size=int(cfg.temporal_kernel_size),
        )

        metrics = run_p0_from_caches(
            clip_ids_train=train_ids,
            clip_ids_eval=eval_ids,
            labels_by_clip=labels_by_clip,
            caches_dir=caches_dir,
            audio_dir=processed_dir,
            cfg=cfg_gate,
            baselines=["uniform", "random_top2", "anchored_top2", "oracle_top2"],
            seeds=seeds,
            train_cfg=train_cfg,
            train_device=str(train_device),
            num_classes=index.num_classes,
            class_names=[str(index.idx_to_label[i]) for i in range(int(index.num_classes))],
            num_segments=10,
            eventness_method=str(eventness_method),
            audio_device=str(audio_device),
            ast_pretrained=bool(ast_pretrained),
            scores_by_clip_override=scores_by_clip_override,
        )

        out_metrics = out_dir / f"metrics_gate_{str(gate_metric)}_{_fmt_thr(float(thr))}.json"
        out_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        summary = metrics.get("summary") or {}
        anchored = float(summary.get("anchored_top2", {}).get("mean", float("nan")))
        uniform = float(summary.get("uniform", {}).get("mean", float("nan")))
        oracle = float(summary.get("oracle_top2", {}).get("mean", float("nan")))
        delta = anchored - uniform

        p = None
        try:
            p = float(metrics.get("paired_ttest", {})["anchored_vs_uniform"]["p"])
        except Exception:
            p = None

        fb = None
        try:
            dbg = (metrics.get("debug_eval") or {}).get("anchored_top2") or {}
            if isinstance(dbg, dict) and dbg:
                n = len(dbg)
                used = sum(1 for v in dbg.values() if isinstance(v, dict) and v.get("fallback_used"))
                fb = float(used) / float(n) if n > 0 else None
        except Exception:
            fb = None

        row = {
            "gate_metric": str(gate_metric),
            "gate_threshold": float(thr),
            "anchored_top2_mean": anchored,
            "uniform_mean": uniform,
            "oracle_top2_mean": oracle,
            "delta_mean": delta,
            "p_anchored_vs_uniform": p,
            "oracle_minus_predicted": (oracle - anchored) if (not math.isnan(oracle) and not math.isnan(anchored)) else None,
            "fallback_used_frac": fb,
            "raw_metrics_path": str(out_metrics),
        }
        rows.append(row)

        key = (float(delta), float(-p) if p is not None else 0.0)
        if best_key is None or key > best_key:
            best_key = key
            best = dict(row)

    out_json = out_dir / "gate_sweep.json"
    best_gate_json = out_dir / "best_gate.json"
    payload = {
        "ok": True,
        "mode": "ave_official",
        "meta_dir": str(meta_dir),
        "processed_dir": str(processed_dir),
        "caches_dir": str(caches_dir),
        "split_train": str(split_train),
        "split_eval": str(split_eval),
        "num_train_ids": int(len(train_ids)),
        "num_eval_ids": int(len(eval_ids)),
        "seeds": [int(x) for x in seeds],
        "train_cfg": {
            "epochs": int(train_cfg.epochs),
            "batch_size": int(train_cfg.batch_size),
            "lr": float(train_cfg.lr),
            "weight_decay": float(train_cfg.weight_decay),
        },
        "p0_cfg_base": {
            "k": int(cfg.k),
            "low_res": int(cfg.low_res),
            "base_res": int(cfg.base_res),
            "high_res": int(cfg.high_res),
            "anchor_shift": int(cfg.anchor_shift),
            "anchor_select": str(cfg.anchor_select),
            "anchor_nms_radius": int(cfg.anchor_nms_radius),
            "anchor_nms_strong_gap": float(cfg.anchor_nms_strong_gap),
            "anchor_std_threshold": float(cfg.anchor_std_threshold),
            "anchor_high_policy": str(cfg.anchor_high_policy),
            "anchor_high_adjacent_dist": int(cfg.anchor_high_adjacent_dist),
            "anchor_high_gap_threshold": float(cfg.anchor_high_gap_threshold),
            "anchor_base_alloc": str(cfg.anchor_base_alloc),
            "head": str(cfg.head),
            "temporal_kernel_size": int(cfg.temporal_kernel_size),
        },
        "eventness_method": str(eventness_method),
        "ast_pretrained": bool(ast_pretrained),
        "audio_device": str(audio_device),
        "gate_metric": str(gate_metric),
        "gate_thresholds": [float(x) for x in gate_thresholds],
        "rows": rows,
        "best": best,
        "select_rule": "max(delta_mean), tie-break by min(p)",
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if best is None:
        best_gate_json.write_text(json.dumps({"ok": False, "reason": "no_rows"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        best_gate_json.write_text(
            json.dumps(
                {
                    "ok": True,
                    "anchor_conf_metric": str(best["gate_metric"]),
                    "anchor_conf_threshold": float(best["gate_threshold"]),
                    "selected_from": str(out_json),
                    "best_row": best,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    return {"out_dir": str(out_dir), "out_json": str(out_json), "best_gate_json": str(best_gate_json)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Listen-then-Look MDE harness (toy-first).")
    sub = p.add_subparsers(dest="cmd", required=True)

    toy = sub.add_parser("toy", help="Generate a toy Pareto report (smoke-only).")
    toy.add_argument("--out-dir", type=Path, default=Path("runs") / f"E0200_mde_toy_{time.strftime('%Y%m%d-%H%M%S')}")

    pg = sub.add_parser("pareto_grid", help="Multi-budget Pareto report on AVE (Oracle→Predicted + controls).")
    pg.add_argument("--mode", type=str, default="ave_official", choices=["ave_official"])
    pg.add_argument("--out-dir", type=Path, default=Path("runs") / f"E0330_pareto_grid_{time.strftime('%Y%m%d-%H%M%S')}")
    pg.add_argument("--meta-dir", type=Path, default=ave_paths().meta_dir)
    pg.add_argument("--processed-dir", type=Path, default=ave_paths().processed_dir)
    pg.add_argument("--caches-dir", type=Path, required=False, default=None)
    pg.add_argument("--train-ids-file", type=Path, default=None)
    pg.add_argument("--eval-ids-file", type=Path, default=None)
    pg.add_argument("--split-train", type=str, default="train", choices=["train", "val", "test"])
    pg.add_argument("--split-eval", type=str, default="test", choices=["train", "val", "test"])
    pg.add_argument("--limit-train", type=int, default=None)
    pg.add_argument("--limit-eval", type=int, default=None)
    pg.add_argument("--allow-missing", action="store_true")
    pg.add_argument("--seeds", type=str, default="0,1,2")
    pg.add_argument("--epochs", type=int, default=5)
    pg.add_argument("--batch-size", type=int, default=16)
    pg.add_argument("--lr", type=float, default=2e-3)
    pg.add_argument("--weight-decay", type=float, default=0.0)
    pg.add_argument("--train-device", type=str, default="cuda:0")
    pg.add_argument("--eventness-method", type=str, default="energy")
    pg.add_argument("--include-cheap-visual", action="store_true", help="Also run cheap-visual anchors as a control.")
    pg.add_argument("--audio-device", type=str, default="cpu", help="Device for audio probes like PANNs/AST (e.g., cpu, cuda:0).")
    pg.add_argument("--ast-pretrained", action="store_true", help="Use pretrained AST weights (downloads from HF).")
    pg.add_argument(
        "--base-config-json",
        type=Path,
        default=None,
        help="Optional best_config.json from `ave_p0_sweep` to fix all non-budget P0 knobs. "
        "If omitted, uses a reasonable default config.",
    )
    pg.add_argument(
        "--triads",
        type=str,
        default="112,160,224;160,224,352;224,352,448",
        help="Semicolon-separated triads: 'low,base,high;...'. Each triad defines one budget point.",
    )
    pg.add_argument(
        "--budget-mode",
        type=str,
        default="auto",
        choices=["auto", "exact", "band"],
        help="Budget solver: exact (strict), band (approx), or auto (exact if possible else band).",
    )
    pg.add_argument("--budget-epsilon-frac", type=float, default=0.05)
    pg.add_argument(
        "--budget-extra-resolutions",
        type=str,
        default="112,160,224,352",
        help="Comma-separated extra resolutions for band-budget plans (excludes high_res automatically).",
    )
    pg.add_argument(
        "--scores-json",
        type=Path,
        default=None,
        help="Optional Stage-1 score cache; missing ids will be filled and written back.",
    )

    ovp = sub.add_parser("oracle_vs_predicted", help="MDE-2 report: Oracle vs Predicted (and controls).")
    ovp.add_argument("--mode", type=str, default="toy", choices=["toy", "ave_official"])
    ovp.add_argument("--out-dir", type=Path, default=Path("runs") / f"E0201_oracle_vs_predicted_{time.strftime('%Y%m%d-%H%M%S')}")
    ovp.add_argument("--meta-dir", type=Path, default=ave_paths().meta_dir)
    ovp.add_argument("--processed-dir", type=Path, default=ave_paths().processed_dir)
    ovp.add_argument("--caches-dir", type=Path, required=False, default=None)
    ovp.add_argument("--train-ids-file", type=Path, default=None)
    ovp.add_argument("--eval-ids-file", type=Path, default=None)
    ovp.add_argument("--split-train", type=str, default="train", choices=["train", "val", "test"])
    ovp.add_argument("--split-eval", type=str, default="val", choices=["train", "val", "test"])
    ovp.add_argument("--limit-train", type=int, default=None)
    ovp.add_argument("--limit-eval", type=int, default=None)
    ovp.add_argument("--allow-missing", action="store_true")

    ovp.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ovp.add_argument("--epochs", type=int, default=5)
    ovp.add_argument("--batch-size", type=int, default=16)
    ovp.add_argument("--lr", type=float, default=2e-3)
    ovp.add_argument("--weight-decay", type=float, default=0.0)
    ovp.add_argument("--train-device", type=str, default="cuda:0")
    ovp.add_argument(
        "--eventness-method",
        type=str,
        default="energy",
        choices=[
            "energy",
            "energy_delta",
            "energy_stride_max",
            "asr_vad",
            "energy_nonspeech_ast",
            "energy_autoshift_clipdiff",
            "energy_autoshift_clipdiff_pos",
            "av_clap_clip_agree",
            "clap_evt",
            "clap_lr",
            "clap_mlp_cls",
            "clap_mlp_cls_target",
            "av_fused",
            "av_fused_prod",
            "ast",
            "ast_nonspeech_max",
            "ast_lr",
            "ast_emb_lr",
            "ast_evt_mlp",
            "ast_mlp_cls",
            "ast_mlp_cls_target",
            "av_fused_clipdiff",
            "av_fused_clipdiff_prod",
            "moe_energy_clipdiff",
            "vision_clipdiff",
            "panns",
            # External supervised AVE temporal localizer as Stage-1 (use with --scores-json).
            "psp_avel_evt",
            # Supervised audio-only eventness (computed inside P0).
            "audio_basic_lr",
            "audio_basic_mlp",
            "audio_basic_tcn",
            "audio_fbank_mlp",
            "audio_fbank_tcn",
            "audio_basic_mlp_cls",
            "audio_basic_mlp_cls_target",
            # Supervised cheap AV eventness (audio basic feats + cheap visual diff).
            "av_basic_lr",
            "av_basic_mlp",
            "av_clipdiff_lr",
            "av_clipdiff_mlp",
            "av_clipdiff_accflip_mlp",
            "av_clipdiff_speech_mlp",
            "av_clipdiff_framediff_mlp",
            "av_clipdiff_flow_mlp",
            "av_clipdiff_flow_mlp_stride",
            "av_clipdiff_fbank_mlp",
            "av_clipdiff_mlp_cls",
            "av_clipdiff_mlp_cls_target",
            "av_clipdiff_vec_mlp",
            "av_clip_mlp_cls",
            "av_clip_mlp_cls_target",
            "av_clipdiff_tcn",
            "av_ast_clipalign_bce",
            # Vision-only learned anchors on cached low-res CLIP features (strong cheap-visual control).
            "vision_mlp_cls",
            "vision_mlp_cls_target",
            "vision_binary_lr",
            "vision_binary_mlp",
        ],
    )
    ovp.add_argument(
        "--base-config-json",
        type=Path,
        default=None,
        help="Optional best_config.json from `ave_p0_sweep` to fix all P0 knobs (including the low/base/high triad).",
    )
    ovp.add_argument(
        "--scores-json",
        type=Path,
        default=None,
        help="Optional Stage-1 score cache; missing ids will be filled and written back. Required for external Stage-1 methods like psp_avel_evt.",
    )
    ovp.add_argument("--include-cheap-visual", action="store_true", help="Also run cheap-visual anchors as a control.")
    ovp.add_argument("--audio-device", type=str, default="cpu", help="Device for audio probes like PANNs (e.g., cpu, cuda:0).")
    ovp.add_argument("--ast-pretrained", action="store_true", help="Use pretrained AST weights (downloads from HF).")

    # Fixed P0 config knobs (keep explicit to avoid hidden coupling).
    ovp.add_argument("--k", type=int, default=2)
    ovp.add_argument("--low-res", type=int, default=160)
    ovp.add_argument("--base-res", type=int, default=224)
    ovp.add_argument("--high-res", type=int, default=352)
    ovp.add_argument("--anchor-shift", type=int, default=1)
    ovp.add_argument("--anchor-std-threshold", type=float, default=1.0)
    ovp.add_argument(
        "--anchor-conf-metric",
        type=str,
        default=None,
        choices=["std", "top1_med", "top12_gap", "gini"],
        help="Optional confidence gate metric. When set, `--anchor-conf-threshold` must also be set. "
        "Overrides legacy `--anchor-std-threshold`.",
    )
    ovp.add_argument(
        "--anchor-conf-threshold",
        type=float,
        default=None,
        help="Confidence threshold for `--anchor-conf-metric` (e.g., gini in [0,1]).",
    )
    ovp.add_argument("--anchor-select", type=str, default="topk", choices=["topk", "nms", "nms_strong", "window_topk"])
    ovp.add_argument("--anchor-nms-radius", type=int, default=2)
    ovp.add_argument("--anchor-nms-strong-gap", type=float, default=0.6)
    ovp.add_argument("--anchor-window", type=int, default=3)
    ovp.add_argument("--anchor-smooth-window", type=int, default=0)
    ovp.add_argument("--anchor-smooth-mode", type=str, default="mean", choices=["mean", "sum"])
    ovp.add_argument(
        "--anchor-base-alloc",
        type=str,
        default="distance",
        choices=["distance", "balanced", "score", "farthest", "mixed"],
    )
    ovp.add_argument("--anchor-high-policy", type=str, default="fixed", choices=["fixed", "adaptive_v1"])
    ovp.add_argument("--anchor-high-adjacent-dist", type=int, default=1)
    ovp.add_argument("--anchor-high-gap-threshold", type=float, default=0.0)
    ovp.add_argument("--max-high-anchors", type=int, default=None)
    ovp.add_argument("--temporal-kernel-size", type=int, default=3)

    gs = sub.add_parser("gate_sweep", help="Select a confidence gate on val (pre-registered grid).")
    gs.add_argument("--mode", type=str, default="toy", choices=["toy", "ave_official"])
    gs.add_argument("--out-dir", type=Path, default=Path("runs") / f"E0204_gate_sweep_{time.strftime('%Y%m%d-%H%M%S')}")
    gs.add_argument("--meta-dir", type=Path, default=ave_paths().meta_dir)
    gs.add_argument("--processed-dir", type=Path, default=ave_paths().processed_dir)
    gs.add_argument("--caches-dir", type=Path, required=False, default=None)
    gs.add_argument("--train-ids-file", type=Path, default=None)
    gs.add_argument("--eval-ids-file", type=Path, default=None)
    gs.add_argument("--split-train", type=str, default="train", choices=["train", "val", "test"])
    gs.add_argument("--split-eval", type=str, default="val", choices=["train", "val", "test"])
    gs.add_argument("--limit-train", type=int, default=None)
    gs.add_argument("--limit-eval", type=int, default=None)
    gs.add_argument("--allow-missing", action="store_true")
    gs.add_argument("--seeds", type=str, default="0,1,2")
    gs.add_argument("--epochs", type=int, default=5)
    gs.add_argument("--batch-size", type=int, default=16)
    gs.add_argument("--lr", type=float, default=2e-3)
    gs.add_argument("--weight-decay", type=float, default=0.0)
    gs.add_argument("--train-device", type=str, default="cuda:0")
    gs.add_argument(
        "--eventness-method",
        type=str,
        default="energy",
        choices=[
            "energy",
            "energy_delta",
            "energy_stride_max",
            "asr_vad",
            "energy_nonspeech_ast",
            "energy_autoshift_clipdiff",
            "energy_autoshift_clipdiff_pos",
            "av_clap_clip_agree",
            "clap_evt",
            "clap_lr",
            "clap_mlp_cls",
            "clap_mlp_cls_target",
            "av_fused",
            "av_fused_prod",
            "ast",
            "ast_nonspeech_max",
            "ast_lr",
            "ast_emb_lr",
            "ast_evt_mlp",
            "ast_mlp_cls",
            "ast_mlp_cls_target",
            "av_fused_clipdiff",
            "av_fused_clipdiff_prod",
            "moe_energy_clipdiff",
            "vision_clipdiff",
            "panns",
            "audio_basic_lr",
            "audio_basic_mlp",
            "audio_basic_tcn",
            "audio_fbank_mlp",
            "audio_fbank_tcn",
            "audio_basic_mlp_cls",
            "audio_basic_mlp_cls_target",
            "av_basic_lr",
            "av_basic_mlp",
            "av_clipdiff_lr",
            "av_clipdiff_mlp",
            "av_clipdiff_accflip_mlp",
            "av_clipdiff_speech_mlp",
            "av_clipdiff_framediff_mlp",
            "av_clipdiff_flow_mlp",
            "av_clipdiff_flow_mlp_stride",
            "av_clipdiff_fbank_mlp",
            "av_clipdiff_vec_mlp",
            "av_clip_mlp_cls",
            "av_clip_mlp_cls_target",
            "av_clipdiff_tcn",
            "av_ast_clipalign_bce",
            "vision_mlp_cls",
            "vision_mlp_cls_target",
            "vision_binary_lr",
            "vision_binary_mlp",
        ],
    )
    gs.add_argument("--audio-device", type=str, default="cpu", help="Device for audio probes like PANNs (e.g., cpu, cuda:0).")
    gs.add_argument("--ast-pretrained", action="store_true", help="Use pretrained AST weights (downloads from HF).")

    # Same P0 config knobs as E0201 (kept explicit and fixed).
    gs.add_argument("--k", type=int, default=2)
    gs.add_argument("--low-res", type=int, default=160)
    gs.add_argument("--base-res", type=int, default=224)
    gs.add_argument("--high-res", type=int, default=352)
    gs.add_argument("--anchor-shift", type=int, default=1)
    gs.add_argument("--anchor-std-threshold", type=float, default=1.0)
    gs.add_argument("--anchor-select", type=str, default="topk", choices=["topk", "nms", "nms_strong", "window_topk"])
    gs.add_argument("--anchor-nms-radius", type=int, default=2)
    gs.add_argument("--anchor-nms-strong-gap", type=float, default=0.6)
    gs.add_argument("--anchor-window", type=int, default=3)
    gs.add_argument("--anchor-smooth-window", type=int, default=0)
    gs.add_argument("--anchor-smooth-mode", type=str, default="mean", choices=["mean", "sum"])
    gs.add_argument(
        "--anchor-base-alloc",
        type=str,
        default="distance",
        choices=["distance", "balanced", "score", "farthest", "mixed"],
    )
    gs.add_argument("--anchor-high-policy", type=str, default="fixed", choices=["fixed", "adaptive_v1"])
    gs.add_argument("--anchor-high-adjacent-dist", type=int, default=1)
    gs.add_argument("--anchor-high-gap-threshold", type=float, default=0.0)
    gs.add_argument("--max-high-anchors", type=int, default=None)
    gs.add_argument("--temporal-kernel-size", type=int, default=3)

    # Gate sweep grid (pre-registered defaults).
    gs.add_argument(
        "--gate-metric",
        type=str,
        default="gini",
        choices=["std", "top1_med", "top12_gap", "gini"],
        help="Confidence metric used for fallback gating.",
    )
    gs.add_argument(
        "--gate-thresholds",
        type=str,
        default="0,0.1,0.2,0.3,0.4",
        help="Comma-separated thresholds for --gate-metric (0 disables gating).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "toy":
        rep = write_toy_pareto_report(out_dir=args.out_dir)
        print(rep["out_json"])
        print(rep["out_png"])
        return 0
    if args.cmd == "pareto_grid":
        caches_dir = Path(args.caches_dir) if args.caches_dir is not None else (Path("runs") / "REAL_AVE_LOCAL" / "caches")
        seeds = [int(x) for x in str(args.seeds).split(",") if str(x).strip()]
        if len(seeds) < 2:
            raise SystemExit("--seeds must contain at least 2 seeds to compute error bars")

        triads = _parse_triads(str(args.triads))
        extra_res = _parse_csv_ints(str(args.budget_extra_resolutions))
        scores_json = args.scores_json
        if scores_json is None:
            scores_json = Path(args.out_dir) / "eventness_scores.json"

        cfg_base = P0Config(
            k=2,
            low_res=160,
            base_res=224,
            high_res=352,
            patch_size=16,
            anchor_shift=1,
            anchor_select="topk",
            anchor_nms_radius=2,
            anchor_nms_strong_gap=0.6,
            anchor_window=3,
            anchor_smooth_window=0,
            anchor_smooth_mode="mean",
            anchor_conf_metric="top1_med",
            anchor_conf_threshold=0.6,
            anchor_base_alloc="distance",
            anchor_high_policy="adaptive_v1",
            anchor_high_adjacent_dist=1,
            anchor_high_gap_threshold=0.0,
            head="temporal_conv",
            temporal_kernel_size=3,
        )
        if args.base_config_json is not None:
            cfg_obj = json.loads(Path(args.base_config_json).read_text(encoding="utf-8"))
            cfg_base = _p0_config_from_json(cfg_obj)
            cfg_base = replace(cfg_base, patch_size=16)  # enforce the project-wide default

        rep = run_pareto_grid_ave_official(
            out_dir=args.out_dir,
            meta_dir=Path(args.meta_dir),
            processed_dir=Path(args.processed_dir),
            caches_dir=caches_dir,
            train_ids_file=args.train_ids_file,
            eval_ids_file=args.eval_ids_file,
            split_train=str(args.split_train),
            split_eval=str(args.split_eval),
            limit_train=int(args.limit_train) if args.limit_train is not None else None,
            limit_eval=int(args.limit_eval) if args.limit_eval is not None else None,
            allow_missing=bool(args.allow_missing),
            seeds=seeds,
            train_cfg=TrainConfig(
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
            ),
            train_device=str(args.train_device),
            audio_device=str(args.audio_device),
            ast_pretrained=bool(args.ast_pretrained),
            eventness_method=str(args.eventness_method),
            cfg_base=cfg_base,
            triads=triads,
            budget_mode=str(args.budget_mode),
            budget_epsilon_frac=float(args.budget_epsilon_frac),
            budget_extra_resolutions=extra_res,
            include_cheap_visual=bool(args.include_cheap_visual),
            scores_json=scores_json,
        )
        print(rep["out_json"])
        print(rep["out_png"])
        return 0
    if args.cmd == "oracle_vs_predicted":
        if args.mode == "toy":
            rep = _toy_oracle_vs_predicted(out_dir=args.out_dir)
            print(rep["out_json"])
            return 0

        caches_dir = Path(args.caches_dir) if args.caches_dir is not None else (Path("runs") / "REAL_AVE_LOCAL" / "caches")
        seeds = [int(x) for x in str(args.seeds).split(",") if str(x).strip()]
        if len(seeds) < 2:
            raise SystemExit("--seeds must contain at least 2 seeds to compute paired p-values")

        if (args.anchor_conf_metric is None) != (args.anchor_conf_threshold is None):
            raise SystemExit("--anchor-conf-metric and --anchor-conf-threshold must be set together (or both unset)")

        cfg = P0Config(
            k=int(args.k),
            low_res=int(args.low_res),
            base_res=int(args.base_res),
            high_res=int(args.high_res),
            patch_size=16,
            max_high_anchors=int(args.max_high_anchors) if args.max_high_anchors is not None else None,
            anchor_shift=int(args.anchor_shift),
            anchor_std_threshold=float(args.anchor_std_threshold),
            anchor_select=str(args.anchor_select),
            anchor_nms_radius=int(args.anchor_nms_radius),
            anchor_nms_strong_gap=float(args.anchor_nms_strong_gap),
            anchor_window=int(args.anchor_window),
            anchor_smooth_window=int(args.anchor_smooth_window),
            anchor_smooth_mode=str(args.anchor_smooth_mode),
            anchor_conf_metric=str(args.anchor_conf_metric) if args.anchor_conf_metric is not None else None,
            anchor_conf_threshold=float(args.anchor_conf_threshold) if args.anchor_conf_threshold is not None else None,
            anchor_base_alloc=str(args.anchor_base_alloc),
            anchor_high_policy=str(args.anchor_high_policy),
            anchor_high_adjacent_dist=int(args.anchor_high_adjacent_dist),
            anchor_high_gap_threshold=float(args.anchor_high_gap_threshold),
            head="temporal_conv",
            temporal_kernel_size=int(args.temporal_kernel_size),
        )
        if args.base_config_json is not None:
            cfg_obj = json.loads(Path(args.base_config_json).read_text(encoding="utf-8"))
            cfg = _p0_config_from_json(cfg_obj)
            cfg = replace(cfg, patch_size=16)  # enforce the project-wide default

        rep = run_oracle_vs_predicted_ave_official(
            out_dir=args.out_dir,
            meta_dir=Path(args.meta_dir),
            processed_dir=Path(args.processed_dir),
            caches_dir=caches_dir,
            train_ids_file=args.train_ids_file,
            eval_ids_file=args.eval_ids_file,
            split_train=str(args.split_train),
            split_eval=str(args.split_eval),
            limit_train=int(args.limit_train) if args.limit_train is not None else None,
            limit_eval=int(args.limit_eval) if args.limit_eval is not None else None,
            allow_missing=bool(args.allow_missing),
            seeds=seeds,
            train_cfg=TrainConfig(
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
            ),
            train_device=str(args.train_device),
            audio_device=str(args.audio_device),
            ast_pretrained=bool(args.ast_pretrained),
            eventness_method=str(args.eventness_method),
            cfg=cfg,
            include_cheap_visual=bool(args.include_cheap_visual),
            scores_json=Path(args.scores_json) if args.scores_json is not None else None,
        )
        print(rep["out_json"])
        return 0
    if args.cmd == "gate_sweep":
        if args.mode == "toy":
            rep = {"ok": True, "mode": "toy", "note": "gate_sweep is only implemented for ave_official"}
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_json = out_dir / "gate_sweep.json"
            out_json.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(str(out_json))
            return 0

        caches_dir = Path(args.caches_dir) if args.caches_dir is not None else (Path("runs") / "REAL_AVE_LOCAL" / "caches")
        seeds = [int(x) for x in str(args.seeds).split(",") if str(x).strip()]
        if len(seeds) < 2:
            raise SystemExit("--seeds must contain at least 2 seeds to compute paired p-values")

        cfg = P0Config(
            k=int(args.k),
            low_res=int(args.low_res),
            base_res=int(args.base_res),
            high_res=int(args.high_res),
            patch_size=16,
            max_high_anchors=int(args.max_high_anchors) if args.max_high_anchors is not None else None,
            anchor_shift=int(args.anchor_shift),
            anchor_std_threshold=float(args.anchor_std_threshold),
            anchor_select=str(args.anchor_select),
            anchor_nms_radius=int(args.anchor_nms_radius),
            anchor_nms_strong_gap=float(args.anchor_nms_strong_gap),
            anchor_window=int(args.anchor_window),
            anchor_smooth_window=int(args.anchor_smooth_window),
            anchor_smooth_mode=str(args.anchor_smooth_mode),
            anchor_conf_metric=None,
            anchor_conf_threshold=None,
            anchor_base_alloc=str(args.anchor_base_alloc),
            anchor_high_policy=str(args.anchor_high_policy),
            anchor_high_adjacent_dist=int(args.anchor_high_adjacent_dist),
            anchor_high_gap_threshold=float(args.anchor_high_gap_threshold),
            head="temporal_conv",
            temporal_kernel_size=int(args.temporal_kernel_size),
        )

        gate_thresholds = [float(x) for x in str(args.gate_thresholds).split(",") if str(x).strip()]
        rep = run_gate_sweep_ave_official(
            out_dir=args.out_dir,
            meta_dir=Path(args.meta_dir),
            processed_dir=Path(args.processed_dir),
            caches_dir=caches_dir,
            train_ids_file=args.train_ids_file,
            eval_ids_file=args.eval_ids_file,
            split_train=str(args.split_train),
            split_eval=str(args.split_eval),
            limit_train=int(args.limit_train) if args.limit_train is not None else None,
            limit_eval=int(args.limit_eval) if args.limit_eval is not None else None,
            allow_missing=bool(args.allow_missing),
            seeds=seeds,
            train_cfg=TrainConfig(
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
            ),
            train_device=str(args.train_device),
            audio_device=str(args.audio_device),
            ast_pretrained=bool(args.ast_pretrained),
            eventness_method=str(args.eventness_method),
            cfg=cfg,
            gate_metric=str(args.gate_metric),
            gate_thresholds=gate_thresholds,
        )
        print(rep["out_json"])
        print(rep["best_gate_json"])
        return 0
    raise SystemExit(f"unknown cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
