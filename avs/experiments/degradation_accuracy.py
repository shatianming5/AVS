from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from avs.audio.augment import add_noise_snr_db, apply_silence_ratio, shift_audio
from avs.audio.eventness import (
    anchors_from_scores_with_debug,
    eventness_energy_delta_per_second,
    eventness_energy_per_second,
    eventness_energy_stride_max_per_second,
    load_wav_mono,
)
from avs.audio.features import audio_features_per_second, audio_features_per_second_from_array
from avs.datasets.ave import AVEIndex, ensure_ave_meta
from avs.datasets.layout import ave_paths
from avs.experiments.ave_p0 import (
    P0Config,
    PerSegmentMLP,
    TemporalConvHead,
    _plan_for_baseline,
    _train_audio_basic_mlp_eventness,
    features_from_cache,
)
from avs.metrics.anchors import recall_at_k
from avs.utils.scores import minmax_01, stride_max_pool_per_second
from avs.vision.cheap_eventness import clip_feature_diff_eventness, list_frames, optical_flow_mag_eventness
from avs.vision.feature_cache import FeatureCache
from avs.train.train_loop import TrainConfig, segment_accuracy, train_per_segment_classifier


def _parse_csv_ints(value: str) -> list[int]:
    out: list[int] = []
    for s in str(value).split(","):
        s = s.strip()
        if not s:
            continue
        out.append(int(s))
    return out


def _parse_csv_floats(value: str) -> list[float]:
    out: list[float] = []
    for s in str(value).split(","):
        s = s.strip()
        if not s:
            continue
        out.append(float(s))
    return out


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
        out[cid] = [int(x) for x in index.segment_labels(clip, num_segments=10)]
    return out


def _u01(key: str, *, salt: str = "alpha_fallback_v1") -> float:
    h = hashlib.sha1(f"{salt}:{key}".encode("utf-8")).digest()
    v = int.from_bytes(h[:8], byteorder="big", signed=False)
    return float(v) / float(2**64)


def _force_uniform_for_alpha(cid: str, alpha: float) -> bool:
    a = float(alpha)
    if a <= 0.0:
        return False
    if a >= 1.0:
        return True
    return _u01(str(cid)) < a


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


def _p0_config_from_json(obj: dict) -> P0Config:
    fields = set(P0Config.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    kwargs: dict = {}
    for k, v in obj.items():
        if k in fields:
            kwargs[k] = v
    return replace(P0Config(), **kwargs)


@torch.no_grad()
def _eval_acc(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    device = next(model.parameters()).device
    logits = model(x.to(device))
    return float(segment_accuracy(logits, y.to(device)))


def _build_model(*, cfg: P0Config, in_dim: int, num_classes: int, device: str) -> torch.nn.Module:
    if str(cfg.head) == "mlp":
        return PerSegmentMLP(
            in_dim=int(in_dim),
            num_classes=int(num_classes),
            hidden_dim=int(cfg.head_hidden_dim),
            dropout=float(cfg.head_dropout),
        ).to(torch.device(device))
    if str(cfg.head) == "temporal_conv":
        return TemporalConvHead(
            in_dim=int(in_dim),
            num_classes=int(num_classes),
            hidden_dim=int(cfg.head_hidden_dim),
            kernel_size=int(cfg.temporal_kernel_size),
            dropout=float(cfg.head_dropout),
        ).to(torch.device(device))
    raise ValueError(f"unknown head: {cfg.head!r}")


def _anchors_for_scores(scores: list[float], *, cfg: P0Config, num_segments: int) -> tuple[list[int], dict]:
    res = anchors_from_scores_with_debug(
        scores,
        k=int(cfg.k),
        num_segments=int(num_segments),
        shift=int(cfg.anchor_shift),
        std_threshold=float(cfg.anchor_std_threshold),
        select=str(cfg.anchor_select),
        nms_radius=int(cfg.anchor_nms_radius),
        nms_strong_gap=float(cfg.anchor_nms_strong_gap),
        anchor_window=int(cfg.anchor_window),
        smooth_window=int(cfg.anchor_smooth_window),
        smooth_mode=str(cfg.anchor_smooth_mode),
        conf_metric=str(cfg.anchor_conf_metric) if cfg.anchor_conf_metric is not None else None,
        conf_threshold=float(cfg.anchor_conf_threshold) if cfg.anchor_conf_threshold is not None else None,
    )
    dbg = {
        "fallback_used": bool(res.fallback_used),
        "fallback_reason": res.fallback_reason,
        "conf_metric": str(res.conf_metric),
        "conf_value": float(res.conf_value),
        "conf_threshold": float(res.conf_threshold),
    }
    return [int(x) for x in (res.anchors or [])], dbg


def _scores_energy_augmented(
    *, wav_path: Path, shift_s: float, snr_db: float, silence_ratio: float, rng: np.random.Generator, num_segments: int
) -> list[float]:
    audio, sr = load_wav_mono(wav_path)
    x = shift_audio(audio=audio, sample_rate=int(sr), shift_s=float(shift_s))
    x = add_noise_snr_db(audio=x, snr_db=float(snr_db), rng=rng)
    x = apply_silence_ratio(audio=x, silence_ratio=float(silence_ratio), rng=rng)
    return [float(s) for s in eventness_energy_per_second(x, int(sr), num_segments=int(num_segments))]


def _scores_energy_delta_augmented(
    *, wav_path: Path, shift_s: float, snr_db: float, silence_ratio: float, rng: np.random.Generator, num_segments: int
) -> list[float]:
    audio, sr = load_wav_mono(wav_path)
    x = shift_audio(audio=audio, sample_rate=int(sr), shift_s=float(shift_s))
    x = add_noise_snr_db(audio=x, snr_db=float(snr_db), rng=rng)
    x = apply_silence_ratio(audio=x, silence_ratio=float(silence_ratio), rng=rng)
    return [float(s) for s in eventness_energy_delta_per_second(x, int(sr), num_segments=int(num_segments))]


def _scores_energy_stride_max_augmented(
    *, wav_path: Path, shift_s: float, snr_db: float, silence_ratio: float, rng: np.random.Generator, num_segments: int
) -> list[float]:
    audio, sr = load_wav_mono(wav_path)
    x = shift_audio(audio=audio, sample_rate=int(sr), shift_s=float(shift_s))
    x = add_noise_snr_db(audio=x, snr_db=float(snr_db), rng=rng)
    x = apply_silence_ratio(audio=x, silence_ratio=float(silence_ratio), rng=rng)
    return [
        float(s)
        for s in eventness_energy_stride_max_per_second(x, int(sr), num_segments=int(num_segments), stride_s=0.2, win_s=0.4)
    ]


def _build_visual_side_features(
    *,
    clip_ids: list[str],
    cache_by_clip: dict[str, FeatureCache],
    processed_dir: Path,
    num_segments: int,
    include_flow: bool,
) -> dict[str, np.ndarray]:
    visual_by_clip: dict[str, np.ndarray] = {}
    for i, cid in enumerate(clip_ids):
        cache = cache_by_clip[cid]
        vis_res = 112 if 112 in cache.features_by_resolution else int(min(cache.features_by_resolution))
        feats = cache.features_by_resolution[int(vis_res)]
        cd = clip_feature_diff_eventness(feats, metric="cosine")
        clipdiff01 = minmax_01([float(x) for x in cd])
        v_clipdiff = np.zeros((int(num_segments), 1), dtype=np.float32)
        for t, s in enumerate(clipdiff01[: int(num_segments)]):
            v_clipdiff[int(t), 0] = float(s)

        parts = [v_clipdiff]
        if bool(include_flow):
            frames_dir = processed_dir / cid / "frames"
            frames = list_frames(frames_dir) if frames_dir.exists() else []
            flow = optical_flow_mag_eventness(frames, size=64) if frames else []
            flow01 = minmax_01([float(x) for x in flow])
            v_flow = np.zeros((int(num_segments), 1), dtype=np.float32)
            for t, s in enumerate(flow01[: int(num_segments)]):
                v_flow[int(t), 0] = float(s)
            parts.append(v_flow)
        visual_by_clip[cid] = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
        if (i + 1) % 500 == 0 or (i + 1) == len(clip_ids):
            print(f"[degradation-visual] built {i+1}/{len(clip_ids)} clips", flush=True)
    return visual_by_clip


def _train_av_clipdiff_mlp(
    *,
    clip_ids_train: list[str],
    clip_ids_all: list[str],
    labels_by_clip: dict[str, list[int]],
    processed_dir: Path,
    cache_by_clip: dict[str, FeatureCache],
    num_segments: int,
) -> tuple[torch.nn.Module, dict[str, np.ndarray]]:
    include_flow = False
    feats_by_train: dict[str, np.ndarray] = {}

    visual_by_clip = _build_visual_side_features(
        clip_ids=clip_ids_all,
        cache_by_clip=cache_by_clip,
        processed_dir=processed_dir,
        num_segments=int(num_segments),
        include_flow=bool(include_flow),
    )

    for i, cid in enumerate(clip_ids_train):
        wav_path = processed_dir / cid / "audio.wav"
        a = audio_features_per_second(wav_path, num_segments=int(num_segments), feature_set="basic").astype(np.float32, copy=False)
        v = visual_by_clip[cid]
        feats_by_train[cid] = np.concatenate([a, v], axis=1).astype(np.float32, copy=False)

        if (i + 1) % 500 == 0 or (i + 1) == len(clip_ids_train):
            print(f"[av_clipdiff_mlp] train feats {i+1}/{len(clip_ids_train)}", flush=True)

    model = _train_audio_basic_mlp_eventness(
        clip_ids_train=clip_ids_train,
        labels_by_clip=labels_by_clip,
        audio_feats_by_clip=feats_by_train,
        device="cpu",
        hidden_dim=128,
    )
    model = model.to(torch.device("cpu"))
    model.eval()
    return model, visual_by_clip


def _train_av_clipdiff_flow_mlp(
    *,
    clip_ids_train: list[str],
    clip_ids_all: list[str],
    labels_by_clip: dict[str, list[int]],
    processed_dir: Path,
    cache_by_clip: dict[str, FeatureCache],
    num_segments: int,
) -> tuple[torch.nn.Module, dict[str, np.ndarray]]:
    feats_by_train: dict[str, np.ndarray] = {}
    visual_by_clip = _build_visual_side_features(
        clip_ids=clip_ids_all,
        cache_by_clip=cache_by_clip,
        processed_dir=processed_dir,
        num_segments=int(num_segments),
        include_flow=True,
    )

    for i, cid in enumerate(clip_ids_train):
        wav_path = processed_dir / cid / "audio.wav"
        a = audio_features_per_second(wav_path, num_segments=int(num_segments), feature_set="basic").astype(np.float32, copy=False)
        v = visual_by_clip[cid]
        feats_by_train[cid] = np.concatenate([a, v], axis=1).astype(np.float32, copy=False)
        if (i + 1) % 500 == 0 or (i + 1) == len(clip_ids_train):
            print(f"[av_clipdiff_flow_mlp] train feats {i+1}/{len(clip_ids_train)}", flush=True)

    model = _train_audio_basic_mlp_eventness(
        clip_ids_train=clip_ids_train,
        labels_by_clip=labels_by_clip,
        audio_feats_by_clip=feats_by_train,
        device="cpu",
        hidden_dim=128,
    )
    model = model.to(torch.device("cpu"))
    model.eval()
    return model, visual_by_clip


def _scores_av_clipdiff_mlp_augmented(
    *,
    wav_path: Path,
    shift_s: float,
    snr_db: float,
    silence_ratio: float,
    rng: np.random.Generator,
    num_segments: int,
    model: torch.nn.Module,
    visual_side: np.ndarray,
    apply_stride_max: bool = False,
) -> list[float]:
    audio, sr = load_wav_mono(wav_path)
    x = shift_audio(audio=audio, sample_rate=int(sr), shift_s=float(shift_s))
    x = add_noise_snr_db(audio=x, snr_db=float(snr_db), rng=rng)
    x = apply_silence_ratio(audio=x, silence_ratio=float(silence_ratio), rng=rng)
    a = audio_features_per_second_from_array(x, int(sr), num_segments=int(num_segments), feature_set="basic").astype(
        np.float32, copy=False
    )
    v = np.asarray(visual_side, dtype=np.float32)
    if v.ndim != 2:
        raise ValueError(f"unexpected visual_side shape: {v.shape}")
    if int(v.shape[0]) < int(num_segments):
        v_pad = np.zeros((int(num_segments), int(v.shape[1])), dtype=np.float32)
        if int(v.shape[0]) > 0:
            v_pad[: int(v.shape[0]), :] = v
        v = v_pad
    else:
        v = v[: int(num_segments), :]

    feats = np.concatenate([a, v], axis=1).astype(np.float32, copy=False)
    with torch.no_grad():
        logits = model(torch.from_numpy(feats).float()).squeeze(-1)
    s = [float(x) for x in logits.detach().cpu().numpy().astype("float32").tolist()]
    if bool(apply_stride_max):
        s = stride_max_pool_per_second(s, num_segments=int(num_segments), stride_s=0.2, win_s=0.6)
    return [float(x) for x in s]


def _build_xy(
    *,
    clip_ids: list[str],
    labels_by_clip: dict[str, list[int]],
    cache_by_clip: dict[str, FeatureCache],
    cfg: P0Config,
    baseline: str,
    rng: random.Random,
    scores_by_clip: dict[str, list[float]] | None,
    alpha: float,
    num_segments: int,
    force_uniform_eval_only: bool,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    tokens: list[int] = []
    fallback_gate = 0
    fallback_alpha = 0
    used = 0
    recalls_by_delta: dict[str, list[float]] = {k: [] for k in ("0", "1", "2")}

    for cid in clip_ids:
        labs = labels_by_clip[cid][: int(num_segments)]
        y_rows.append(np.asarray(labs, dtype=np.int64))

        force_uniform = False
        if bool(force_uniform_eval_only) and split != "eval":
            force_uniform = False
        else:
            force_uniform = _force_uniform_for_alpha(cid, float(alpha))

        scores = None if scores_by_clip is None else scores_by_clip.get(cid)
        anchors: list[int] | None = None
        scores_list: list[float] | None = None
        if baseline == "anchored_top2":
            if scores is None:
                raise ValueError("scores_by_clip required for anchored_top2")
            scores_list = [float(x) for x in scores]
            if force_uniform:
                fallback_alpha += 1
                anchors = []
            else:
                anchors, dbg = _anchors_for_scores(scores_list, cfg=cfg, num_segments=int(num_segments))
                fallback_gate += 1 if dbg.get("fallback_used") else 0
            used += 1
            # Stage-1 recall (anchored baseline only).
            gt = [i for i, lab in enumerate(labs) if int(lab) != 0]
            for d in (0, 1, 2):
                recalls_by_delta[str(d)].append(float(recall_at_k(gt, anchors or [], num_segments=int(num_segments), delta=int(d)).recall))

        if baseline == "random_top2" and force_uniform:
            fallback_alpha += 1
            plan = _plan_for_baseline(
                "uniform",
                cfg=cfg,
                num_segments=int(num_segments),
                rng=rng,
                anchors=None,
                oracle_segments=None,
                scores=None,
            )
        elif baseline == "anchored_top2" and force_uniform:
            plan = _plan_for_baseline(
                "uniform",
                cfg=cfg,
                num_segments=int(num_segments),
                rng=rng,
                anchors=None,
                oracle_segments=None,
                scores=None,
            )
        else:
            plan = _plan_for_baseline(
                baseline,
                cfg=cfg,
                num_segments=int(num_segments),
                rng=rng,
                anchors=anchors,
                oracle_segments=None,
                scores=scores_list,
            )

        tokens.append(int(plan.total_tokens()))
        cache = cache_by_clip[cid]
        x = features_from_cache(cache, plan, res_feature=str(cfg.res_feature), base_res=int(cfg.base_res))
        x_rows.append(x.astype(np.float32, copy=False))

    x_t = torch.from_numpy(np.stack(x_rows, axis=0)).float()
    y_t = torch.from_numpy(np.stack(y_rows, axis=0)).long()
    meta = {
        "token_usage": {"mean": float(np.mean(np.asarray(tokens, dtype=np.float32))), "min": int(min(tokens)), "max": int(max(tokens))},
        "alpha_force_uniform_frac": float(fallback_alpha / max(1, len(clip_ids))) if split == "eval" else 0.0,
        "gate_fallback_used_frac": float(fallback_gate / max(1, used)) if used else 0.0,
        "recall_by_delta": {
            str(d): float(np.mean(np.asarray(recalls_by_delta[str(d)], dtype=np.float32))) if recalls_by_delta[str(d)] else 0.0
            for d in (0, 1, 2)
        },
    }
    return x_t, y_t, meta


def run_ave_official_degradation_accuracy(
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
    eventness_method: str,
    base_config_json: Path,
    scores_json: Path | None,
    seeds: list[int],
    train_cfg: TrainConfig,
    train_device: str,
    shift_grid: list[float],
    snr_grid: list[float],
    silence_grid: list[float],
    alpha_grid: list[float],
    num_segments: int = 10,
) -> dict:
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

    cfg_obj = json.loads(base_config_json.read_text(encoding="utf-8"))
    cfg = _p0_config_from_json(cfg_obj)

    # Cache preload (dominant IO).
    all_ids = sorted(set(train_ids + eval_ids))
    cache_by_clip: dict[str, FeatureCache] = {}
    t0 = time.time()
    for i, cid in enumerate(all_ids):
        cache_by_clip[cid] = FeatureCache.load_npz(caches_dir / f"{cid}.npz")
        if (i + 1) % 500 == 0 or (i + 1) == len(all_ids):
            print(f"[E0331] loaded {i+1}/{len(all_ids)} caches", flush=True)
    print(f"[E0331] cache preload done: {len(all_ids)} clips in {time.time() - t0:.1f}s", flush=True)

    clean_scores_by_clip: dict[str, list[float]] | None = None
    if scores_json is not None:
        clean_scores_by_clip = _load_scores_json(scores_json)
        missing = [cid for cid in all_ids if cid not in clean_scores_by_clip]
        if missing:
            raise ValueError(f"scores_json missing {len(missing)} ids (e.g. {missing[:3]}); re-run score cache first")

    # Stage-1 model for corrupted scoring (only needed for learned methods).
    av_model = None
    visual_side_by_clip: dict[str, np.ndarray] | None = None
    if str(eventness_method) == "av_clipdiff_mlp":
        av_model, visual_side_by_clip = _train_av_clipdiff_mlp(
            clip_ids_train=train_ids,
            clip_ids_all=all_ids,
            labels_by_clip=labels_by_clip,
            processed_dir=processed_dir,
            cache_by_clip=cache_by_clip,
            num_segments=int(num_segments),
        )
    elif str(eventness_method) in ("av_clipdiff_flow_mlp", "av_clipdiff_flow_mlp_stride"):
        av_model, visual_side_by_clip = _train_av_clipdiff_flow_mlp(
            clip_ids_train=train_ids,
            clip_ids_all=all_ids,
            labels_by_clip=labels_by_clip,
            processed_dir=processed_dir,
            cache_by_clip=cache_by_clip,
            num_segments=int(num_segments),
        )

    # Precompute fixed train/eval tensors for clean training.
    x_train_uniform, y_train_t, _ = _build_xy(
        clip_ids=train_ids,
        labels_by_clip=labels_by_clip,
        cache_by_clip=cache_by_clip,
        cfg=cfg,
        baseline="uniform",
        rng=random.Random(0),
        scores_by_clip=None,
        alpha=0.0,
        num_segments=int(num_segments),
        force_uniform_eval_only=True,
        split="train",
    )
    x_eval_uniform, y_eval_t, _ = _build_xy(
        clip_ids=eval_ids,
        labels_by_clip=labels_by_clip,
        cache_by_clip=cache_by_clip,
        cfg=cfg,
        baseline="uniform",
        rng=random.Random(0),
        scores_by_clip=None,
        alpha=0.0,
        num_segments=int(num_segments),
        force_uniform_eval_only=True,
        split="eval",
    )

    if clean_scores_by_clip is None:
        raise ValueError("scores_json is required for this runner (clean train anchors)")

    x_train_anchored_clean, _, meta_anchor_train = _build_xy(
        clip_ids=train_ids,
        labels_by_clip=labels_by_clip,
        cache_by_clip=cache_by_clip,
        cfg=cfg,
        baseline="anchored_top2",
        rng=random.Random(0),
        scores_by_clip=clean_scores_by_clip,
        alpha=0.0,
        num_segments=int(num_segments),
        force_uniform_eval_only=True,
        split="train",
    )
    x_eval_anchored_clean, _, meta_anchor_eval_clean = _build_xy(
        clip_ids=eval_ids,
        labels_by_clip=labels_by_clip,
        cache_by_clip=cache_by_clip,
        cfg=cfg,
        baseline="anchored_top2",
        rng=random.Random(0),
        scores_by_clip=clean_scores_by_clip,
        alpha=0.0,
        num_segments=int(num_segments),
        force_uniform_eval_only=True,
        split="eval",
    )

    # Train per-seed heads once on clean data.
    models_uniform: dict[int, torch.nn.Module] = {}
    models_random: dict[int, torch.nn.Module] = {}
    models_anchored: dict[int, torch.nn.Module] = {}

    acc_clean: dict[str, dict[str, list[float]]] = {"uniform": {}, "random_top2": {}, "anchored_top2": {}}

    for seed in seeds:
        torch.manual_seed(int(seed))
        mu = _build_model(cfg=cfg, in_dim=int(x_train_uniform.shape[-1]), num_classes=int(index.num_classes), device=str(train_device))
        _ = train_per_segment_classifier(model=mu, x_train=x_train_uniform, y_train=y_train_t, x_val=x_eval_uniform, y_val=y_eval_t, cfg=train_cfg)
        models_uniform[int(seed)] = mu
        acc_clean["uniform"][str(seed)] = [_eval_acc(mu, x_eval_uniform, y_eval_t)]

        # Random baseline: generated per seed.
        xr_tr, _, _ = _build_xy(
            clip_ids=train_ids,
            labels_by_clip=labels_by_clip,
            cache_by_clip=cache_by_clip,
            cfg=cfg,
            baseline="random_top2",
            rng=random.Random(int(seed)),
            scores_by_clip=clean_scores_by_clip,
            alpha=0.0,
            num_segments=int(num_segments),
            force_uniform_eval_only=True,
            split="train",
        )
        xr_ev, _, _ = _build_xy(
            clip_ids=eval_ids,
            labels_by_clip=labels_by_clip,
            cache_by_clip=cache_by_clip,
            cfg=cfg,
            baseline="random_top2",
            rng=random.Random(int(seed)),
            scores_by_clip=clean_scores_by_clip,
            alpha=0.0,
            num_segments=int(num_segments),
            force_uniform_eval_only=True,
            split="eval",
        )

        torch.manual_seed(int(seed))
        mr = _build_model(cfg=cfg, in_dim=int(xr_tr.shape[-1]), num_classes=int(index.num_classes), device=str(train_device))
        _ = train_per_segment_classifier(model=mr, x_train=xr_tr, y_train=y_train_t, x_val=xr_ev, y_val=y_eval_t, cfg=train_cfg)
        models_random[int(seed)] = mr
        acc_clean["random_top2"][str(seed)] = [_eval_acc(mr, xr_ev, y_eval_t)]

        torch.manual_seed(int(seed))
        ma = _build_model(cfg=cfg, in_dim=int(x_train_anchored_clean.shape[-1]), num_classes=int(index.num_classes), device=str(train_device))
        _ = train_per_segment_classifier(
            model=ma, x_train=x_train_anchored_clean, y_train=y_train_t, x_val=x_eval_anchored_clean, y_val=y_eval_t, cfg=train_cfg
        )
        models_anchored[int(seed)] = ma
        acc_clean["anchored_top2"][str(seed)] = [_eval_acc(ma, x_eval_anchored_clean, y_eval_t)]

    # Precompute eval tensors for random baseline under alpha (independent of corruption).
    random_eval_by_alpha: dict[str, dict[int, torch.Tensor]] = {}
    for alpha in alpha_grid:
        key = str(alpha)
        random_eval_by_alpha[key] = {}
        for seed in seeds:
            x_ev, _, _ = _build_xy(
                clip_ids=eval_ids,
                labels_by_clip=labels_by_clip,
                cache_by_clip=cache_by_clip,
                cfg=cfg,
                baseline="random_top2",
                rng=random.Random(int(seed)),
                scores_by_clip=clean_scores_by_clip,
                alpha=float(alpha),
                num_segments=int(num_segments),
                force_uniform_eval_only=True,
                split="eval",
            )
            random_eval_by_alpha[key][int(seed)] = x_ev

    rows: list[dict] = []
    for shift_s in shift_grid:
        for snr_db in snr_grid:
            for silence_ratio in silence_grid:
                cond_seed = int((float(shift_s) + 10.0) * 1000 + (float(snr_db) + 100.0) * 10 + float(silence_ratio) * 100)
                rng = np.random.default_rng(int(cond_seed))

                scores_eval_aug: dict[str, list[float]] = {}
                for i, cid in enumerate(eval_ids):
                    wav_path = processed_dir / cid / "audio.wav"
                    if str(eventness_method) == "energy":
                        scores = _scores_energy_augmented(
                            wav_path=wav_path,
                            shift_s=float(shift_s),
                            snr_db=float(snr_db),
                            silence_ratio=float(silence_ratio),
                            rng=rng,
                            num_segments=int(num_segments),
                        )
                    elif str(eventness_method) == "energy_delta":
                        scores = _scores_energy_delta_augmented(
                            wav_path=wav_path,
                            shift_s=float(shift_s),
                            snr_db=float(snr_db),
                            silence_ratio=float(silence_ratio),
                            rng=rng,
                            num_segments=int(num_segments),
                        )
                    elif str(eventness_method) == "energy_stride_max":
                        scores = _scores_energy_stride_max_augmented(
                            wav_path=wav_path,
                            shift_s=float(shift_s),
                            snr_db=float(snr_db),
                            silence_ratio=float(silence_ratio),
                            rng=rng,
                            num_segments=int(num_segments),
                        )
                    elif str(eventness_method) in ("av_clipdiff_mlp", "av_clipdiff_flow_mlp", "av_clipdiff_flow_mlp_stride"):
                        if av_model is None or visual_side_by_clip is None:
                            raise ValueError(f"{eventness_method} model not initialized")
                        vis = visual_side_by_clip.get(cid)
                        if vis is None:
                            raise ValueError(f"missing visual-side features for {cid}")
                        scores = _scores_av_clipdiff_mlp_augmented(
                            wav_path=wav_path,
                            shift_s=float(shift_s),
                            snr_db=float(snr_db),
                            silence_ratio=float(silence_ratio),
                            rng=rng,
                            num_segments=int(num_segments),
                            model=av_model,
                            visual_side=vis,
                            apply_stride_max=bool(str(eventness_method) == "av_clipdiff_flow_mlp_stride"),
                        )
                    else:
                        raise ValueError(f"unsupported eventness_method for E0331: {eventness_method}")
                    scores_eval_aug[str(cid)] = [float(x) for x in scores]
                    if (i + 1) % 200 == 0 or (i + 1) == len(eval_ids):
                        print(f"[E0331] scored {i+1}/{len(eval_ids)} eval clips (shift={shift_s}, snr={snr_db}, silence={silence_ratio})", flush=True)

                for alpha in alpha_grid:
                    scores_mix = {**{cid: clean_scores_by_clip[cid] for cid in train_ids}, **scores_eval_aug}
                    x_eval_anchored_aug, _, meta_aug = _build_xy(
                        clip_ids=eval_ids,
                        labels_by_clip=labels_by_clip,
                        cache_by_clip=cache_by_clip,
                        cfg=cfg,
                        baseline="anchored_top2",
                        rng=random.Random(0),
                        scores_by_clip=scores_mix,
                        alpha=float(alpha),
                        num_segments=int(num_segments),
                        force_uniform_eval_only=True,
                        split="eval",
                    )

                    # Downstream accuracy per seed (uniform + random + anchored).
                    u_acc: list[float] = []
                    r_acc: list[float] = []
                    a_acc: list[float] = []
                    for seed in seeds:
                        u_acc.append(_eval_acc(models_uniform[int(seed)], x_eval_uniform, y_eval_t))
                        r_acc.append(_eval_acc(models_random[int(seed)], random_eval_by_alpha[str(alpha)][int(seed)], y_eval_t))
                        a_acc.append(_eval_acc(models_anchored[int(seed)], x_eval_anchored_aug, y_eval_t))

                    def _summ(xs: list[float]) -> dict:
                        arr = np.asarray(xs, dtype=np.float32)
                        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0}

                    rows.append(
                        {
                            "shift_s": float(shift_s),
                            "snr_db": float(snr_db),
                            "silence_ratio": float(silence_ratio),
                            "alpha": float(alpha),
                            "acc": {
                                "uniform": _summ(u_acc),
                                "random_top2": _summ(r_acc),
                                "anchored_top2": _summ(a_acc),
                            },
                            "acc_delta": {
                                "anchored_minus_uniform": float(np.mean(np.asarray(a_acc, dtype=np.float32) - np.asarray(u_acc, dtype=np.float32))),
                                "anchored_minus_random": float(np.mean(np.asarray(a_acc, dtype=np.float32) - np.asarray(r_acc, dtype=np.float32))),
                            },
                            "stage1": meta_aug,
                        }
                    )

    out = {
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
        "scores_json": str(scores_json) if scores_json is not None else None,
        "seeds": [int(x) for x in seeds],
        "train_cfg": {"epochs": int(train_cfg.epochs), "batch_size": int(train_cfg.batch_size), "lr": float(train_cfg.lr), "weight_decay": float(train_cfg.weight_decay)},
        "train_device": str(train_device),
        "p0_cfg": json.loads(json.dumps(cfg_obj)),
        "grid": {"shift_s": shift_grid, "snr_db": snr_grid, "silence_ratio": silence_grid, "alpha": alpha_grid, "deltas": [0, 1, 2]},
        "clean": {
            "uniform": {"acc_by_seed": acc_clean["uniform"]},
            "random_top2": {"acc_by_seed": acc_clean["random_top2"]},
            "anchored_top2": {"acc_by_seed": acc_clean["anchored_top2"], "stage1_train": meta_anchor_train, "stage1_eval_clean": meta_anchor_eval_clean},
        },
        "rows": rows,
    }

    floor_rows: list[dict] = []
    for r in rows:
        alpha = float(r["alpha"])
        uniform = float(r["acc"]["uniform"]["mean"])
        anchored = float(r["acc"]["anchored_top2"]["mean"])
        floor = float(alpha * uniform)
        margin = float(anchored - floor)
        floor_rows.append(
            {
                "shift_s": float(r["shift_s"]),
                "snr_db": float(r["snr_db"]),
                "silence_ratio": float(r["silence_ratio"]),
                "alpha": alpha,
                "uniform_mean": uniform,
                "anchored_mean": anchored,
                "alpha_floor": floor,
                "margin": margin,
                "pass": bool(margin >= -1e-8),
            }
        )
    floor_pass = [x for x in floor_rows if bool(x["pass"])]
    floor_fail = [x for x in floor_rows if not bool(x["pass"])]
    worst_fail = sorted(floor_fail, key=lambda x: float(x["margin"]))[:5]
    out["alpha_floor_checks"] = {
        "rule": "anchored_top2_mean >= alpha * uniform_mean",
        "num_rows": int(len(floor_rows)),
        "num_pass": int(len(floor_pass)),
        "num_fail": int(len(floor_fail)),
        "min_margin": float(min((float(x["margin"]) for x in floor_rows), default=0.0)),
        "worst_failures": worst_fail,
    }

    # Plots (minimal oral-friendly heatmaps).
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: E402

        plots_dir = out_dir / "degradation_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Fast lookup by grid key.
        by_key: dict[tuple[float, float, float, float], dict] = {}
        for r in rows:
            by_key[(float(r["shift_s"]), float(r["snr_db"]), float(r["silence_ratio"]), float(r["alpha"]))] = r

        def _heatmap(
            *,
            value_fn,
            title: str,
            out_path: Path,
        ) -> None:
            fig, axes = plt.subplots(1, len(silence_grid), figsize=(4.2 * len(silence_grid), 3.2), squeeze=False)
            vmin = None
            vmax = None
            mats: list[np.ndarray] = []
            for sil in silence_grid:
                mat = np.full((len(shift_grid), len(snr_grid)), np.nan, dtype=np.float32)
                for i, sh in enumerate(shift_grid):
                    for j, snr in enumerate(snr_grid):
                        rr = by_key.get((float(sh), float(snr), float(sil), float(alpha)))
                        if rr is None:
                            continue
                        mat[i, j] = float(value_fn(rr))
                mats.append(mat)
                if np.isfinite(mat).any():
                    lo = float(np.nanmin(mat))
                    hi = float(np.nanmax(mat))
                    vmin = lo if vmin is None else float(min(float(vmin), lo))
                    vmax = hi if vmax is None else float(max(float(vmax), hi))

            for k, sil in enumerate(silence_grid):
                ax = axes[0][k]
                mat = mats[k]
                im = ax.imshow(
                    mat,
                    aspect="auto",
                    origin="lower",
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                )
                ax.set_title(f"silence={sil}")
                ax.set_xlabel("SNR(dB)")
                ax.set_ylabel("Shift(s)")
                ax.set_xticks(list(range(len(snr_grid))))
                ax.set_xticklabels([str(x) for x in snr_grid])
                ax.set_yticks(list(range(len(shift_grid))))
                ax.set_yticklabels([str(x) for x in shift_grid])
                fig.colorbar(im, ax=ax, shrink=0.85)

            fig.suptitle(title)
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

        for alpha in alpha_grid:
            a_tag = str(alpha).replace(".", "p")
            _heatmap(
                value_fn=lambda r: float(r["acc_delta"]["anchored_minus_uniform"]),
                title=f"Δacc (anchored - uniform), alpha={alpha}",
                out_path=plots_dir / f"delta_acc_alpha{a_tag}.png",
            )
            _heatmap(
                value_fn=lambda r: float(r["stage1"]["recall_by_delta"]["0"]),
                title=f"Recall@K,Δ0 (anchored), alpha={alpha}",
                out_path=plots_dir / f"recall_d0_alpha{a_tag}.png",
            )
    except Exception as e:
        # Plotting must never block experiment runs; record failure in the JSON for debugging.
        out["plots_error"] = str(e)

    out_json = out_dir / "degradation_accuracy.json"
    out_json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"out_dir": str(out_dir), "out_json": str(out_json), "num_rows": len(rows)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="E0331: Degradation suite with downstream accuracy + alpha lower bound.")
    p.add_argument("--mode", type=str, default="ave_official", choices=["ave_official"])
    p.add_argument("--out-dir", type=Path, default=Path("runs") / f"E0331_degradation_accuracy_{time.strftime('%Y%m%d-%H%M%S')}")

    p.add_argument("--meta-dir", type=Path, default=ave_paths().meta_dir)
    p.add_argument("--processed-dir", type=Path, default=ave_paths().processed_dir)
    p.add_argument("--caches-dir", type=Path, required=True)
    p.add_argument("--train-ids-file", type=Path, default=None)
    p.add_argument("--eval-ids-file", type=Path, default=None)
    p.add_argument("--split-train", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--split-eval", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-eval", type=int, default=None)
    p.add_argument("--allow-missing", action="store_true")

    p.add_argument(
        "--eventness-method",
        type=str,
        default="av_clipdiff_mlp",
        choices=["energy", "energy_delta", "energy_stride_max", "av_clipdiff_mlp", "av_clipdiff_flow_mlp", "av_clipdiff_flow_mlp_stride"],
    )
    p.add_argument("--base-config-json", type=Path, required=True, help="best_config.json from the current AVE-P0 winner (e.g. E0223).")
    p.add_argument("--scores-json", type=Path, required=True, help="Clean Stage-1 score cache (eventness_scores.json).")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--train-device", type=str, default="cuda:0")

    p.add_argument("--shift-grid", type=str, default="-0.5,0.0,0.5")
    p.add_argument("--snr-grid", type=str, default="20,10,0")
    p.add_argument("--silence-grid", type=str, default="0.0,0.5")
    p.add_argument("--alpha-grid", type=str, default="0.0,0.5,1.0")

    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.mode != "ave_official":
        raise SystemExit(f"unsupported mode: {args.mode}")

    rep = run_ave_official_degradation_accuracy(
        out_dir=args.out_dir,
        meta_dir=args.meta_dir,
        processed_dir=args.processed_dir,
        caches_dir=args.caches_dir,
        train_ids_file=args.train_ids_file,
        eval_ids_file=args.eval_ids_file,
        split_train=args.split_train,
        split_eval=args.split_eval,
        limit_train=args.limit_train,
        limit_eval=args.limit_eval,
        allow_missing=bool(args.allow_missing),
        eventness_method=str(args.eventness_method),
        base_config_json=args.base_config_json,
        scores_json=args.scores_json,
        seeds=_parse_csv_ints(args.seeds),
        train_cfg=TrainConfig(epochs=int(args.epochs), batch_size=int(args.batch_size), lr=float(args.lr), weight_decay=float(args.weight_decay)),
        train_device=str(args.train_device),
        shift_grid=_parse_csv_floats(args.shift_grid),
        snr_grid=_parse_csv_floats(args.snr_grid),
        silence_grid=_parse_csv_floats(args.silence_grid),
        alpha_grid=_parse_csv_floats(args.alpha_grid),
        num_segments=10,
    )
    print(rep.get("out_json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
