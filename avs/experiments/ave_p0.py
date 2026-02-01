from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from avs.audio.features import audio_features_per_second
from avs.audio.eventness import (
    anchors_from_scores,
    anchors_from_scores_with_debug,
    compute_eventness_wav_energy,
    compute_eventness_wav_energy_delta,
    topk_anchors,
)
from avs.audio.ast_probe import ASTEventnessProbe, ASTProbeConfig
from avs.datasets.ave import AVEIndex, ensure_ave_meta
from avs.datasets.layout import ave_paths
from avs.models.per_segment_mlp import PerSegmentMLP
from avs.models.temporal_conv import TemporalConvHead
from avs.sampling.plans import SamplingPlan, equal_token_budget_anchored_plan, equal_token_budget_anchored_plan_scored, uniform_plan
from avs.train.train_loop import TrainConfig, train_per_segment_classifier
from avs.vision.feature_cache import FeatureCache


@dataclass(frozen=True)
class P0Config:
    k: int = 2
    low_res: int = 112
    base_res: int = 224
    high_res: int = 448
    patch_size: int = 16
    max_high_anchors: int | None = None
    anchor_shift: int = 0
    anchor_std_threshold: float = 0.0
    anchor_select: str = "topk"  # topk|nms|nms_strong
    anchor_nms_radius: int = 1
    anchor_nms_strong_gap: float = 0.6
    anchor_window: int = 3
    anchor_smooth_window: int = 0
    anchor_smooth_mode: str = "mean"
    anchor_conf_metric: str | None = None  # std|top1_med|top12_gap|gini (None => legacy std_threshold)
    anchor_conf_threshold: float | None = None  # None => legacy std_threshold
    anchor_base_alloc: str = "distance"  # distance|score|farthest|mixed
    anchor_high_policy: str = "fixed"  # fixed|adaptive_v1
    anchor_high_adjacent_dist: int = 1
    anchor_high_gap_threshold: float = 0.0
    head: str = "mlp"  # "mlp" | "temporal_conv"
    head_hidden_dim: int = 128
    head_dropout: float = 0.0
    temporal_kernel_size: int = 3

    def token_budget(self, *, num_segments: int = 10) -> int:
        plan = equal_token_budget_anchored_plan(
            num_segments=num_segments,
            anchors=[],
            low_res=self.low_res,
            base_res=self.base_res,
            high_res=self.high_res,
            patch_size=self.patch_size,
        )
        return plan.total_tokens()


def _segments_from_labels(segment_labels: list[int]) -> list[int]:
    return [i for i, lab in enumerate(segment_labels) if int(lab) != 0]


def _top2_gap(scores: list[float] | None) -> float:
    if scores is None or len(scores) < 2:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    return float(scores[order[0]] - scores[order[1]])


def _max_high_anchors_for_clip(*, cfg: P0Config, anchors: list[int], scores: list[float] | None) -> int | None:
    if cfg.anchor_high_policy == "fixed":
        return cfg.max_high_anchors

    if cfg.anchor_high_policy == "adaptive_v1":
        if len(anchors) < 2:
            return cfg.max_high_anchors

        dist = abs(int(anchors[0]) - int(anchors[1]))
        gap = _top2_gap(scores)

        if dist <= int(cfg.anchor_high_adjacent_dist):
            return 1
        if float(cfg.anchor_high_gap_threshold) > 0.0 and gap >= float(cfg.anchor_high_gap_threshold):
            return 1
        return cfg.max_high_anchors

    raise ValueError(f"unknown anchor_high_policy: {cfg.anchor_high_policy}")


def _train_audio_basic_lr_eventness(
    *,
    clip_ids_train: list[str],
    labels_by_clip: dict[str, list[int]],
    audio_feats_by_clip: dict[str, np.ndarray],
    device: str = "cpu",
    epochs: int = 50,
    batch_size: int = 2048,
    lr: float = 2e-2,
) -> torch.nn.Module:
    """
    Train a tiny supervised audio-only eventness model on top of `audio_features_per_second(..., feature_set='basic')`.

    Target is binary: 1 if segment label != 0 else 0.
    Returns a 1-layer logistic regression model that outputs per-second logits (higher => more likely event).
    """
    import torch.nn as nn

    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    for cid in clip_ids_train:
        feats = audio_feats_by_clip[cid]  # [T, F]
        labs = np.asarray(labels_by_clip[cid], dtype=np.int64)
        y = (labs != 0).astype(np.float32)
        x_rows.append(feats)
        y_rows.append(y.reshape(-1, 1))

    x_np = np.concatenate(x_rows, axis=0).astype(np.float32, copy=False)
    y_np = np.concatenate(y_rows, axis=0).astype(np.float32, copy=False)

    x = torch.from_numpy(x_np).to(device=torch.device(device), dtype=torch.float32)
    y = torch.from_numpy(y_np).to(device=torch.device(device), dtype=torch.float32)

    torch.manual_seed(0)
    model = nn.Linear(int(x.shape[-1]), 1).to(torch.device(device))

    pos = float((y > 0.5).sum().item())
    neg = float((y <= 0.5).sum().item())
    pos_weight = torch.tensor([neg / max(1.0, pos)], device=torch.device(device), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr))

    n = int(x.shape[0])
    steps = max(1, (n + int(batch_size) - 1) // int(batch_size))
    for _epoch in range(int(epochs)):
        perm = torch.randperm(n, device=torch.device(device))
        for i in range(steps):
            idx = perm[i * int(batch_size) : (i + 1) * int(batch_size)]
            xb = x[idx]
            yb = y[idx]
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    return model


def _train_audio_basic_mlp_eventness(
    *,
    clip_ids_train: list[str],
    labels_by_clip: dict[str, list[int]],
    audio_feats_by_clip: dict[str, np.ndarray],
    device: str = "cpu",
    epochs: int = 50,
    batch_size: int = 2048,
    lr: float = 2e-3,
    hidden_dim: int = 64,
) -> torch.nn.Module:
    """
    Train a tiny supervised audio-only eventness model on top of `audio_features_per_second(..., feature_set='basic')`.

    Target is binary: 1 if segment label != 0 else 0.
    Returns a 2-layer MLP that outputs per-second logits (higher => more likely event).
    """
    import torch.nn as nn

    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    for cid in clip_ids_train:
        feats = audio_feats_by_clip[cid]  # [T, F]
        labs = np.asarray(labels_by_clip[cid], dtype=np.int64)
        y = (labs != 0).astype(np.float32)
        x_rows.append(feats)
        y_rows.append(y.reshape(-1, 1))

    x_np = np.concatenate(x_rows, axis=0).astype(np.float32, copy=False)
    y_np = np.concatenate(y_rows, axis=0).astype(np.float32, copy=False)

    x = torch.from_numpy(x_np).to(device=torch.device(device), dtype=torch.float32)
    y = torch.from_numpy(y_np).to(device=torch.device(device), dtype=torch.float32)

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(int(x.shape[-1]), int(hidden_dim)),
        nn.ReLU(),
        nn.Linear(int(hidden_dim), 1),
    ).to(torch.device(device))

    pos = float((y > 0.5).sum().item())
    neg = float((y <= 0.5).sum().item())
    pos_weight = torch.tensor([neg / max(1.0, pos)], device=torch.device(device), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr))

    n = int(x.shape[0])
    steps = max(1, (n + int(batch_size) - 1) // int(batch_size))
    for _epoch in range(int(epochs)):
        perm = torch.randperm(n, device=torch.device(device))
        for i in range(steps):
            idx = perm[i * int(batch_size) : (i + 1) * int(batch_size)]
            xb = x[idx]
            yb = y[idx]
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    return model


def _train_audio_basic_mlp_cls_eventness(
    *,
    clip_ids_train: list[str],
    labels_by_clip: dict[str, list[int]],
    audio_feats_by_clip: dict[str, np.ndarray],
    num_classes: int,
    device: str = "cpu",
    epochs: int = 30,
    batch_size: int = 2048,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    dropout: float = 0.0,
) -> torch.nn.Module:
    """
    Train a tiny supervised audio-only per-second classifier on top of
    `audio_features_per_second(..., feature_set='basic')`.

    Target is multi-class (AVE segment label): 0=background, >0=event class.

    The returned model outputs per-second logits over `num_classes`.
    Use `eventness = 1 - softmax(logits)[..., 0]` to derive a scalar score.
    """
    import torch.nn as nn

    if int(num_classes) <= 1:
        raise ValueError(f"num_classes must be > 1, got {num_classes}")

    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    for cid in clip_ids_train:
        feats = audio_feats_by_clip[cid]  # [T, F]
        labs = np.asarray(labels_by_clip[cid], dtype=np.int64)  # [T]
        x_rows.append(feats)
        y_rows.append(labs)

    x_np = np.concatenate(x_rows, axis=0).astype(np.float32, copy=False)
    y_np = np.concatenate(y_rows, axis=0).astype(np.int64, copy=False)
    if x_np.shape[0] != y_np.shape[0]:
        raise AssertionError(f"bug: x rows {x_np.shape[0]} != y rows {y_np.shape[0]}")

    x = torch.from_numpy(x_np).to(device=torch.device(device), dtype=torch.float32)
    y = torch.from_numpy(y_np).to(device=torch.device(device), dtype=torch.long)

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(int(x.shape[-1]), int(hidden_dim)),
        nn.ReLU(),
        nn.Dropout(p=float(dropout)),
        nn.Linear(int(hidden_dim), int(num_classes)),
    ).to(torch.device(device))

    # Class reweighting to combat background dominance (common in AVE).
    counts = np.bincount(y_np, minlength=int(num_classes)).astype(np.float32)
    total = float(counts.sum())
    weights = np.zeros((int(num_classes),), dtype=np.float32)
    nz = counts > 0
    weights[nz] = total / (float(num_classes) * counts[nz])
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device=torch.device(device), dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr))

    n = int(x.shape[0])
    steps = max(1, (n + int(batch_size) - 1) // int(batch_size))
    for _epoch in range(int(epochs)):
        perm = torch.randperm(n, device=torch.device(device))
        for i in range(steps):
            idx = perm[i * int(batch_size) : (i + 1) * int(batch_size)]
            xb = x[idx]
            yb = y[idx]
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    return model


def _plan_for_baseline(
    baseline: str,
    *,
    cfg: P0Config,
    num_segments: int,
    rng: random.Random,
    anchors: list[int] | None,
    oracle_segments: list[int] | None,
    scores: list[float] | None,
) -> SamplingPlan:
    if baseline == "uniform_low":
        return uniform_plan(num_segments=num_segments, resolution=cfg.low_res, patch_size=cfg.patch_size)

    if baseline in ("uniform", "audio_concat_uniform", "audio_feat_concat_uniform"):
        return uniform_plan(num_segments=num_segments, resolution=cfg.base_res, patch_size=cfg.patch_size)

    elif baseline == "random_top2":
        use_anchors = rng.sample(range(num_segments), k=min(cfg.k, num_segments))
    elif baseline in ("anchored_top2", "audio_concat_anchored_top2", "audio_feat_concat_anchored_top2"):
        use_anchors = list(anchors or [])[: cfg.k]
    elif baseline == "oracle_top2":
        use_anchors = list(oracle_segments or [])[: cfg.k]
    else:
        raise ValueError(f"unknown baseline: {baseline}")

    max_high_anchors: int | None = cfg.max_high_anchors
    if baseline in ("anchored_top2", "audio_concat_anchored_top2", "audio_feat_concat_anchored_top2"):
        max_high_anchors = _max_high_anchors_for_clip(cfg=cfg, anchors=use_anchors, scores=scores)

    alloc = str(cfg.anchor_base_alloc)
    use_scores_for_base: list[float] | None = None

    if alloc == "score":
        # Base allocation by score only makes sense when the baseline is audio-guided; avoid leaking
        # audio signal into random/oracle baselines (and avoid requiring scores for uniform/oracle).
        if baseline in ("anchored_top2", "audio_concat_anchored_top2", "audio_feat_concat_anchored_top2"):
            if scores is None:
                raise ValueError("anchor_base_alloc=score requires per-second scores to be provided")
            use_scores_for_base = scores
        else:
            alloc = "distance"

    # NOTE: Default `anchor_base_alloc=distance` keeps `anchored_top2` backward-compatible:
    # base segments are allocated by distance-to-anchor (not by score).
    return equal_token_budget_anchored_plan_scored(
        num_segments=num_segments,
        anchors=use_anchors,
        scores=use_scores_for_base,
        base_alloc=str(alloc),
        low_res=cfg.low_res,
        base_res=cfg.base_res,
        high_res=cfg.high_res,
        max_high_anchors=max_high_anchors,
        patch_size=cfg.patch_size,
    )


def features_from_cache(cache: FeatureCache, plan: SamplingPlan) -> np.ndarray:
    t = len(plan.resolutions)
    # Use the first resolution present to infer embedding dim.
    any_r = cache.resolutions[0]
    d = cache.features_by_resolution[any_r].shape[-1]
    out = np.empty((t, d), dtype=np.float32)
    for i, r in enumerate(plan.resolutions):
        out[i] = cache.features_by_resolution[int(r)][i]
    return out


def run_p0_from_caches(
    *,
    clip_ids_train: list[str],
    clip_ids_eval: list[str],
    labels_by_clip: dict[str, list[int]],
    caches_dir: Path,
    audio_dir: Path | None,
    cfg: P0Config,
    baselines: list[str],
    seeds: list[int],
    train_cfg: TrainConfig,
    train_device: str = "cpu",
    num_classes: int,
    num_segments: int = 10,
    eventness_method: str = "energy",
    audio_device: str = "cpu",
    ast_pretrained: bool = False,
    panns_random: bool = False,
    panns_checkpoint: Path | None = None,
    audiomae_random: bool = False,
    audiomae_checkpoint: Path | None = None,
) -> dict:
    results_by_seed: list[dict] = []
    train_device = str(train_device)
    all_ids = sorted(set(clip_ids_train + clip_ids_eval))

    ast_probe = None
    if eventness_method == "ast":
        ast_probe = ASTEventnessProbe(ASTProbeConfig(pretrained=ast_pretrained, device=str(audio_device)))

    panns_probe = None
    if eventness_method == "panns":
        from avs.audio.panns_probe import PANNsEventnessProbe, PANNsProbeConfig

        panns_probe = PANNsEventnessProbe(
            PANNsProbeConfig(pretrained=not panns_random, checkpoint_path=panns_checkpoint, device=str(audio_device))
        )

    audiomae_probe = None
    if eventness_method == "audiomae":
        from avs.audio.audiomae_probe import AudioMAEEventnessProbe, AudioMAEProbeConfig

        audiomae_probe = AudioMAEEventnessProbe(
            AudioMAEProbeConfig(
                pretrained=(audiomae_checkpoint is not None) and (not audiomae_random),
                checkpoint_path=audiomae_checkpoint if (audiomae_checkpoint is not None) and (not audiomae_random) else None,
                device=str(audio_device),
            )
        )

    audio_feats_basic_by_clip: dict[str, np.ndarray] | None = None
    if audio_dir is not None and any(b in baselines for b in ("audio_feat_concat_uniform", "audio_feat_concat_anchored_top2")):
        audio_feats_basic_by_clip = {}
        for cid in all_ids:
            wav_path = audio_dir / cid / "audio.wav"
            audio_feats_basic_by_clip[cid] = audio_features_per_second(wav_path, num_segments=num_segments, feature_set="basic")

    audio_eventness_feats_by_clip: dict[str, np.ndarray] | None = None
    if audio_dir is not None and eventness_method in (
        "audio_basic_lr",
        "audio_basic_mlp",
        "audio_basic_mlp_cls",
        "audio_basic_mlp_cls_target",
        "audio_fbank_mlp",
    ):
        if eventness_method.startswith("audio_basic"):
            if audio_feats_basic_by_clip is not None:
                audio_eventness_feats_by_clip = audio_feats_basic_by_clip
            else:
                audio_eventness_feats_by_clip = {}
                for cid in all_ids:
                    wav_path = audio_dir / cid / "audio.wav"
                    audio_eventness_feats_by_clip[cid] = audio_features_per_second(
                        wav_path, num_segments=num_segments, feature_set="basic"
                    )
        else:
            # audio_fbank_mlp
            audio_eventness_feats_by_clip = {}
            for cid in all_ids:
                wav_path = audio_dir / cid / "audio.wav"
                audio_eventness_feats_by_clip[cid] = audio_features_per_second(
                    wav_path, num_segments=num_segments, feature_set="fbank_stats"
                )

    scores_by_clip: dict[str, list[float]] | None = None
    if audio_dir is not None and any(
        b in baselines for b in ("anchored_top2", "audio_concat_anchored_top2", "audio_feat_concat_anchored_top2", "audio_concat_uniform")
    ):
        scores_by_clip = {}
        if eventness_method == "audio_basic_lr":
            if audio_eventness_feats_by_clip is None:
                raise ValueError("internal error: audio_eventness_feats_by_clip should be precomputed for audio_basic_lr")
            # Train a supervised audio eventness model on train split, then score train+eval clips.
            audio_model = _train_audio_basic_lr_eventness(
                clip_ids_train=clip_ids_train,
                labels_by_clip=labels_by_clip,
                audio_feats_by_clip=audio_eventness_feats_by_clip,
                device="cpu",
            )
            audio_model_cpu = audio_model.to(torch.device("cpu"))
            for cid in all_ids:
                feats = torch.from_numpy(audio_eventness_feats_by_clip[cid]).float()
                with torch.no_grad():
                    logits = audio_model_cpu(feats).squeeze(-1).numpy().astype("float32")
                scores_by_clip[cid] = [float(x) for x in logits.tolist()]
        elif eventness_method == "audio_basic_mlp":
            if audio_eventness_feats_by_clip is None:
                raise ValueError("internal error: audio_eventness_feats_by_clip should be precomputed for audio_basic_mlp")
            audio_model = _train_audio_basic_mlp_eventness(
                clip_ids_train=clip_ids_train,
                labels_by_clip=labels_by_clip,
                audio_feats_by_clip=audio_eventness_feats_by_clip,
                device="cpu",
            )
            audio_model_cpu = audio_model.to(torch.device("cpu"))
            for cid in all_ids:
                feats = torch.from_numpy(audio_eventness_feats_by_clip[cid]).float()
                with torch.no_grad():
                    logits = audio_model_cpu(feats).squeeze(-1)
                scores_by_clip[cid] = [float(x) for x in logits.detach().cpu().numpy().astype("float32").tolist()]
        elif eventness_method == "audio_fbank_mlp":
            if audio_eventness_feats_by_clip is None:
                raise ValueError("internal error: audio_eventness_feats_by_clip should be precomputed for audio_fbank_mlp")
            audio_model = _train_audio_basic_mlp_eventness(
                clip_ids_train=clip_ids_train,
                labels_by_clip=labels_by_clip,
                audio_feats_by_clip=audio_eventness_feats_by_clip,
                device="cpu",
                hidden_dim=128,
            )
            audio_model_cpu = audio_model.to(torch.device("cpu"))
            for cid in all_ids:
                feats = torch.from_numpy(audio_eventness_feats_by_clip[cid]).float()
                with torch.no_grad():
                    logits = audio_model_cpu(feats).squeeze(-1)
                scores_by_clip[cid] = [float(x) for x in logits.detach().cpu().numpy().astype("float32").tolist()]
        elif eventness_method in ("audio_basic_mlp_cls", "audio_basic_mlp_cls_target"):
            if audio_eventness_feats_by_clip is None:
                raise ValueError(
                    f"internal error: audio_eventness_feats_by_clip should be precomputed for {eventness_method}"
                )
            audio_model = _train_audio_basic_mlp_cls_eventness(
                clip_ids_train=clip_ids_train,
                labels_by_clip=labels_by_clip,
                audio_feats_by_clip=audio_eventness_feats_by_clip,
                num_classes=int(num_classes),
                device="cpu",
            )
            audio_model_cpu = audio_model.to(torch.device("cpu"))
            for cid in all_ids:
                feats = torch.from_numpy(audio_eventness_feats_by_clip[cid]).float()
                with torch.no_grad():
                    logits = audio_model_cpu(feats)
                    if eventness_method == "audio_basic_mlp_cls":
                        probs = torch.softmax(logits, dim=-1)
                        # Scalar eventness: P(non-background).
                        scores = 1.0 - probs[:, 0]
                    else:
                        # Class-conditional eventness:
                        # infer a clip-level event class (exclude background=0), then score per-second by that class.
                        clip_logits = logits.mean(dim=0)
                        if int(clip_logits.shape[0]) < 2:
                            raise ValueError(f"audio_basic_mlp_cls_target requires num_classes>=2, got {clip_logits.shape[0]}")
                        clip_logits = clip_logits.clone()
                        clip_logits[0] = float("-inf")
                        cls = int(torch.argmax(clip_logits).item())
                        scores = logits[:, cls]

                scores_by_clip[cid] = [float(x) for x in scores.detach().cpu().numpy().astype("float32").tolist()]
        else:
            for cid in all_ids:
                wav_path = audio_dir / cid / "audio.wav"
                if eventness_method == "energy":
                    ev = compute_eventness_wav_energy(wav_path, num_segments=num_segments)
                    scores_by_clip[cid] = [float(x) for x in ev.scores]
                elif eventness_method == "energy_delta":
                    ev = compute_eventness_wav_energy_delta(wav_path, num_segments=num_segments)
                    scores_by_clip[cid] = [float(x) for x in ev.scores]
                elif eventness_method == "ast":
                    assert ast_probe is not None
                    scores_by_clip[cid] = [float(x) for x in ast_probe.eventness_per_second(wav_path, num_segments=num_segments)]
                elif eventness_method == "panns":
                    assert panns_probe is not None
                    scores_by_clip[cid] = [float(x) for x in panns_probe.eventness_per_second(wav_path, num_segments=num_segments)]
                elif eventness_method == "audiomae":
                    assert audiomae_probe is not None
                    scores_by_clip[cid] = [float(x) for x in audiomae_probe.eventness_per_second(wav_path, num_segments=num_segments)]
                else:
                    raise ValueError(f"unsupported eventness_method: {eventness_method}")

    debug_eval = None
    anchors_by_clip: dict[str, list[int]] | None = None
    anchor_debug_by_clip: dict[str, dict] | None = None
    if scores_by_clip is not None:
        anchors_by_clip = {}
        anchor_debug_by_clip = {}
        for cid in all_ids:
            scores = scores_by_clip[cid]
            sel = anchors_from_scores_with_debug(
                scores,
                k=cfg.k,
                num_segments=num_segments,
                shift=cfg.anchor_shift,
                std_threshold=cfg.anchor_std_threshold,
                select=cfg.anchor_select,
                nms_radius=cfg.anchor_nms_radius,
                nms_strong_gap=cfg.anchor_nms_strong_gap,
                anchor_window=cfg.anchor_window,
                smooth_window=cfg.anchor_smooth_window,
                smooth_mode=cfg.anchor_smooth_mode,
                conf_metric=cfg.anchor_conf_metric,
                conf_threshold=cfg.anchor_conf_threshold,
            )
            anchors_by_clip[cid] = [int(x) for x in sel.anchors]
            anchor_debug_by_clip[cid] = {
                "fallback_used": bool(sel.fallback_used),
                "fallback_reason": sel.fallback_reason,
                "conf_metric": str(sel.conf_metric),
                "conf_value": float(sel.conf_value),
                "conf_threshold": float(sel.conf_threshold),
            }

    if audio_dir is not None and scores_by_clip is not None and "anchored_top2" in baselines:
        by_clip: dict[str, dict] = {}
        for cid in sorted(set(clip_ids_eval)):
            scores = scores_by_clip.get(cid)
            if scores is None:
                continue
            if anchors_by_clip is None or anchor_debug_by_clip is None:
                raise ValueError("internal error: anchors_by_clip should be precomputed")
            anchors = anchors_by_clip.get(cid) or []
            max_high_anchors = _max_high_anchors_for_clip(cfg=cfg, anchors=anchors, scores=scores)
            alloc = str(cfg.anchor_base_alloc)
            if alloc == "distance":
                plan_scores = None
            elif alloc == "score":
                plan_scores = scores
            elif alloc == "farthest":
                plan_scores = None
            elif alloc == "mixed":
                plan_scores = None
            else:
                raise ValueError(f"unknown anchor_base_alloc: {alloc!r}; expected 'distance', 'score', 'farthest', or 'mixed'")
            plan = equal_token_budget_anchored_plan_scored(
                num_segments=num_segments,
                anchors=anchors,
                scores=plan_scores,
                base_alloc=alloc,
                low_res=cfg.low_res,
                base_res=cfg.base_res,
                high_res=cfg.high_res,
                max_high_anchors=max_high_anchors,
                patch_size=cfg.patch_size,
            )
            by_clip[str(cid)] = {
                "scores": [float(x) for x in scores],
                "anchors": [int(x) for x in anchors],
                **(anchor_debug_by_clip.get(cid) or {}),
                "max_high_anchors": max_high_anchors,
                "plan_resolutions": plan.resolutions,
            }
        debug_eval = {"anchored_top2": by_clip}

    # Cache IO is the dominant cost in full runs. Preload caches once and reuse across
    # baselines/seeds so the sweep is compute-bound (GPU) instead of disk-bound.
    t0 = time.time()
    cache_by_clip: dict[str, FeatureCache] = {}
    for i, cid in enumerate(all_ids):
        cache_by_clip[cid] = FeatureCache.load_npz(caches_dir / f"{cid}.npz")
        if (i + 1) % 500 == 0 or (i + 1) == len(all_ids):
            print(f"[P0] loaded {i+1}/{len(all_ids)} caches", flush=True)
    print(f"[P0] cache preload done: {len(all_ids)} clips in {time.time() - t0:.1f}s", flush=True)

    # Labels are shared across all baselines.
    y_train_np = np.stack([np.asarray(labels_by_clip[cid], dtype=np.int64) for cid in clip_ids_train], axis=0)
    y_eval_np = np.stack([np.asarray(labels_by_clip[cid], dtype=np.int64) for cid in clip_ids_eval], axis=0)
    y_train_t = torch.from_numpy(y_train_np).long()
    y_eval_t = torch.from_numpy(y_eval_np).long()

    oracle_segments_by_clip = {cid: _segments_from_labels(labels_by_clip[cid]) for cid in all_ids}

    fixed_data: dict[str, dict[str, torch.Tensor]] = {}
    token_budget_by_baseline: dict[str, int] = {}
    fixed_rng = random.Random(0)
    for baseline in baselines:
        if baseline == "random_top2":
            continue

        x_train: list[np.ndarray] = []
        x_eval: list[np.ndarray] = []

        token_budget: int | None = None
        for split, clip_ids in (("train", clip_ids_train), ("eval", clip_ids_eval)):
            for cid in clip_ids:
                cache = cache_by_clip[cid]
                oracle_segments = oracle_segments_by_clip[cid]

                anchors = None
                if baseline in ("anchored_top2", "audio_concat_anchored_top2", "audio_feat_concat_anchored_top2"):
                    if audio_dir is None:
                        raise ValueError("audio_dir is required for anchored_top2")
                    if anchors_by_clip is None:
                        raise ValueError("internal error: anchors_by_clip should be precomputed")
                    anchors = anchors_by_clip[cid]

                scores = scores_by_clip[cid] if scores_by_clip is not None else None
                plan = _plan_for_baseline(
                    baseline,
                    cfg=cfg,
                    num_segments=num_segments,
                    rng=fixed_rng,
                    anchors=anchors,
                    oracle_segments=oracle_segments,
                    scores=scores,
                )
                if token_budget is None:
                    token_budget = int(plan.total_tokens())

                x = features_from_cache(cache, plan)
                if baseline in ("audio_concat_uniform", "audio_concat_anchored_top2"):
                    if audio_dir is None:
                        raise ValueError("audio_dir is required for audio_concat_uniform")
                    if scores_by_clip is None:
                        raise ValueError("internal error: scores_by_clip should be precomputed")
                    audio_feat = np.asarray(scores_by_clip[cid], dtype=np.float32).reshape(num_segments, 1)
                    x = np.concatenate([x, audio_feat], axis=-1)
                if baseline in ("audio_feat_concat_uniform", "audio_feat_concat_anchored_top2"):
                    if audio_dir is None:
                        raise ValueError("audio_dir is required for audio_feat_concat_*")
                    if audio_feats_basic_by_clip is None:
                        raise ValueError("internal error: audio_feats_basic_by_clip should be precomputed")
                    x = np.concatenate([x, audio_feats_basic_by_clip[cid]], axis=-1)

                if split == "train":
                    x_train.append(x)
                else:
                    x_eval.append(x)

        fixed_data[baseline] = {
            "x_train": torch.from_numpy(np.stack(x_train, axis=0)).float(),
            "x_eval": torch.from_numpy(np.stack(x_eval, axis=0)).float(),
        }
        token_budget_by_baseline[baseline] = int(token_budget if token_budget is not None else cfg.token_budget(num_segments=num_segments))

    for seed in seeds:
        rng = random.Random(seed)
        per_baseline: dict[str, dict] = {}

        for baseline in baselines:
            print(f"[P0] seed={seed} baseline={baseline}", flush=True)

            if baseline != "random_top2":
                xtr = fixed_data[baseline]["x_train"]
                xev = fixed_data[baseline]["x_eval"]
                ytr = y_train_t
                yev = y_eval_t
                token_budget = token_budget_by_baseline[baseline]
            else:
                x_train: list[np.ndarray] = []
                x_eval: list[np.ndarray] = []
                token_budget = None
                for split, clip_ids in (("train", clip_ids_train), ("eval", clip_ids_eval)):
                    for cid in clip_ids:
                        cache = cache_by_clip[cid]
                        oracle_segments = oracle_segments_by_clip[cid]
                        scores = scores_by_clip[cid] if scores_by_clip is not None else None
                        plan = _plan_for_baseline(
                            baseline,
                            cfg=cfg,
                            num_segments=num_segments,
                            rng=rng,
                            anchors=None,
                            oracle_segments=oracle_segments,
                            scores=scores,
                        )
                        if token_budget is None:
                            token_budget = int(plan.total_tokens())
                        x = features_from_cache(cache, plan)
                        if split == "train":
                            x_train.append(x)
                        else:
                            x_eval.append(x)
                xtr = torch.from_numpy(np.stack(x_train, axis=0)).float()
                xev = torch.from_numpy(np.stack(x_eval, axis=0)).float()
                ytr = y_train_t
                yev = y_eval_t
                token_budget = int(token_budget if token_budget is not None else cfg.token_budget(num_segments=num_segments))

            # Make baseline comparisons fair: same init/shuffle for all baselines within a seed.
            torch.manual_seed(int(seed))
            if cfg.head == "mlp":
                model = PerSegmentMLP(
                    in_dim=xtr.shape[-1],
                    num_classes=num_classes,
                    hidden_dim=int(cfg.head_hidden_dim),
                    dropout=float(cfg.head_dropout),
                ).to(torch.device(train_device))
            elif cfg.head == "temporal_conv":
                model = TemporalConvHead(
                    in_dim=xtr.shape[-1],
                    num_classes=num_classes,
                    hidden_dim=int(cfg.head_hidden_dim),
                    kernel_size=int(cfg.temporal_kernel_size),
                    dropout=float(cfg.head_dropout),
                ).to(torch.device(train_device))
            else:
                raise ValueError(f"unknown head: {cfg.head}")
            m = train_per_segment_classifier(model=model, x_train=xtr, y_train=ytr, x_val=xev, y_val=yev, cfg=train_cfg)

            per_baseline[baseline] = {
                "val_acc": float(m["val_acc"]),
                "val_acc_event": m.get("val_acc_event"),
                "val_acc_by_sample": m.get("val_acc_by_sample"),
                "val_acc_event_by_sample": m.get("val_acc_event_by_sample"),
                "token_budget": int(token_budget),
            }

        results_by_seed.append({"seed": seed, "baselines": per_baseline})

    # Aggregate mean/std.
    summary: dict[str, dict[str, float]] = {}
    for baseline in baselines:
        vals = [r["baselines"][baseline]["val_acc"] for r in results_by_seed]
        arr = np.asarray(vals, dtype=np.float64)
        summary[baseline] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(vals) > 1 else 0.0,
        }

    token_budget = cfg.token_budget(num_segments=num_segments)
    paired_ttest = None
    if len(seeds) > 1:
        try:
            # Avoid optional scipy dependency: compute paired t-test via StudentT CDF from torch.
            def _ttest_rel(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
                # t-test on (y - x)
                dx = torch.as_tensor(y - x, dtype=torch.float64)
                n = int(dx.numel())
                if n < 2:
                    return {"t": float("nan"), "p": float("nan")}
                mean = float(dx.mean().item())
                std = float(dx.std(unbiased=True).item())
                if std <= 0.0:
                    if mean == 0.0:
                        return {"t": 0.0, "p": 1.0}
                    return {"t": float("inf") if mean > 0.0 else float("-inf"), "p": 0.0}
                t = mean / (std / math.sqrt(float(n)))
                dist = torch.distributions.StudentT(df=float(n - 1))
                p = float(2.0 * (1.0 - float(dist.cdf(torch.tensor(abs(t), dtype=torch.float64)).item())))
                return {"t": float(t), "p": float(max(0.0, min(1.0, p)))}

            paired_ttest = {}

            if "anchored_top2" in baselines and "uniform" in baselines:
                anchored = np.asarray([r["baselines"]["anchored_top2"]["val_acc"] for r in results_by_seed], dtype=np.float64)
                uniform = np.asarray([r["baselines"]["uniform"]["val_acc"] for r in results_by_seed], dtype=np.float64)
                paired_ttest["anchored_vs_uniform"] = _ttest_rel(uniform, anchored)

                if "random_top2" in baselines:
                    random_top2 = np.asarray([r["baselines"]["random_top2"]["val_acc"] for r in results_by_seed], dtype=np.float64)
                    paired_ttest["anchored_vs_random"] = _ttest_rel(random_top2, anchored)

                if "audio_concat_uniform" in baselines:
                    audio_concat = np.asarray(
                        [r["baselines"]["audio_concat_uniform"]["val_acc"] for r in results_by_seed], dtype=np.float64
                    )
                    paired_ttest["anchored_vs_audio_concat_uniform"] = _ttest_rel(audio_concat, anchored)

                if "audio_feat_concat_uniform" in baselines:
                    audio_feat_concat = np.asarray(
                        [r["baselines"]["audio_feat_concat_uniform"]["val_acc"] for r in results_by_seed], dtype=np.float64
                    )
                    paired_ttest["anchored_vs_audio_feat_concat_uniform"] = _ttest_rel(audio_feat_concat, anchored)

            if "oracle_top2" in baselines and "uniform" in baselines:
                oracle = np.asarray([r["baselines"]["oracle_top2"]["val_acc"] for r in results_by_seed], dtype=np.float64)
                uniform = np.asarray([r["baselines"]["uniform"]["val_acc"] for r in results_by_seed], dtype=np.float64)
                paired_ttest["oracle_vs_uniform"] = _ttest_rel(uniform, oracle)

            if "audio_concat_anchored_top2" in baselines and "audio_concat_uniform" in baselines:
                audio_concat_anchored = np.asarray(
                    [r["baselines"]["audio_concat_anchored_top2"]["val_acc"] for r in results_by_seed], dtype=np.float64
                )
                audio_concat = np.asarray(
                    [r["baselines"]["audio_concat_uniform"]["val_acc"] for r in results_by_seed], dtype=np.float64
                )
                paired_ttest["audio_concat_anchored_vs_audio_concat_uniform"] = _ttest_rel(audio_concat, audio_concat_anchored)
                if "anchored_top2" in baselines:
                    anchored = np.asarray([r["baselines"]["anchored_top2"]["val_acc"] for r in results_by_seed], dtype=np.float64)
                    paired_ttest["audio_concat_anchored_vs_anchored"] = _ttest_rel(anchored, audio_concat_anchored)

            if "audio_feat_concat_anchored_top2" in baselines and "audio_feat_concat_uniform" in baselines:
                audio_feat_concat_anchored = np.asarray(
                    [r["baselines"]["audio_feat_concat_anchored_top2"]["val_acc"] for r in results_by_seed], dtype=np.float64
                )
                audio_feat_concat = np.asarray(
                    [r["baselines"]["audio_feat_concat_uniform"]["val_acc"] for r in results_by_seed], dtype=np.float64
                )
                paired_ttest["audio_feat_concat_anchored_vs_audio_feat_concat_uniform"] = _ttest_rel(
                    audio_feat_concat, audio_feat_concat_anchored
                )
                if "anchored_top2" in baselines:
                    anchored = np.asarray([r["baselines"]["anchored_top2"]["val_acc"] for r in results_by_seed], dtype=np.float64)
                    paired_ttest["audio_feat_concat_anchored_vs_anchored"] = _ttest_rel(anchored, audio_feat_concat_anchored)
        except Exception:
            paired_ttest = None

    out = {
        "baselines": baselines,
        "seeds": seeds,
        "token_budget": int(token_budget),
        "results_by_seed": results_by_seed,
        "summary": summary,
        "paired_ttest": paired_ttest,
    }
    if debug_eval is not None:
        out["debug_eval"] = debug_eval
        out["debug_eval_clip_ids"] = [str(x) for x in clip_ids_eval]
    return out


def _build_ave_labels(index: AVEIndex, split: str, limit: int | None) -> tuple[list[str], dict[str, list[int]]]:
    ids = index.splits[split]
    if limit is not None:
        ids = ids[:limit]

    clip_ids: list[str] = []
    labels_by_clip: dict[str, list[int]] = {}
    for idx in ids:
        clip = index.clips[int(idx)]
        seg_labels = index.segment_labels(clip)
        clip_ids.append(clip.video_id)
        labels_by_clip[clip.video_id] = [int(x) for x in seg_labels]
    return clip_ids, labels_by_clip


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AVE-P0 runner on cached features (equal token budget baselines).")
    p.add_argument("--caches-dir", type=Path, required=True, help="Dir containing <clip_id>.npz feature caches")
    p.add_argument("--processed-dir", type=Path, default=ave_paths().processed_dir)
    p.add_argument("--meta-dir", type=Path, default=ave_paths().meta_dir)
    p.add_argument("--split-train", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--split-eval", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-eval", type=int, default=None)
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--k", type=int, default=2, help="Top-K audio anchors to use (budget may cap effective K).")
    p.add_argument("--low-res", type=int, default=112)
    p.add_argument("--base-res", type=int, default=224)
    p.add_argument("--high-res", type=int, default=448)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument(
        "--max-high-anchors",
        type=int,
        default=None,
        help="Optional cap on how many anchors get high-res allocation (budget-aware). Default: use as many as budget allows.",
    )
    p.add_argument("--anchor-shift", type=int, default=0, help="Shift anchor indices by this many segments (A/V misalignment).")
    p.add_argument(
        "--anchor-std-threshold",
        type=float,
        default=0.0,
        help="If std(scores) < threshold, fall back to uniform sampling (anchored baseline). 0 disables.",
    )
    p.add_argument(
        "--anchor-select",
        type=str,
        default="topk",
        choices=["topk", "nms", "nms_strong", "window_topk"],
        help="Anchor selection strategy on per-second eventness scores.",
    )
    p.add_argument(
        "--anchor-window",
        type=int,
        default=3,
        help="For --anchor-select window_topk: window size for score aggregation (odd; e.g., 3 or 5).",
    )
    p.add_argument(
        "--anchor-smooth-window",
        type=int,
        default=0,
        help="Optional score smoothing window (odd). Applied before anchor selection. 0 disables.",
    )
    p.add_argument(
        "--anchor-smooth-mode",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help="For --anchor-smooth-window: how to aggregate scores inside the smoothing window.",
    )
    p.add_argument(
        "--anchor-nms-radius",
        type=int,
        default=1,
        help="For --anchor-select nms: suppress anchors within ±radius segments of a selected anchor.",
    )
    p.add_argument(
        "--anchor-nms-strong-gap",
        type=float,
        default=0.6,
        help="For --anchor-select nms_strong: accept a far anchor only if (top1_score - best_far_score) <= gap.",
    )
    p.add_argument(
        "--anchor-conf-metric",
        type=str,
        default=None,
        choices=["std", "top1_med", "top12_gap", "gini"],
        help="Anchor confidence metric. If set, uses --anchor-conf-threshold to decide fallback to uniform (replaces std-only fallback).",
    )
    p.add_argument(
        "--anchor-conf-threshold",
        type=float,
        default=None,
        help="For --anchor-conf-metric: if confidence < threshold, fall back to uniform (return empty anchors).",
    )
    p.add_argument(
        "--anchor-base-alloc",
        type=str,
        default="distance",
        choices=["distance", "score", "farthest", "mixed"],
        help="How to allocate base-res segments in the equal-budget anchored plan. distance=closest-to-anchor (legacy); score=highest eventness scores; farthest=farthest-from-anchor (preserve context); mixed=half near anchors + half far (context).",
    )
    p.add_argument(
        "--anchor-high-policy",
        type=str,
        default="fixed",
        choices=["fixed", "adaptive_v1"],
        help="How many anchors get high-res allocation: fixed uses --max-high-anchors; adaptive_v1 demotes the 2nd high anchor when anchors are adjacent or the 2nd peak is weak.",
    )
    p.add_argument(
        "--anchor-high-adjacent-dist",
        type=int,
        default=1,
        help="For --anchor-high-policy adaptive_v1: if top2 anchors are within this distance, allocate high-res to only 1 anchor.",
    )
    p.add_argument(
        "--anchor-high-gap-threshold",
        type=float,
        default=0.0,
        help="For --anchor-high-policy adaptive_v1: if (top1_score - top2_score) >= threshold, allocate high-res to only 1 anchor. 0 disables.",
    )
    p.add_argument("--head", type=str, default="mlp", choices=["mlp", "temporal_conv"])
    p.add_argument("--head-hidden-dim", type=int, default=128)
    p.add_argument("--head-dropout", type=float, default=0.0)
    p.add_argument("--temporal-kernel-size", type=int, default=3, help="Only for --head temporal_conv; must be odd.")
    p.add_argument("--train-device", type=str, default="cpu", help="Device for training the classifier head (cpu or cuda:<i>).")
    p.add_argument(
        "--eventness-method",
        type=str,
        default="energy",
        choices=[
            "energy",
            "energy_delta",
            "ast",
            "panns",
            "audiomae",
            "audio_basic_lr",
            "audio_basic_mlp",
            "audio_fbank_mlp",
            "audio_basic_mlp_cls",
            "audio_basic_mlp_cls_target",
        ],
    )
    p.add_argument("--audio-device", type=str, default="cpu", help="Device for audio probe inference (e.g., cuda:0).")
    p.add_argument("--ast-pretrained", action="store_true", help="Use pretrained AST weights (downloads from HF)")
    p.add_argument("--panns-checkpoint", type=Path, default=None, help="Path to PANNs Cnn14 checkpoint (.pth)")
    p.add_argument("--panns-random", action="store_true", help="Use random PANNs weights (no checkpoint; smoke/debug only)")
    p.add_argument("--audiomae-checkpoint", type=Path, default=None, help="Path to AudioMAE(-style) checkpoint (optional)")
    p.add_argument("--audiomae-random", action="store_true", help="Use random AudioMAE(-style) weights (no checkpoint; smoke/debug only)")
    p.add_argument("--out-dir", type=Path, default=Path("runs") / f"AVE_P0_{time.strftime('%Y%m%d-%H%M%S')}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_ave_meta(args.meta_dir)
    index = AVEIndex.from_meta_dir(args.meta_dir)

    train_ids, train_labels = _build_ave_labels(index, args.split_train, args.limit_train)
    eval_ids, eval_labels = _build_ave_labels(index, args.split_eval, args.limit_eval)
    labels_by_clip = {**train_labels, **eval_labels}

    seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip()]
    baselines = [
        "uniform",
        "uniform_low",
        "audio_concat_uniform",
        "audio_feat_concat_uniform",
        "random_top2",
        "anchored_top2",
        "audio_concat_anchored_top2",
        "audio_feat_concat_anchored_top2",
        "oracle_top2",
    ]

    metrics = run_p0_from_caches(
        clip_ids_train=train_ids,
        clip_ids_eval=eval_ids,
        labels_by_clip=labels_by_clip,
        caches_dir=args.caches_dir,
        audio_dir=args.processed_dir,
        cfg=P0Config(
            k=int(args.k),
            low_res=int(args.low_res),
            base_res=int(args.base_res),
            high_res=int(args.high_res),
            patch_size=int(args.patch_size),
            max_high_anchors=args.max_high_anchors,
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
            head=str(args.head),
            head_hidden_dim=int(args.head_hidden_dim),
            head_dropout=float(args.head_dropout),
            temporal_kernel_size=int(args.temporal_kernel_size),
        ),
        baselines=baselines,
        seeds=seeds,
        train_cfg=TrainConfig(epochs=5, batch_size=32, lr=2e-3),
        train_device=str(args.train_device),
        num_classes=index.num_classes,
        num_segments=10,
        eventness_method=str(args.eventness_method),
        audio_device=str(args.audio_device),
        ast_pretrained=bool(args.ast_pretrained),
        panns_random=bool(args.panns_random),
        panns_checkpoint=args.panns_checkpoint,
        audiomae_random=bool(args.audiomae_random),
        audiomae_checkpoint=args.audiomae_checkpoint,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
