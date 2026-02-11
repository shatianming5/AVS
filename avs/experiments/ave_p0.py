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
    AnchorSelectionResult,
    anchors_from_scores,
    anchors_from_scores_with_debug,
    compute_eventness_wav_energy,
    compute_eventness_wav_energy_delta,
    compute_eventness_wav_energy_stride_max,
    topk_anchors,
)
from avs.audio.ast_probe import ASTEventnessProbe, ASTProbeConfig
from avs.datasets.ave import AVEIndex, ensure_ave_meta
from avs.datasets.layout import ave_paths
from avs.models.per_segment_mlp import PerSegmentMLP
from avs.models.temporal_conv import TemporalConvHead
from avs.sampling.plans import (
    SamplingPlan,
    budget_band_anchored_plan_scored,
    equal_token_budget_anchored_plan_scored,
    uniform_plan,
)
from avs.sampling.token_budget import TokenBudget
from avs.train.train_loop import TrainConfig, train_per_segment_classifier
from avs.utils.scores import (
    AV_FUSED_SCORE_SCALE,
    best_shift_by_corr,
    fuse_max,
    fuse_prod,
    minmax_01,
    scale,
    shift_scores,
    stride_max_pool_per_second,
)
from avs.vision.feature_cache import FeatureCache


@dataclass(frozen=True)
class P0Config:
    k: int = 2
    low_res: int = 112
    base_res: int = 224
    high_res: int = 448
    patch_size: int = 16
    res_feature: str = "none"  # none|scalar (append per-segment resolution indicator to vision features)
    max_high_anchors: int | None = None
    anchor_shift: int = 0
    anchor_std_threshold: float = 0.0
    anchor_select: str = "topk"  # topk|nms|nms_strong|adjacent_top2|window_topk
    anchor_drop_far_dist: int = 0  # 0 disables; if >0 and dist(top1,top2) > threshold, drop anchor2
    anchor_fallback_far_dist: int = 0  # 0 disables; if >0 and dist(top1,top2) > threshold, force fallback to uniform
    anchor_fallback_mode: str = "uniform"  # uniform|cheap_visual_clipdiff|cheap_visual_framediff
    anchor_fallback_visual_conf_metric: str = "top1_med"  # confidence metric on the fallback visual scores
    anchor_fallback_visual_conf_threshold: float = 0.0  # 0 disables (always use visual fallback when available)
    anchor_nms_radius: int = 1
    anchor_nms_strong_gap: float = 0.6
    anchor_window: int = 3
    anchor_smooth_window: int = 0
    anchor_smooth_mode: str = "mean"
    anchor2_veto_method: str = "none"  # none|top2med_norm_v1|lr_v1
    anchor2_veto_threshold: float = 0.5
    anchor2_veto_label_radius: int = 1
    anchor_gate_method: str = "none"  # none|lr_top1hit_v1|lr_top1hit_all_v1
    anchor_gate_threshold: float = 0.0  # 0 disables
    anchor_gate_label_radius: int = 1
    anchor_conf_metric: str | None = None  # std|top1_med|top12_gap|gini (None => legacy std_threshold)
    anchor_conf_threshold: float | None = None  # None => legacy std_threshold
    anchor_base_alloc: str = "distance"  # distance|balanced|bridge|score|farthest|mixed (+ optional suffix "_high")
    anchor_high_policy: str = "fixed"  # fixed|adaptive_v1|adaptive_v2|adaptive_v3
    anchor_high_adjacent_dist: int = 1
    anchor_high_gap_threshold: float = 0.0
    anchor_high_conf_metric: str | None = None  # None => use anchor_conf_metric (or std)
    anchor_high_conf_threshold: float = 0.0  # 0 disables conf-based demotion
    head: str = "mlp"  # "mlp" | "temporal_conv"
    head_hidden_dim: int = 128
    head_dropout: float = 0.0
    temporal_kernel_size: int = 3
    triad_policy: str = "fixed"  # fixed|top1med_tiered_v1
    triad_alt_conf_threshold: float = 0.0  # 0 disables
    triad_alt_low_res: int = 112
    triad_alt_high_res: int = 448
    triad_alt_max_high_anchors: int | None = 1  # None => no extra cap beyond `max_high_anchors`
    budget_mode: str = "exact"  # exact|band
    budget_epsilon_frac: float = 0.01
    budget_extra_resolutions: tuple[int, ...] = ()

    def token_budget(self, *, num_segments: int = 10) -> int:
        budget = TokenBudget(patch_size=self.patch_size)
        return int(num_segments) * int(budget.tokens_for_resolution(int(self.base_res)))


def _segments_from_labels(segment_labels: list[int]) -> list[int]:
    return [i for i, lab in enumerate(segment_labels) if int(lab) != 0]


def _top2_gap(scores: list[float] | None) -> float:
    if scores is None or len(scores) < 2:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    return float(scores[order[0]] - scores[order[1]])


@dataclass(frozen=True)
class _Anchor2VetoLR:
    mean: np.ndarray  # [D]
    std: np.ndarray  # [D]
    weight: np.ndarray  # [D]
    bias: float


def _gini01(x01: np.ndarray) -> float:
    if x01.size == 0:
        return 0.0
    x = np.asarray(x01, dtype=np.float64)
    x = x - float(np.min(x))
    s = float(np.sum(x))
    if s <= 0.0:
        return 0.0
    x = np.sort(x)
    n = int(x.size)
    idx = np.arange(1, n + 1, dtype=np.float64)
    g = (2.0 * float(np.sum(idx * x)) / (float(n) * s)) - (float(n) + 1.0) / float(n)
    return float(max(0.0, min(1.0, g)))


def _anchor2_veto_features(scores: list[float], anchors: list[int]) -> np.ndarray:
    """
    Scale-free features for deciding whether to keep the 2nd anchor (k-adaptive).

    Uses per-clip min-max normalization to [0,1] so thresholds/models generalize across different
    Stage-1 score scales (BCE logits, MIL logits, etc.).
    """
    if not scores or len(scores) < 2 or len(anchors) < 2:
        return np.zeros((12,), dtype=np.float32)

    s = np.asarray(minmax_01([float(x) for x in scores]), dtype=np.float32)
    n = int(s.size)
    a1 = int(anchors[0])
    a2 = int(anchors[1])
    if not (0 <= a1 < n and 0 <= a2 < n):
        return np.zeros((12,), dtype=np.float32)

    order = sorted(range(n), key=lambda i: (-float(s[int(i)]), int(i)))
    top1 = float(s[a1])
    top2 = float(s[a2])
    top3 = float(s[int(order[2])]) if n >= 3 else float(0.0)
    med = float(np.median(s))
    mean = float(np.mean(s))
    std = float(np.std(s))
    gini = _gini01(s.astype(np.float64))
    gap12 = float(top1 - top2)
    gap23 = float(top2 - top3)
    dist12 = float(abs(int(a1) - int(a2))) / float(max(1, n - 1))

    def _local_mean(idx: int) -> float:
        lo = max(0, int(idx) - 1)
        hi = min(n, int(idx) + 2)
        return float(np.mean(s[lo:hi])) if hi > lo else float(s[int(idx)])

    loc1 = _local_mean(a1)
    loc2 = _local_mean(a2)
    top1_med = float(top1 - med)
    top2_med = float(top2 - med)

    return np.asarray(
        [
            top1,
            top2,
            gap12,
            gap23,
            top1_med,
            top2_med,
            std,
            gini,
            dist12,
            loc1,
            loc2,
            mean,
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class _AnchorGateLR:
    mean: np.ndarray  # [D]
    std: np.ndarray  # [D]
    weight: np.ndarray  # [D]
    bias: float


def _anchor_gate_features(scores: list[float], anchors: list[int], *, num_segments: int) -> np.ndarray:
    """
    Scale-free features for a clip-level gate that predicts whether the top-1 selected anchor is correct.

    Uses per-clip min-max normalization to [0,1] so the gate is robust across Stage-1 score scales.
    """
    nseg = int(num_segments)
    if nseg <= 0 or not scores:
        return np.zeros((16,), dtype=np.float32)

    s = np.asarray(minmax_01([float(x) for x in scores[:nseg]]), dtype=np.float32)
    n = int(s.size)
    if n <= 0:
        return np.zeros((16,), dtype=np.float32)

    order = sorted(range(n), key=lambda i: (-float(s[int(i)]), int(i)))
    a1 = int(anchors[0]) if anchors else int(order[0])
    if not (0 <= a1 < n):
        a1 = int(order[0])

    top1 = float(s[a1])
    top2 = float(s[int(order[1])]) if n >= 2 else float(0.0)
    top3 = float(s[int(order[2])]) if n >= 3 else float(0.0)
    bot1 = float(s[int(order[-1])]) if n >= 1 else float(0.0)
    bot2 = float(s[int(order[-2])]) if n >= 2 else float(0.0)
    med = float(np.median(s))
    mean = float(np.mean(s))
    std = float(np.std(s))
    gini = float(_gini01(s.astype(np.float64)))

    gap12 = float(top1 - top2)
    gap13 = float(top1 - top3)
    top1_med = float(top1 - med)
    top1_mean = float(top1 - mean)
    bot12_gap = float(bot2 - bot1)

    frac_above_med = float(np.mean((s > med).astype(np.float32)))
    frac_above_p80 = float(np.mean((s >= 0.8).astype(np.float32)))

    def _local_mean(idx: int) -> float:
        lo = max(0, int(idx) - 1)
        hi = min(n, int(idx) + 2)
        return float(np.mean(s[lo:hi])) if hi > lo else float(s[int(idx)])

    loc1 = _local_mean(a1)
    # Second-best location mean as a proxy for multi-peak ambiguity.
    a2 = int(order[1]) if n >= 2 else int(a1)
    loc2 = _local_mean(a2)
    dist12 = float(abs(int(a1) - int(a2))) / float(max(1, n - 1))

    return np.asarray(
        [
            top1,
            top2,
            top3,
            gap12,
            gap13,
            top1_med,
            top1_mean,
            std,
            gini,
            frac_above_med,
            frac_above_p80,
            loc1,
            loc2,
            dist12,
            bot1,
            bot12_gap,
        ],
        dtype=np.float32,
    )


def _train_anchor_gate_lr_top1hit_v1(
    *,
    clip_ids_train: list[str],
    labels_by_clip: dict[str, list[int]],
    scores_by_clip: dict[str, list[float]],
    sel_by_clip: dict[str, "AnchorSelectionResult"],
    cfg: P0Config,
    num_segments: int,
) -> _AnchorGateLR | None:
    """
    Train a tiny clip-level gate to predict whether the top-1 selected anchor hits an event second.

    Label: 1 if `anchor1` overlaps any positive segment within `anchor_gate_label_radius`, else 0.

    This gate is intended as a **rescue** mechanism for broad/multi-second events where `top1_med` is low
    (causing heavy fallback), but the selected anchor is still usually correct.
    """
    xs: list[np.ndarray] = []
    ys: list[int] = []
    rad = max(0, int(cfg.anchor_gate_label_radius))
    base_thr = float(cfg.anchor_conf_threshold) if cfg.anchor_conf_threshold is not None else float(cfg.anchor_std_threshold)
    if float(base_thr) <= 0.0:
        # No base gate => nothing to "rescue".
        return None

    for cid in clip_ids_train:
        sel = sel_by_clip.get(cid)
        if sel is None:
            continue
        # Train the rescue gate only on clips that would fall back under the base gate.
        if float(sel.conf_value) >= float(base_thr):
            continue
        anchors = [int(x) for x in (sel.anchors or [])]
        if not anchors:
            continue
        scores = scores_by_clip.get(cid)
        labs = labels_by_clip.get(cid)
        if scores is None or labs is None:
            continue
        if len(labs) < int(num_segments):
            continue

        a1 = int(anchors[0])
        if not (0 <= a1 < int(num_segments)):
            continue

        lo = max(0, int(a1) - int(rad))
        hi = min(int(num_segments), int(a1) + int(rad) + 1)
        y = 1 if any(int(labs[t]) != 0 for t in range(int(lo), int(hi))) else 0
        x = _anchor_gate_features(scores, anchors, num_segments=int(num_segments))
        xs.append(x)
        ys.append(int(y))

    if not xs:
        return None

    x = torch.from_numpy(np.stack(xs, axis=0)).float()
    y = torch.from_numpy(np.asarray(ys, dtype=np.float32).reshape(-1, 1)).float()

    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    std = torch.clamp(std, min=1e-6)
    x = (x - mean) / std

    import torch.nn as nn

    torch.manual_seed(0)
    model = nn.Linear(int(x.shape[1]), 1)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)

    pos = float((y > 0.5).sum().item())
    neg = float((y <= 0.5).sum().item())
    pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-2, weight_decay=1e-3)

    epochs = 200
    for _epoch in range(int(epochs)):
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        prob = torch.sigmoid(model(x))
        pred = (prob >= 0.5).float()
        acc = float((pred == y).float().mean().item())

    w = model.weight.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
    b = float(model.bias.detach().cpu().numpy().reshape(-1)[0])
    print(
        f"[anchor_gate] trained lr_top1hit_v1(rescue): n={len(xs)} pos={int(pos)} neg={int(neg)} "
        f"acc={acc:.3f} rad={rad} base_thr={base_thr}",
        flush=True,
    )
    return _AnchorGateLR(mean=mean.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False), std=std.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False), weight=w, bias=float(b))


def _train_anchor_gate_lr_top1hit_all_v1(
    *,
    clip_ids_train: list[str],
    labels_by_clip: dict[str, list[int]],
    scores_by_clip: dict[str, list[float]],
    sel_by_clip: dict[str, "AnchorSelectionResult"],
    cfg: P0Config,
    num_segments: int,
) -> _AnchorGateLR | None:
    """
    Train a tiny clip-level gate to predict whether the top-1 selected anchor hits an event second.

    Label: 1 if `anchor1` overlaps any positive segment within `anchor_gate_label_radius`, else 0.

    Unlike `lr_top1hit_v1` (rescue-only), this trains on **all** train clips and is intended to be used
    as a *veto* on top of the base confidence gate: accept clips only when (base_pass AND gate_pass).
    """
    xs: list[np.ndarray] = []
    ys: list[int] = []
    rad = max(0, int(cfg.anchor_gate_label_radius))

    for cid in clip_ids_train:
        sel = sel_by_clip.get(cid)
        if sel is None:
            continue
        anchors = [int(x) for x in (sel.anchors or [])]
        if not anchors:
            continue
        scores = scores_by_clip.get(cid)
        labs = labels_by_clip.get(cid)
        if scores is None or labs is None:
            continue
        if len(labs) < int(num_segments):
            continue

        a1 = int(anchors[0])
        if not (0 <= a1 < int(num_segments)):
            continue

        lo = max(0, int(a1) - int(rad))
        hi = min(int(num_segments), int(a1) + int(rad) + 1)
        y = 1 if any(int(labs[t]) != 0 for t in range(int(lo), int(hi))) else 0
        x = _anchor_gate_features(scores, anchors, num_segments=int(num_segments))
        xs.append(x)
        ys.append(int(y))

    if len(xs) < 64:
        print(f"[anchor_gate] skip lr_top1hit_all_v1: only {len(xs)} train samples (need >=64)", flush=True)
        return None

    x = torch.from_numpy(np.stack(xs, axis=0)).float()
    y = torch.from_numpy(np.asarray(ys, dtype=np.float32).reshape(-1, 1)).float()

    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    std = torch.clamp(std, min=1e-6)
    x = (x - mean) / std

    import torch.nn as nn

    torch.manual_seed(0)
    model = nn.Linear(int(x.shape[1]), 1)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)

    pos = float((y > 0.5).sum().item())
    neg = float((y <= 0.5).sum().item())
    pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-2, weight_decay=1e-3)

    epochs = 200
    for _epoch in range(int(epochs)):
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        prob = torch.sigmoid(model(x))
        pred = (prob >= 0.5).float()
        acc = float((pred == y).float().mean().item())

    w = model.weight.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
    b = float(model.bias.detach().cpu().numpy().reshape(-1)[0])
    print(
        f"[anchor_gate] trained lr_top1hit_all_v1(veto): n={len(xs)} pos={int(pos)} neg={int(neg)} acc={acc:.3f} rad={rad}",
        flush=True,
    )
    return _AnchorGateLR(
        mean=mean.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False),
        std=std.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False),
        weight=w,
        bias=float(b),
    )


def _anchor_gate_prob_lr(model: _AnchorGateLR, *, scores: list[float], anchors: list[int], num_segments: int) -> float:
    x = _anchor_gate_features(scores, anchors, num_segments=int(num_segments)).astype(np.float32, copy=False)
    x = (x - model.mean) / model.std
    logit = float(np.dot(x.astype(np.float64), model.weight.astype(np.float64)) + float(model.bias))
    return float(1.0 / (1.0 + math.exp(-logit)))


def _train_anchor2_veto_lr(
    *,
    clip_ids_train: list[str],
    labels_by_clip: dict[str, list[int]],
    scores_by_clip: dict[str, list[float]],
    sel_by_clip: dict[str, "AnchorSelectionResult"],
    cfg: P0Config,
    num_segments: int,
) -> _Anchor2VetoLR | None:
    """
    Train a tiny clip-level gate to predict whether the 2nd selected anchor is a true event second.

    Label: 1 if `anchor2` overlaps any positive segment within `anchor2_veto_label_radius`, else 0.
    This targets the failure mode where a spurious 2nd peak wastes budget/context and hurts transfer.
    """
    xs: list[np.ndarray] = []
    ys: list[int] = []
    rad = max(0, int(cfg.anchor2_veto_label_radius))

    for cid in clip_ids_train:
        sel = sel_by_clip.get(cid)
        if sel is None or bool(sel.fallback_used):
            continue
        anchors = [int(x) for x in (sel.anchors or [])]
        if len(anchors) < 2:
            continue
        scores = scores_by_clip.get(cid)
        labs = labels_by_clip.get(cid)
        if scores is None or labs is None:
            continue
        if len(labs) < int(num_segments):
            continue

        a2 = int(anchors[1])
        if not (0 <= a2 < int(num_segments)):
            continue

        lo = max(0, int(a2) - int(rad))
        hi = min(int(num_segments), int(a2) + int(rad) + 1)
        y = 1 if any(int(labs[t]) != 0 for t in range(int(lo), int(hi))) else 0

        xs.append(_anchor2_veto_features(scores, anchors))
        ys.append(int(y))

    if len(xs) < 64:
        print(f"[anchor2_veto] skip lr_v1: only {len(xs)} train samples (need >=64)", flush=True)
        return None

    x_np = np.stack(xs, axis=0).astype(np.float32, copy=False)
    y_np = np.asarray(ys, dtype=np.float32).reshape(-1, 1)

    mean = x_np.mean(axis=0)
    std = x_np.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32, copy=False)
    x_np = (x_np - mean) / std

    x = torch.from_numpy(x_np).to(dtype=torch.float32)
    y = torch.from_numpy(y_np).to(dtype=torch.float32)

    import torch.nn as nn

    torch.manual_seed(0)
    model = nn.Linear(int(x.shape[1]), 1)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)

    pos = float((y > 0.5).sum().item())
    neg = float((y <= 0.5).sum().item())
    pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-2, weight_decay=1e-3)

    epochs = 200
    for _epoch in range(int(epochs)):
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        prob = torch.sigmoid(model(x))
        pred = (prob >= 0.5).float()
        acc = float((pred == y).float().mean().item())

    w = model.weight.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
    b = float(model.bias.detach().cpu().numpy().reshape(-1)[0])
    print(
        f"[anchor2_veto] trained lr_v1: n={len(xs)} pos={int(pos)} neg={int(neg)} acc={acc:.3f} rad={rad}",
        flush=True,
    )
    return _Anchor2VetoLR(mean=mean.astype(np.float32, copy=False), std=std, weight=w, bias=float(b))


def _anchor2_keep_prob_lr(model: _Anchor2VetoLR, *, scores: list[float], anchors: list[int]) -> float:
    x = _anchor2_veto_features(scores, anchors).astype(np.float32, copy=False)
    x = (x - model.mean) / model.std
    logit = float(np.dot(x.astype(np.float64), model.weight.astype(np.float64)) + float(model.bias))
    return float(1.0 / (1.0 + math.exp(-logit)))


def _confidence_value_for_scores(*, scores: list[float] | None, metric: str | None) -> float:
    if not scores:
        return 0.0
    from avs.audio import eventness as _ev

    m = str(metric or "std")
    if m == "std":
        return float(_ev.confidence_std(scores))
    if m == "std_norm":
        return float(_ev.confidence_std_norm(scores))
    if m == "top1_med":
        return float(_ev.confidence_top1_minus_median(scores))
    if m == "top1_med_norm":
        return float(_ev.confidence_top1_minus_median_norm(scores))
    if m == "top12_gap":
        return float(_ev.confidence_top12_gap(scores))
    if m == "top12_gap_norm":
        return float(_ev.confidence_top12_gap_norm(scores))
    if m == "top3_bottom3_gap_norm":
        return float(_ev.confidence_top3_bottom3_gap_norm(scores))
    if m == "gini":
        return float(_ev.confidence_gini(scores))
    raise ValueError(
        f"unknown conf_metric: {m!r}; expected 'std', 'std_norm', 'top1_med', 'top1_med_norm', "
        "'top12_gap', 'top12_gap_norm', 'top3_bottom3_gap_norm', or 'gini'"
    )


def _resolve_anchored_triad_for_clip(
    *,
    cfg: P0Config,
    scores: list[float] | None,
    max_high_anchors: int | None,
) -> tuple[int, int, int | None, bool]:
    """
    Resolve (low_res, high_res, max_high_anchors, alt_used) for anchored baselines under a triad policy.

    - Always preserves the equal-token budget against `base_res` (enforced by plan generator).
    - Backward-compatible default: `triad_policy="fixed"` keeps (cfg.low_res, cfg.high_res) unchanged.
    """
    policy = str(cfg.triad_policy)
    if policy == "fixed":
        return int(cfg.low_res), int(cfg.high_res), max_high_anchors, False
    if policy != "top1med_tiered_v1":
        raise ValueError(f"unknown triad_policy={policy!r}; expected 'fixed' or 'top1med_tiered_v1'")

    thr = float(cfg.triad_alt_conf_threshold)
    if thr <= 0.0:
        return int(cfg.low_res), int(cfg.high_res), max_high_anchors, False

    conf = _confidence_value_for_scores(scores=scores, metric=cfg.anchor_conf_metric)
    if float(conf) < float(thr):
        return int(cfg.low_res), int(cfg.high_res), max_high_anchors, False

    alt_low = int(cfg.triad_alt_low_res)
    alt_high = int(cfg.triad_alt_high_res)
    cap = cfg.triad_alt_max_high_anchors
    if cap is not None:
        max_high_anchors = int(cap) if max_high_anchors is None else min(int(max_high_anchors), int(cap))
    return alt_low, alt_high, max_high_anchors, True


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

    if cfg.anchor_high_policy == "adaptive_v2":
        if len(anchors) < 2:
            return cfg.max_high_anchors

        dist = abs(int(anchors[0]) - int(anchors[1]))
        gap = _top2_gap(scores)

        if dist <= int(cfg.anchor_high_adjacent_dist):
            return 1

        # Confidence-based demotion: for "medium confidence" clips, keep anchors but limit to 1 high-res anchor
        # to preserve context (more base_res seconds instead of low_res).
        if float(cfg.anchor_high_conf_threshold) > 0.0 and scores is not None:
            from avs.audio import eventness as _ev

            metric = cfg.anchor_high_conf_metric or cfg.anchor_conf_metric or "std"
            metric = str(metric)
            if metric == "std":
                conf = _ev.confidence_std(scores)
            elif metric == "top1_med":
                conf = _ev.confidence_top1_minus_median(scores)
            elif metric == "top1_med_norm":
                conf = _ev.confidence_top1_minus_median_norm(scores)
            elif metric == "top12_gap":
                conf = _ev.confidence_top12_gap(scores)
            elif metric == "top12_gap_norm":
                conf = _ev.confidence_top12_gap_norm(scores)
            elif metric == "top3_bottom3_gap_norm":
                conf = _ev.confidence_top3_bottom3_gap_norm(scores)
            elif metric == "gini":
                conf = _ev.confidence_gini(scores)
            else:
                raise ValueError(
                    f"unknown anchor_high_conf_metric: {metric!r}; expected 'std', 'top1_med', 'top1_med_norm', "
                    "'top12_gap', 'top12_gap_norm', 'top3_bottom3_gap_norm', or 'gini'"
                )
            if float(conf) < float(cfg.anchor_high_conf_threshold):
                return 1

        if float(cfg.anchor_high_gap_threshold) > 0.0 and gap >= float(cfg.anchor_high_gap_threshold):
            return 1
        return cfg.max_high_anchors

    if cfg.anchor_high_policy == "adaptive_v3":
        """
        Keep-when-adjacent policy (inverse of adaptive_v1).

        Motivation (AVE/P0): diagnostics show the 2-high regime can be harmful when the two selected anchors
        are far apart (context loss), but can be beneficial when anchors are adjacent (multi-second evidence).

        Behavior:
          - If anchors are farther than `anchor_high_adjacent_dist`, demote to 1 high-res anchor (but keep both
            anchors for base allocation).
          - Optional confidence/gap demotion matches adaptive_v2 to avoid wasting high-res on weak 2nd peaks.
        """
        if len(anchors) < 2:
            return cfg.max_high_anchors

        dist = abs(int(anchors[0]) - int(anchors[1]))
        gap = _top2_gap(scores)

        # Only allow 2-high when anchors are within the adjacency threshold.
        if dist > int(cfg.anchor_high_adjacent_dist):
            return 1

        # Optional confidence-based demotion (same as adaptive_v2).
        if float(cfg.anchor_high_conf_threshold) > 0.0 and scores is not None:
            from avs.audio import eventness as _ev

            metric = cfg.anchor_high_conf_metric or cfg.anchor_conf_metric or "std"
            metric = str(metric)
            if metric == "std":
                conf = _ev.confidence_std(scores)
            elif metric == "top1_med":
                conf = _ev.confidence_top1_minus_median(scores)
            elif metric == "top1_med_norm":
                conf = _ev.confidence_top1_minus_median_norm(scores)
            elif metric == "top12_gap":
                conf = _ev.confidence_top12_gap(scores)
            elif metric == "top12_gap_norm":
                conf = _ev.confidence_top12_gap_norm(scores)
            elif metric == "top3_bottom3_gap_norm":
                conf = _ev.confidence_top3_bottom3_gap_norm(scores)
            elif metric == "gini":
                conf = _ev.confidence_gini(scores)
            else:
                raise ValueError(
                    f"unknown anchor_high_conf_metric: {metric!r}; expected 'std', 'top1_med', 'top1_med_norm', "
                    "'top12_gap', 'top12_gap_norm', 'top3_bottom3_gap_norm', or 'gini'"
                )
            if float(conf) < float(cfg.anchor_high_conf_threshold):
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


def _train_audio_tcn_eventness(
    *,
    clip_ids_train: list[str],
    labels_by_clip: dict[str, list[int]],
    audio_feats_by_clip: dict[str, np.ndarray],
    device: str = "cpu",
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden_channels: int = 64,
    kernel_size: int = 3,
    dropout: float = 0.1,
) -> torch.nn.Module:
    """
    Train a tiny temporal conv net (TCN-like) for per-second eventness.

    - Input: per-second audio features [T, F]
    - Target: binary eventness per second (label != 0)
    - Output: per-second logits [T, 1]

    Compared to per-second LR/MLP, this can use temporal context to improve anchor stability.
    """
    import torch.nn as nn

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for cid in clip_ids_train:
        feats = audio_feats_by_clip[cid]  # [T, F]
        labs = np.asarray(labels_by_clip[cid], dtype=np.int64)
        y = (labs != 0).astype(np.float32)
        xs.append(feats.astype(np.float32, copy=False))
        ys.append(y.astype(np.float32, copy=False))

    x_np = np.stack(xs, axis=0)  # [N, T, F]
    y_np = np.stack(ys, axis=0)  # [N, T]

    x = torch.from_numpy(x_np).to(device=torch.device(device), dtype=torch.float32)
    y = torch.from_numpy(y_np).to(device=torch.device(device), dtype=torch.float32).unsqueeze(-1)  # [N, T, 1]

    in_dim = int(x.shape[-1])
    ks = int(kernel_size)
    if ks % 2 != 1:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")

    class _AudioTCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_dim, int(hidden_channels), kernel_size=ks, padding=ks // 2),
                nn.ReLU(),
                nn.Dropout(p=float(dropout)),
                nn.Conv1d(int(hidden_channels), int(hidden_channels), kernel_size=ks, padding=ks // 2),
                nn.ReLU(),
                nn.Dropout(p=float(dropout)),
                nn.Conv1d(int(hidden_channels), 1, kernel_size=1),
            )

        def forward(self, feats: torch.Tensor) -> torch.Tensor:
            # feats: [B, T, F] or [T, F]
            squeeze = False
            if feats.ndim == 2:
                feats = feats.unsqueeze(0)
                squeeze = True
            if feats.ndim != 3:
                raise ValueError(f"expected feats with shape [B,T,F] or [T,F], got {tuple(feats.shape)}")
            x1 = feats.transpose(1, 2)  # [B, F, T]
            y1 = self.net(x1).transpose(1, 2)  # [B, T, 1]
            return y1.squeeze(0) if squeeze else y1

    torch.manual_seed(0)
    model = _AudioTCN().to(torch.device(device))

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
            xb = x[idx]  # [B,T,F]
            yb = y[idx]  # [B,T,1]
            logits = model(xb)  # [B,T,1]
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    return model


def _train_ast_lr_eventness(
    *,
    clip_ids_train: list[str],
    labels_by_clip: dict[str, list[int]],
    audio_dir: Path,
    ast_probe: ASTEventnessProbe,
    num_segments: int = 10,
    device: str = "cpu",
    epochs: int = 30,
    batch_size: int = 4096,
    lr: float = 2e-2,
) -> tuple[torch.nn.Module, dict[str, np.ndarray]]:
    """
    Train a tiny supervised eventness model on top of pretrained AST per-second logits.

    Target is binary: 1 if segment label != 0 else 0.
    Returns:
      - a 1-layer logistic regression (per-second logits)
      - cached AST logits for train clips (to reuse during scoring)
    """
    import torch.nn as nn

    train_ids = [str(x) for x in clip_ids_train]
    feats_by_train: dict[str, np.ndarray] = {}
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []

    for i, cid in enumerate(train_ids):
        wav_path = audio_dir / cid / "audio.wav"
        feats = ast_probe.logits_per_second(wav_path, num_segments=int(num_segments))  # [T, C]
        feats_by_train[cid] = feats

        labs = np.asarray(labels_by_clip[cid], dtype=np.int64)[: int(num_segments)]
        y = (labs != 0).astype(np.float32).reshape(-1, 1)

        x_rows.append(feats.astype(np.float32, copy=False))
        y_rows.append(y)

        if (i + 1) % 200 == 0 or (i + 1) == len(train_ids):
            print(f"[ast_lr] logits train {i+1}/{len(train_ids)} clips", flush=True)

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
    return model, feats_by_train


def _train_clap_lr_eventness(
    *,
    clip_ids_train: list[str],
    labels_by_clip: dict[str, list[int]],
    audio_dir: Path,
    clap_probe: object,
    clap_text: np.ndarray,
    num_segments: int = 10,
    device: str = "cpu",
    epochs: int = 30,
    batch_size: int = 4096,
    lr: float = 2e-2,
) -> tuple[torch.nn.Module, dict[str, np.ndarray]]:
    """
    Train a tiny supervised eventness model on top of CLAP per-second audio↔text cosine similarities.

    Target is binary: 1 if segment label != 0 else 0.
    Returns:
      - a 1-layer logistic regression (per-second logits)
      - cached CLAP similarity features for train clips (to reuse during scoring)
    """
    import torch.nn as nn

    if clap_text.ndim != 2:
        raise ValueError(f"clap_text must have shape [C,D], got {tuple(clap_text.shape)}")

    train_ids = [str(x) for x in clip_ids_train]
    feats_by_train: dict[str, np.ndarray] = {}
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []

    for i, cid in enumerate(train_ids):
        wav_path = audio_dir / cid / "audio.wav"
        aud_feat = clap_probe.audio_embeddings_per_second(wav_path, num_segments=int(num_segments))  # [T, D]
        feats = aud_feat @ clap_text.T  # [T, C]
        feats = feats.astype(np.float32, copy=False)
        feats_by_train[cid] = feats

        labs = np.asarray(labels_by_clip[cid], dtype=np.int64)[: int(num_segments)]
        y = (labs != 0).astype(np.float32).reshape(-1, 1)

        x_rows.append(feats)
        y_rows.append(y)

        if (i + 1) % 200 == 0 or (i + 1) == len(train_ids):
            print(f"[clap_lr] feats train {i+1}/{len(train_ids)} clips", flush=True)

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
    return model, feats_by_train


def _train_ast_evt_mlp_eventness(
    *,
    clip_ids_train: list[str],
    labels_by_clip: dict[str, list[int]],
    audio_dir: Path,
    ast_probe: ASTEventnessProbe,
    num_segments: int = 10,
    device: str = "cpu",
    epochs: int = 30,
    batch_size: int = 4096,
    lr: float = 1e-3,
    hidden_dim: int = 128,
) -> tuple[torch.nn.Module, dict[str, np.ndarray]]:
    """
    Train a small MLP eventness model on top of pretrained AST per-second logits.

    Target is binary: 1 if segment label != 0 else 0.
    Returns:
      - a 2-layer MLP (per-second logits)
      - cached AST logits for train clips (to reuse during scoring)
    """
    import torch.nn as nn

    train_ids = [str(x) for x in clip_ids_train]
    feats_by_train: dict[str, np.ndarray] = {}
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []

    for i, cid in enumerate(train_ids):
        wav_path = audio_dir / cid / "audio.wav"
        feats = ast_probe.logits_per_second(wav_path, num_segments=int(num_segments))  # [T, C]
        feats_by_train[cid] = feats

        labs = np.asarray(labels_by_clip[cid], dtype=np.int64)[: int(num_segments)]
        y = (labs != 0).astype(np.float32).reshape(-1, 1)

        x_rows.append(feats.astype(np.float32, copy=False))
        y_rows.append(y)

        if (i + 1) % 200 == 0 or (i + 1) == len(train_ids):
            print(f"[ast_evt_mlp] logits train {i+1}/{len(train_ids)} clips", flush=True)

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
    return model, feats_by_train


def _train_ast_emb_lr_eventness(
    *,
    clip_ids_train: list[str],
    labels_by_clip: dict[str, list[int]],
    audio_dir: Path,
    ast_probe: ASTEventnessProbe,
    num_segments: int = 10,
    device: str = "cpu",
    epochs: int = 30,
    batch_size: int = 4096,
    lr: float = 2e-2,
) -> tuple[torch.nn.Module, dict[str, np.ndarray]]:
    """
    Train a tiny supervised eventness model on top of pretrained AST per-second embeddings.

    Target is binary: 1 if segment label != 0 else 0.
    Returns:
      - a 1-layer logistic regression (per-second logits)
      - cached AST embeddings for train clips (to reuse during scoring)
    """
    import torch.nn as nn

    train_ids = [str(x) for x in clip_ids_train]
    feats_by_train: dict[str, np.ndarray] = {}
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []

    for i, cid in enumerate(train_ids):
        wav_path = audio_dir / cid / "audio.wav"
        feats = ast_probe.embeddings_per_second(wav_path, num_segments=int(num_segments))  # [T, D]
        feats_by_train[cid] = feats

        labs = np.asarray(labels_by_clip[cid], dtype=np.int64)[: int(num_segments)]
        y = (labs != 0).astype(np.float32).reshape(-1, 1)

        x_rows.append(feats.astype(np.float32, copy=False))
        y_rows.append(y)

        if (i + 1) % 200 == 0 or (i + 1) == len(train_ids):
            print(f"[ast_emb_lr] embeds train {i+1}/{len(train_ids)} clips", flush=True)

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
    return model, feats_by_train


def _train_ast_mlp_cls_eventness(
    *,
    clip_ids_train: list[str],
    labels_by_clip: dict[str, list[int]],
    audio_dir: Path,
    ast_probe: ASTEventnessProbe,
    num_classes: int,
    num_segments: int = 10,
    device: str = "cpu",
    epochs: int = 30,
    batch_size: int = 2048,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    dropout: float = 0.0,
) -> tuple[torch.nn.Module, dict[str, np.ndarray]]:
    """
    Train a tiny supervised per-second classifier on top of pretrained AST logits.

    Target is multi-class (AVE segment labels): 0=background, >0=event class.
    Returns:
      - a 2-layer MLP classifier over `num_classes`
      - cached AST logits for train clips (to reuse during scoring)
    """
    train_ids = [str(x) for x in clip_ids_train]
    feats_by_train: dict[str, np.ndarray] = {}
    for i, cid in enumerate(train_ids):
        wav_path = audio_dir / cid / "audio.wav"
        feats = ast_probe.logits_per_second(wav_path, num_segments=int(num_segments))  # [T, C_ast]
        feats_by_train[cid] = feats.astype(np.float32, copy=False)
        if (i + 1) % 200 == 0 or (i + 1) == len(train_ids):
            print(f"[ast_mlp_cls] logits train {i+1}/{len(train_ids)} clips", flush=True)

    model = _train_audio_basic_mlp_cls_eventness(
        clip_ids_train=train_ids,
        labels_by_clip=labels_by_clip,
        audio_feats_by_clip=feats_by_train,
        num_classes=int(num_classes),
        device=str(device),
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
    )
    model.eval()
    return model, feats_by_train


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
    dropout: float = 0.0,
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
        nn.Dropout(p=float(dropout)),
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


def _train_audio_basic_mlp_visgain_eventness(
    *,
    clip_ids_train: list[str],
    targets_by_clip: dict[str, np.ndarray],
    audio_feats_by_clip: dict[str, np.ndarray],
    device: str = "cpu",
    epochs: int = 50,
    batch_size: int = 2048,
    lr: float = 2e-3,
    hidden_dim: int = 64,
    dropout: float = 0.0,
) -> torch.nn.Module:
    """
    Train a tiny per-second regressor to predict a *visual usefulness* teacher signal.

    This is intended for "teacher-student Stage-1" experiments where the teacher is derived from
    (expensive) visual features (e.g., resolution sensitivity) but the student is deployable
    (audio-only or cheap A+V inputs).

    Target is a scalar per second (typically >=0), e.g.:
      teacher[t] = (label!=0) * (1 - cos_sim(vision_base[t], vision_high[t])).

    Returns a 2-layer MLP that outputs per-second scores (higher => more visually-useful).
    """
    import torch.nn as nn

    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    for cid in clip_ids_train:
        feats = audio_feats_by_clip[cid]  # [T, F]
        tgt = targets_by_clip[cid]  # [T] or [T,1]
        tgt = np.asarray(tgt, dtype=np.float32)
        if tgt.ndim == 1:
            tgt = tgt.reshape(-1, 1)
        if feats.shape[0] != tgt.shape[0]:
            raise ValueError(f"target length mismatch for {cid}: feats={feats.shape}, tgt={tgt.shape}")
        x_rows.append(feats)
        y_rows.append(tgt)

    x_np = np.concatenate(x_rows, axis=0).astype(np.float32, copy=False)
    y_np = np.concatenate(y_rows, axis=0).astype(np.float32, copy=False)

    x = torch.from_numpy(x_np).to(device=torch.device(device), dtype=torch.float32)
    y = torch.from_numpy(y_np).to(device=torch.device(device), dtype=torch.float32)

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(int(x.shape[-1]), int(hidden_dim)),
        nn.ReLU(),
        nn.Dropout(p=float(dropout)),
        nn.Linear(int(hidden_dim), 1),
    ).to(torch.device(device))

    # Reweight non-zero teacher seconds (usually sparse) to avoid the trivial all-zeros solution.
    pos = float((y > 1e-6).sum().item())
    neg = float((y <= 1e-6).sum().item())
    pos_weight = neg / max(1.0, pos)

    opt = torch.optim.AdamW(model.parameters(), lr=float(lr))

    n = int(x.shape[0])
    steps = max(1, (n + int(batch_size) - 1) // int(batch_size))
    for _epoch in range(int(epochs)):
        perm = torch.randperm(n, device=torch.device(device))
        for i in range(steps):
            idx = perm[i * int(batch_size) : (i + 1) * int(batch_size)]
            xb = x[idx]
            yb = y[idx]
            pred = model(xb)
            w = torch.where(yb > 1e-6, float(pos_weight), 1.0)
            loss = torch.mean(w * (pred - yb) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    return model


def _train_audio_basic_mil_mlp_eventness(
    *,
    clip_ids_train: list[str],
    labels_by_clip: dict[str, list[int]],
    audio_feats_by_clip: dict[str, np.ndarray],
    device: str = "cpu",
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 2e-3,
    hidden_dim: int = 64,
    dropout: float = 0.0,
) -> torch.nn.Module:
    """
    Train a per-second eventness scorer with a multi-instance learning (MIL) objective.

    Motivation: Stage-1 is used for Top-K *anchor selection*, so what matters most is that at least one of the
    true event seconds ranks near the top (peaky, clip-wise separation). BCE treats each second independently
    and can produce flatter score distributions, which increases fallback and far-anchor errors.

    Setup:
      - Inputs: per-second features [T, F] (audio or A+V concatenated) per clip
      - Targets: binary event mask per second (label != 0)
      - Loss per clip (if any positives): -log sum_{t in P} softmax(scores)[t]
        = -(logsumexp(scores_pos) - logsumexp(scores_all))

    Returns a 2-layer MLP that outputs per-second logits (higher => more likely event).
    """
    import torch.nn as nn

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for cid in clip_ids_train:
        feats = audio_feats_by_clip[cid]  # [T, F]
        labs = np.asarray(labels_by_clip[cid], dtype=np.int64)
        y = (labs != 0).astype(np.float32)  # [T]
        xs.append(feats.astype(np.float32, copy=False))
        ys.append(y.astype(np.float32, copy=False))

    x_np = np.stack(xs, axis=0)  # [N, T, F]
    y_np = np.stack(ys, axis=0)  # [N, T]

    x = torch.from_numpy(x_np).to(device=torch.device(device), dtype=torch.float32)
    y = torch.from_numpy(y_np).to(device=torch.device(device), dtype=torch.float32)

    in_dim = int(x.shape[-1])

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(in_dim, int(hidden_dim)),
        nn.ReLU(),
        nn.Dropout(p=float(dropout)),
        nn.Linear(int(hidden_dim), 1),
    ).to(torch.device(device))

    opt = torch.optim.AdamW(model.parameters(), lr=float(lr))

    n = int(x.shape[0])
    steps = max(1, (n + int(batch_size) - 1) // int(batch_size))

    for _epoch in range(int(epochs)):
        perm = torch.randperm(n, device=torch.device(device))
        for i in range(steps):
            idx = perm[i * int(batch_size) : (i + 1) * int(batch_size)]
            xb = x[idx]  # [B, T, F]
            yb = y[idx]  # [B, T]

            b, t, f = xb.shape
            scores = model(xb.reshape(int(b * t), int(f))).reshape(int(b), int(t))  # [B, T]
            pos = yb > 0.5

            # log p(pos) under a per-clip softmax distribution.
            scores_all = torch.logsumexp(scores, dim=1)  # [B]
            scores_pos = scores.masked_fill(~pos, float("-inf"))
            scores_pos_lse = torch.logsumexp(scores_pos, dim=1)  # [B], -inf when no positives

            valid = torch.isfinite(scores_pos_lse)
            if not bool(valid.any()):
                continue

            loss = -(scores_pos_lse[valid] - scores_all[valid]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    return model


class _AVClipAlignNCE(torch.nn.Module):
    def __init__(self, *, audio_dim: int, vision_dim: int, proj_dim: int, dropout: float = 0.0):
        import torch.nn as nn

        super().__init__()
        self.audio_proj = nn.Sequential(
            nn.Linear(int(audio_dim), int(proj_dim), bias=False),
            nn.LayerNorm(int(proj_dim)),
            nn.Dropout(p=float(dropout)),
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(int(vision_dim), int(proj_dim), bias=False),
            nn.LayerNorm(int(proj_dim)),
            nn.Dropout(p=float(dropout)),
        )

    def project_audio(self, audio: torch.Tensor) -> torch.Tensor:
        z = self.audio_proj(audio)
        return torch.nn.functional.normalize(z, dim=-1)

    def project_vision(self, vision: torch.Tensor) -> torch.Tensor:
        z = self.vision_proj(vision)
        return torch.nn.functional.normalize(z, dim=-1)

    def similarity_matrix(self, audio: torch.Tensor, vision: torch.Tensor, *, temperature: float) -> torch.Tensor:
        """
        audio: [B, T, A]
        vision: [B, T, V]
        returns: [B, T, T] similarity logits (audio->vision)
        """
        a = self.project_audio(audio)
        v = self.project_vision(vision)
        return torch.matmul(a, v.transpose(1, 2)) / float(temperature)

    def diag_scores(self, audio: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:
        """
        audio: [T, A]
        vision: [T, V]
        returns: [T] cosine similarity between projected audio[t] and vision[t].
        """
        a = self.project_audio(audio)
        v = self.project_vision(vision)
        return (a * v).sum(dim=-1)


def _train_av_clipalign_nce_eventness(
    *,
    clip_ids_train: list[str],
    labels_by_clip: dict[str, list[int]],
    audio_emb_by_clip: dict[str, np.ndarray],
    vision_emb_by_clip: dict[str, np.ndarray],
    device: str = "cpu",
    proj_dim: int = 128,
    temperature: float = 0.07,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 2e-3,
    weight_decay: float = 0.0,
    dropout: float = 0.1,
    seed: int = 0,
) -> _AVClipAlignNCE:
    """
    Train a cross-modal within-clip alignment model (InfoNCE) to score audio-visual alignment per second.

    Supervision uses only positive (non-background) timestamps; negatives are other timestamps within the
    same clip. This targets the AVE failure mode where audio is present but visual evidence is off-screen.
    """
    import torch.nn.functional as F

    if float(temperature) <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    clip_ids_train = [str(x) for x in clip_ids_train]
    if not clip_ids_train:
        raise ValueError("clip_ids_train is empty")

    a0 = audio_emb_by_clip[clip_ids_train[0]]
    v0 = vision_emb_by_clip[clip_ids_train[0]]
    if a0.ndim != 2 or v0.ndim != 2:
        raise ValueError("expected per-clip embeddings with shape [T, D]")
    if int(a0.shape[0]) != int(v0.shape[0]):
        raise ValueError(f"audio/vision T mismatch: {a0.shape} vs {v0.shape}")

    t = int(a0.shape[0])
    audio_dim = int(a0.shape[1])
    vision_dim = int(v0.shape[1])

    torch.manual_seed(int(seed))
    model = _AVClipAlignNCE(audio_dim=audio_dim, vision_dim=vision_dim, proj_dim=int(proj_dim), dropout=float(dropout)).to(
        torch.device(device)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    n = int(len(clip_ids_train))
    steps = max(1, (n + int(batch_size) - 1) // int(batch_size))
    diag_idx = torch.arange(t, device=torch.device(device), dtype=torch.long)

    for _epoch in range(int(epochs)):
        perm = torch.randperm(n, device=torch.device("cpu"))
        for i in range(steps):
            idx = perm[i * int(batch_size) : (i + 1) * int(batch_size)].tolist()
            batch_ids = [clip_ids_train[j] for j in idx]

            a_list: list[np.ndarray] = []
            v_list: list[np.ndarray] = []
            pos_list: list[np.ndarray] = []
            for cid in batch_ids:
                a = audio_emb_by_clip[cid].astype(np.float32, copy=False)
                v = vision_emb_by_clip[cid].astype(np.float32, copy=False)
                labs = np.asarray(labels_by_clip[cid], dtype=np.int64)[:t]
                pos_list.append((labs != 0).astype(np.float32))
                a_list.append(a)
                v_list.append(v)

            a_b = torch.from_numpy(np.stack(a_list, axis=0)).to(device=torch.device(device), dtype=torch.float32)  # [B,T,A]
            v_b = torch.from_numpy(np.stack(v_list, axis=0)).to(device=torch.device(device), dtype=torch.float32)  # [B,T,V]
            pos_b = torch.from_numpy(np.stack(pos_list, axis=0)).to(device=torch.device(device), dtype=torch.float32)  # [B,T]

            sim_av = model.similarity_matrix(a_b, v_b, temperature=float(temperature))  # [B,T,T]
            sim_va = sim_av.transpose(1, 2)  # [B,T,T]

            logp_av = F.log_softmax(sim_av, dim=-1)
            logp_va = F.log_softmax(sim_va, dim=-1)

            diag_av = logp_av[:, diag_idx, diag_idx]  # [B,T]
            diag_va = logp_va[:, diag_idx, diag_idx]  # [B,T]

            mask = pos_b > 0.5
            if not bool(mask.any()):
                continue

            loss = -(diag_av[mask].mean() + diag_va[mask].mean()) * 0.5
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    return model


def _train_av_clipalign_bce_eventness(
    *,
    clip_ids_train: list[str],
    labels_by_clip: dict[str, list[int]],
    audio_emb_by_clip: dict[str, np.ndarray],
    vision_emb_by_clip: dict[str, np.ndarray],
    device: str = "cpu",
    proj_dim: int = 128,
    temperature: float = 0.07,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 2e-3,
    weight_decay: float = 0.0,
    dropout: float = 0.1,
    seed: int = 0,
) -> _AVClipAlignNCE:
    """
    Train a cross-modal diagonal correspondence model (BCE) to score A/V match per second.

    Label: y=1 for non-background seconds (labs != 0), else 0.

    Motivation:
      - Off-screen audio activity is a dominant false-anchor source.
      - A/V diagonal similarity is a *different signal* from audio eventness: it can suppress peaks that
        have no visual corroboration.
      - Compared to InfoNCE-only training, BCE directly optimizes per-second separability of the diagonal.
    """
    if float(temperature) <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    clip_ids_train = [str(x) for x in clip_ids_train]
    if not clip_ids_train:
        raise ValueError("clip_ids_train is empty")

    a0 = audio_emb_by_clip[clip_ids_train[0]]
    v0 = vision_emb_by_clip[clip_ids_train[0]]
    if a0.ndim != 2 or v0.ndim != 2:
        raise ValueError("expected per-clip embeddings with shape [T, D]")
    if int(a0.shape[0]) != int(v0.shape[0]):
        raise ValueError(f"audio/vision T mismatch: {a0.shape} vs {v0.shape}")

    t = int(a0.shape[0])
    audio_dim = int(a0.shape[1])
    vision_dim = int(v0.shape[1])

    import torch.nn.functional as F

    torch.manual_seed(int(seed))
    model = _AVClipAlignNCE(audio_dim=audio_dim, vision_dim=vision_dim, proj_dim=int(proj_dim), dropout=float(dropout)).to(
        torch.device(device)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    # Global pos_weight for class imbalance (background dominates in AVE).
    pos = 0.0
    neg = 0.0
    for cid in clip_ids_train:
        labs = labels_by_clip[cid][:t]
        p = float(sum(1 for x in labs if int(x) != 0))
        pos += p
        neg += float(t) - p
    pos_weight = torch.tensor([neg / max(1.0, pos)], device=torch.device(device), dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n = int(len(clip_ids_train))
    steps = max(1, (n + int(batch_size) - 1) // int(batch_size))
    inv_temp = float(1.0 / float(temperature))

    for _epoch in range(int(epochs)):
        perm = torch.randperm(n, device=torch.device("cpu"))
        for i in range(steps):
            idx = perm[i * int(batch_size) : (i + 1) * int(batch_size)].tolist()
            batch_ids = [clip_ids_train[j] for j in idx]

            a_list: list[np.ndarray] = []
            v_list: list[np.ndarray] = []
            y_list: list[np.ndarray] = []
            for cid in batch_ids:
                a = audio_emb_by_clip[cid].astype(np.float32, copy=False)
                v = vision_emb_by_clip[cid].astype(np.float32, copy=False)
                labs = np.asarray(labels_by_clip[cid], dtype=np.int64)[:t]
                y = (labs != 0).astype(np.float32)
                a_list.append(a)
                v_list.append(v)
                y_list.append(y)

            a_b = torch.from_numpy(np.stack(a_list, axis=0)).to(device=torch.device(device), dtype=torch.float32)  # [B,T,A]
            v_b = torch.from_numpy(np.stack(v_list, axis=0)).to(device=torch.device(device), dtype=torch.float32)  # [B,T,V]
            y_b = torch.from_numpy(np.stack(y_list, axis=0)).to(device=torch.device(device), dtype=torch.float32)  # [B,T]

            a_p = model.project_audio(a_b)
            v_p = model.project_vision(v_b)
            logits = (a_p * v_p).sum(dim=-1) * float(inv_temp)  # [B,T]

            loss = loss_fn(logits, y_b)
            if not torch.isfinite(loss):
                continue
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
    alloc_core = alloc[: -len("_high")] if alloc.endswith("_high") else alloc
    use_scores_for_base: list[float] | None = None

    if alloc_core == "score":
        # Base allocation by score only makes sense when the baseline is audio-guided; avoid leaking
        # audio signal into random/oracle baselines (and avoid requiring scores for uniform/oracle).
        if baseline in ("anchored_top2", "audio_concat_anchored_top2", "audio_feat_concat_anchored_top2"):
            if scores is None:
                raise ValueError("anchor_base_alloc=score requires per-second scores to be provided")
            use_scores_for_base = scores
        else:
            alloc = "distance"
            alloc_core = "distance"

    low_res = int(cfg.low_res)
    high_res = int(cfg.high_res)
    if baseline in ("anchored_top2", "audio_concat_anchored_top2", "audio_feat_concat_anchored_top2"):
        low_res, high_res, max_high_anchors, _ = _resolve_anchored_triad_for_clip(
            cfg=cfg,
            scores=scores,
            max_high_anchors=max_high_anchors,
        )

    mode = str(cfg.budget_mode or "exact")
    if mode not in ("exact", "band"):
        raise ValueError(f"unknown budget_mode={mode!r}; expected 'exact' or 'band'")

    extra = [int(r) for r in (cfg.budget_extra_resolutions or ())]
    if mode == "band":
        return budget_band_anchored_plan_scored(
            num_segments=num_segments,
            anchors=use_anchors,
            scores=use_scores_for_base,
            base_alloc=str(alloc),
            low_res=int(low_res),
            base_res=int(cfg.base_res),
            high_res=int(high_res),
            extra_resolutions=extra if extra else None,
            max_high_anchors=max_high_anchors,
            patch_size=cfg.patch_size,
            epsilon_frac=float(cfg.budget_epsilon_frac),
        )

    # NOTE: Default `anchor_base_alloc=distance` keeps `anchored_top2` backward-compatible:
    # base segments are allocated by distance-to-anchor (not by score).
    return equal_token_budget_anchored_plan_scored(
        num_segments=num_segments,
        anchors=use_anchors,
        scores=use_scores_for_base,
        base_alloc=str(alloc),
        low_res=int(low_res),
        base_res=int(cfg.base_res),
        high_res=int(high_res),
        max_high_anchors=max_high_anchors,
        patch_size=cfg.patch_size,
    )


def features_from_cache(cache: FeatureCache, plan: SamplingPlan, *, res_feature: str = "none", base_res: int = 224) -> np.ndarray:
    t = len(plan.resolutions)
    # Use the first resolution present to infer embedding dim.
    any_r = cache.resolutions[0]
    d = cache.features_by_resolution[any_r].shape[-1]
    mode = str(res_feature)
    extra = 0
    if mode == "none":
        extra = 0
    elif mode == "scalar":
        extra = 1
    else:
        raise ValueError(f"unknown res_feature={mode!r}; expected 'none' or 'scalar'")

    out = np.empty((t, d + extra), dtype=np.float32)
    for i, r in enumerate(plan.resolutions):
        out[i, :d] = cache.features_by_resolution[int(r)][i]
        if extra:
            # Center at base_res (base_res => 0.0).
            out[i, d] = (float(int(r)) - float(int(base_res))) / float(max(1, int(base_res)))
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
    class_names: list[str] | None = None,
    num_segments: int = 10,
    eventness_method: str = "energy",
    audio_device: str = "cpu",
    ast_pretrained: bool = False,
    panns_random: bool = False,
    panns_checkpoint: Path | None = None,
    audiomae_random: bool = False,
    audiomae_checkpoint: Path | None = None,
    scores_by_clip_override: dict[str, list[float]] | None = None,
) -> dict:
    results_by_seed: list[dict] = []
    train_device = str(train_device)
    all_ids = sorted(set(clip_ids_train + clip_ids_eval))

    ast_probe = None
    if scores_by_clip_override is None and eventness_method in (
        "ast",
        "ast_nonspeech_max",
        "energy_nonspeech_ast",
        "ast_lr",
        "ast_emb_lr",
        "ast_evt_mlp",
        "ast_mlp_cls",
        "ast_mlp_cls_target",
        "av_clipdiff_speech_mlp",
        "av_ast_clipdiff_mlp",
        "av_ast_clipdiff_mil_mlp",
        "av_ast_clipdiff_tcn",
        "av_ast_clipalign_nce",
        "av_ast_clipalign_bce",
    ):
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

    if str(eventness_method) in (
        "av_clap_clip_agree",
        "av_fused",
        "av_fused_prod",
        "energy_autoshift_clipdiff",
        "energy_autoshift_clipdiff_pos",
        "av_fused_clipdiff",
        "av_fused_clipdiff_prod",
        "moe_energy_clipdiff",
        "av_basic_lr",
        "av_basic_mlp",
        "av_clipdiff_lr",
        "av_clipdiff_mlp",
        "av_clipdiff_accflip_mlp",
        "av_clipdiff_speech_mlp",
        "av_clipdiff_visgain_mlp",
        "av_clipdiff_lossgain_mlp",
        "av_clipdiff_mil_mlp",
        "av_clipdiff_framediff_mlp",
        "av_clipdiff_flow_mlp",
        "av_clipdiff_flow_mlp_stride",
        "av_clipdiff_fbank_mlp",
        "av_clipdiff_mlp_r160",
        "av_ast_clipdiff_mlp",
        "av_ast_clipdiff_mil_mlp",
        "av_ast_clipdiff_tcn",
        "av_ast_clipalign_nce",
        "av_ast_clipalign_bce",
        "av_clipdiff_vec_mlp",
        "av_clip_mlp_cls",
        "av_clip_mlp_cls_target",
        "av_clipdiff_tcn",
        "vision_clipdiff",
        "vision_binary_lr",
        "vision_binary_mlp",
        "vision_mlp_cls",
        "vision_mlp_cls_target",
        "imagebind_av_sim",
    ) and any(
        b in baselines for b in ("audio_concat_uniform", "audio_concat_anchored_top2")
    ):
        raise ValueError(
            "eventness_method in {'av_fused','av_fused_prod','av_basic_*'} is intended for anchor proposals and can use visual signals; "
            "it would leak visual information into audio_concat_* baselines. Use eventness_method='energy' or 'energy_stride_max' instead."
        )

    audio_eventness_feats_by_clip: dict[str, np.ndarray] | None = None
    if audio_dir is not None and scores_by_clip_override is None and eventness_method in (
        "audio_basic_lr",
        "audio_basic_mlp",
        "audio_basic_tcn",
        "audio_basic_mlp_cls",
        "audio_basic_mlp_cls_target",
        "audio_fbank_mlp",
        "audio_fbank_tcn",
        "av_basic_lr",
        "av_basic_mlp",
        "av_clipdiff_lr",
        "av_clipdiff_mlp",
        "av_clipdiff_accflip_mlp",
        "av_clipdiff_speech_mlp",
        "av_clipdiff_visgain_mlp",
        "av_clipdiff_lossgain_mlp",
        "av_clipdiff_mil_mlp",
        "av_clipdiff_framediff_mlp",
        "av_clipdiff_flow_mlp",
        "av_clipdiff_flow_mlp_stride",
        "av_clipdiff_fbank_mlp",
        "av_clipdiff_mlp_r160",
        "av_ast_clipdiff_mlp",
        "av_ast_clipdiff_mil_mlp",
        "av_ast_clipdiff_tcn",
        "av_ast_clipalign_nce",
        "av_ast_clipalign_bce",
        "av_clipdiff_vec_mlp",
        "av_clip_mlp_cls",
        "av_clip_mlp_cls_target",
        "av_clipdiff_tcn",
    ):
        if str(eventness_method) in (
            "av_ast_clipdiff_mlp",
            "av_ast_clipdiff_mil_mlp",
            "av_ast_clipdiff_tcn",
            "av_ast_clipalign_nce",
            "av_ast_clipalign_bce",
        ):
            assert ast_probe is not None
            audio_eventness_feats_by_clip = {}
            for i, cid in enumerate(all_ids):
                wav_path = audio_dir / cid / "audio.wav"
                audio_eventness_feats_by_clip[cid] = ast_probe.embeddings_per_second(
                    wav_path, num_segments=int(num_segments)
                ).astype(np.float32, copy=False)
                if (i + 1) % 200 == 0 or (i + 1) == len(all_ids):
                    print(f"[P0] extracted {i+1}/{len(all_ids)} ast_emb clips", flush=True)
        elif str(eventness_method) in ("audio_fbank_mlp", "audio_fbank_tcn", "av_clipdiff_fbank_mlp"):
            audio_eventness_feats_by_clip = {}
            for i, cid in enumerate(all_ids):
                wav_path = audio_dir / cid / "audio.wav"
                audio_eventness_feats_by_clip[cid] = audio_features_per_second(
                    wav_path, num_segments=num_segments, feature_set="fbank_stats"
                )
                if (i + 1) % 200 == 0 or (i + 1) == len(all_ids):
                    print(f"[P0] extracted {i+1}/{len(all_ids)} fbank_stats clips", flush=True)
        elif (
            str(eventness_method).startswith("audio_basic")
            or str(eventness_method).startswith("av_basic")
            or str(eventness_method).startswith("av_clipdiff")
        ):
            # Per-second audio features.
            if audio_feats_basic_by_clip is not None:
                audio_eventness_feats_by_clip = {k: v for k, v in audio_feats_basic_by_clip.items()}
            else:
                audio_eventness_feats_by_clip = {}
                for i, cid in enumerate(all_ids):
                    wav_path = audio_dir / cid / "audio.wav"
                    audio_eventness_feats_by_clip[cid] = audio_features_per_second(
                        wav_path, num_segments=num_segments, feature_set="basic"
                    )
                    if (i + 1) % 500 == 0 or (i + 1) == len(all_ids):
                        print(f"[P0] extracted {i+1}/{len(all_ids)} basic audio clips", flush=True)

            # Optional cheap visual feature concat (frame-diff eventness).
            if str(eventness_method).startswith("av_basic"):
                from avs.vision.cheap_eventness import frame_diff_eventness, list_frames

                for cid in all_ids:
                    frames_dir = audio_dir / cid / "frames"
                    frames = list_frames(frames_dir) if frames_dir.exists() else []
                    vis = frame_diff_eventness(frames, size=32) if frames else []

                    v = np.zeros((int(num_segments), 1), dtype=np.float32)
                    for t, s in enumerate(vis[: int(num_segments)]):
                        v[int(t), 0] = float(s)

                    a = audio_eventness_feats_by_clip[cid]
                    if a.shape[0] != int(num_segments):
                        raise ValueError(f"unexpected audio feature shape for {cid}: {a.shape}")
                    audio_eventness_feats_by_clip[cid] = np.concatenate([a, v], axis=1).astype(np.float32, copy=False)
        else:
            raise ValueError(f"internal error: unexpected eventness_method requiring audio_eventness_feats_by_clip: {eventness_method}")

    # Cache IO is the dominant cost in full runs. Preload caches once and reuse across
    # baselines/seeds so the sweep is compute-bound (GPU) instead of disk-bound.
    t0 = time.time()
    cache_by_clip: dict[str, FeatureCache] = {}
    for i, cid in enumerate(all_ids):
        cache_by_clip[cid] = FeatureCache.load_npz(caches_dir / f"{cid}.npz")
        if (i + 1) % 500 == 0 or (i + 1) == len(all_ids):
            print(f"[P0] loaded {i+1}/{len(all_ids)} caches", flush=True)
    print(f"[P0] cache preload done: {len(all_ids)} clips in {time.time() - t0:.1f}s", flush=True)

    scores_by_clip: dict[str, list[float]] | None = None
    autoshift_by_clip: dict[str, int] | None = None
    if audio_dir is not None and any(
        b in baselines for b in ("anchored_top2", "audio_concat_anchored_top2", "audio_feat_concat_anchored_top2", "audio_concat_uniform")
    ):
        if scores_by_clip_override is not None:
            missing = [cid for cid in all_ids if cid not in scores_by_clip_override]
            if missing:
                raise ValueError(f"scores_by_clip_override missing {len(missing)} clip_ids (e.g. {missing[:3]})")
            scores_by_clip = {cid: [float(x) for x in scores_by_clip_override[cid]] for cid in all_ids}
        else:
            scores_by_clip = {}
            if str(eventness_method) in ("energy_autoshift_clipdiff", "energy_autoshift_clipdiff_pos"):
                autoshift_by_clip = {}
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
            elif eventness_method == "av_basic_lr":
                if audio_eventness_feats_by_clip is None:
                    raise ValueError("internal error: audio_eventness_feats_by_clip should be precomputed for av_basic_lr")
                av_model = _train_audio_basic_lr_eventness(
                    clip_ids_train=clip_ids_train,
                    labels_by_clip=labels_by_clip,
                    audio_feats_by_clip=audio_eventness_feats_by_clip,
                    device="cpu",
                )
                av_model_cpu = av_model.to(torch.device("cpu"))
                for cid in all_ids:
                    feats = torch.from_numpy(audio_eventness_feats_by_clip[cid]).float()
                    with torch.no_grad():
                        logits = av_model_cpu(feats).squeeze(-1).numpy().astype("float32")
                    scores_by_clip[cid] = [float(x) for x in logits.tolist()]
            elif eventness_method == "av_clipdiff_lr":
                if audio_eventness_feats_by_clip is None:
                    raise ValueError(
                        "internal error: audio_eventness_feats_by_clip should be precomputed for av_clipdiff_lr"
                    )
                from avs.vision.cheap_eventness import clip_feature_diff_eventness

                av_feats_by_clip: dict[str, np.ndarray] = {}
                for i, cid in enumerate(all_ids):
                    a = audio_eventness_feats_by_clip[cid]
                    cache = cache_by_clip[cid]
                    # Stage-1 uses a fixed cheap visual resolution (prefer 112 when available) to avoid coupling
                    # Stage-1 scores to Stage-2 plan knobs (e.g., sweeps that vary low_res).
                    vis_res = 112 if 112 in cache.features_by_resolution else int(min(cache.features_by_resolution))
                    vis_feats = cache.features_by_resolution[int(vis_res)]

                    vis = clip_feature_diff_eventness(vis_feats, metric="cosine")
                    vis_scores = minmax_01([float(x) for x in vis])
                    v = np.zeros((int(num_segments), 1), dtype=np.float32)
                    for t, s in enumerate(vis_scores[: int(num_segments)]):
                        v[int(t), 0] = float(s)

                    if a.shape[0] != int(num_segments):
                        raise ValueError(f"unexpected audio feature shape for {cid}: {a.shape}")
                    av_feats_by_clip[cid] = np.concatenate([a, v], axis=1).astype(np.float32, copy=False)
                    if (i + 1) % 500 == 0 or (i + 1) == len(all_ids):
                        print(f"[av_clipdiff_lr] built feats {i+1}/{len(all_ids)} clips", flush=True)

                av_model = _train_audio_basic_lr_eventness(
                    clip_ids_train=clip_ids_train,
                    labels_by_clip=labels_by_clip,
                    audio_feats_by_clip=av_feats_by_clip,
                    device="cpu",
                )
                av_model_cpu = av_model.to(torch.device("cpu"))
                for i, cid in enumerate(all_ids):
                    feats = torch.from_numpy(av_feats_by_clip[cid]).float()
                    with torch.no_grad():
                        logits = av_model_cpu(feats).squeeze(-1).numpy().astype("float32")
                    scores_by_clip[cid] = [float(x) for x in logits.tolist()]
                    if (i + 1) % 500 == 0 or (i + 1) == len(all_ids):
                        print(f"[av_clipdiff_lr] scored {i+1}/{len(all_ids)} clips", flush=True)
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
            elif eventness_method == "av_basic_mlp":
                if audio_eventness_feats_by_clip is None:
                    raise ValueError("internal error: audio_eventness_feats_by_clip should be precomputed for av_basic_mlp")
                av_model = _train_audio_basic_mlp_eventness(
                    clip_ids_train=clip_ids_train,
                    labels_by_clip=labels_by_clip,
                    audio_feats_by_clip=audio_eventness_feats_by_clip,
                    device="cpu",
                )
                av_model_cpu = av_model.to(torch.device("cpu"))
                for cid in all_ids:
                    feats = torch.from_numpy(audio_eventness_feats_by_clip[cid]).float()
                    with torch.no_grad():
                        logits = av_model_cpu(feats).squeeze(-1)
                    scores_by_clip[cid] = [float(x) for x in logits.detach().cpu().numpy().astype("float32").tolist()]
            elif eventness_method in (
                "av_clipdiff_mlp",
                "av_clipdiff_accflip_mlp",
                "av_clipdiff_speech_mlp",
                "av_clipdiff_visgain_mlp",
                "av_clipdiff_lossgain_mlp",
                "av_clipdiff_mil_mlp",
                "av_clipdiff_framediff_mlp",
                "av_clipdiff_flow_mlp",
                "av_clipdiff_flow_mlp_stride",
                "av_clipdiff_fbank_mlp",
                "av_clipdiff_mlp_r160",
                "av_clipdiff_mlp_r224",
            ):
                if audio_eventness_feats_by_clip is None:
                    raise ValueError(
                        "internal error: audio_eventness_feats_by_clip should be precomputed for av_clipdiff_mlp*"
                    )
                from avs.vision.cheap_eventness import (
                    clip_feature_diff_eventness,
                    frame_diff_eventness,
                    list_frames,
                    optical_flow_mag_eventness,
                )

                av_feats_by_clip: dict[str, np.ndarray] = {}
                use_ast_speech = str(eventness_method) == "av_clipdiff_speech_mlp"
                speech_idx: list[int] = []
                if use_ast_speech:
                    assert ast_probe is not None
                    for k, v in ast_probe.model.config.id2label.items():
                        name = str(v).strip().lower()
                        if ("speech" in name) or ("conversation" in name) or ("narration" in name) or ("talking" in name):
                            speech_idx.append(int(k))
                    speech_idx = sorted(set(int(x) for x in speech_idx))
                use_visgain_teacher = str(eventness_method) == "av_clipdiff_visgain_mlp"
                use_lossgain_teacher = str(eventness_method) == "av_clipdiff_lossgain_mlp"
                use_accflip_teacher = str(eventness_method) == "av_clipdiff_accflip_mlp"
                teacher_by_clip: dict[str, np.ndarray] | None = (
                    {} if (use_visgain_teacher or use_lossgain_teacher or use_accflip_teacher) else None
                )
                loss_cids: list[str] = []
                loss_base_rows: list[np.ndarray] = []
                loss_high_rows: list[np.ndarray] = []
                loss_y_rows: list[np.ndarray] = []
                acc_cids: list[str] = []
                acc_base_rows: list[np.ndarray] = []
                acc_high_rows: list[np.ndarray] = []
                acc_y_rows: list[np.ndarray] = []
                train_set = set(str(x) for x in clip_ids_train)
                for i, cid in enumerate(all_ids):
                    a = audio_eventness_feats_by_clip[cid]
                    cache = cache_by_clip[cid]
                    # Stage-1 uses a fixed cheap visual resolution (prefer 112 when available) to avoid coupling
                    # Stage-1 scores to Stage-2 plan knobs (e.g., sweeps that vary low_res).
                    prefer_res = 112
                    if str(eventness_method).endswith("_r160"):
                        prefer_res = 160
                    elif str(eventness_method).endswith("_r224"):
                        prefer_res = 224
                    if int(prefer_res) in cache.features_by_resolution:
                        vis_res = int(prefer_res)
                    elif 112 in cache.features_by_resolution:
                        vis_res = 112
                    else:
                        vis_res = int(min(cache.features_by_resolution))
                    vis_feats = cache.features_by_resolution[int(vis_res)]

                    vis = clip_feature_diff_eventness(vis_feats, metric="cosine")
                    vis_scores = minmax_01([float(x) for x in vis])
                    v_clipdiff = np.zeros((int(num_segments), 1), dtype=np.float32)
                    for t, s in enumerate(vis_scores[: int(num_segments)]):
                        v_clipdiff[int(t), 0] = float(s)

                    v_extras: list[np.ndarray] = []
                    if str(eventness_method) in ("av_clipdiff_framediff_mlp", "av_clipdiff_flow_mlp", "av_clipdiff_flow_mlp_stride"):
                        frames_dir = audio_dir / cid / "frames"
                        frames = list_frames(frames_dir) if frames_dir.exists() else []
                        if str(eventness_method) == "av_clipdiff_framediff_mlp":
                            fd = frame_diff_eventness(frames, size=32) if frames else []
                            fd_scores = minmax_01([float(x) for x in fd])
                            v_frame = np.zeros((int(num_segments), 1), dtype=np.float32)
                            for t, s in enumerate(fd_scores[: int(num_segments)]):
                                v_frame[int(t), 0] = float(s)
                            v_extras.append(v_frame)
                        else:
                            flow = optical_flow_mag_eventness(frames, size=64) if frames else []
                            flow_scores = minmax_01([float(x) for x in flow])
                            v_flow = np.zeros((int(num_segments), 1), dtype=np.float32)
                            for t, s in enumerate(flow_scores[: int(num_segments)]):
                                v_flow[int(t), 0] = float(s)
                            v_extras.append(v_flow)

                    if a.shape[0] != int(num_segments):
                        raise ValueError(f"unexpected audio feature shape for {cid}: {a.shape}")
                    speech = None
                    if use_ast_speech:
                        assert ast_probe is not None
                        wav_path = audio_dir / cid / "audio.wav"
                        logits = ast_probe.logits_per_second(wav_path, num_segments=int(num_segments))  # [T,C]
                        probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float32)))
                        s = (
                            probs[:, speech_idx].max(axis=1)
                            if speech_idx
                            else np.zeros((int(num_segments),), dtype=np.float32)
                        )
                        speech = s.reshape(int(num_segments), 1).astype(np.float32, copy=False)

                    parts = [a, v_clipdiff]
                    parts.extend(v_extras)
                    if speech is not None:
                        parts.append(speech)
                    av_feats_by_clip[cid] = np.concatenate(parts, axis=1).astype(np.float32, copy=False)

                    if (use_visgain_teacher or use_lossgain_teacher or use_accflip_teacher) and str(cid) in train_set:
                        # Teacher inputs from cached vision features (base=224 vs high=352; fallback to closest available).
                        avail = sorted(int(r) for r in cache.features_by_resolution.keys())
                        if not avail:
                            raise ValueError(f"no cached resolutions for {cid}")

                        def _closest(target: int) -> int:
                            return min(avail, key=lambda r: (abs(int(r) - int(target)), int(r)))

                        base_r = _closest(224)
                        high_r = _closest(352)
                        if int(high_r) == int(base_r):
                            higher = [int(r) for r in avail if int(r) > int(base_r)]
                            high_r = int(higher[0]) if higher else int(base_r)

                        v_base = cache.features_by_resolution[int(base_r)].astype(np.float32, copy=False)[: int(num_segments)]
                        v_high = cache.features_by_resolution[int(high_r)].astype(np.float32, copy=False)[: int(num_segments)]
                        if v_base.shape != v_high.shape:
                            raise ValueError(f"teacher res mismatch for {cid}: base={v_base.shape}, high={v_high.shape}")

                        labs = np.asarray(labels_by_clip[cid], dtype=np.int64)[: int(num_segments)]
                        if use_visgain_teacher:
                            v_base_n = v_base / (np.linalg.norm(v_base, axis=1, keepdims=True) + 1e-12)
                            v_high_n = v_high / (np.linalg.norm(v_high, axis=1, keepdims=True) + 1e-12)
                            cos = np.sum(v_base_n * v_high_n, axis=1)
                            cos = np.clip(cos, -1.0, 1.0)
                            gain = np.clip(1.0 - cos, 0.0, 1.0).astype(np.float32, copy=False)
                            mask = (labs != 0).astype(np.float32, copy=False)
                            assert teacher_by_clip is not None
                            teacher_by_clip[cid] = (gain * mask).astype(np.float32, copy=False)

                        if use_lossgain_teacher:
                            loss_cids.append(str(cid))
                            loss_base_rows.append(v_base.astype(np.float32, copy=False))
                            loss_high_rows.append(v_high.astype(np.float32, copy=False))
                            loss_y_rows.append(labs.astype(np.int64, copy=False))
                        if use_accflip_teacher:
                            acc_cids.append(str(cid))
                            acc_base_rows.append(v_base.astype(np.float32, copy=False))
                            acc_high_rows.append(v_high.astype(np.float32, copy=False))
                            acc_y_rows.append(labs.astype(np.int64, copy=False))
                    if (i + 1) % 500 == 0 or (i + 1) == len(all_ids):
                        print(f"[{eventness_method}] built feats {i+1}/{len(all_ids)} clips", flush=True)

                if use_lossgain_teacher:
                    # Train a cheap base-res teacher head on vision features, then define the teacher target as
                    # per-second loss reduction when swapping in high-res features (event seconds only).
                    if not loss_cids:
                        raise ValueError(f"{eventness_method}: no lossgain teacher samples collected")
                    if set(loss_cids) != train_set:
                        missing = sorted(train_set.difference(set(loss_cids)))
                        raise ValueError(f"{eventness_method}: lossgain teacher missing {len(missing)} train ids (e.g. {missing[:3]})")

                    import torch.nn.functional as F

                    num_classes = 1
                    for cid in clip_ids_train:
                        labs = labels_by_clip.get(cid) or []
                        if labs:
                            num_classes = max(int(num_classes), int(max(int(x) for x in labs)) + 1)

                    x_base_t = torch.from_numpy(np.stack(loss_base_rows, axis=0)).float()
                    x_high_t = torch.from_numpy(np.stack(loss_high_rows, axis=0)).float()
                    y_t = torch.from_numpy(np.stack(loss_y_rows, axis=0)).long()

                    torch.manual_seed(0)
                    teacher = PerSegmentMLP(
                        in_dim=int(x_base_t.shape[-1]),
                        num_classes=int(num_classes),
                        hidden_dim=128,
                        dropout=0.1,
                    ).to(torch.device("cpu"))
                    _ = train_per_segment_classifier(
                        model=teacher,
                        x_train=x_base_t,
                        y_train=y_t,
                        x_val=x_base_t,
                        y_val=y_t,
                        cfg=TrainConfig(epochs=5, batch_size=256, lr=2e-3, weight_decay=0.0),
                    )

                    teacher.eval()
                    with torch.no_grad():
                        logits_base = teacher(x_base_t)
                        logits_high = teacher(x_high_t)
                        c = int(logits_base.shape[-1])
                        loss_base = F.cross_entropy(logits_base.view(-1, c), y_t.view(-1), reduction="none").view_as(y_t)
                        loss_high = F.cross_entropy(logits_high.view(-1, c), y_t.view(-1), reduction="none").view_as(y_t)
                        gain = (loss_base - loss_high).clamp(min=0.0)
                        gain = gain * (y_t != 0).float()
                        denom = gain.max(dim=1, keepdim=True).values
                        gain = gain / (denom + 1e-6)

                    assert teacher_by_clip is not None
                    teacher_by_clip = {str(cid): gain[int(i)].detach().cpu().numpy().astype(np.float32) for i, cid in enumerate(loss_cids)}

                if use_accflip_teacher:
                    # Train two cheap vision teachers (base-res and high-res), then define the teacher target as
                    # per-second "accuracy flip" where high-res predicts the correct label but base-res does not
                    # (event seconds only). This aligns Stage-1 to the downstream budgeted resolution benefit.
                    if not acc_cids:
                        raise ValueError(f"{eventness_method}: no accflip teacher samples collected")
                    if set(acc_cids) != train_set:
                        missing = sorted(train_set.difference(set(acc_cids)))
                        raise ValueError(f"{eventness_method}: accflip teacher missing {len(missing)} train ids (e.g. {missing[:3]})")

                    num_classes = 1
                    for cid in clip_ids_train:
                        labs = labels_by_clip.get(cid) or []
                        if labs:
                            num_classes = max(int(num_classes), int(max(int(x) for x in labs)) + 1)

                    x_base_t = torch.from_numpy(np.stack(acc_base_rows, axis=0)).float()
                    x_high_t = torch.from_numpy(np.stack(acc_high_rows, axis=0)).float()
                    y_t = torch.from_numpy(np.stack(acc_y_rows, axis=0)).long()

                    torch.manual_seed(0)
                    teacher_base = PerSegmentMLP(
                        in_dim=int(x_base_t.shape[-1]),
                        num_classes=int(num_classes),
                        hidden_dim=128,
                        dropout=0.1,
                    ).to(torch.device("cpu"))
                    _ = train_per_segment_classifier(
                        model=teacher_base,
                        x_train=x_base_t,
                        y_train=y_t,
                        x_val=x_base_t,
                        y_val=y_t,
                        cfg=TrainConfig(epochs=5, batch_size=256, lr=2e-3, weight_decay=0.0),
                    )

                    torch.manual_seed(0)
                    teacher_high = PerSegmentMLP(
                        in_dim=int(x_high_t.shape[-1]),
                        num_classes=int(num_classes),
                        hidden_dim=128,
                        dropout=0.1,
                    ).to(torch.device("cpu"))
                    _ = train_per_segment_classifier(
                        model=teacher_high,
                        x_train=x_high_t,
                        y_train=y_t,
                        x_val=x_high_t,
                        y_val=y_t,
                        cfg=TrainConfig(epochs=5, batch_size=256, lr=2e-3, weight_decay=0.0),
                    )

                    teacher_base.eval()
                    teacher_high.eval()
                    with torch.no_grad():
                        pred_base = teacher_base(x_base_t).argmax(dim=-1)
                        pred_high = teacher_high(x_high_t).argmax(dim=-1)
                        flip = ((pred_high == y_t) & (pred_base != y_t) & (y_t != 0)).float()

                    pos = float((flip > 0.5).sum().item())
                    total = float(flip.numel())
                    print(f"[accflip teacher] pos={int(pos)}/{int(total)} ({(pos / max(1.0, total)):.6f})", flush=True)

                    assert teacher_by_clip is not None
                    teacher_by_clip = {
                        str(cid): flip[int(i)].detach().cpu().numpy().astype(np.float32, copy=False)
                        for i, cid in enumerate(acc_cids)
                    }

                if str(eventness_method) in ("av_clipdiff_visgain_mlp", "av_clipdiff_lossgain_mlp"):
                    assert teacher_by_clip is not None
                    av_model = _train_audio_basic_mlp_visgain_eventness(
                        clip_ids_train=clip_ids_train,
                        targets_by_clip=teacher_by_clip,
                        audio_feats_by_clip=av_feats_by_clip,
                        device="cpu",
                        epochs=60,
                        batch_size=2048,
                        lr=2e-3,
                        hidden_dim=64,
                        dropout=0.0,
                    )
                elif str(eventness_method) == "av_clipdiff_accflip_mlp":
                    assert teacher_by_clip is not None
                    # Convert accflip targets into a 0/1 label map and reuse the standard binary Stage-1 trainer.
                    teacher_labels: dict[str, list[int]] = {}
                    for cid in clip_ids_train:
                        tgt = teacher_by_clip.get(str(cid))
                        if tgt is None:
                            raise ValueError(f"{eventness_method}: missing teacher target for train id {cid}")
                        t = np.asarray(tgt, dtype=np.float32).reshape(-1)[: int(num_segments)]
                        teacher_labels[str(cid)] = [int(float(x) > 0.5) for x in t.tolist()]

                    av_model = _train_audio_basic_mlp_eventness(
                        clip_ids_train=clip_ids_train,
                        labels_by_clip=teacher_labels,
                        audio_feats_by_clip=av_feats_by_clip,
                        device="cpu",
                        hidden_dim=128,
                    )
                elif str(eventness_method) == "av_clipdiff_mil_mlp":
                    av_model = _train_audio_basic_mil_mlp_eventness(
                        clip_ids_train=clip_ids_train,
                        labels_by_clip=labels_by_clip,
                        audio_feats_by_clip=av_feats_by_clip,
                        device="cpu",
                        epochs=50,
                        batch_size=128,
                        lr=2e-3,
                        hidden_dim=128,
                        dropout=0.0,
                    )
                else:
                    av_model = _train_audio_basic_mlp_eventness(
                        clip_ids_train=clip_ids_train,
                        labels_by_clip=labels_by_clip,
                        audio_feats_by_clip=av_feats_by_clip,
                        device="cpu",
                        hidden_dim=128,
                    )
                av_model_cpu = av_model.to(torch.device("cpu"))
                for i, cid in enumerate(all_ids):
                    feats = torch.from_numpy(av_feats_by_clip[cid]).float()
                    with torch.no_grad():
                        logits = av_model_cpu(feats).squeeze(-1)
                    s_np = logits.detach().cpu().numpy().astype("float32")
                    if str(eventness_method) == "av_clipdiff_flow_mlp_stride":
                        s_np = np.asarray(
                            stride_max_pool_per_second(
                                [float(x) for x in s_np.tolist()],
                                num_segments=int(num_segments),
                                stride_s=0.2,
                                win_s=0.6,
                            ),
                            dtype=np.float32,
                        )
                    scores_by_clip[cid] = [float(x) for x in s_np.tolist()]
                    if (i + 1) % 500 == 0 or (i + 1) == len(all_ids):
                        print(f"[{eventness_method}] scored {i+1}/{len(all_ids)} clips", flush=True)
            elif eventness_method in ("av_ast_clipdiff_mlp", "av_ast_clipdiff_mil_mlp", "av_ast_clipdiff_tcn"):
                if audio_eventness_feats_by_clip is None:
                    raise ValueError(
                        "internal error: audio_eventness_feats_by_clip should be precomputed for av_ast_clipdiff_*"
                    )
                from avs.vision.cheap_eventness import clip_feature_diff_eventness

                av_feats_by_clip: dict[str, np.ndarray] = {}
                for i, cid in enumerate(all_ids):
                    a = audio_eventness_feats_by_clip[cid]
                    cache = cache_by_clip[cid]
                    # Keep Stage-1 cheap and decoupled: always prefer the smallest cached resolution (112 when present).
                    vis_res = 112 if 112 in cache.features_by_resolution else int(min(cache.features_by_resolution))
                    vis_feats = cache.features_by_resolution[int(vis_res)]

                    vis = clip_feature_diff_eventness(vis_feats, metric="cosine")
                    vis_scores = minmax_01([float(x) for x in vis])
                    v_clipdiff = np.zeros((int(num_segments), 1), dtype=np.float32)
                    for t, s in enumerate(vis_scores[: int(num_segments)]):
                        v_clipdiff[int(t), 0] = float(s)

                    if a.shape[0] != int(num_segments):
                        raise ValueError(f"unexpected AST embedding shape for {cid}: {a.shape}")
                    av_feats_by_clip[cid] = np.concatenate([a, v_clipdiff], axis=1).astype(np.float32, copy=False)
                    if (i + 1) % 500 == 0 or (i + 1) == len(all_ids):
                        print(f"[{eventness_method}] built feats {i+1}/{len(all_ids)} clips", flush=True)

                if eventness_method == "av_ast_clipdiff_mil_mlp":
                    av_model = _train_audio_basic_mil_mlp_eventness(
                        clip_ids_train=clip_ids_train,
                        labels_by_clip=labels_by_clip,
                        audio_feats_by_clip=av_feats_by_clip,
                        device="cpu",
                        hidden_dim=128,
                        dropout=0.1,
                    )
                elif eventness_method == "av_ast_clipdiff_tcn":
                    av_model = _train_audio_tcn_eventness(
                        clip_ids_train=clip_ids_train,
                        labels_by_clip=labels_by_clip,
                        audio_feats_by_clip=av_feats_by_clip,
                        device="cpu",
                        hidden_channels=128,
                        kernel_size=3,
                        dropout=0.1,
                    )
                else:
                    av_model = _train_audio_basic_mlp_eventness(
                        clip_ids_train=clip_ids_train,
                        labels_by_clip=labels_by_clip,
                        audio_feats_by_clip=av_feats_by_clip,
                        device="cpu",
                        hidden_dim=128,
                        dropout=0.1,
                    )

                av_model_cpu = av_model.to(torch.device("cpu"))
                for i, cid in enumerate(all_ids):
                    feats = torch.from_numpy(av_feats_by_clip[cid]).float()
                    with torch.no_grad():
                        logits = av_model_cpu(feats).squeeze(-1)
                    scores_by_clip[cid] = [float(x) for x in logits.detach().cpu().numpy().astype("float32").tolist()]
                    if (i + 1) % 500 == 0 or (i + 1) == len(all_ids):
                        print(f"[{eventness_method}] scored {i+1}/{len(all_ids)} clips", flush=True)
            elif eventness_method == "av_ast_clipalign_nce":
                if audio_eventness_feats_by_clip is None:
                    raise ValueError(
                        "internal error: audio_eventness_feats_by_clip should be precomputed for av_ast_clipalign_nce"
                    )

                vision_emb_by_clip: dict[str, np.ndarray] = {}
                for cid in all_ids:
                    cache = cache_by_clip[cid]
                    vis_res = 112 if 112 in cache.features_by_resolution else int(min(cache.features_by_resolution))
                    vision_emb_by_clip[cid] = cache.features_by_resolution[int(vis_res)].astype(np.float32, copy=False)

                av_model = _train_av_clipalign_nce_eventness(
                    clip_ids_train=clip_ids_train,
                    labels_by_clip=labels_by_clip,
                    audio_emb_by_clip=audio_eventness_feats_by_clip,
                    vision_emb_by_clip=vision_emb_by_clip,
                    device="cpu",
                    proj_dim=128,
                    temperature=0.07,
                    epochs=60,
                    batch_size=64,
                    lr=2e-3,
                    weight_decay=0.0,
                    dropout=0.1,
                    seed=0,
                )
                av_model_cpu = av_model.to(torch.device("cpu"))
                for i, cid in enumerate(all_ids):
                    a = torch.from_numpy(audio_eventness_feats_by_clip[cid]).float()
                    v = torch.from_numpy(vision_emb_by_clip[cid]).float()
                    with torch.no_grad():
                        s = av_model_cpu.diag_scores(a, v)
                    scores_by_clip[cid] = [float(x) for x in s.detach().cpu().numpy().astype("float32").tolist()]
                    if (i + 1) % 500 == 0 or (i + 1) == len(all_ids):
                        print(f"[av_ast_clipalign_nce] scored {i+1}/{len(all_ids)} clips", flush=True)
            elif eventness_method == "av_ast_clipalign_bce":
                if audio_eventness_feats_by_clip is None:
                    raise ValueError(
                        "internal error: audio_eventness_feats_by_clip should be precomputed for av_ast_clipalign_bce"
                    )

                vision_emb_by_clip: dict[str, np.ndarray] = {}
                for cid in all_ids:
                    cache = cache_by_clip[cid]
                    vis_res = 112 if 112 in cache.features_by_resolution else int(min(cache.features_by_resolution))
                    vision_emb_by_clip[cid] = cache.features_by_resolution[int(vis_res)].astype(np.float32, copy=False)

                temperature = 0.07
                av_model = _train_av_clipalign_bce_eventness(
                    clip_ids_train=clip_ids_train,
                    labels_by_clip=labels_by_clip,
                    audio_emb_by_clip=audio_eventness_feats_by_clip,
                    vision_emb_by_clip=vision_emb_by_clip,
                    device="cpu",
                    proj_dim=128,
                    temperature=float(temperature),
                    epochs=80,
                    batch_size=64,
                    lr=2e-3,
                    weight_decay=0.0,
                    dropout=0.1,
                    seed=0,
                )
                av_model_cpu = av_model.to(torch.device("cpu"))
                inv_temp = float(1.0 / float(temperature))
                for i, cid in enumerate(all_ids):
                    a = torch.from_numpy(audio_eventness_feats_by_clip[cid]).float()
                    v = torch.from_numpy(vision_emb_by_clip[cid]).float()
                    with torch.no_grad():
                        s = av_model_cpu.diag_scores(a, v) * float(inv_temp)
                    scores_by_clip[cid] = [float(x) for x in s.detach().cpu().numpy().astype("float32").tolist()]
                    if (i + 1) % 500 == 0 or (i + 1) == len(all_ids):
                        print(f"[av_ast_clipalign_bce] scored {i+1}/{len(all_ids)} clips", flush=True)
            elif eventness_method == "av_clipdiff_vec_mlp":
                if audio_eventness_feats_by_clip is None:
                    raise ValueError(
                        "internal error: audio_eventness_feats_by_clip should be precomputed for av_clipdiff_vec_mlp"
                    )

                av_feats_by_clip: dict[str, np.ndarray] = {}
                for i, cid in enumerate(all_ids):
                    a = audio_eventness_feats_by_clip[cid]
                    cache = cache_by_clip[cid]
                    # Stage-1 uses a fixed cheap visual resolution (prefer 112 when available) to avoid coupling
                    # Stage-1 scores to Stage-2 plan knobs (e.g., sweeps that vary low_res).
                    vis_res = 112 if 112 in cache.features_by_resolution else int(min(cache.features_by_resolution))
                    vis_feats = cache.features_by_resolution[int(vis_res)].astype(np.float32, copy=False)
                    # Normalize then take a first-order diff vector (semantic motion direction, not just magnitude).
                    vis_feats = vis_feats / (np.linalg.norm(vis_feats, axis=1, keepdims=True) + 1e-12)
                    d = np.zeros_like(vis_feats, dtype=np.float32)
                    d[1:] = vis_feats[1:] - vis_feats[:-1]

                    if a.shape[0] != int(num_segments):
                        raise ValueError(f"unexpected audio feature shape for {cid}: {a.shape}")
                    if d.shape[0] != int(num_segments):
                        raise ValueError(f"unexpected clip feature shape for {cid}: {d.shape}")
                    av_feats_by_clip[cid] = np.concatenate([a, d], axis=1).astype(np.float32, copy=False)
                    if (i + 1) % 500 == 0 or (i + 1) == len(all_ids):
                        print(f"[av_clipdiff_vec_mlp] built feats {i+1}/{len(all_ids)} clips", flush=True)

                av_model = _train_audio_basic_mlp_eventness(
                    clip_ids_train=clip_ids_train,
                    labels_by_clip=labels_by_clip,
                    audio_feats_by_clip=av_feats_by_clip,
                    device="cpu",
                    hidden_dim=128,
                    dropout=0.1,
                )
                av_model_cpu = av_model.to(torch.device("cpu"))
                for i, cid in enumerate(all_ids):
                    feats = torch.from_numpy(av_feats_by_clip[cid]).float()
                    with torch.no_grad():
                        logits = av_model_cpu(feats).squeeze(-1)
                    scores_by_clip[cid] = [float(x) for x in logits.detach().cpu().numpy().astype("float32").tolist()]
                    if (i + 1) % 500 == 0 or (i + 1) == len(all_ids):
                        print(f"[av_clipdiff_vec_mlp] scored {i+1}/{len(all_ids)} clips", flush=True)
            elif eventness_method in ("av_clipdiff_mlp_cls", "av_clipdiff_mlp_cls_target"):
                if audio_eventness_feats_by_clip is None:
                    raise ValueError(
                        f"internal error: audio_eventness_feats_by_clip should be precomputed for {eventness_method}"
                    )
                from avs.vision.cheap_eventness import clip_feature_diff_eventness

                av_feats_by_clip: dict[str, np.ndarray] = {}
                for i, cid in enumerate(all_ids):
                    a = audio_eventness_feats_by_clip[cid]
                    cache = cache_by_clip[cid]
                    # Stage-1 uses a fixed cheap visual resolution (prefer 112 when available) to avoid coupling
                    # Stage-1 scores to Stage-2 plan knobs (e.g., sweeps that vary low_res).
                    vis_res = 112 if 112 in cache.features_by_resolution else int(min(cache.features_by_resolution))
                    vis_feats = cache.features_by_resolution[int(vis_res)]

                    vis = clip_feature_diff_eventness(vis_feats, metric="cosine")
                    vis_scores = minmax_01([float(x) for x in vis])
                    v = np.zeros((int(num_segments), 1), dtype=np.float32)
                    for t, s in enumerate(vis_scores[: int(num_segments)]):
                        v[int(t), 0] = float(s)

                    if a.shape[0] != int(num_segments):
                        raise ValueError(f"unexpected audio feature shape for {cid}: {a.shape}")
                    av_feats_by_clip[cid] = np.concatenate([a, v], axis=1).astype(np.float32, copy=False)
                    if (i + 1) % 500 == 0 or (i + 1) == len(all_ids):
                        print(f"[{eventness_method}] built feats {i+1}/{len(all_ids)} clips", flush=True)

                av_model = _train_audio_basic_mlp_cls_eventness(
                    clip_ids_train=clip_ids_train,
                    labels_by_clip=labels_by_clip,
                    audio_feats_by_clip=av_feats_by_clip,
                    num_classes=int(num_classes),
                    device="cpu",
                )
                av_model_cpu = av_model.to(torch.device("cpu"))
                for i, cid in enumerate(all_ids):
                    feats = torch.from_numpy(av_feats_by_clip[cid]).float()
                    with torch.no_grad():
                        logits = av_model_cpu(feats)  # [T, C]
                        bg = logits[:, 0]
                        if eventness_method == "av_clipdiff_mlp_cls":
                            # Margin eventness: best non-background logit vs background logit.
                            scores = logits[:, 1:].max(dim=-1).values - bg
                        else:
                            # Class-conditional margin eventness: infer a clip-level class (exclude background) then
                            # score per-second by (logit_cls - logit_bg).
                            clip_logits = logits.mean(dim=0)
                            if int(clip_logits.shape[0]) < 2:
                                raise ValueError(
                                    f"av_clipdiff_mlp_cls_target requires num_classes>=2, got {clip_logits.shape[0]}"
                                )
                            clip_logits = clip_logits.clone()
                            clip_logits[0] = float("-inf")
                            cls = int(torch.argmax(clip_logits).item())
                            scores = logits[:, cls] - bg

                    scores_by_clip[cid] = [float(x) for x in scores.detach().cpu().numpy().astype("float32").tolist()]
                    if (i + 1) % 500 == 0 or (i + 1) == len(all_ids):
                        print(f"[{eventness_method}] scored {i+1}/{len(all_ids)} clips", flush=True)
            elif eventness_method in ("av_clip_mlp_cls", "av_clip_mlp_cls_target"):
                if audio_eventness_feats_by_clip is None:
                    raise ValueError(
                        f"internal error: audio_eventness_feats_by_clip should be precomputed for {eventness_method}"
                    )

                av_feats_by_clip: dict[str, np.ndarray] = {}
                for i, cid in enumerate(all_ids):
                    a = audio_eventness_feats_by_clip[cid]
                    cache = cache_by_clip[cid]
                    # Stage-1 uses a fixed cheap visual resolution (prefer 112 when available) to avoid coupling
                    # Stage-1 scores to Stage-2 plan knobs (e.g., sweeps that vary low_res).
                    vis_res = 112 if 112 in cache.features_by_resolution else int(min(cache.features_by_resolution))
                    v = cache.features_by_resolution[int(vis_res)].astype(np.float32, copy=False)
                    # Normalize CLIP features for stability; cached features are post-LN but not L2-normalized.
                    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

                    if a.shape[0] != int(num_segments):
                        raise ValueError(f"unexpected audio feature shape for {cid}: {a.shape}")
                    if v.shape[0] != int(num_segments):
                        raise ValueError(f"unexpected clip feature shape for {cid}: {v.shape}")
                    av_feats_by_clip[cid] = np.concatenate([a, v], axis=1).astype(np.float32, copy=False)
                    if (i + 1) % 500 == 0 or (i + 1) == len(all_ids):
                        print(f"[{eventness_method}] built feats {i+1}/{len(all_ids)} clips", flush=True)

                av_model = _train_audio_basic_mlp_cls_eventness(
                    clip_ids_train=clip_ids_train,
                    labels_by_clip=labels_by_clip,
                    audio_feats_by_clip=av_feats_by_clip,
                    num_classes=int(num_classes),
                    device="cpu",
                    hidden_dim=128,
                    dropout=0.1,
                )
                av_model_cpu = av_model.to(torch.device("cpu"))
                for i, cid in enumerate(all_ids):
                    feats = torch.from_numpy(av_feats_by_clip[cid]).float()
                    with torch.no_grad():
                        logits = av_model_cpu(feats)  # [T, C]
                        bg = logits[:, 0]
                        if eventness_method == "av_clip_mlp_cls":
                            scores = logits[:, 1:].max(dim=-1).values - bg
                        else:
                            clip_logits = logits.mean(dim=0)
                            if int(clip_logits.shape[0]) < 2:
                                raise ValueError(
                                    f"av_clip_mlp_cls_target requires num_classes>=2, got {clip_logits.shape[0]}"
                                )
                            clip_logits = clip_logits.clone()
                            clip_logits[0] = float("-inf")
                            cls = int(torch.argmax(clip_logits).item())
                            scores = logits[:, cls] - bg

                    scores_by_clip[cid] = [float(x) for x in scores.detach().cpu().numpy().astype("float32").tolist()]
                    if (i + 1) % 500 == 0 or (i + 1) == len(all_ids):
                        print(f"[{eventness_method}] scored {i+1}/{len(all_ids)} clips", flush=True)
            elif eventness_method == "av_clipdiff_tcn":
                if audio_eventness_feats_by_clip is None:
                    raise ValueError(
                        "internal error: audio_eventness_feats_by_clip should be precomputed for av_clipdiff_tcn"
                    )
                from avs.vision.cheap_eventness import clip_feature_diff_eventness

                av_feats_by_clip: dict[str, np.ndarray] = {}
                for i, cid in enumerate(all_ids):
                    a = audio_eventness_feats_by_clip[cid]
                    cache = cache_by_clip[cid]
                    # Stage-1 uses a fixed cheap visual resolution (prefer 112 when available) to avoid coupling
                    # Stage-1 scores to Stage-2 plan knobs (e.g., sweeps that vary low_res).
                    vis_res = 112 if 112 in cache.features_by_resolution else int(min(cache.features_by_resolution))
                    vis_feats = cache.features_by_resolution[int(vis_res)]

                    vis = clip_feature_diff_eventness(vis_feats, metric="cosine")
                    vis_scores = minmax_01([float(x) for x in vis])
                    v = np.zeros((int(num_segments), 1), dtype=np.float32)
                    for t, s in enumerate(vis_scores[: int(num_segments)]):
                        v[int(t), 0] = float(s)

                    if a.shape[0] != int(num_segments):
                        raise ValueError(f"unexpected audio feature shape for {cid}: {a.shape}")
                    av_feats_by_clip[cid] = np.concatenate([a, v], axis=1).astype(np.float32, copy=False)
                    if (i + 1) % 500 == 0 or (i + 1) == len(all_ids):
                        print(f"[av_clipdiff_tcn] built feats {i+1}/{len(all_ids)} clips", flush=True)

                av_model = _train_audio_tcn_eventness(
                    clip_ids_train=clip_ids_train,
                    labels_by_clip=labels_by_clip,
                    audio_feats_by_clip=av_feats_by_clip,
                    device="cpu",
                    epochs=50,
                    batch_size=128,
                    lr=1e-3,
                    hidden_channels=64,
                    kernel_size=3,
                    dropout=0.1,
                )
                av_model_cpu = av_model.to(torch.device("cpu"))
                av_model_cpu.eval()
                for i, cid in enumerate(all_ids):
                    feats = torch.from_numpy(av_feats_by_clip[cid]).float()
                    with torch.no_grad():
                        logits = av_model_cpu(feats).squeeze(-1)
                    scores_by_clip[cid] = [float(x) for x in logits.detach().cpu().numpy().astype("float32").tolist()]
                    if (i + 1) % 500 == 0 or (i + 1) == len(all_ids):
                        print(f"[av_clipdiff_tcn] scored {i+1}/{len(all_ids)} clips", flush=True)
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
            elif eventness_method in ("audio_basic_tcn", "audio_fbank_tcn"):
                if audio_eventness_feats_by_clip is None:
                    raise ValueError(f"internal error: audio_eventness_feats_by_clip should be precomputed for {eventness_method}")
                audio_model = _train_audio_tcn_eventness(
                    clip_ids_train=clip_ids_train,
                    labels_by_clip=labels_by_clip,
                    audio_feats_by_clip=audio_eventness_feats_by_clip,
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
                                raise ValueError(
                                    f"audio_basic_mlp_cls_target requires num_classes>=2, got {clip_logits.shape[0]}"
                                )
                            clip_logits = clip_logits.clone()
                            clip_logits[0] = float("-inf")
                            cls = int(torch.argmax(clip_logits).item())
                            scores = logits[:, cls]

                    scores_by_clip[cid] = [float(x) for x in scores.detach().cpu().numpy().astype("float32").tolist()]
            elif eventness_method in ("vision_binary_lr", "vision_binary_mlp"):
                # Supervised binary eventness on cheap visual features (frozen CLIP cache at low_res):
                # predict event vs background, then use logits as Stage-1 scores.
                #
                # This is a strong cheap-visual baseline/control rather than the core audio-only proposal.
                vision_feats_by_clip: dict[str, np.ndarray] = {}
                vis_res = int(cfg.low_res)
                for cid in all_ids:
                    cache = cache_by_clip[cid]
                    if vis_res not in cache.features_by_resolution:
                        raise ValueError(
                            f"cache missing resolution={vis_res} for {cid} (available={sorted(cache.features_by_resolution)})"
                        )
                    vision_feats_by_clip[cid] = cache.features_by_resolution[vis_res].astype(np.float32, copy=False)

                if eventness_method == "vision_binary_lr":
                    vision_model = _train_audio_basic_lr_eventness(
                        clip_ids_train=clip_ids_train,
                        labels_by_clip=labels_by_clip,
                        audio_feats_by_clip=vision_feats_by_clip,
                        device="cpu",
                    )
                else:
                    vision_model = _train_audio_basic_mlp_eventness(
                        clip_ids_train=clip_ids_train,
                        labels_by_clip=labels_by_clip,
                        audio_feats_by_clip=vision_feats_by_clip,
                        device="cpu",
                        hidden_dim=128,
                    )

                vision_model_cpu = vision_model.to(torch.device("cpu"))
                for cid in all_ids:
                    feats = torch.from_numpy(vision_feats_by_clip[cid]).float()
                    with torch.no_grad():
                        logits = vision_model_cpu(feats).squeeze(-1)
                    scores_by_clip[cid] = [float(x) for x in logits.detach().cpu().numpy().astype("float32").tolist()]
            elif eventness_method in ("vision_mlp_cls", "vision_mlp_cls_target"):
                # Train a tiny supervised per-second classifier on cheap visual features (frozen CLIP cache at low_res).
                #
                # Note: This uses *visual* features to generate anchors; treat as a strong cheap-visual baseline/control
                # rather than the core Listen-then-Look audio-only proposal.
                vision_feats_by_clip: dict[str, np.ndarray] = {}
                vis_res = int(cfg.low_res)
                for cid in all_ids:
                    cache = cache_by_clip[cid]
                    if vis_res not in cache.features_by_resolution:
                        raise ValueError(f"cache missing resolution={vis_res} for {cid} (available={sorted(cache.features_by_resolution)})")
                    vision_feats_by_clip[cid] = cache.features_by_resolution[vis_res].astype(np.float32, copy=False)

                vision_model = _train_audio_basic_mlp_cls_eventness(
                    clip_ids_train=clip_ids_train,
                    labels_by_clip=labels_by_clip,
                    audio_feats_by_clip=vision_feats_by_clip,
                    num_classes=int(num_classes),
                    device="cpu",
                    hidden_dim=128,
                )
                vision_model_cpu = vision_model.to(torch.device("cpu"))

                for cid in all_ids:
                    feats = torch.from_numpy(vision_feats_by_clip[cid]).float()
                    with torch.no_grad():
                        logits = vision_model_cpu(feats)
                        if eventness_method == "vision_mlp_cls":
                            probs = torch.softmax(logits, dim=-1)
                            scores = 1.0 - probs[:, 0]
                        else:
                            clip_logits = logits.mean(dim=0)
                            if int(clip_logits.shape[0]) < 2:
                                raise ValueError(
                                    f"vision_mlp_cls_target requires num_classes>=2, got {clip_logits.shape[0]}"
                                )
                            clip_logits = clip_logits.clone()
                            clip_logits[0] = float("-inf")
                            cls = int(torch.argmax(clip_logits).item())
                            scores = logits[:, cls]

                    scores_by_clip[cid] = [float(x) for x in scores.detach().cpu().numpy().astype("float32").tolist()]
            else:
                if eventness_method == "ast_evt_mlp":
                    assert ast_probe is not None

                    ast_evt_mlp_model, train_logits_by_clip = _train_ast_evt_mlp_eventness(
                        clip_ids_train=clip_ids_train,
                        labels_by_clip=labels_by_clip,
                        audio_dir=audio_dir,
                        ast_probe=ast_probe,
                        num_segments=num_segments,
                        device="cpu",
                        hidden_dim=128,
                    )
                    ast_evt_mlp_cpu = ast_evt_mlp_model.to(torch.device("cpu"))
                    ast_evt_mlp_cpu.eval()

                    for i, cid in enumerate(all_ids):
                        feats = train_logits_by_clip.get(cid)
                        if feats is None:
                            wav_path = audio_dir / cid / "audio.wav"
                            feats = ast_probe.logits_per_second(wav_path, num_segments=int(num_segments))

                        feats_t = torch.from_numpy(feats).float()
                        with torch.no_grad():
                            s = ast_evt_mlp_cpu(feats_t).squeeze(-1)
                        scores_by_clip[cid] = [float(x) for x in s.detach().cpu().numpy().astype("float32").tolist()]

                        if (i + 1) % 200 == 0 or (i + 1) == len(all_ids):
                            print(f"[ast_evt_mlp] scored {i+1}/{len(all_ids)} clips", flush=True)
                elif eventness_method == "ast_lr":
                    assert ast_probe is not None

                    ast_lr_model, train_logits_by_clip = _train_ast_lr_eventness(
                        clip_ids_train=clip_ids_train,
                        labels_by_clip=labels_by_clip,
                        audio_dir=audio_dir,
                        ast_probe=ast_probe,
                        num_segments=num_segments,
                        device="cpu",
                    )
                    ast_lr_model_cpu = ast_lr_model.to(torch.device("cpu"))
                    ast_lr_model_cpu.eval()

                    for i, cid in enumerate(all_ids):
                        if cid in train_logits_by_clip:
                            feats = train_logits_by_clip[cid]
                        else:
                            wav_path = audio_dir / cid / "audio.wav"
                            feats = ast_probe.logits_per_second(wav_path, num_segments=int(num_segments))

                        feats_t = torch.from_numpy(feats).float()
                        with torch.no_grad():
                            s = ast_lr_model_cpu(feats_t).squeeze(-1)
                        scores_by_clip[cid] = [float(x) for x in s.detach().cpu().numpy().astype("float32").tolist()]

                        if (i + 1) % 200 == 0 or (i + 1) == len(all_ids):
                            print(f"[ast_lr] scored {i+1}/{len(all_ids)} clips", flush=True)
                elif eventness_method == "ast_emb_lr":
                    assert ast_probe is not None

                    ast_emb_lr_model, train_emb_by_clip = _train_ast_emb_lr_eventness(
                        clip_ids_train=clip_ids_train,
                        labels_by_clip=labels_by_clip,
                        audio_dir=audio_dir,
                        ast_probe=ast_probe,
                        num_segments=num_segments,
                        device="cpu",
                    )
                    ast_emb_lr_cpu = ast_emb_lr_model.to(torch.device("cpu"))
                    ast_emb_lr_cpu.eval()

                    for i, cid in enumerate(all_ids):
                        feats = train_emb_by_clip.get(cid)
                        if feats is None:
                            wav_path = audio_dir / cid / "audio.wav"
                            feats = ast_probe.embeddings_per_second(wav_path, num_segments=int(num_segments))

                        feats_t = torch.from_numpy(feats).float()
                        with torch.no_grad():
                            s = ast_emb_lr_cpu(feats_t).squeeze(-1)
                        scores_by_clip[cid] = [float(x) for x in s.detach().cpu().numpy().astype("float32").tolist()]

                        if (i + 1) % 200 == 0 or (i + 1) == len(all_ids):
                            print(f"[ast_emb_lr] scored {i+1}/{len(all_ids)} clips", flush=True)
                elif eventness_method in ("ast_mlp_cls", "ast_mlp_cls_target"):
                    assert ast_probe is not None

                    ast_mlp_model, train_logits_by_clip = _train_ast_mlp_cls_eventness(
                        clip_ids_train=clip_ids_train,
                        labels_by_clip=labels_by_clip,
                        audio_dir=audio_dir,
                        ast_probe=ast_probe,
                        num_classes=int(num_classes),
                        num_segments=num_segments,
                        device="cpu",
                        hidden_dim=128,
                    )
                    ast_mlp_cpu = ast_mlp_model.to(torch.device("cpu"))
                    ast_mlp_cpu.eval()

                    for i, cid in enumerate(all_ids):
                        if cid in train_logits_by_clip:
                            feats = train_logits_by_clip[cid]
                        else:
                            wav_path = audio_dir / cid / "audio.wav"
                            feats = ast_probe.logits_per_second(wav_path, num_segments=int(num_segments))

                        feats_t = torch.from_numpy(feats).float()
                        with torch.no_grad():
                            logits = ast_mlp_cpu(feats_t)  # [T, num_classes]
                            bg = logits[:, 0]
                            if eventness_method == "ast_mlp_cls":
                                mx = logits[:, 1:].max(dim=-1).values
                                scores = mx - bg
                            else:
                                clip_logits = logits.mean(dim=0)
                                clip_logits = clip_logits.clone()
                                clip_logits[0] = float("-inf")
                                cls = int(torch.argmax(clip_logits).item())
                                scores = logits[:, cls] - bg

                        scores_by_clip[cid] = [float(x) for x in scores.detach().cpu().numpy().astype("float32").tolist()]
                        if (i + 1) % 200 == 0 or (i + 1) == len(all_ids):
                            print(f"[ast_mlp_cls] scored {i+1}/{len(all_ids)} clips", flush=True)
                else:
                    speech_idx: list[int] | None = None
                    if eventness_method in ("ast_nonspeech_max", "energy_nonspeech_ast"):
                        assert ast_probe is not None
                        speech_idx = []
                        for k, v in ast_probe.model.config.id2label.items():
                            name = str(v).strip().lower()
                            if ("speech" in name) or ("conversation" in name) or ("narration" in name) or ("talking" in name):
                                speech_idx.append(int(k))
                        speech_idx = sorted(set(int(x) for x in speech_idx))

                    clap_probe = None
                    clap_text = None
                    clip_probe = None
                    clip_text = None
                    clip_scale = 1.0
                    audio_scale = 10.0
                    clap_lr_model_cpu = None
                    clap_lr_feats_by_train: dict[str, np.ndarray] | None = None
                    clap_mlp_cls_model_cpu = None
                    clap_mlp_cls_emb_by_train: dict[str, np.ndarray] | None = None

                    def _softmax_np(x: np.ndarray, *, axis: int = -1) -> np.ndarray:
                        x = np.asarray(x, dtype=np.float32)
                        x = x - np.max(x, axis=axis, keepdims=True)
                        ex = np.exp(x)
                        return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-12)

                    if eventness_method in (
                        "av_clap_clip_agree",
                        "clap_evt",
                        "clap_lr",
                        "clap_mlp_cls",
                        "clap_mlp_cls_target",
                    ):
                        from avs.audio.clap_probe import ClapProbe, ClapProbeConfig

                        # CLAP is expensive; run it on `audio_device`.
                        clap_probe = ClapProbe(ClapProbeConfig(pretrained=True, device=str(audio_device), dtype="float32"))

                        if eventness_method in ("av_clap_clip_agree", "clap_evt", "clap_lr"):
                            if class_names is None:
                                raise ValueError(
                                    f"{eventness_method} requires class_names (idx->label) to build text prompts"
                                )
                            if len(class_names) != int(num_classes):
                                raise ValueError(
                                    f"{eventness_method}: expected len(class_names)==num_classes ({num_classes}), got {len(class_names)}"
                                )
                            event_labels = [str(x) for x in class_names[1:]]  # exclude background
                            clap_prompts = [f"a sound of {lab}" for lab in event_labels]
                            clap_text = clap_probe.text_embeddings(clap_prompts)  # [C, D]
                            if eventness_method == "av_clap_clip_agree":
                                from avs.vision.clip_text import ClipTextProbe, ClipTextProbeConfig

                                # CLIP text+projection can stay on CPU.
                                clip_probe = ClipTextProbe(
                                    ClipTextProbeConfig(pretrained=True, device="cpu", dtype="float32")
                                )
                                clip_prompts = [f"a photo of {lab}" for lab in event_labels]
                                clip_text = clip_probe.text_features(clip_prompts)  # [C, 512]
                                clip_scale = float(clip_probe.logit_scale())
                            elif eventness_method == "clap_lr":
                                if clap_probe is None or clap_text is None:
                                    raise ValueError("internal error: clap_lr probes are not initialized")
                                clap_lr_model, clap_lr_feats_by_train = _train_clap_lr_eventness(
                                    clip_ids_train=clip_ids_train,
                                    labels_by_clip=labels_by_clip,
                                    audio_dir=audio_dir,
                                    clap_probe=clap_probe,
                                    clap_text=clap_text,
                                    num_segments=int(num_segments),
                                    device="cpu",
                                )
                                clap_lr_model_cpu = clap_lr_model.to(torch.device("cpu"))
                                clap_lr_model_cpu.eval()

                        if eventness_method in ("clap_mlp_cls", "clap_mlp_cls_target"):
                            if clap_probe is None:
                                raise ValueError("internal error: clap_mlp_cls requires clap_probe")
                            clap_mlp_cls_emb_by_train = {}
                            for i, cid in enumerate(clip_ids_train):
                                wav_path = audio_dir / cid / "audio.wav"
                                emb = clap_probe.audio_embeddings_per_second(wav_path, num_segments=int(num_segments))
                                clap_mlp_cls_emb_by_train[cid] = emb.astype(np.float32, copy=False)
                                if (i + 1) % 200 == 0 or (i + 1) == len(clip_ids_train):
                                    print(f"[clap_mlp_cls] feats train {i+1}/{len(clip_ids_train)} clips", flush=True)

                            clap_mlp_cls_model = _train_audio_basic_mlp_cls_eventness(
                                clip_ids_train=clip_ids_train,
                                labels_by_clip=labels_by_clip,
                                audio_feats_by_clip=clap_mlp_cls_emb_by_train,
                                num_classes=int(num_classes),
                                device="cpu",
                                hidden_dim=128,
                                dropout=0.1,
                            )
                            clap_mlp_cls_model_cpu = clap_mlp_cls_model.to(torch.device("cpu"))
                            clap_mlp_cls_model_cpu.eval()

                    for i, cid in enumerate(all_ids):
                        wav_path = audio_dir / cid / "audio.wav"
                        if eventness_method == "energy":
                            ev = compute_eventness_wav_energy(wav_path, num_segments=num_segments)
                            scores_by_clip[cid] = [float(x) for x in ev.scores]
                        elif eventness_method == "energy_delta":
                            ev = compute_eventness_wav_energy_delta(wav_path, num_segments=num_segments)
                            scores_by_clip[cid] = [float(x) for x in ev.scores]
                        elif eventness_method == "energy_stride_max":
                            ev = compute_eventness_wav_energy_stride_max(
                                wav_path, num_segments=num_segments, stride_s=0.2, win_s=0.4
                            )
                            scores_by_clip[cid] = [float(x) for x in ev.scores]
                        elif eventness_method == "clap_lr":
                            if clap_probe is None or clap_text is None or clap_lr_model_cpu is None:
                                raise ValueError("internal error: clap_lr probes are not initialized")
                            feats = None if clap_lr_feats_by_train is None else clap_lr_feats_by_train.get(cid)
                            if feats is None:
                                aud_feat = clap_probe.audio_embeddings_per_second(wav_path, num_segments=int(num_segments))  # [T, D]
                                feats = (aud_feat @ clap_text.T).astype(np.float32, copy=False)  # [T, C]
                            with torch.no_grad():
                                s = clap_lr_model_cpu(torch.from_numpy(feats).float()).squeeze(-1)
                            scores_by_clip[cid] = [float(x) for x in s.detach().cpu().numpy().astype("float32").tolist()]
                            if (i + 1) % 200 == 0 or (i + 1) == len(all_ids):
                                print(f"[clap_lr] scored {i+1}/{len(all_ids)} clips", flush=True)
                        elif eventness_method == "clap_evt":
                            if clap_probe is None or clap_text is None:
                                raise ValueError("internal error: clap_evt probes are not initialized")
                            aud_feat = clap_probe.audio_embeddings_per_second(wav_path, num_segments=int(num_segments))  # [T, D]
                            aud_logits = aud_feat @ clap_text.T  # [T, C]
                            aud_probs = _softmax_np(aud_logits * float(audio_scale), axis=1)
                            scores = aud_probs.max(axis=1).astype(np.float32, copy=False)
                            scores_by_clip[cid] = [float(x) for x in scores.tolist()]
                        elif eventness_method in ("clap_mlp_cls", "clap_mlp_cls_target"):
                            if clap_probe is None or clap_mlp_cls_model_cpu is None:
                                raise ValueError("internal error: clap_mlp_cls probes are not initialized")
                            emb = None if clap_mlp_cls_emb_by_train is None else clap_mlp_cls_emb_by_train.get(cid)
                            if emb is None:
                                emb = clap_probe.audio_embeddings_per_second(wav_path, num_segments=int(num_segments))
                            with torch.no_grad():
                                logits = clap_mlp_cls_model_cpu(torch.from_numpy(emb).float())  # [T,C]
                                bg = logits[:, 0]
                                if eventness_method == "clap_mlp_cls":
                                    probs = torch.softmax(logits, dim=-1)
                                    scores = 1.0 - probs[:, 0]
                                else:
                                    clip_logits = logits.mean(dim=0)
                                    if int(clip_logits.shape[0]) < 2:
                                        raise ValueError(
                                            f"clap_mlp_cls_target requires num_classes>=2, got {clip_logits.shape[0]}"
                                        )
                                    clip_logits = clip_logits.clone()
                                    clip_logits[0] = float("-inf")
                                    cls = int(torch.argmax(clip_logits).item())
                                    scores = logits[:, cls] - bg
                            scores_by_clip[cid] = [
                                float(x) for x in scores.detach().cpu().numpy().astype("float32").tolist()
                            ]
                            if (i + 1) % 200 == 0 or (i + 1) == len(all_ids):
                                print(f"[clap_mlp_cls] scored {i+1}/{len(all_ids)} clips", flush=True)
                        elif eventness_method == "av_clap_clip_agree":
                            if clap_probe is None or clap_text is None or clip_probe is None or clip_text is None:
                                raise ValueError("internal error: av_clap_clip_agree probes are not initialized")

                            cache = cache_by_clip[cid]
                            vis_res = 112 if 112 in cache.features_by_resolution else int(min(cache.features_by_resolution))
                            vis_pooled = cache.features_by_resolution[int(vis_res)]

                            vis_feat = clip_probe.project_image_features(vis_pooled)  # [T, 512]
                            vis_logits = vis_feat @ clip_text.T  # [T, C]
                            vis_probs = _softmax_np(vis_logits * float(clip_scale), axis=1)

                            aud_feat = clap_probe.audio_embeddings_per_second(wav_path, num_segments=int(num_segments))  # [T, D]
                            aud_logits = aud_feat @ clap_text.T  # [T, C]
                            aud_probs = _softmax_np(aud_logits * float(audio_scale), axis=1)

                            agree_tc = aud_probs * vis_probs  # [T, C]
                            cls = int(np.argmax(np.max(agree_tc, axis=0)))
                            scores = agree_tc[:, cls].astype(np.float32, copy=False)
                            scores_by_clip[cid] = [float(x) for x in scores.tolist()]
                            if (i + 1) % 100 == 0 or (i + 1) == len(all_ids):
                                print(f"[av_clap_clip_agree] scored {i+1}/{len(all_ids)} clips", flush=True)
                        elif eventness_method == "energy_nonspeech_ast":
                            # Speech-aware suppression: non-speech audio peaks are more likely to correspond to audio-visual events.
                            assert ast_probe is not None
                            if speech_idx is None:
                                raise ValueError("internal error: speech_idx must be initialized for energy_nonspeech_ast")
                            ev = compute_eventness_wav_energy_stride_max(
                                wav_path, num_segments=num_segments, stride_s=0.2, win_s=0.4
                            )
                            base = minmax_01([float(x) for x in ev.scores])
                            logits = ast_probe.logits_per_second(wav_path, num_segments=int(num_segments))  # [T,C]
                            probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float32)))
                            speech = (
                                probs[:, speech_idx].max(axis=1)
                                if speech_idx
                                else np.zeros((int(num_segments),), dtype=np.float32)
                            )
                            scores_by_clip[cid] = [
                                float(b) * (1.0 - float(s)) for b, s in zip(base, speech.tolist(), strict=True)
                            ]
                        elif eventness_method == "asr_vad":
                            # Speech-aware anchor probe (deployable): suppress speech-dominant seconds.
                            #
                            # Implementation (v1): energy_stride_max (normalized) × (1 - VAD_speech_ratio).
                            # This targets a common failure mode in AVE (YouTube narration / off-screen speech)
                            # where audio peaks are not evidence-bearing for the visual label.
                            from avs.audio.vad_webrtc import WebRtcVadConfig, webrtcvad_speech_ratio_per_second

                            ev = compute_eventness_wav_energy_stride_max(
                                wav_path, num_segments=num_segments, stride_s=0.2, win_s=0.4
                            )
                            base = minmax_01([float(x) for x in ev.scores])
                            speech = webrtcvad_speech_ratio_per_second(
                                wav_path, num_segments=num_segments, cfg=WebRtcVadConfig(aggressiveness=2, frame_ms=30)
                            )
                            scores_by_clip[cid] = [float(b) * (1.0 - float(s)) for b, s in zip(base, speech, strict=True)]
                        elif eventness_method == "av_fused":
                            ev = compute_eventness_wav_energy_stride_max(
                                wav_path, num_segments=num_segments, stride_s=0.2, win_s=0.4
                            )
                            audio_scores = minmax_01([float(x) for x in ev.scores])

                            frames_dir = audio_dir / cid / "frames"
                            try:
                                from avs.vision.cheap_eventness import frame_diff_eventness, list_frames

                                frames = list_frames(frames_dir) if frames_dir.exists() else []
                                vis = frame_diff_eventness(frames, size=32)
                            except Exception:
                                vis = []

                            vis_scores = minmax_01([float(x) for x in vis])
                            scores_by_clip[cid] = scale(
                                fuse_max(audio_scores, vis_scores, num_segments=num_segments),
                                AV_FUSED_SCORE_SCALE,
                            )
                        elif eventness_method == "av_fused_prod":
                            ev = compute_eventness_wav_energy_stride_max(
                                wav_path, num_segments=num_segments, stride_s=0.2, win_s=0.4
                            )
                            audio_scores = minmax_01([float(x) for x in ev.scores])

                            frames_dir = audio_dir / cid / "frames"
                            try:
                                from avs.vision.cheap_eventness import frame_diff_eventness, list_frames

                                frames = list_frames(frames_dir) if frames_dir.exists() else []
                                vis = frame_diff_eventness(frames, size=32)
                            except Exception:
                                vis = []

                            vis_scores = minmax_01([float(x) for x in vis])
                            scores_by_clip[cid] = scale(
                                fuse_prod(audio_scores, vis_scores, num_segments=num_segments),
                                AV_FUSED_SCORE_SCALE,
                            )
                        elif eventness_method in (
                            "vision_clipdiff",
                            "energy_autoshift_clipdiff",
                            "energy_autoshift_clipdiff_pos",
                            "av_fused_clipdiff",
                            "av_fused_clipdiff_prod",
                            "moe_energy_clipdiff",
                        ):
                            from avs.vision.cheap_eventness import clip_feature_diff_eventness

                            cache = cache_by_clip[cid]
                            vis_res = int(cfg.low_res)
                            if vis_res not in cache.features_by_resolution:
                                vis_res = int(min(cache.features_by_resolution))
                            vis_feats = cache.features_by_resolution[int(vis_res)]

                            vis = clip_feature_diff_eventness(vis_feats, metric="cosine")
                            vis_scores = minmax_01([float(x) for x in vis])

                            if eventness_method == "vision_clipdiff":
                                scores_by_clip[cid] = scale(vis_scores, AV_FUSED_SCORE_SCALE)
                            elif eventness_method == "energy_autoshift_clipdiff":
                                ev = compute_eventness_wav_energy(wav_path, num_segments=num_segments)
                                audio_raw = [float(x) for x in ev.scores]
                                audio_scores = minmax_01(audio_raw)
                                s = best_shift_by_corr(audio_scores, vis_scores, shifts=[-2, -1, 0, 1, 2])
                                if autoshift_by_clip is not None:
                                    autoshift_by_clip[cid] = int(s)
                                scores_by_clip[cid] = shift_scores(audio_raw, int(s))
                            elif eventness_method == "energy_autoshift_clipdiff_pos":
                                ev = compute_eventness_wav_energy(wav_path, num_segments=num_segments)
                                audio_raw = [float(x) for x in ev.scores]
                                audio_scores = minmax_01(audio_raw)
                                s = best_shift_by_corr(audio_scores, vis_scores, shifts=[0, 1, 2])
                                if autoshift_by_clip is not None:
                                    autoshift_by_clip[cid] = int(s)
                                scores_by_clip[cid] = shift_scores(audio_raw, int(s))
                            elif eventness_method == "moe_energy_clipdiff":
                                ev = compute_eventness_wav_energy(wav_path, num_segments=num_segments)
                                audio_raw = [float(x) for x in ev.scores]
                                # If audio would fall back under the legacy std gate, fall back to semantic visual motion.
                                if float(np.asarray(audio_raw, dtype=np.float32).std()) < float(cfg.anchor_std_threshold):
                                    scores_by_clip[cid] = scale(vis_scores, AV_FUSED_SCORE_SCALE)
                                else:
                                    scores_by_clip[cid] = audio_raw
                            else:
                                ev = compute_eventness_wav_energy_stride_max(
                                    wav_path, num_segments=num_segments, stride_s=0.2, win_s=0.4
                                )
                                audio_scores = minmax_01([float(x) for x in ev.scores])
                                if eventness_method == "av_fused_clipdiff":
                                    fused = fuse_max(audio_scores, vis_scores, num_segments=num_segments)
                                else:
                                    fused = fuse_prod(audio_scores, vis_scores, num_segments=num_segments)
                                scores_by_clip[cid] = scale(fused, AV_FUSED_SCORE_SCALE)
                        elif eventness_method == "ast":
                            assert ast_probe is not None
                            scores_by_clip[cid] = [
                                float(x) for x in ast_probe.eventness_per_second(wav_path, num_segments=num_segments)
                            ]
                        elif eventness_method == "ast_nonspeech_max":
                            assert ast_probe is not None
                            logits = ast_probe.logits_per_second(wav_path, num_segments=int(num_segments))  # [T, C]
                            probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float32)))
                            if speech_idx:
                                probs[:, speech_idx] = 0.0
                            scores = probs.max(axis=1).astype(np.float32, copy=False)
                            scores_by_clip[cid] = [float(x) for x in scores.tolist()]
                        elif eventness_method == "panns":
                            assert panns_probe is not None
                            scores_by_clip[cid] = [
                                float(x) for x in panns_probe.eventness_per_second(wav_path, num_segments=num_segments)
                            ]
                        elif eventness_method == "audiomae":
                            assert audiomae_probe is not None
                            scores_by_clip[cid] = [
                                float(x) for x in audiomae_probe.eventness_per_second(wav_path, num_segments=num_segments)
                            ]
                        else:
                            raise ValueError(f"unsupported eventness_method: {eventness_method}")

                    if eventness_method in (
                        "av_clap_clip_agree",
                        "clap_evt",
                        "clap_lr",
                        "clap_mlp_cls",
                        "clap_mlp_cls_target",
                    ):
                        # Free GPU memory before head training; CLAP is large and not needed beyond scoring.
                        clap_probe = None
                        clip_probe = None
                        if str(audio_device).startswith("cuda"):
                            torch.cuda.empty_cache()

    debug_eval = None
    anchors_by_clip: dict[str, list[int]] | None = None
    anchor_debug_by_clip: dict[str, dict] | None = None
    if scores_by_clip is not None:
        sel_by_clip: dict[str, AnchorSelectionResult] = {}
        sel_raw_by_clip: dict[str, AnchorSelectionResult] = {}
        sel_base_by_clip: dict[str, AnchorSelectionResult] = {}
        shift_by_clip: dict[str, int] = {}
        gate_method = str(cfg.anchor_gate_method or "none")
        gate_thr = float(cfg.anchor_gate_threshold)
        gate_requested = (gate_method != "none") and (float(gate_thr) > 0.0)
        if gate_method != "none" and not gate_requested:
            print(f"[anchor_gate] disabled: method={gate_method!r} requires anchor_gate_threshold>0", flush=True)
        gate_prob_by_clip: dict[str, float] = {}
        gate_rescued_by_clip: dict[str, bool] = {}

        for cid in all_ids:
            scores = scores_by_clip[cid]
            # Some Stage-1 methods already bake in per-clip shifts; do not apply global anchor_shift again.
            shift = 0 if str(eventness_method) in ("energy_autoshift_clipdiff", "energy_autoshift_clipdiff_pos") else int(cfg.anchor_shift)
            shift_by_clip[cid] = int(shift)
            if gate_requested:
                # Always select anchors first (disable hard fallback inside the selector),
                # then apply the rescue policy below.
                sel_raw_by_clip[cid] = anchors_from_scores_with_debug(
                    scores,
                    k=int(cfg.k),
                    num_segments=num_segments,
                    shift=int(shift),
                    std_threshold=float(0.0),
                    select=str(cfg.anchor_select),
                    nms_radius=int(cfg.anchor_nms_radius),
                    nms_strong_gap=float(cfg.anchor_nms_strong_gap),
                    anchor_window=int(cfg.anchor_window),
                    smooth_window=int(cfg.anchor_smooth_window),
                    smooth_mode=str(cfg.anchor_smooth_mode),
                    conf_metric=str(cfg.anchor_conf_metric) if cfg.anchor_conf_metric is not None else None,
                    conf_threshold=float(0.0),
                )

            sel_base_by_clip[cid] = anchors_from_scores_with_debug(
                scores,
                k=int(cfg.k),
                num_segments=num_segments,
                shift=int(shift),
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

        gate_model: _AnchorGateLR | None = None
        if gate_requested:
            if gate_method == "lr_top1hit_v1":
                gate_model = _train_anchor_gate_lr_top1hit_v1(
                    clip_ids_train=clip_ids_train,
                    labels_by_clip=labels_by_clip,
                    scores_by_clip=scores_by_clip,
                    sel_by_clip=sel_raw_by_clip,
                    cfg=cfg,
                    num_segments=num_segments,
                )
            elif gate_method == "lr_top1hit_all_v1":
                gate_model = _train_anchor_gate_lr_top1hit_all_v1(
                    clip_ids_train=clip_ids_train,
                    labels_by_clip=labels_by_clip,
                    scores_by_clip=scores_by_clip,
                    sel_by_clip=sel_raw_by_clip,
                    cfg=cfg,
                    num_segments=num_segments,
                )
            else:
                raise ValueError(
                    f"unknown anchor_gate_method: {gate_method!r}; expected 'none', 'lr_top1hit_v1', or 'lr_top1hit_all_v1'"
                )

            if gate_model is None:
                print(f"[anchor_gate] WARNING: gate training returned None; disabling gate (method={gate_method!r})", flush=True)
                gate_requested = False

        # Default: use the base selector behavior (hard fallback by conf/std threshold).
        sel_by_clip = dict(sel_base_by_clip)

        if gate_requested and gate_model is not None:
            base_thr = float(cfg.anchor_conf_threshold) if cfg.anchor_conf_threshold is not None else float(cfg.anchor_std_threshold)
            # Gate policy:
            #   - lr_top1hit_v1: rescue-only (accept if base_pass OR gate_pass).
            #   - lr_top1hit_all_v1: veto/primary gate (accept if gate_pass).
            gate_vetoed_by_clip: dict[str, bool] = {}
            for cid in all_ids:
                sel = sel_raw_by_clip.get(cid) or sel_base_by_clip[cid]
                scores = scores_by_clip[cid]
                anchors = [int(x) for x in (sel.anchors or [])]
                p = _anchor_gate_prob_lr(gate_model, scores=scores, anchors=anchors, num_segments=num_segments)
                gate_prob_by_clip[cid] = float(p)
                base_pass = (float(base_thr) <= 0.0) or (float(sel.conf_value) >= float(base_thr))
                gate_pass = float(p) >= float(gate_thr)
                if gate_method == "lr_top1hit_v1":
                    accept = base_pass or gate_pass
                    rescued = (not base_pass) and gate_pass
                    vetoed = False
                elif gate_method == "lr_top1hit_all_v1":
                    accept = base_pass and gate_pass
                    rescued = False
                    vetoed = base_pass and (not gate_pass)
                else:
                    raise ValueError(
                        f"internal error: unexpected gate_method {gate_method!r} in gate policy (training should have validated)"
                    )
                gate_rescued_by_clip[cid] = bool(rescued)
                gate_vetoed_by_clip[cid] = bool(vetoed)

                if not accept:
                    if bool(vetoed):
                        reason = "gate_veto"
                    else:
                        reason = "conf_and_gate_below_threshold" if float(base_thr) > 0.0 else "gate_below_threshold"
                    sel_by_clip[cid] = AnchorSelectionResult(
                        anchors=[],
                        conf_metric=str(sel.conf_metric),
                        conf_value=float(sel.conf_value),
                        conf_threshold=float(base_thr),
                        fallback_used=True,
                        fallback_reason=reason,
                    )
                else:
                    sel_by_clip[cid] = AnchorSelectionResult(
                        anchors=anchors,
                        conf_metric=str(sel.conf_metric),
                        conf_value=float(sel.conf_value),
                        conf_threshold=float(base_thr),
                        fallback_used=False,
                        fallback_reason=None,
                    )

        veto_method = str(cfg.anchor2_veto_method or "none")
        anchor2_lr: _Anchor2VetoLR | None = None
        if veto_method == "lr_v1":
            anchor2_lr = _train_anchor2_veto_lr(
                clip_ids_train=clip_ids_train,
                labels_by_clip=labels_by_clip,
                scores_by_clip=scores_by_clip,
                sel_by_clip=sel_by_clip,
                cfg=cfg,
                num_segments=num_segments,
            )

        anchors_by_clip = {}
        anchor_debug_by_clip = {}
        for cid in all_ids:
            scores = scores_by_clip[cid]
            shift = int(shift_by_clip.get(cid, 0))
            sel = sel_by_clip[cid]
            anchors = [int(x) for x in (sel.anchors or [])]
            fallback_used = bool(sel.fallback_used)
            fallback_reason = sel.fallback_reason
            fallback_mode = str(cfg.anchor_fallback_mode or "uniform")
            fallback_visual_mode_used: str | None = None
            fallback_visual_anchors: list[int] | None = None
            fallback_visual_conf_metric: str | None = None
            fallback_visual_conf_threshold: float | None = None
            fallback_visual_conf_value: float | None = None

            # Optional fallback *plan* that uses cheap-visual anchors instead of uniform when the audio gate falls back.
            # Note: we keep `fallback_used=True` to indicate "audio gate failed" for downstream diagnostics and to
            # avoid mixing these clips into audio-specific veto training.
            if fallback_used and fallback_mode != "uniform":
                try:
                    vis_scores: list[float] = []
                    if fallback_mode == "cheap_visual_clipdiff":
                        from avs.vision.cheap_eventness import clip_feature_diff_eventness

                        cache = cache_by_clip[cid]
                        vis_res = 112 if 112 in cache.features_by_resolution else int(min(cache.features_by_resolution))
                        vis = clip_feature_diff_eventness(cache.features_by_resolution[int(vis_res)], metric="cosine")
                        vis_scores = minmax_01([float(x) for x in vis])
                    elif fallback_mode == "cheap_visual_framediff":
                        if audio_dir is None:
                            raise ValueError("cheap_visual_framediff fallback requires audio_dir/processed_dir")
                        from avs.vision.cheap_eventness import frame_diff_eventness, list_frames

                        frames_dir = Path(audio_dir) / str(cid) / "frames"
                        frames = list_frames(frames_dir) if frames_dir.exists() else []
                        vis = frame_diff_eventness(frames, size=32)
                        vis_scores = minmax_01([float(x) for x in vis])
                    else:
                        raise ValueError(
                            f"unknown anchor_fallback_mode: {fallback_mode!r}; expected 'uniform', "
                            "'cheap_visual_clipdiff', or 'cheap_visual_framediff'"
                        )

                    if vis_scores:
                        fallback_visual_conf_metric = str(cfg.anchor_fallback_visual_conf_metric or "std")
                        fallback_visual_conf_threshold = float(cfg.anchor_fallback_visual_conf_threshold)
                        fallback_visual_conf_value = float(
                            _confidence_value_for_scores(scores=vis_scores, metric=fallback_visual_conf_metric)
                        )
                        if float(fallback_visual_conf_threshold) > 0.0 and float(fallback_visual_conf_value) < float(
                            fallback_visual_conf_threshold
                        ):
                            # Too unconfident; keep uniform fallback.
                            pass
                        else:
                            vis_sel = anchors_from_scores_with_debug(
                                vis_scores,
                                k=int(cfg.k),
                                num_segments=num_segments,
                                shift=0,  # visual fallback is not subject to A/V shift
                                std_threshold=0.0,
                                select=str(cfg.anchor_select),
                                nms_radius=int(cfg.anchor_nms_radius),
                                nms_strong_gap=float(cfg.anchor_nms_strong_gap),
                                anchor_window=int(cfg.anchor_window),
                                smooth_window=int(cfg.anchor_smooth_window),
                                smooth_mode=str(cfg.anchor_smooth_mode),
                                conf_metric="std",
                                conf_threshold=0.0,
                            )
                            fb_anchors = [int(x) for x in (vis_sel.anchors or [])]
                            if fb_anchors:
                                anchors = fb_anchors
                                fallback_visual_mode_used = str(fallback_mode)
                                fallback_visual_anchors = list(anchors)
                except Exception as e:
                    print(f"[anchor_fallback] WARNING: {cid} fallback_mode={fallback_mode!r} failed: {e}", flush=True)

            anchor2_veto_applied = False
            anchor2_veto_dropped = False
            anchor2_keep_prob: float | None = None
            anchor2_veto_score: float | None = None
            if (not fallback_used) and len(anchors) >= 2 and veto_method != "none":
                thr = float(cfg.anchor2_veto_threshold)
                if veto_method == "top2med_norm_v1":
                    s = np.asarray(minmax_01([float(x) for x in scores]), dtype=np.float32)
                    med = float(np.median(s))
                    a2 = int(anchors[1])
                    v = float(s[a2] - med) if 0 <= a2 < int(s.size) else 0.0
                    anchor2_veto_score = float(v)
                    anchor2_veto_applied = True
                    if float(v) < float(thr):
                        anchors = anchors[:1]
                        anchor2_veto_dropped = True
                elif veto_method == "lr_v1":
                    if anchor2_lr is not None:
                        p = _anchor2_keep_prob_lr(anchor2_lr, scores=scores, anchors=anchors)
                        anchor2_keep_prob = float(p)
                        anchor2_veto_applied = True
                        if float(p) < float(thr):
                            anchors = anchors[:1]
                            anchor2_veto_dropped = True
                else:
                    raise ValueError(f"unknown anchor2_veto_method: {veto_method!r}")

            fallback_far_applied = False
            fallback_far_actual_dist: int | None = None
            fallback_far_thr = int(cfg.anchor_fallback_far_dist)
            if int(fallback_far_thr) > 0 and len(anchors) >= 2:
                dist = int(abs(int(anchors[0]) - int(anchors[1])))
                if dist > int(fallback_far_thr):
                    anchors = []
                    fallback_used = True
                    fallback_reason = "far_anchor_dist"
                    fallback_far_applied = True
                    fallback_far_actual_dist = int(dist)
            drop_far_applied = False
            drop_far_actual_dist: int | None = None
            drop_far_thr = int(cfg.anchor_drop_far_dist)
            if int(drop_far_thr) > 0 and len(anchors) >= 2:
                dist = int(abs(int(anchors[0]) - int(anchors[1])))
                if dist > int(drop_far_thr):
                    anchors = anchors[:1]
                    drop_far_applied = True
                    drop_far_actual_dist = int(dist)

            anchors_by_clip[cid] = anchors
            anchor_debug_by_clip[cid] = {
                "fallback_used": bool(fallback_used),
                "fallback_reason": fallback_reason,
                "fallback_mode": str(fallback_mode),
                "fallback_visual_mode_used": str(fallback_visual_mode_used) if fallback_visual_mode_used is not None else None,
                "fallback_visual_anchors": [int(x) for x in (fallback_visual_anchors or [])] if fallback_visual_anchors is not None else None,
                "fallback_visual_conf_metric": str(fallback_visual_conf_metric) if fallback_visual_conf_metric is not None else None,
                "fallback_visual_conf_threshold": float(fallback_visual_conf_threshold)
                if fallback_visual_conf_threshold is not None
                else None,
                "fallback_visual_conf_value": float(fallback_visual_conf_value) if fallback_visual_conf_value is not None else None,
                "conf_metric": str(sel.conf_metric),
                "conf_value": float(sel.conf_value),
                "conf_threshold": float(sel.conf_threshold),
                "gate_method": str(gate_method),
                "gate_threshold": float(gate_thr),
                "gate_prob": float(gate_prob_by_clip[cid]) if cid in gate_prob_by_clip else None,
                "gate_rescued": bool(gate_rescued_by_clip.get(cid, False)),
                "gate_vetoed": bool(gate_vetoed_by_clip.get(cid, False)) if gate_requested else False,
                "anchor_shift_used": int(shift),
                "anchor2_veto_method": str(veto_method),
                "anchor2_veto_threshold": float(cfg.anchor2_veto_threshold),
                "anchor2_veto_applied": bool(anchor2_veto_applied),
                "anchor2_veto_dropped": bool(anchor2_veto_dropped),
                "anchor2_keep_prob": float(anchor2_keep_prob) if anchor2_keep_prob is not None else None,
                "anchor2_veto_score": float(anchor2_veto_score) if anchor2_veto_score is not None else None,
                "drop_far_dist_threshold": int(drop_far_thr),
                "drop_far_applied": bool(drop_far_applied),
                "drop_far_actual_dist": int(drop_far_actual_dist) if drop_far_actual_dist is not None else None,
                "fallback_far_dist_threshold": int(fallback_far_thr),
                "fallback_far_applied": bool(fallback_far_applied),
                "fallback_far_actual_dist": int(fallback_far_actual_dist) if fallback_far_actual_dist is not None else None,
                "eventness_autoshift": (
                    int((autoshift_by_clip or {}).get(cid, 0))
                    if str(eventness_method) in ("energy_autoshift_clipdiff", "energy_autoshift_clipdiff_pos")
                    else None
                ),
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
            plan = _plan_for_baseline(
                "anchored_top2",
                cfg=cfg,
                num_segments=num_segments,
                rng=random.Random(0),
                anchors=anchors,
                oracle_segments=None,
                scores=scores,
            )
            low_used, high_used, max_high_used, triad_alt_used = _resolve_anchored_triad_for_clip(
                cfg=cfg,
                scores=scores,
                max_high_anchors=max_high_anchors,
            )
            by_clip[str(cid)] = {
                "scores": [float(x) for x in scores],
                "anchors": [int(x) for x in anchors],
                **(anchor_debug_by_clip.get(cid) or {}),
                "max_high_anchors": max_high_anchors,
                "max_high_anchors_used": int(max_high_used) if max_high_used is not None else None,
                "triad_policy": str(cfg.triad_policy),
                "triad_alt_conf_threshold": float(cfg.triad_alt_conf_threshold),
                "triad_alt_low_res": int(cfg.triad_alt_low_res),
                "triad_alt_high_res": int(cfg.triad_alt_high_res),
                "triad_alt_max_high_anchors": int(cfg.triad_alt_max_high_anchors) if cfg.triad_alt_max_high_anchors is not None else None,
                "triad_alt_used": bool(triad_alt_used),
                "triad_low_res_used": int(low_used),
                "triad_high_res_used": int(high_used),
                "plan_resolutions": plan.resolutions,
            }
        debug_eval = {"anchored_top2": by_clip}

    # Labels are shared across all baselines.
    y_train_np = np.stack([np.asarray(labels_by_clip[cid], dtype=np.int64) for cid in clip_ids_train], axis=0)
    y_eval_np = np.stack([np.asarray(labels_by_clip[cid], dtype=np.int64) for cid in clip_ids_eval], axis=0)
    y_train_t = torch.from_numpy(y_train_np).long()
    y_eval_t = torch.from_numpy(y_eval_np).long()

    oracle_segments_by_clip = {cid: _segments_from_labels(labels_by_clip[cid]) for cid in all_ids}

    def _tok_stats(tokens: list[int]) -> dict[str, float | int]:
        if not tokens:
            return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0, "max": 0}
        arr = np.asarray(tokens, dtype=np.float64)
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if int(arr.size) > 1 else 0.0,
            "min": int(arr.min()),
            "max": int(arr.max()),
        }

    fixed_data: dict[str, dict[str, torch.Tensor]] = {}
    token_budget_by_baseline: dict[str, int] = {}
    token_usage: dict[str, dict[str, dict[str, float | int]]] = {}
    fixed_rng = random.Random(0)
    for baseline in baselines:
        if baseline == "random_top2":
            continue

        x_train: list[np.ndarray] = []
        x_eval: list[np.ndarray] = []
        tokens_train: list[int] = []
        tokens_eval: list[int] = []

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
                tok = int(plan.total_tokens())
                if split == "train":
                    tokens_train.append(tok)
                else:
                    tokens_eval.append(tok)

                x = features_from_cache(cache, plan, res_feature=str(cfg.res_feature), base_res=int(cfg.base_res))
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
        token_usage[baseline] = {
            "train": _tok_stats(tokens_train),
            "eval": _tok_stats(tokens_eval),
        }

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
                tokens_train: list[int] = []
                tokens_eval: list[int] = []
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
                        tok = int(plan.total_tokens())
                        if split == "train":
                            tokens_train.append(tok)
                        else:
                            tokens_eval.append(tok)
                        x = features_from_cache(cache, plan, res_feature=str(cfg.res_feature), base_res=int(cfg.base_res))
                        if split == "train":
                            x_train.append(x)
                        else:
                            x_eval.append(x)
                xtr = torch.from_numpy(np.stack(x_train, axis=0)).float()
                xev = torch.from_numpy(np.stack(x_eval, axis=0)).float()
                ytr = y_train_t
                yev = y_eval_t
                token_budget = int(token_budget if token_budget is not None else cfg.token_budget(num_segments=num_segments))
                if "random_top2" not in token_usage:
                    token_usage["random_top2"] = {"train": _tok_stats(tokens_train), "eval": _tok_stats(tokens_eval)}

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
                try:
                    from scipy.stats import t as student_t  # type: ignore

                    p = float(2.0 * float(student_t.sf(abs(t), df=float(n - 1))))
                    p = float(max(0.0, min(1.0, p)))
                except Exception:
                    p = float("nan")

                return {"t": float(t), "p": float(p)}

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
        "token_usage": token_usage,
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
        choices=[
            "std",
            "std_norm",
            "top1_med",
            "top1_med_norm",
            "top12_gap",
            "top12_gap_norm",
            "top3_bottom3_gap_norm",
            "gini",
        ],
        help="Anchor confidence metric. If set, uses --anchor-conf-threshold to decide fallback to uniform (replaces std-only fallback).",
    )
    p.add_argument(
        "--anchor-conf-threshold",
        type=float,
        default=None,
        help="For --anchor-conf-metric: if confidence < threshold, fall back to uniform (return empty anchors).",
    )
    p.add_argument(
        "--anchor-gate-method",
        type=str,
        default="none",
        choices=["none", "lr_top1hit_v1", "lr_top1hit_all_v1"],
        help="Optional clip-level gate for anchored sampling. "
        "If enabled, acceptance policy depends on the method: "
        "lr_top1hit_v1 = rescue (accept if base_pass OR gate_pass); "
        "lr_top1hit_all_v1 = veto (accept only if base_pass AND gate_pass).",
    )
    p.add_argument(
        "--anchor-gate-threshold",
        type=float,
        default=0.0,
        help="For --anchor-gate-method: gate probability threshold. 0 disables the gate.",
    )
    p.add_argument(
        "--anchor-gate-label-radius",
        type=int,
        default=1,
        help="For --anchor-gate-method lr_top1hit_v1: label radius (in seconds) for training the gate on train split.",
    )
    p.add_argument(
        "--anchor-base-alloc",
        type=str,
        default="distance",
        choices=[
            "distance",
            "distance_high",
            "balanced",
            "balanced_high",
            "bridge",
            "bridge_high",
            "score",
            "score_high",
            "farthest",
            "farthest_high",
            "mixed",
            "mixed_high",
        ],
        help="How to allocate base-res segments in the equal-budget anchored plan. distance=closest-to-anchor (legacy); balanced=distance but round-robin around anchors; score=highest eventness scores; farthest=farthest-from-anchor (preserve context); mixed=half near anchors + half far (context).",
    )
    p.add_argument(
        "--anchor-high-policy",
        type=str,
        default="fixed",
        choices=["fixed", "adaptive_v1", "adaptive_v2", "adaptive_v3"],
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
            "av_fused_clipdiff",
            "av_fused_clipdiff_prod",
            "moe_energy_clipdiff",
            "av_basic_lr",
            "av_basic_mlp",
            "av_clipdiff_lr",
            "av_clipdiff_mlp",
            "av_clipdiff_accflip_mlp",
            "av_clipdiff_speech_mlp",
            "av_clipdiff_visgain_mlp",
            "av_clipdiff_lossgain_mlp",
            "av_clipdiff_flow_mlp",
            "av_clipdiff_flow_mlp_stride",
            "av_clipdiff_fbank_mlp",
            "av_ast_clipdiff_mlp",
            "av_ast_clipdiff_mil_mlp",
            "av_ast_clipdiff_tcn",
            "av_ast_clipalign_nce",
            "av_ast_clipalign_bce",
            "av_clipdiff_vec_mlp",
            "av_clipdiff_mlp_cls",
            "av_clipdiff_mlp_cls_target",
            "av_clip_mlp_cls",
            "av_clip_mlp_cls_target",
            "av_clipdiff_tcn",
            "vision_clipdiff",
            "vision_binary_lr",
            "vision_binary_mlp",
            "vision_mlp_cls",
            "vision_mlp_cls_target",
            "ast",
            "ast_nonspeech_max",
            "ast_lr",
            "ast_emb_lr",
            "ast_evt_mlp",
            "ast_mlp_cls",
            "ast_mlp_cls_target",
            "panns",
            "audiomae",
            "audio_basic_lr",
            "audio_basic_mlp",
            "audio_basic_tcn",
            "audio_fbank_mlp",
            "audio_fbank_tcn",
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
            anchor_gate_method=str(args.anchor_gate_method),
            anchor_gate_threshold=float(args.anchor_gate_threshold),
            anchor_gate_label_radius=int(args.anchor_gate_label_radius),
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
        class_names=[str(index.idx_to_label[i]) for i in range(int(index.num_classes))],
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
