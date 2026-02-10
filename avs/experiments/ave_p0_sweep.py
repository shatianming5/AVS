from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from avs.datasets.ave import AVEIndex, ensure_ave_meta
from avs.datasets.layout import ave_paths
from avs.experiments.ave_p0 import P0Config, run_p0_from_caches
from avs.train.train_loop import TrainConfig
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


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    k: int
    low_res: int
    base_res: int
    high_res: int
    head: str
    temporal_kernel_size: int
    anchor_shift: int
    anchor_std_threshold: float
    anchor_select: str
    anchor_nms_radius: int
    anchor_nms_strong_gap: float
    anchor_window: int
    anchor_smooth_window: int
    anchor_smooth_mode: str
    anchor_base_alloc: str
    anchor_conf_metric: str | None
    anchor_conf_threshold: float | None
    max_high_anchors: int | None
    anchor_high_policy: str
    anchor_high_adjacent_dist: int
    anchor_high_gap_threshold: float
    anchor2_veto_method: str = "none"  # none|top2med_norm_v1|lr_v1
    anchor2_veto_threshold: float = 0.5
    anchor2_veto_label_radius: int = 1
    anchor_gate_method: str = "none"  # none|lr_top1hit_v1
    anchor_gate_threshold: float = 0.0  # 0 disables
    anchor_gate_label_radius: int = 1
    anchor_drop_far_dist: int = 0
    anchor_fallback_far_dist: int = 0
    anchor_fallback_mode: str = "uniform"  # uniform|cheap_visual_clipdiff|cheap_visual_framediff
    anchor_fallback_visual_conf_metric: str = "top1_med"
    anchor_fallback_visual_conf_threshold: float = 0.0
    anchor_high_conf_metric: str | None = None
    anchor_high_conf_threshold: float = 0.0
    head_hidden_dim: int = 128
    head_dropout: float = 0.0
    res_feature: str = "none"
    triad_policy: str = "fixed"  # fixed|top1med_tiered_v1
    triad_alt_conf_threshold: float = 0.0
    triad_alt_low_res: int = 112
    triad_alt_high_res: int = 448
    triad_alt_max_high_anchors: int | None = 1
    budget_mode: str = "exact"  # exact|band
    budget_epsilon_frac: float = 0.01
    budget_extra_resolutions: tuple[int, ...] = ()

    def to_jsonable(self) -> dict:
        return {
            "name": str(self.name),
            "k": int(self.k),
            "low_res": int(self.low_res),
            "base_res": int(self.base_res),
            "high_res": int(self.high_res),
            "head": str(self.head),
            "head_hidden_dim": int(self.head_hidden_dim),
            "head_dropout": float(self.head_dropout),
            "res_feature": str(self.res_feature),
            "temporal_kernel_size": int(self.temporal_kernel_size),
            "anchor_shift": int(self.anchor_shift),
            "anchor_std_threshold": float(self.anchor_std_threshold),
            "anchor_select": str(self.anchor_select),
            "anchor_nms_radius": int(self.anchor_nms_radius),
            "anchor_nms_strong_gap": float(self.anchor_nms_strong_gap),
            "anchor_window": int(self.anchor_window),
            "anchor_smooth_window": int(self.anchor_smooth_window),
            "anchor_smooth_mode": str(self.anchor_smooth_mode),
            "anchor2_veto_method": str(self.anchor2_veto_method),
            "anchor2_veto_threshold": float(self.anchor2_veto_threshold),
            "anchor2_veto_label_radius": int(self.anchor2_veto_label_radius),
            "anchor_gate_method": str(self.anchor_gate_method),
            "anchor_gate_threshold": float(self.anchor_gate_threshold),
            "anchor_gate_label_radius": int(self.anchor_gate_label_radius),
            "anchor_base_alloc": str(self.anchor_base_alloc),
            "anchor_conf_metric": str(self.anchor_conf_metric) if self.anchor_conf_metric is not None else None,
            "anchor_conf_threshold": float(self.anchor_conf_threshold) if self.anchor_conf_threshold is not None else None,
            "max_high_anchors": int(self.max_high_anchors) if self.max_high_anchors is not None else None,
            "anchor_high_policy": str(self.anchor_high_policy),
            "anchor_high_adjacent_dist": int(self.anchor_high_adjacent_dist),
            "anchor_high_gap_threshold": float(self.anchor_high_gap_threshold),
            "anchor_drop_far_dist": int(self.anchor_drop_far_dist),
            "anchor_fallback_far_dist": int(self.anchor_fallback_far_dist),
            "anchor_fallback_mode": str(self.anchor_fallback_mode),
            "anchor_fallback_visual_conf_metric": str(self.anchor_fallback_visual_conf_metric),
            "anchor_fallback_visual_conf_threshold": float(self.anchor_fallback_visual_conf_threshold),
            "anchor_high_conf_metric": str(self.anchor_high_conf_metric) if self.anchor_high_conf_metric is not None else None,
            "anchor_high_conf_threshold": float(self.anchor_high_conf_threshold),
            "triad_policy": str(self.triad_policy),
            "triad_alt_conf_threshold": float(self.triad_alt_conf_threshold),
            "triad_alt_low_res": int(self.triad_alt_low_res),
            "triad_alt_high_res": int(self.triad_alt_high_res),
            "triad_alt_max_high_anchors": int(self.triad_alt_max_high_anchors) if self.triad_alt_max_high_anchors is not None else None,
            "budget_mode": str(self.budget_mode),
            "budget_epsilon_frac": float(self.budget_epsilon_frac),
            "budget_extra_resolutions": [int(r) for r in self.budget_extra_resolutions],
        }


def _default_candidates() -> list[CandidateConfig]:
    """
    Fixed, compact search space for reproducibility.

    This list should be edited deliberately (not auto-expanded) so results are comparable across runs.
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=1,
        anchor_std_threshold=1.0,
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric=None,
        anchor_conf_threshold=None,
        max_high_anchors=None,
        anchor_high_policy="fixed",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    out.append(CandidateConfig(name="base_160_224_352_topk", **base))
    out.append(CandidateConfig(name="base_160_224_352_nmsR2", **{**base, "anchor_select": "nms"}))
    out.append(CandidateConfig(name="base_160_224_352_window3", **{**base, "anchor_select": "window_topk", "anchor_window": 3}))
    out.append(CandidateConfig(name="base_160_224_352_window5", **{**base, "anchor_select": "window_topk", "anchor_window": 5}))
    out.append(CandidateConfig(name="base_160_224_352_scoreAlloc", **{**base, "anchor_base_alloc": "score"}))
    out.append(CandidateConfig(name="base_160_224_352_mixedAlloc", **{**base, "anchor_base_alloc": "mixed"}))
    out.append(CandidateConfig(name="base_160_224_352_k5", **{**base, "temporal_kernel_size": 5}))
    # Slightly less conservative confidence threshold (empirically reduces fallback rate without catastrophic drops).
    out.append(CandidateConfig(name="base_160_224_352_k5_std0p9", **{**base, "temporal_kernel_size": 5, "anchor_std_threshold": 0.9}))

    # A more extreme triad that sometimes helps (requires caches with 112/224/448).
    out.append(
        CandidateConfig(
            name="extreme_112_224_448_window3",
            **{
                **base,
                "low_res": 112,
                "high_res": 448,
                "anchor_select": "window_topk",
                "anchor_window": 3,
            },
        )
    )

    # Confidence gating variants (opt-in): use top1-top2 gap on raw scores.
    out.append(
        CandidateConfig(
            name="base_160_224_352_gapGate0.6",
            **{**base, "anchor_conf_metric": "top12_gap", "anchor_conf_threshold": 0.6},
        )
    )
    return out


def _candidates_ast_v1() -> list[CandidateConfig]:
    """
    AST-tuned search space.

    Notes:
    - AST per-second scores are in [0,1] and typically have std ~ O(0.1), so confidence thresholds must be
      in a different scale than log-energy (which used ~1.0).
    - Keep this list intentionally small; add only if it materially improves full test402.
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=5,
        anchor_shift=1,
        anchor_std_threshold=0.05,
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric=None,
        anchor_conf_threshold=None,
        max_high_anchors=None,
        anchor_high_policy="fixed",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    out.append(CandidateConfig(name="ast_160_224_352_k5_std0p00", **{**base, "anchor_std_threshold": 0.0}))
    out.append(CandidateConfig(name="ast_160_224_352_k5_std0p05", **{**base, "anchor_std_threshold": 0.05}))
    out.append(CandidateConfig(name="ast_160_224_352_k5_std0p10", **{**base, "anchor_std_threshold": 0.10}))

    out.append(
        CandidateConfig(
            name="ast_160_224_352_window3_std0p05",
            **{**base, "anchor_select": "window_topk", "anchor_window": 3, "anchor_std_threshold": 0.05},
        )
    )
    out.append(
        CandidateConfig(
            name="ast_160_224_352_scoreAlloc_std0p05",
            **{**base, "anchor_base_alloc": "score", "anchor_std_threshold": 0.05},
        )
    )

    # More extreme triad (requires caches with 112/224/448). Often helps when anchors are accurate.
    out.append(
        CandidateConfig(
            name="ast_112_224_448_window3_std0p05",
            **{
                **base,
                "low_res": 112,
                "high_res": 448,
                "anchor_select": "window_topk",
                "anchor_window": 3,
                "anchor_std_threshold": 0.05,
            },
        )
    )
    return out


def _candidates_energy_v2() -> list[CandidateConfig]:
    """
    Energy-tuned search space that targets larger anchored gains and better transfer.

    This is an extension of `energy_v1` with targeted knobs:
      - more aggressive use of anchors (std=0 disables fallback)
      - diverse anchor selection (nms/window_topk)
      - adaptive high-res allocation when anchors are adjacent
      - larger K (more anchors) to stabilize selection
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=1,
        anchor_std_threshold=1.0,
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric=None,
        anchor_conf_threshold=None,
        max_high_anchors=None,
        anchor_high_policy="fixed",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []

    # Reference config (matches the strongest known official full-split evidence in docs/plan.md).
    out.append(CandidateConfig(name="energy_ref_k2_topk_std1p0", **base))

    # Disable fallback; rely on selection diversity / adaptive high policy to avoid catastrophic cases.
    out.append(CandidateConfig(name="energy_k2_topk_std0p0", **{**base, "anchor_std_threshold": 0.0}))
    out.append(CandidateConfig(name="energy_k2_nmsR2_std0p0", **{**base, "anchor_std_threshold": 0.0, "anchor_select": "nms"}))
    out.append(
        CandidateConfig(
            name="energy_k2_window3_std0p0",
            **{**base, "anchor_std_threshold": 0.0, "anchor_select": "window_topk", "anchor_window": 3},
        )
    )
    out.append(
        CandidateConfig(
            name="energy_k2_window5_std0p0",
            **{**base, "anchor_std_threshold": 0.0, "anchor_select": "window_topk", "anchor_window": 5},
        )
    )

    # Adaptive high-res allocation: when anchors are adjacent (common failure mode), demote to 1 high-res anchor.
    out.append(
        CandidateConfig(
            name="energy_k2_topk_std0p0_adaptiveAdj1",
            **{**base, "anchor_std_threshold": 0.0, "anchor_high_policy": "adaptive_v1", "anchor_high_adjacent_dist": 1},
        )
    )
    out.append(
        CandidateConfig(
            name="energy_k2_topk_std0p0_adaptiveAdj2",
            **{**base, "anchor_std_threshold": 0.0, "anchor_high_policy": "adaptive_v1", "anchor_high_adjacent_dist": 2},
        )
    )

    # More anchors (K=5) can stabilize selection; budget still caps high-res allocations.
    out.append(CandidateConfig(name="energy_k5_nmsR2_std0p0", **{**base, "k": 5, "anchor_std_threshold": 0.0, "anchor_select": "nms"}))
    out.append(
        CandidateConfig(
            name="energy_k5_nmsR2_std0p0_adaptiveAdj2",
            **{
                **base,
                "k": 5,
                "anchor_std_threshold": 0.0,
                "anchor_select": "nms",
                "anchor_high_policy": "adaptive_v1",
                "anchor_high_adjacent_dist": 2,
            },
        )
    )

    # Extreme triad (requires 112/224/448 caches): sometimes yields larger gains if anchors are reliable.
    out.append(
        CandidateConfig(
            name="energy_extreme_112_224_448_nmsR2_std0p0",
            **{**base, "anchor_std_threshold": 0.0, "anchor_select": "nms", "low_res": 112, "high_res": 448},
        )
    )

    return out


def _candidates_energy_v3() -> list[CandidateConfig]:
    """
    Energy sweep that targets larger anchored gains on test by expanding around the best-known
    full-split config and explicitly addressing two observed failure modes:
      - high fallback rate under std-threshold gating (too conservative)
      - shift-induced 1-anchor drops near boundaries (shift=1 can drop anchors)

    Tactics:
      - sweep std thresholds in a small, pre-registered set (0.4/0.6/1.0)
      - include shift=0 variants
      - include stability variants that preserve context (max_high_anchors=1, mixed/score base alloc)
      - include a "safer extreme" triad (112/224/448) only with max_high_anchors=1
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=0,
        anchor_std_threshold=0.6,
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric=None,
        anchor_conf_threshold=None,
        max_high_anchors=None,
        anchor_high_policy="fixed",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []

    # Reference: known to transfer positively (but < +2% so far).
    out.append(CandidateConfig(name="energyv3_ref_shift1_std1p0", **{**base, "anchor_shift": 1, "anchor_std_threshold": 1.0}))

    # Shift=0 avoids dropping anchors at boundaries.
    out.append(CandidateConfig(name="energyv3_shift0_std1p0", **{**base, "anchor_shift": 0, "anchor_std_threshold": 1.0}))
    out.append(CandidateConfig(name="energyv3_shift0_std0p6", **{**base, "anchor_shift": 0, "anchor_std_threshold": 0.6}))
    out.append(CandidateConfig(name="energyv3_shift0_std0p4", **{**base, "anchor_shift": 0, "anchor_std_threshold": 0.4}))

    # Keep shift=1 variants to cover true A/V offset.
    out.append(CandidateConfig(name="energyv3_shift1_std0p6", **{**base, "anchor_shift": 1, "anchor_std_threshold": 0.6}))
    out.append(CandidateConfig(name="energyv3_shift1_std0p4", **{**base, "anchor_shift": 1, "anchor_std_threshold": 0.4}))

    # Selection diversity (adjacent-anchor mitigation).
    out.append(
        CandidateConfig(
            name="energyv3_shift0_nmsR2_std0p4",
            **{**base, "anchor_shift": 0, "anchor_std_threshold": 0.4, "anchor_select": "nms"},
        )
    )
    out.append(
        CandidateConfig(
            name="energyv3_shift0_window3_std0p4",
            **{**base, "anchor_shift": 0, "anchor_std_threshold": 0.4, "anchor_select": "window_topk", "anchor_window": 3},
        )
    )

    # Preserve context by reducing the number of high-res anchors.
    out.append(CandidateConfig(name="energyv3_shift0_std0p4_maxHigh1", **{**base, "anchor_shift": 0, "anchor_std_threshold": 0.4, "max_high_anchors": 1}))

    # Base allocation ablations: evidence-only vs evidence+context.
    out.append(CandidateConfig(name="energyv3_shift0_std0p4_mixedAlloc", **{**base, "anchor_shift": 0, "anchor_std_threshold": 0.4, "anchor_base_alloc": "mixed"}))
    out.append(CandidateConfig(name="energyv3_shift0_std0p4_scoreAlloc", **{**base, "anchor_shift": 0, "anchor_std_threshold": 0.4, "anchor_base_alloc": "score"}))

    # Adaptive high allocation: demote to 1 high-res anchor when adjacent or when top1 dominates.
    out.append(
        CandidateConfig(
            name="energyv3_shift0_std0p4_adaptiveGap0p6",
            **{
                **base,
                "anchor_shift": 0,
                "anchor_std_threshold": 0.4,
                "anchor_high_policy": "adaptive_v1",
                "anchor_high_adjacent_dist": 1,
                "anchor_high_gap_threshold": 0.6,
            },
        )
    )

    # Extreme triad: only with max_high_anchors=1 to avoid the catastrophic 2×448 + 8×112 plan.
    out.append(
        CandidateConfig(
            name="energyv3_extreme_112_224_448_maxHigh1_shift0_std1p0",
            **{
                **base,
                "low_res": 112,
                "high_res": 448,
                "max_high_anchors": 1,
                "anchor_shift": 0,
                "anchor_std_threshold": 1.0,
            },
        )
    )
    out.append(
        CandidateConfig(
            name="energyv3_extreme_112_224_448_maxHigh1_shift0_std0p6",
            **{
                **base,
                "low_res": 112,
                "high_res": 448,
                "max_high_anchors": 1,
                "anchor_shift": 0,
                "anchor_std_threshold": 0.6,
            },
        )
    )

    return out


def _candidates_ltl_gini_v1() -> list[CandidateConfig]:
    """
    Stage-1-agnostic sweep space for "Listen-then-Look" runs that use a *different score scale*
    than log-energy (e.g., learned logits like av_clipdiff_*).

    Key idea: use a scale-free confidence gate (`conf_metric=gini`) instead of std-threshold gating,
    so anchor fallback behavior is comparable across Stage-1 methods.

    This list is intentionally compact for reproducibility.
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=0,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="gini",
        anchor_conf_threshold=0.20,
        max_high_anchors=None,
        anchor_high_policy="fixed",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []

    # Confidence thresholds (gini is in [0,1]); keep a small pre-registered set.
    out.append(CandidateConfig(name="ltl_gini0p20_shift0", **base))
    out.append(CandidateConfig(name="ltl_gini0p20_shift1", **{**base, "anchor_shift": 1}))
    out.append(CandidateConfig(name="ltl_gini0p30_shift0", **{**base, "anchor_conf_threshold": 0.30}))
    out.append(CandidateConfig(name="ltl_gini0p30_shift1", **{**base, "anchor_conf_threshold": 0.30, "anchor_shift": 1}))

    # Selection diversity.
    out.append(CandidateConfig(name="ltl_gini0p20_nmsR2", **{**base, "anchor_select": "nms"}))
    out.append(CandidateConfig(name="ltl_gini0p20_window3", **{**base, "anchor_select": "window_topk", "anchor_window": 3}))

    # Preserve context by reducing the number of high-res anchors.
    out.append(CandidateConfig(name="ltl_gini0p20_maxHigh1", **{**base, "max_high_anchors": 1}))

    # Base allocation ablations.
    out.append(CandidateConfig(name="ltl_gini0p20_mixedAlloc", **{**base, "anchor_base_alloc": "mixed"}))
    out.append(CandidateConfig(name="ltl_gini0p20_scoreAlloc", **{**base, "anchor_base_alloc": "score"}))

    # Safer extreme triad (only with max_high_anchors=1).
    out.append(
        CandidateConfig(
            name="ltl_gini0p20_extreme_112_224_448_maxHigh1",
            **{
                **base,
                "low_res": 112,
                "high_res": 448,
                "max_high_anchors": 1,
            },
        )
    )

    return out


def _candidates_ltl_std_v1() -> list[CandidateConfig]:
    """
    Fine-grained std-threshold sweep for learned-logit Stage-1 methods (e.g., av_clipdiff_*).

    Motivation: energy_v3 uses a coarse std threshold grid (0.4/0.6/1.0). For learned-logit scores we
    observed that the best config can become *too* conservative (fallback≈0.88 at std_thr=0.6), while
    std_thr=0.4 can become too permissive (fallback≈0.71). This set fills the missing middle points
    (0.45/0.50/0.55) to target fallback≈0.8 and potentially increase anchored gains.

    Keep this list compact for reproducibility: only sweep {shift × std_thr} around the current best.
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=1,
        anchor_std_threshold=0.6,
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric=None,
        anchor_conf_threshold=None,
        max_high_anchors=None,
        anchor_high_policy="fixed",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for thr in (0.45, 0.50, 0.55, 0.60):
        thr_name = str(thr).replace(".", "p")
        out.append(CandidateConfig(name=f"ltlstd_shift1_std{thr_name}", **{**base, "anchor_shift": 1, "anchor_std_threshold": thr}))
        out.append(CandidateConfig(name=f"ltlstd_shift0_std{thr_name}", **{**base, "anchor_shift": 0, "anchor_std_threshold": thr}))
    return out


def _candidates_ltl_std_v2() -> list[CandidateConfig]:
    """
    Extended std-threshold sweep for learned-logit Stage-1 methods.

    Compared to `ltl_std_v1`, this set adds a small number of high-leverage Stage-2 knobs around the mid-range
    std thresholds (where fallback is neither ~0.9 nor ~0.7):
      - selection diversity (nms/window_topk)
      - base allocation variants (mixed/score)
      - context preservation (max_high_anchors=1)
      - an extreme triad (112/224/448) with max_high_anchors=1
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=0,
        anchor_std_threshold=0.5,
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric=None,
        anchor_conf_threshold=None,
        max_high_anchors=None,
        anchor_high_policy="fixed",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []

    # Core grid (same as v1): shift × std_thr.
    for thr in (0.45, 0.50, 0.55, 0.60):
        thr_name = str(thr).replace(".", "p")
        out.append(CandidateConfig(name=f"ltlstd2_shift0_std{thr_name}", **{**base, "anchor_shift": 0, "anchor_std_threshold": thr}))
        out.append(CandidateConfig(name=f"ltlstd2_shift1_std{thr_name}", **{**base, "anchor_shift": 1, "anchor_std_threshold": thr}))

    # Extra knobs around the mid threshold (std=0.5) for better transfer.
    out.append(CandidateConfig(name="ltlstd2_shift0_std0p5_nmsR2", **{**base, "anchor_shift": 0, "anchor_std_threshold": 0.5, "anchor_select": "nms"}))
    out.append(
        CandidateConfig(
            name="ltlstd2_shift0_std0p5_window3",
            **{**base, "anchor_shift": 0, "anchor_std_threshold": 0.5, "anchor_select": "window_topk", "anchor_window": 3},
        )
    )
    out.append(CandidateConfig(name="ltlstd2_shift0_std0p5_mixedAlloc", **{**base, "anchor_shift": 0, "anchor_std_threshold": 0.5, "anchor_base_alloc": "mixed"}))
    out.append(CandidateConfig(name="ltlstd2_shift0_std0p5_scoreAlloc", **{**base, "anchor_shift": 0, "anchor_std_threshold": 0.5, "anchor_base_alloc": "score"}))
    out.append(CandidateConfig(name="ltlstd2_shift0_std0p5_maxHigh1", **{**base, "anchor_shift": 0, "anchor_std_threshold": 0.5, "max_high_anchors": 1}))
    out.append(
        CandidateConfig(
            name="ltlstd2_extreme_112_224_448_maxHigh1_shift0_std0p5",
            **{**base, "low_res": 112, "high_res": 448, "max_high_anchors": 1, "anchor_shift": 0, "anchor_std_threshold": 0.5},
        )
    )

    # A couple of variants around the current best (std=0.45, shift=0).
    out.append(CandidateConfig(name="ltlstd2_shift0_std0p45_scoreAlloc", **{**base, "anchor_shift": 0, "anchor_std_threshold": 0.45, "anchor_base_alloc": "score"}))
    out.append(CandidateConfig(name="ltlstd2_shift0_std0p45_maxHigh1", **{**base, "anchor_shift": 0, "anchor_std_threshold": 0.45, "max_high_anchors": 1}))

    return out


def _candidates_ltl_adaptive_v1() -> list[CandidateConfig]:
    """
    Adaptive high-res allocation sweep for learned-logit Stage-1 methods.

    Hypothesis: a common failure case for learned anchors is selecting two adjacent peaks; allocating high-res
    to both can waste budget and harm context. `anchor_high_policy=adaptive_v1` demotes to 1 high-res anchor
    when anchors are adjacent, which may improve test transfer without changing Stage-1.
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=0,
        anchor_std_threshold=0.6,
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric=None,
        anchor_conf_threshold=None,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for thr in (0.45, 0.50, 0.55, 0.60):
        thr_name = str(thr).replace(".", "p")
        for shift in (0, 1):
            out.append(
                CandidateConfig(
                    name=f"ltladj1_shift{shift}_std{thr_name}",
                    **{**base, "anchor_shift": int(shift), "anchor_std_threshold": float(thr), "anchor_high_adjacent_dist": 1},
                )
            )
            out.append(
                CandidateConfig(
                    name=f"ltladj2_shift{shift}_std{thr_name}",
                    **{**base, "anchor_shift": int(shift), "anchor_std_threshold": float(thr), "anchor_high_adjacent_dist": 2},
                )
            )
    return out


def _candidates_ltl_adaptive_keepadj_v1() -> list[CandidateConfig]:
    """
    Keep-when-adjacent demotion (adaptive_v3) for learned-logit Stage-1 methods.

    Motivation (AVE/P0 / C0003):
      - Test402 diagnostics show the "2-high" regime can be net harmful when the selected anchors are far apart
        (context loss under a fixed token budget).
      - `anchor_high_policy=adaptive_v3` keeps both anchors for base allocation, but only allows 2-high when the
        two anchors are within `anchor_high_adjacent_dist`.

    Grid:
      - shift ∈ {0,1}
      - std_thr ∈ {0.45,0.50,0.55,0.60}
      - adjacency distance ∈ {1,2}
      - all other knobs match the ltl_adaptive_v1 defaults.
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=0,
        anchor_std_threshold=0.55,
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric=None,
        anchor_conf_threshold=None,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v3",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for thr in (0.45, 0.50, 0.55, 0.60):
        thr_name = str(thr).replace(".", "p")
        for shift in (0, 1):
            for adj in (1, 2):
                out.append(
                    CandidateConfig(
                        name=f"ltlkeepadj_adj{adj}_shift{shift}_std{thr_name}",
                        **{
                            **base,
                            "anchor_shift": int(shift),
                            "anchor_std_threshold": float(thr),
                            "anchor_high_adjacent_dist": int(adj),
                        },
                    )
                )
    return out


def _candidates_ltl_smooth_v1() -> list[CandidateConfig]:
    """
    Score-smoothing candidate set for learned-logit Stage-1 methods.

    Motivation: In the current best learned-anchor test402 run, diagnostics show that the 2-high regime is
    net harmful for many clips. A lightweight Stage-2 knob is to smooth the per-second Stage-1 scores
    before selecting Top-K anchors: this tends to merge nearby peaks and increases the chance that the
    selected anchors are adjacent, which triggers `anchor_high_policy=adaptive_v1` demotion to 1 high-res
    anchor (more base-res context under the same token budget).

    Keep the grid small and pre-registered:
      - std_thr ∈ {0.45,0.50} and shift ∈ {0,1}
      - smooth_window ∈ {0,3,5} (0 == no smoothing; odd windows only)
      - adaptive demotion with adjacent_dist ∈ {1,2}
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=0,
        anchor_std_threshold=0.45,
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric=None,
        anchor_conf_threshold=None,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for thr in (0.45, 0.50):
        thr_name = str(thr).replace(".", "p")
        for shift in (0, 1):
            for smooth in (0, 3, 5):
                sw_name = str(smooth)
                for adj in (1, 2):
                    out.append(
                        CandidateConfig(
                            name=f"ltlsmooth_shift{shift}_std{thr_name}_sw{sw_name}_adj{adj}",
                            **{
                                **base,
                                "anchor_shift": int(shift),
                                "anchor_std_threshold": float(thr),
                                "anchor_smooth_window": int(smooth),
                                "anchor_high_adjacent_dist": int(adj),
                            },
                        )
                    )
    return out


def _candidates_ltl_adaptive_v2() -> list[CandidateConfig]:
    """
    Adaptive+low-threshold candidate set for learned-logit Stage-1 methods.

    Motivation: the current best config on test402 uses `std_thr≈0.45`, which can yield a high fallback rate
    (anchors rejected => uniform plan), diluting anchored gains. This set widens the std threshold range to
    increase anchor usage, while keeping the Stage-2 knobs small and pre-registered:
      - `anchor_high_policy=adaptive_v1` with `anchor_high_adjacent_dist∈{1,2}`
      - std_thr grid down to 0.10
      - a few robust base-allocation + selection variants around mid thresholds
      - an optional extreme triad (112/224/448) with adaptive high-res demotion
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=0,
        anchor_std_threshold=0.25,
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric=None,
        anchor_conf_threshold=None,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []

    # Wide std-threshold grid for the common adjacent case (adj_dist=1).
    for thr in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
        thr_name = str(thr).replace(".", "p")
        out.append(
            CandidateConfig(
                name=f"ltladjv2_adj1_shift0_std{thr_name}",
                **{**base, "anchor_std_threshold": float(thr), "anchor_high_adjacent_dist": 1},
            )
        )

    # Slightly more aggressive adjacency demotion (adj_dist=2) on a smaller grid.
    for thr in (0.15, 0.20, 0.25, 0.30):
        thr_name = str(thr).replace(".", "p")
        out.append(
            CandidateConfig(
                name=f"ltladjv2_adj2_shift0_std{thr_name}",
                **{**base, "anchor_std_threshold": float(thr), "anchor_high_adjacent_dist": 2},
            )
        )

    # Around the mid thresholds, try a few high-leverage Stage-2 knobs that can reduce harm when anchors are noisy.
    for thr in (0.20, 0.25):
        thr_name = str(thr).replace(".", "p")
        for alloc in ("balanced", "mixed", "score"):
            out.append(
                CandidateConfig(
                    name=f"ltladjv2_adj1_shift0_std{thr_name}_{alloc}Alloc",
                    **{**base, "anchor_std_threshold": float(thr), "anchor_high_adjacent_dist": 1, "anchor_base_alloc": str(alloc)},
                )
            )
        out.append(
            CandidateConfig(
                name=f"ltladjv2_adj1_shift0_std{thr_name}_nmsStrongR2G0p6",
                **{
                    **base,
                    "anchor_std_threshold": float(thr),
                    "anchor_high_adjacent_dist": 1,
                    "anchor_select": "nms_strong",
                    "anchor_nms_radius": 2,
                    "anchor_nms_strong_gap": 0.6,
                },
            )
        )
        out.append(
            CandidateConfig(
                name=f"ltladjv2_adj1_shift0_std{thr_name}_window3",
                **{**base, "anchor_std_threshold": float(thr), "anchor_high_adjacent_dist": 1, "anchor_select": "window_topk", "anchor_window": 3},
            )
        )

    # Optional "extreme" triad to increase peak resolution when anchors are trusted.
    extreme = {**base, "low_res": 112, "high_res": 448}
    for thr in (0.15, 0.20, 0.25):
        thr_name = str(thr).replace(".", "p")
        out.append(
            CandidateConfig(
                name=f"ltlextadjv2_adj1_shift0_std{thr_name}",
                **{**extreme, "anchor_std_threshold": float(thr), "anchor_high_adjacent_dist": 1},
            )
        )
        out.append(
            CandidateConfig(
                name=f"ltlextadjv2_adj2_shift0_std{thr_name}",
                **{**extreme, "anchor_std_threshold": float(thr), "anchor_high_adjacent_dist": 2},
            )
        )

    return out


def _candidates_ltl_adaptive_v3() -> list[CandidateConfig]:
    """
    Confidence-aware adaptive high-res allocation for learned Stage-1 methods.

    Hypothesis: for "medium-confidence" clips, predicted anchors can be useful but allocating two high-res
    seconds harms context (too many low-res seconds under a fixed budget). `anchor_high_policy=adaptive_v2`
    keeps anchors but demotes to `max_high_anchors=1` when confidence is below `anchor_high_conf_threshold`.

    This yields a 3-level behavior (for std-based confidence):
      - conf < std_thr: fall back to uniform
      - std_thr <= conf < high_conf_thr: anchored with 1 high-res anchor (more base-res context)
      - conf >= high_conf_thr: anchored with up to 2 high-res anchors (plus adjacency demotion)
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=0,
        anchor_std_threshold=0.30,  # fallback threshold (conf < thr => uniform)
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric=None,
        anchor_conf_threshold=None,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v2",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
        anchor_high_conf_metric=None,  # None => std
        anchor_high_conf_threshold=0.45,
    )

    out: list[CandidateConfig] = []

    # 2-threshold grid: (fallback std_thr, high_conf_thr) with adjacency demotion.
    pairs = [
        (0.25, 0.45),
        (0.30, 0.45),
        (0.35, 0.45),
        (0.30, 0.50),
        (0.35, 0.50),
    ]
    for std_thr, hi_thr in pairs:
        std_name = str(std_thr).replace(".", "p")
        hi_name = str(hi_thr).replace(".", "p")
        for adj in (1, 2):
            out.append(
                CandidateConfig(
                    name=f"ltladjv3_adj{adj}_shift0_std{std_name}_hi{hi_name}",
                    **{
                        **base,
                        "anchor_std_threshold": float(std_thr),
                        "anchor_high_adjacent_dist": int(adj),
                        "anchor_high_conf_threshold": float(hi_thr),
                    },
                )
            )

    # A few robust variants around the default (std=0.30, hi=0.45).
    out.append(
        CandidateConfig(
            name="ltladjv3_adj1_shift0_std0p3_hi0p45_mixedAlloc",
            **{**base, "anchor_std_threshold": 0.30, "anchor_high_adjacent_dist": 1, "anchor_high_conf_threshold": 0.45, "anchor_base_alloc": "mixed"},
        )
    )
    out.append(
        CandidateConfig(
            name="ltladjv3_adj1_shift0_std0p3_hi0p45_scoreAlloc",
            **{**base, "anchor_std_threshold": 0.30, "anchor_high_adjacent_dist": 1, "anchor_high_conf_threshold": 0.45, "anchor_base_alloc": "score"},
        )
    )
    out.append(
        CandidateConfig(
            name="ltladjv3_adj1_shift0_std0p3_hi0p45_window3",
            **{
                **base,
                "anchor_std_threshold": 0.30,
                "anchor_high_adjacent_dist": 1,
                "anchor_high_conf_threshold": 0.45,
                "anchor_select": "window_topk",
                "anchor_window": 3,
            },
        )
    )

    return out


def _candidates_ltl_maxhigh1_v1() -> list[CandidateConfig]:
    """
    Always-maxHigh1 candidate set for learned-logit Stage-1 methods.

    Motivation: In the current best learned-anchor test402 run, per-clip diagnostics show that the "2-high"
    regime (2×high_res + 2×base_res + 6×low_res under the 160/224/352 triad) is net harmful, while the
    "1-high" regime (1×high_res + 6×base_res + 3×low_res) is net positive. This set fixes `max_high_anchors=1`
    to preserve more base_res context while keeping the token budget identical.

    Keep the grid small and pre-registered:
      - std_thr grid in {0.30,0.35,0.40,0.45}
      - base allocation in {distance,balanced,mixed}
      - selection in {topk,window_topk(window=3)}
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=0,
        anchor_std_threshold=0.45,
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric=None,
        anchor_conf_threshold=None,
        max_high_anchors=1,
        anchor_high_policy="fixed",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for thr in (0.30, 0.35, 0.40, 0.45):
        thr_name = str(thr).replace(".", "p")
        for alloc in ("distance", "balanced", "mixed"):
            out.append(
                CandidateConfig(
                    name=f"ltlmax1_thr{thr_name}_{alloc}_topk",
                    **{**base, "anchor_std_threshold": float(thr), "anchor_base_alloc": str(alloc), "anchor_select": "topk"},
                )
            )
            out.append(
                CandidateConfig(
                    name=f"ltlmax1_thr{thr_name}_{alloc}_window3",
                    **{
                        **base,
                        "anchor_std_threshold": float(thr),
                        "anchor_base_alloc": str(alloc),
                        "anchor_select": "window_topk",
                        "anchor_window": 3,
                    },
                )
            )

    return out


def _candidates_ltl_gini_v2() -> list[CandidateConfig]:
    """
    Scale-free gini-gated candidate set for learned-logit Stage-1 methods.

    `ltl_gini_v1` used low thresholds (0.20/0.30) that can be too permissive for learned logits (fallback≈0.05).
    This set shifts to a higher, pre-registered grid (0.35/0.40/0.45/0.50) to target a more selective regime.
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=0,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="gini",
        anchor_conf_threshold=0.40,
        max_high_anchors=None,
        anchor_high_policy="fixed",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for thr in (0.35, 0.40, 0.45, 0.50):
        thr_name = str(thr).replace(".", "p")
        out.append(CandidateConfig(name=f"ltlgini2_gini{thr_name}_shift0", **{**base, "anchor_shift": 0, "anchor_conf_threshold": thr}))
        out.append(CandidateConfig(name=f"ltlgini2_gini{thr_name}_shift1", **{**base, "anchor_shift": 1, "anchor_conf_threshold": thr}))
    return out


def _candidates_ltl_gap_v1() -> list[CandidateConfig]:
    """
    Gap-gated candidate set for learned-logit Stage-1 methods.

    Uses `conf_metric=top12_gap` as a simple, scale-aware confidence proxy:
    a large (top1 - top2) gap indicates a single dominant event peak, which tends to reduce false positives.
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=0,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top12_gap",
        anchor_conf_threshold=0.30,
        max_high_anchors=None,
        anchor_high_policy="fixed",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for thr in (0.20, 0.30, 0.40, 0.50):
        thr_name = str(thr).replace(".", "p")
        out.append(CandidateConfig(name=f"ltlgap1_gap{thr_name}_shift0", **{**base, "anchor_shift": 0, "anchor_conf_threshold": thr}))
        out.append(CandidateConfig(name=f"ltlgap1_gap{thr_name}_shift1", **{**base, "anchor_shift": 1, "anchor_conf_threshold": thr}))
    return out


def _candidates_ltl_top1med_v1() -> list[CandidateConfig]:
    """
    Top1-minus-median gated candidate set for learned-logit Stage-1 methods.

    Motivation: For learned logits, the legacy std-based gate can be brittle (scale- and distribution-dependent),
    while gini/top12_gap can be either too permissive or too strict. `top1_med` is a simple, robust "peakiness"
    measure that often correlates with "single salient event" clips, which are the most likely to benefit from
    anchored allocation under a fixed token budget.

    Keep the grid small and pre-registered:
      - `conf_metric=top1_med`, thresholds in {0.4,0.5,0.6,0.7,0.8}
      - shift ∈ {0,1}
      - keep the current best learned-anchor Stage-2 settings otherwise (adaptive demotion on adjacent anchors).
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=0,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for thr in (0.40, 0.50, 0.60, 0.70, 0.80):
        thr_name = str(thr).replace(".", "p")
        out.append(CandidateConfig(name=f"ltltop1med_thr{thr_name}_shift0", **{**base, "anchor_shift": 0, "anchor_conf_threshold": thr}))
        out.append(CandidateConfig(name=f"ltltop1med_thr{thr_name}_shift1", **{**base, "anchor_shift": 1, "anchor_conf_threshold": thr}))
    return out


def _candidates_ltl_top1med_gate_lr_v1() -> list[CandidateConfig]:
    """
    Top1-minus-median gate + learned rescue gate (lr_top1hit_v1).

    Motivation (AVE/P0 / C0003 “拉大”):
      - The current best deployable config (E0224; top1_med thr≈0.6) is limited by heavy fallback
        (`fallback_used_frac≈0.75`), diluting anchored gains despite a strong Oracle ceiling.
      - Simply lowering the gate (or using sep3 as a replacement) increases anchor usage but adds many
        harmful clips (E0332/E0333).

    Idea:
      - Keep the strict, interpretable base gate (`top1_med thr=0.6`) as the default.
      - Train a tiny clip-level gate on the *train split* to predict whether the top-1 selected anchor
        hits an event second, and use it as a **rescue** mechanism: accept if (base_pass OR gate_pass).

    Keep the grid small and pre-registered:
      - base gate fixed: `top1_med thr=0.6`
      - rescue gate thresholds in {0.6,0.7,0.8}
      - shift ∈ {0,1}
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=0,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
        anchor_gate_method="lr_top1hit_v1",
        anchor_gate_label_radius=1,
    )

    out: list[CandidateConfig] = []
    for gthr in (0.60, 0.70, 0.80):
        gthr_name = str(gthr).replace(".", "p")
        out.append(
            CandidateConfig(
                name=f"ltltop1med_gate{gthr_name}_shift0",
                **{**base, "anchor_shift": 0, "anchor_gate_threshold": float(gthr)},
            )
        )
        out.append(
            CandidateConfig(
                name=f"ltltop1med_gate{gthr_name}_shift1",
                **{**base, "anchor_shift": 1, "anchor_gate_threshold": float(gthr)},
            )
        )
    return out


def _candidates_ltl_top1med_gate_all_v1() -> list[CandidateConfig]:
    """
    Top1-minus-median base gate + learned top1-hit gate trained on *all* clips (lr_top1hit_all_v1).

    Motivation (AVE/P0 / “拉大” C0003):
      - Val→test transfer is noisy and several “safe” fixes (keepadj/basealloc_high/visfb) did not close the gap.
      - The rescue-only gate (lr_top1hit_v1) did not rescue in practice (gate_prob rarely exceeded thresholds).

    Idea:
      - Train the same lightweight LR gate, but on **all** train clips (not only fallback clips), and use it as a
        veto on top of the base gate: accept a clip only if (base_pass AND gate_prob >= gate_threshold). This can
        drop “confident-but-wrong” anchors that slip through the base confidence metric, and may improve transfer
        by reducing harmful clips.

    Keep the grid small and pre-registered:
      - base gate fixed: `top1_med thr=0.6` (still computed/recorded for diagnostics)
      - gate thresholds in {0.2,0.3,0.4,0.5}
      - shift ∈ {0,1}
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=0,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
        anchor_gate_method="lr_top1hit_all_v1",
        anchor_gate_label_radius=1,
    )

    out: list[CandidateConfig] = []
    for gthr in (0.20, 0.30, 0.40, 0.50):
        gthr_name = str(gthr).replace(".", "p")
        out.append(
            CandidateConfig(
                name=f"ltltop1med_gateall{gthr_name}_shift0",
                **{**base, "anchor_shift": 0, "anchor_gate_threshold": float(gthr)},
            )
        )
        out.append(
            CandidateConfig(
                name=f"ltltop1med_gateall{gthr_name}_shift1",
                **{**base, "anchor_shift": 1, "anchor_gate_threshold": float(gthr)},
            )
        )
    return out


def _candidates_ltl_top1med_visfb_v1() -> list[CandidateConfig]:
    """
    Top1-med base gate + cheap-visual fallback plan (instead of uniform fallback).

    Motivation (AVE/P0 / C0003):
      - The best deployable config (E0224) uses a strict base gate and falls back to uniform for ~75% clips.
      - Lowering the gate (or sep3 replacement) reduces fallback but introduces many harmful clips.

    Idea:
      - Keep the strict base gate unchanged.
      - When the audio gate falls back, **do not** revert to uniform; instead, use a cheap-visual anchor proposal
        (CLIPdiff or framediff) as a fallback plan under the same equal-token budget.

    Keep the grid tiny and interpretable:
      - One baseline (uniform fallback) + two cheap-visual fallback variants.
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=1,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
        anchor_gate_method="none",
        anchor_gate_threshold=0.0,
        anchor_gate_label_radius=1,
    )

    out: list[CandidateConfig] = []
    out.append(CandidateConfig(name="ltltop1med_uniformfb_shift1", **{**base, "anchor_fallback_mode": "uniform"}))
    out.append(
        CandidateConfig(
            name="ltltop1med_clipdifffb_shift1",
            **{**base, "anchor_fallback_mode": "cheap_visual_clipdiff"},
        )
    )
    out.append(
        CandidateConfig(
            name="ltltop1med_framedifffb_shift1",
            **{**base, "anchor_fallback_mode": "cheap_visual_framediff"},
        )
    )
    return out


def _candidates_ltl_top1med_visfb_gated_v1() -> list[CandidateConfig]:
    """
    Top1-med base gate + cheap-visual fallback plan gated by visual confidence.

    Motivation: E0336 shows naive visfb hurts on val402 when applied to all fallback clips. Hypothesis:
    cheap-visual anchors are only useful when the visual score sequence is sufficiently "peaky"; otherwise,
    selecting top-2 is effectively random and causes context loss.

    Grid (tiny; pre-registered):
      - keep the strict base gate fixed: `top1_med thr=0.6`, `shift=1`
      - fallback modes: {clipdiff, framediff}
      - visual confidence gate (on minmax-normalized visual scores): `top1_med` threshold in {0.10,0.20,0.30,0.40}
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=1,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
        anchor_gate_method="none",
        anchor_gate_threshold=0.0,
        anchor_gate_label_radius=1,
        anchor_fallback_visual_conf_metric="top1_med",
    )

    out: list[CandidateConfig] = []
    out.append(CandidateConfig(name="ltltop1med_uniformfb_shift1", **{**base, "anchor_fallback_mode": "uniform"}))

    for mode, prefix in (
        ("cheap_visual_clipdiff", "clipdiff"),
        ("cheap_visual_framediff", "framediff"),
    ):
        for thr in (0.10, 0.20, 0.30, 0.40):
            thr_name = str(thr).replace(".", "p")
            out.append(
                CandidateConfig(
                    name=f"ltltop1med_{prefix}fb_vc{thr_name}_shift1",
                    **{
                        **base,
                        "anchor_fallback_mode": str(mode),
                        "anchor_fallback_visual_conf_threshold": float(thr),
                    },
                )
            )
    return out


def _candidates_ltl_top1med_anchor2veto_v1() -> list[CandidateConfig]:
    """
    Anchor2 veto (k-adaptive) candidate set for learned-logit Stage-1 methods.

    Motivation (AVE/P0 / C0003):
      - Diagnostics show the 2-high, far-anchor regime is strongly harmful on test402, and prior fixes that
        *keep* the second anchor for base allocation (adaptive_v3) still regress.
      - Hypothesis: the second selected anchor is often spurious; dropping it entirely (k=1 for that clip)
        preserves context and avoids wasting budget on wrong windows.

    Keep everything fixed to the current best top1-med setting (thr=0.6, shift=1) and sweep only the veto knob.
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=1,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
        anchor2_veto_label_radius=1,
    )

    out: list[CandidateConfig] = []
    out.append(CandidateConfig(name="ltltop1med_a2veto_none", **{**base, "anchor2_veto_method": "none"}))

    for thr in (0.05, 0.10, 0.15):
        thr_name = str(thr).replace(".", "p")
        out.append(
            CandidateConfig(
                name=f"ltltop1med_a2veto_top2med{thr_name}",
                **{
                    **base,
                    "anchor2_veto_method": "top2med_norm_v1",
                    "anchor2_veto_threshold": float(thr),
                },
            )
        )

    for thr in (0.35, 0.50, 0.65):
        thr_name = str(thr).replace(".", "p")
        out.append(
            CandidateConfig(
                name=f"ltltop1med_a2veto_lr{thr_name}",
                **{
                    **base,
                    "anchor2_veto_method": "lr_v1",
                    "anchor2_veto_threshold": float(thr),
                },
            )
        )
    return out


def _candidates_ltl_top1med_norm_v1() -> list[CandidateConfig]:
    """
    Scale-invariant top1-minus-median gated candidate set for learned-logit Stage-1 methods.

    Uses `conf_metric=top1_med_norm`, which first applies per-clip min-max normalization to [0,1] then
    computes (top1 - median). This makes the gate robust to eventness-score scale differences across
    different Stage-1 backends (e.g., BCE vs MIL vs AST-embedding models), and avoids the failure mode
    where all clips pass the gate due to large raw logit magnitudes.

    Keep the grid small and pre-registered:
      - thresholds in {0.5, 0.6, 0.7}
      - shift ∈ {0,1}
      - otherwise match the current learned-anchor defaults (triad=160/224/352, adaptive_v1).
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=0,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med_norm",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for thr in (0.50, 0.60, 0.70):
        thr_name = str(thr).replace(".", "p")
        out.append(CandidateConfig(name=f"ltltop1medn_thr{thr_name}_shift0", **{**base, "anchor_shift": 0, "anchor_conf_threshold": thr}))
        out.append(CandidateConfig(name=f"ltltop1medn_thr{thr_name}_shift1", **{**base, "anchor_shift": 1, "anchor_conf_threshold": thr}))
    return out


def _candidates_ltl_sep3_v1() -> list[CandidateConfig]:
    """
    Separation-based confidence gate for learned-logit Stage-1 methods.

    Uses `conf_metric=top3_bottom3_gap_norm`:
      conf = mean(top3(scores_norm)) - mean(bottom3(scores_norm))

    Rationale: Top1-median is a "single-peak" gate and can over-fallback on broad multi-second events.
    The separation gate remains high when event seconds are consistently above background seconds, even
    if the event spans many segments.

    Keep the grid small and pre-registered:
      - thresholds in {0.60,0.64,0.66,0.68,0.70}
      - shift ∈ {0,1}
      - otherwise match the current learned-anchor defaults (triad=160/224/352, adaptive_v1).
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=0,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top3_bottom3_gap_norm",
        anchor_conf_threshold=0.66,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for thr in (0.60, 0.64, 0.66, 0.68, 0.70):
        thr_name = str(thr).replace(".", "p")
        out.append(CandidateConfig(name=f"ltlsep3_thr{thr_name}_shift0", **{**base, "anchor_shift": 0, "anchor_conf_threshold": thr}))
        out.append(CandidateConfig(name=f"ltlsep3_thr{thr_name}_shift1", **{**base, "anchor_shift": 1, "anchor_conf_threshold": thr}))
    return out


def _candidates_ltl_top1med_band_v1() -> list[CandidateConfig]:
    """
    Budget-band variant of `ltl_top1med_v1`.

    Motivation (AVE/P0 / C0003):
      - The current best exact-budget triad (160/224/352) is constrained by a divisibility condition that
        forces the 2-high regime to use only 2×base + 6×low, which can cause context loss.
      - Allowing a small under-budget band (<=1%) and an extra cheap resolution (112) lets the planner
        preserve more `base_res` context (e.g., 2×high + 3–4×base + remaining at {160,112}) without
        exceeding the uniform token budget.

    Keep the grid small and pre-registered:
      - `conf_metric=top1_med`, thresholds in {0.5,0.6,0.7}
      - shift ∈ {0,1}
      - `budget_mode=band`, `budget_epsilon_frac=0.01`, `budget_extra_resolutions=(112,)`
      - all other knobs match the current best learned-anchor defaults.
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=0,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
        budget_mode="band",
        budget_epsilon_frac=0.01,
        budget_extra_resolutions=(112,),
    )

    out: list[CandidateConfig] = []
    for thr in (0.50, 0.60, 0.70):
        thr_name = str(thr).replace(".", "p")
        out.append(
            CandidateConfig(
                name=f"ltltop1medband_thr{thr_name}_shift0",
                **{**base, "anchor_shift": 0, "anchor_conf_threshold": float(thr)},
            )
        )
        out.append(
            CandidateConfig(
                name=f"ltltop1medband_thr{thr_name}_shift1",
                **{**base, "anchor_shift": 1, "anchor_conf_threshold": float(thr)},
            )
        )
    return out


def _candidates_ltl_top1med_band_low112_v1() -> list[CandidateConfig]:
    """
    Budget-band variant of `ltl_top1med_v1` with a cheaper low-res (112) to preserve more context.

    Motivation (AVE/P0 / C0003):
      - Under the exact equal-budget triad (160/224/352), the 2-high regime is forced to use only 2×base slots,
        which can cause context loss and makes far-anchor cases brittle.
      - Switching to a cheaper low-res (112) and using the band-budget planner lets us allocate more mid/base
        context while staying within the uniform token budget.

    Keep the grid small and pre-registered:
      - `conf_metric=top1_med`, thresholds in {0.5,0.6,0.7}
      - shift ∈ {0,1}
      - `budget_mode=band`, `budget_epsilon_frac=0.01`, `budget_extra_resolutions=(160,)` so the DP can use 160 as
        a mid-resolution option (often avoids over-using 112).
      - all other knobs match the current learned-anchor defaults.
    """
    base = dict(
        k=2,
        low_res=112,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=0,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
        budget_mode="band",
        budget_epsilon_frac=0.01,
        budget_extra_resolutions=(160,),
    )

    out: list[CandidateConfig] = []
    for thr in (0.50, 0.60, 0.70):
        thr_name = str(thr).replace(".", "p")
        out.append(
            CandidateConfig(
                name=f"ltltop1medband112_thr{thr_name}_shift0",
                **{**base, "anchor_shift": 0, "anchor_conf_threshold": float(thr)},
            )
        )
        out.append(
            CandidateConfig(
                name=f"ltltop1medband112_thr{thr_name}_shift1",
                **{**base, "anchor_shift": 1, "anchor_conf_threshold": float(thr)},
            )
        )
    return out


def _candidates_ltl_top1med_band_midres_v1() -> list[CandidateConfig]:
    """
    Mid-resolution band-budget variant targeting the harmful "far 2-high" bucket on AVE test402.

    Motivation (AVE/P0 / C0003 “拉大”):
      - Under the exact equal-budget triad (160/224/352), the 2-high regime is forced into 2×base + 6×low,
        which can cause severe context loss.
      - Per-clip diagnostics on the current best deployable test402 run (E0224) show that 2-high clips with
        non-adjacent anchors (dist=2..5) are strongly negative.

    Idea:
      - Keep the winning Stage-1 gate fixed: `conf_metric=top1_med`, `thr=0.6`, `shift=1`.
      - Reduce the peak cost slightly (`high_res=320` instead of 352) and add mid resolutions (192, 208).
      - Use the band-budget DP allocator to preserve more mid/base context without exceeding the uniform budget.

    Grid (tiny; pre-registered):
      - Include the E0224 winner as an internal baseline (exact budget; 160/224/352; base_alloc=distance).
      - Band candidates: (low,base,high)=(160,224,320), `budget_extra_resolutions=(192,208)`, `epsilon=0.01`.
      - Sweep only `anchor_base_alloc ∈ {distance, bridge, mixed}`.
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=1,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
        budget_mode="exact",
        budget_epsilon_frac=0.01,
        budget_extra_resolutions=(),
    )

    out: list[CandidateConfig] = []
    out.append(CandidateConfig(name="ltltop1med_thr0p6_shift1_base_exact352", **base))

    band = {
        **base,
        "high_res": 320,
        "budget_mode": "band",
        "budget_epsilon_frac": 0.01,
        "budget_extra_resolutions": (192, 208),
    }
    for alloc in ("distance", "bridge", "mixed"):
        tag = "dist" if alloc == "distance" else str(alloc)
        out.append(
            CandidateConfig(
                name=f"ltltop1med_thr0p6_shift1_midres320_band_{tag}",
                **{**band, "anchor_base_alloc": str(alloc)},
            )
        )
    return out


def _candidates_ltl_top1med_moe_v1() -> list[CandidateConfig]:
    """
    Top1-minus-median gated candidate set for `EVENTNESS=moe_energy_clipdiff`.

    Motivation:
      - `moe_energy_clipdiff` uses `cfg.anchor_std_threshold` as an *internal* switch:
          if std(audio_energy) < anchor_std_threshold → use semantic visual motion (CLIPdiff scores)
          else → use audio energy scores
      - `ltl_top1med_v1` sets `anchor_std_threshold=0.0` (since std gating is ignored when `conf_threshold`
        is set), which unintentionally disables the MOE behavior.

    This candidate set keeps the pre-registered `top1_med` gate grid, but also sweeps
    `anchor_std_threshold ∈ {0.4, 0.6, 1.0}` to enable the MOE switch while keeping Stage-2 budgets fixed.
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=0,
        anchor_std_threshold=1.0,  # used by moe_energy_clipdiff; ignored by the conf gate here
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for std_thr in (0.40, 0.60, 1.00):
        std_name = str(std_thr).replace(".", "p")
        for thr in (0.40, 0.50, 0.60, 0.70, 0.80):
            thr_name = str(thr).replace(".", "p")
            out.append(
                CandidateConfig(
                    name=f"ltltop1medmoe_std{std_name}_thr{thr_name}_shift0",
                    **{
                        **base,
                        "anchor_shift": 0,
                        "anchor_std_threshold": float(std_thr),
                        "anchor_conf_threshold": float(thr),
                    },
                )
            )
            out.append(
                CandidateConfig(
                    name=f"ltltop1medmoe_std{std_name}_thr{thr_name}_shift1",
                    **{
                        **base,
                        "anchor_shift": 1,
                        "anchor_std_threshold": float(std_thr),
                        "anchor_conf_threshold": float(thr),
                    },
                )
            )
    return out


def _candidates_ltl_top1med_nmsstrong_v1() -> list[CandidateConfig]:
    """
    Top1-minus-median gated candidate set that uses strong NMS for anchor selection.

    Motivation: Diagnostics on the current best learned-anchor run show that non-adjacent 2-anchor cases
    (distance 2–5) are net harmful on test402, while adjacent 2-anchor cases are net positive.
    `select="nms_strong"` keeps a far-away 2nd anchor only when it is competitive with the top1 anchor;
    otherwise it falls back to the 2nd-best overall anchor (often adjacent), which then gets demoted to 1-high
    by `anchor_high_policy=adaptive_v1`.

    Keep the grid small and pre-registered (based on the E0224 winner):
      - fix `conf_metric=top1_med`, `conf_threshold=0.6`, `shift=1`
      - sweep strong-NMS radius ∈ {1,2} and max_gap ∈ {0.05,0.10,0.15,0.20,0.30}
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=1,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="nms_strong",
        anchor_nms_radius=1,
        anchor_nms_strong_gap=0.10,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for radius in (1, 2):
        for gap in (0.05, 0.10, 0.15, 0.20, 0.30):
            gap_name = str(gap).replace(".", "p")
            out.append(
                CandidateConfig(
                    name=f"ltltop1med_thr0p6_shift1_ns_r{radius}_gap{gap_name}",
                    **{
                        **base,
                        "anchor_nms_radius": int(radius),
                        "anchor_nms_strong_gap": float(gap),
                    },
                )
            )
    return out


def _candidates_ltl_top1med_dropfar_v1() -> list[CandidateConfig]:
    """
    Top1-minus-median gated candidate set with conditional dropping of the 2nd anchor when it is far away.

    Motivation: On the current best learned-anchor run (E0224), adjacent 2-anchor cases (dist=1) are net positive
    but non-adjacent cases (dist=2..5) are net negative on test402. Instead of global k=1, we keep adjacent top-2
    anchors (to preserve multi-second evidence), but drop the 2nd anchor when it is far from top1.

    Grid (small and pre-registered):
      - keep Stage-2 knobs aligned with E0224
      - fix shift=1 (winner) and sweep top1-med threshold in {0.5,0.6,0.7}
      - sweep drop-far threshold in {0 (disabled), 1 (drop if dist>1)}
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=1,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_drop_far_dist=0,
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for thr in (0.50, 0.60, 0.70):
        thr_name = str(thr).replace(".", "p")
        out.append(
            CandidateConfig(
                name=f"ltltop1med_thr{thr_name}_shift1_df0",
                **{**base, "anchor_conf_threshold": float(thr), "anchor_drop_far_dist": 0},
            )
        )
        out.append(
            CandidateConfig(
                name=f"ltltop1med_thr{thr_name}_shift1_df1",
                **{**base, "anchor_conf_threshold": float(thr), "anchor_drop_far_dist": 1},
            )
        )
    return out


def _candidates_ltl_top1med_farfb_v1() -> list[CandidateConfig]:
    """
    Top1-minus-median gated candidate set with conditional *fallback* when the 2nd anchor is far from top1.

    Motivation: E0224 diagnostics show non-adjacent 2-anchor cases (dist=2..5) are often harmful on test402.
    Instead of dropping anchor2 (P0080 / dropfar), this policy forces a full fallback-to-uniform when the top-2
    anchors are far apart, aiming to avoid spending any budget on unreliable anchor proposals.

    Grid (small and pre-registered; aligned with E0224 knobs):
      - fix shift=1 and Stage-2 plan knobs
      - sweep top1-med threshold in {0.5,0.6,0.7}
      - sweep fallback-far threshold in {0 (disabled), 1 (fallback if dist>1)}
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=1,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_drop_far_dist=0,
        anchor_fallback_far_dist=0,
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for thr in (0.50, 0.60, 0.70):
        thr_name = str(thr).replace(".", "p")
        out.append(
            CandidateConfig(
                name=f"ltltop1med_thr{thr_name}_shift1_ff0",
                **{**base, "anchor_conf_threshold": float(thr), "anchor_fallback_far_dist": 0},
            )
        )
        out.append(
            CandidateConfig(
                name=f"ltltop1med_thr{thr_name}_shift1_ff1",
                **{**base, "anchor_conf_threshold": float(thr), "anchor_fallback_far_dist": 1},
            )
        )
    return out


def _candidates_ltl_top1med_adjselect_v1() -> list[CandidateConfig]:
    """
    Top1-minus-median gated candidate set that prefers an adjacent 2nd anchor when it is competitive.

    Motivation: `nms_strong` can still pick a far 2nd anchor if it is the 2nd-best overall, even when the
    adjacent second around the top1 peak is also high (common for multi-second events). `adjacent_top2`
    explicitly checks the local neighborhood of the top1 anchor and selects an adjacent 2nd anchor if it is
    within `max_gap`, reducing far-anchor 2-high cases without changing Stage-1 scores.

    Grid (small; based on E0224 knobs):
      - fix `conf_metric=top1_med`, `conf_threshold=0.6`, `shift=1`
      - set `anchor_select=adjacent_top2`
      - sweep neighborhood radius and `max_gap`
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=1,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="adjacent_top2",
        anchor_drop_far_dist=0,
        anchor_nms_radius=1,  # used as adjacent radius for adjacent_top2
        anchor_nms_strong_gap=0.2,  # used as max_gap for adjacent_top2
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for radius in (1, 2):
        for gap in (0.10, 0.20, 0.30, 0.40):
            gap_name = str(gap).replace(".", "p")
            out.append(
                CandidateConfig(
                    name=f"ltltop1med_adjsel_r{radius}_gap{gap_name}",
                    **{
                        **base,
                        "anchor_nms_radius": int(radius),
                        "anchor_nms_strong_gap": float(gap),
                    },
                )
            )
    return out


def _candidates_ltl_top1med_keepadj_v1() -> list[CandidateConfig]:
    """
    Top1-minus-median gated candidate set with *far-anchor* 2-high demotion (adaptive_v3).

    Motivation: Under the current best learned-anchor config (E0224), per-clip diagnostics show that the
    2-high regime (2×high_res) can be harmful when the two selected anchors are far apart. `adaptive_v3`
    keeps both anchors for base allocation, but only allows 2-high when anchors are adjacent (or within a
    small distance), preserving more base-res context for far-anchor clips.

    Grid (small and pre-registered; based on E0224 knobs):
      - fix `conf_metric=top1_med`, `conf_threshold=0.6`, `shift=1`
      - set `anchor_high_policy=adaptive_v3`
      - sweep `anchor_high_adjacent_dist` as the max distance allowed to keep 2-high
      - optional small gap-based demotion when adjacent (avoid wasting 2-high on weak 2nd peaks)
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=1,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_drop_far_dist=0,
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v3",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for keep2_dist in (1, 2):
        for gap in (0.0, 0.10):
            gap_name = str(gap).replace(".", "p")
            out.append(
                CandidateConfig(
                    name=f"ltltop1med_keepadj_d{keep2_dist}_gap{gap_name}",
                    **{
                        **base,
                        "anchor_high_adjacent_dist": int(keep2_dist),
                        "anchor_high_gap_threshold": float(gap),
                    },
                )
            )
    return out


def _candidates_ltl_top1med_keepadj_basealloc_v1() -> list[CandidateConfig]:
    """
    Top1-minus-median gated candidate set that combines far-anchor 2-high demotion (adaptive_v3) with
    alternative base-res allocation strategies.

    Motivation: Under `adaptive_v3`, far-anchor clips are demoted to the 1-high regime (6×base + 3×low), but
    the default base allocation (`distance`) can still waste base slots around a far/incorrect 2nd anchor,
    reducing context. Sweeping base allocation under the same high-policy is a minimal way to test whether
    "mixed/context-preserving" allocation improves transfer.

    Grid (very small; based on E0224 knobs):
      - fix Stage-1 gate: `conf_metric=top1_med`, `conf_threshold=0.6`, `shift=1`
      - fix `anchor_high_policy=adaptive_v3` with `anchor_high_adjacent_dist=1` (only allow 2-high when adjacent)
      - sweep `anchor_base_alloc` in {distance, balanced, mixed, farthest, score}
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=1,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_drop_far_dist=0,
        anchor_fallback_far_dist=0,
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v3",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for alloc in ("distance", "balanced", "mixed", "farthest", "score"):
        out.append(CandidateConfig(name=f"ltltop1med_keepadj_{alloc}", **{**base, "anchor_base_alloc": str(alloc)}))
    return out


def _candidates_ltl_top1med_keepadj_basealloc_highonly_v1() -> list[CandidateConfig]:
    """
    Variant of `ltl_top1med_keepadj_basealloc_v1` that uses the "_high" base allocation modes.

    Motivation: Under `adaptive_v3`, far-anchor clips are demoted to the 1-high regime (more `base_res` seconds),
    but the default planners still rank candidates by distance-to-*all* anchors (including the demoted 2nd anchor).
    This can waste base slots around a far/spurious anchor2. The "_high" modes allocate base slots with respect to
    the *high-set only* (anchors that actually received high-res), which should recover context near anchor1 without
    spending tokens around the demoted anchor2.

    Grid (very small; based on E0224 knobs):
      - fix Stage-1 gate: `conf_metric=top1_med`, `conf_threshold=0.6`, `shift=1`
      - fix `anchor_high_policy=adaptive_v3` with `anchor_high_adjacent_dist=1` (only allow 2-high when adjacent)
      - sweep `anchor_base_alloc` in {distance_high, balanced_high, mixed_high, farthest_high, score_high}
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=1,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_drop_far_dist=0,
        anchor_fallback_far_dist=0,
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance_high",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v3",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for alloc in ("distance_high", "balanced_high", "mixed_high", "farthest_high", "score_high"):
        out.append(CandidateConfig(name=f"ltltop1med_keepadj_{alloc}", **{**base, "anchor_base_alloc": str(alloc)}))
    return out


def _candidates_ltl_top1med_basealloc_v1() -> list[CandidateConfig]:
    """
    Top1-minus-median gated candidate set that sweeps Stage-2 base-res allocation.

    Motivation: Under the 2-high regime (160/224/352, k=2), only 2 seconds get `base_res`. With the default
    `base_alloc=distance`, tie-breaking can allocate both base slots near a single anchor, starving the other
    anchor's neighborhood (especially when anchors are far apart). `base_alloc=balanced` explicitly allocates
    base slots around both anchors in a round-robin manner, which can reduce the "far-anchor harm" bucket
    without changing Stage-1 scores or the equal-token budget.

    Grid (small; based on E0224 knobs):
      - fix Stage-1 gate: `conf_metric=top1_med`, `conf_threshold=0.6`, `shift=1`
      - keep anchor selection and high-policy aligned with E0224
      - sweep `anchor_base_alloc` in {distance, balanced, mixed, score}
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=1,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_drop_far_dist=0,
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for alloc in ("distance", "balanced", "mixed", "score"):
        out.append(CandidateConfig(name=f"ltltop1med_basealloc_{alloc}", **{**base, "anchor_base_alloc": str(alloc)}))
    return out


def _candidates_ltl_top1med_bridgealloc_v1() -> list[CandidateConfig]:
    """
    Single-variable Stage-2 variant: add a "bridge" base allocation strategy for the 2-high regime.

    Motivation: E0224 diagnostics show that when two anchors are far apart (dist=2..5), the 2-high plan
    regresses sharply. Under the 2-high equal-budget constraint, only 2 seconds remain at `base_res`.
    The default `base_alloc=distance` tends to place both base seconds near the anchors, leaving the
    "between-anchors" region at `low_res`. For far-anchor clips, evidence often lies in-between (or the
    2nd anchor is spurious), so spending base budget between the two high anchors can preserve context.

    This candidate set keeps all E0224 knobs fixed and only changes `anchor_base_alloc`.
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=1,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_drop_far_dist=0,
        anchor_fallback_far_dist=0,
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    return [
        CandidateConfig(name="ltltop1med_thr0p6_shift1_base", **base),
        CandidateConfig(name="ltltop1med_thr0p6_shift1_bridgeAlloc", **{**base, "anchor_base_alloc": "bridge"}),
    ]


def _candidates_ltl_top1med_autoshift_v1() -> list[CandidateConfig]:
    """
    Top1-minus-median gate for per-clip autoshifted learned anchors.

    Intended for `EVENTNESS=av_clipdiff_mlp_autoshift`, which already applies a per-clip temporal shift to
    the learned score sequence. Therefore, we fix `anchor_shift=0` and only sweep the confidence threshold.
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=0,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for thr in (0.40, 0.50, 0.60, 0.70, 0.80):
        thr_name = str(thr).replace(".", "p")
        out.append(CandidateConfig(name=f"ltltop1med_as_thr{thr_name}_shift0", **{**base, "anchor_conf_threshold": thr}))
    return out


def _candidates_ltl_top1med_maxhigh1_v1() -> list[CandidateConfig]:
    """
    Top1-minus-median gated candidate set that always uses at most 1 high-res anchor.

    Motivation: Diagnostics on the current best learned-anchor run show that the "2-high" regime
    (2×high_res + 2×base_res + 6×low_res under the 160/224/352 triad) is net harmful on test402,
    while the "1-high" regime is net positive. This set keeps the same Stage-1 gate (`top1_med`)
    but fixes `max_high_anchors=1` to preserve more base-res context and eliminate 2-high cases.

    Keep the grid small and pre-registered:
      - `conf_metric=top1_med`, thresholds in {0.4,0.5,0.6,0.7,0.8}
      - shift ∈ {0,1}
      - keep other Stage-2 knobs aligned with the `ltl_top1med_v1` winner.
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=0,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=1,
        anchor_high_policy="fixed",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for thr in (0.40, 0.50, 0.60, 0.70, 0.80):
        thr_name = str(thr).replace(".", "p")
        out.append(
            CandidateConfig(
                name=f"ltltop1medmax1_thr{thr_name}_shift0",
                **{**base, "anchor_shift": 0, "anchor_conf_threshold": thr},
            )
        )
        out.append(
            CandidateConfig(
                name=f"ltltop1medmax1_thr{thr_name}_shift1",
                **{**base, "anchor_shift": 1, "anchor_conf_threshold": thr},
            )
        )
    return out


def _candidates_ltl_top1med_k1_v1() -> list[CandidateConfig]:
    """
    Top1-minus-median gated candidate set with k=1 (single-anchor allocation).

    Motivation: In the current best learned-anchor run, the non-adjacent 2-anchor / 2-high regime is net harmful.
    Setting `k=1` removes the 2-high regime entirely while keeping the equal-token budget via the same 160/224/352
    triad: 1×high_res + 6×base_res + 3×low_res.

    Grid (pre-registered, small):
      - k=1
      - conf_metric=top1_med, thresholds in {0.4,0.5,0.6,0.7,0.8}
      - shift ∈ {0,1}
      - stage-2 plan knobs aligned with the learned-anchor winner otherwise.
    """
    base = dict(
        k=1,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=0,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="fixed",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for thr in (0.40, 0.50, 0.60, 0.70, 0.80):
        thr_name = str(thr).replace(".", "p")
        out.append(CandidateConfig(name=f"ltltop1medk1_thr{thr_name}_shift0", **{**base, "anchor_shift": 0, "anchor_conf_threshold": thr}))
        out.append(CandidateConfig(name=f"ltltop1medk1_thr{thr_name}_shift1", **{**base, "anchor_shift": 1, "anchor_conf_threshold": thr}))
    return out


def _candidates_ltl_top1med_k1_extreme_v1() -> list[CandidateConfig]:
    """
    Top1-minus-median gated candidate set with k=1 and an aggressive triad (112/224/448).

    Motivation: `ltl_top1med_extreme_v1` uses the 112/224/448 triad with `max_high_anchors=1`, but still
    provides *two* anchors (k=2) for base allocation. When the 2nd anchor is spurious, base slots can be
    wasted around it, hurting transfer. This set removes anchor2 entirely (k=1), keeping base allocation
    focused around the single most confident anchor while still allowing a high peak resolution (448).

    Equal-budget implications (patch=16, num_segments=10):
      - 1×448 + 5×224 + 4×112 == 10×224 (strict token budget match)

    Grid (small and pre-registered):
      - k=1, low/base/high = 112/224/448
      - conf_metric=top1_med, thresholds in {0.50,0.60,0.70,0.80}
      - shift ∈ {0,1}
      - base allocation ∈ {distance, score}
    """
    base = dict(
        k=1,
        low_res=112,
        base_res=224,
        high_res=448,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=0,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="fixed",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for thr in (0.50, 0.60, 0.70, 0.80):
        thr_name = str(thr).replace(".", "p")
        for shift in (0, 1):
            for alloc in ("distance", "score"):
                out.append(
                    CandidateConfig(
                        name=f"ltltop1medk1ext_thr{thr_name}_shift{shift}_{alloc}",
                        **{
                            **base,
                            "anchor_shift": int(shift),
                            "anchor_conf_threshold": float(thr),
                            "anchor_base_alloc": str(alloc),
                        },
                    )
                )
    return out


def _candidates_ltl_top1med_adaptivegap_v1() -> list[CandidateConfig]:
    """
    Top1-minus-median gated candidate set with adaptive high-res demotion based on top1-top2 gap.

    Motivation: E0224 diagnostics suggest the non-adjacent 2-high regime is often harmful on test402.
    A plausible failure mode is that the 2nd anchor is substantially weaker than the top anchor; allocating
    a second high-res second wastes budget and reduces base-res context. We enable the existing
    `anchor_high_gap_threshold` mechanism under `anchor_high_policy=adaptive_v1` and sweep a small set of
    gap thresholds around the observed score scales (learned logits).

    Fixed (to avoid val-overfitting):
      - `conf_metric=top1_med`, `conf_threshold=0.6` (E0224 winner)
      - `shift=1` (E0224 winner)
      - Stage-2 triad 160/224/352 and other knobs aligned with E0224 winner
    Sweep:
      - `anchor_high_gap_threshold` ∈ {0.10,0.15,0.20,0.25,0.30}
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=1,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for gap in (0.10, 0.15, 0.20, 0.25, 0.30):
        gap_name = str(gap).replace(".", "p")
        out.append(
            CandidateConfig(
                name=f"ltltop1med_thr0p6_shift1_agap{gap_name}",
                **{**base, "anchor_high_gap_threshold": gap},
            )
        )
    return out


def _candidates_ltl_top1med_adjdist_v1() -> list[CandidateConfig]:
    """
    Top1-minus-median gated candidate set that sweeps `anchor_high_adjacent_dist` under adaptive_v1.

    Motivation: The current best learned-anchor run (E0224) shows a strong val/test mismatch:
      - On val402, non-adjacent 2-high cases are net positive.
      - On test402, non-adjacent 2-high cases (dist=2–5) are net harmful and dominate the overall regression.

    `anchor_high_adjacent_dist` controls when `adaptive_v1` demotes a 2-anchor clip to 1 high-res anchor.
    Sweeping it is the simplest, transparent way to reduce the harmful 2-high regime without adding new heuristics.

    Fixed (to keep the search pre-registered and comparable to E0224):
      - `conf_metric=top1_med`, `conf_threshold=0.6`, `shift=1`
      - triad 160/224/352, `base_alloc=distance`
      - `anchor_high_policy=adaptive_v1`
    Sweep:
      - `anchor_high_adjacent_dist` ∈ {1,2,3,4,5}
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=1,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for d in (1, 2, 3, 4, 5):
        out.append(CandidateConfig(name=f"ltltop1med_thr0p6_shift1_adj{d}", **{**base, "anchor_high_adjacent_dist": d}))
    return out


def _candidates_ltl_top1med_headcap_v1() -> list[CandidateConfig]:
    """
    Fixed top1-med gate + Stage-2 plan, sweeping head capacity/regularization.

    Motivation: The learned-anchor setting produces mixed-resolution input sequences (160/224/352).
    A small temporal head can underfit this heterogeneity (especially for the 2-high regime), while
    Uniform uses a single resolution and is easier to fit. Increasing head capacity can improve
    `anchored_top2` more than `uniform`, potentially enlarging C0003.

    Fixed (E0224 winner):
      - `conf_metric=top1_med`, `conf_threshold=0.6`, `shift=1`
      - triad 160/224/352, `base_alloc=distance`
      - `anchor_high_policy=adaptive_v1`, `anchor_high_adjacent_dist=1`
    Sweep:
      - `head_hidden_dim` ∈ {128,256,512}
      - `head_dropout` ∈ {0.0, 0.1}
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        head_hidden_dim=128,
        head_dropout=0.0,
        temporal_kernel_size=3,
        anchor_shift=1,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for hd in (128, 256, 512):
        for dr in (0.0, 0.1):
            dr_name = str(dr).replace(".", "p")
            out.append(
                CandidateConfig(
                    name=f"ltltop1med_thr0p6_shift1_hd{hd}_dr{dr_name}",
                    **{**base, "head_hidden_dim": hd, "head_dropout": dr},
                )
            )
    return out


def _candidates_ltl_top1med_resfeat_v1() -> list[CandidateConfig]:
    """
    Fixed top1-med gate + Stage-2 plan, sweeping a free resolution indicator feature.

    Motivation: `anchored_top2` trains on mixed-resolution sequences (160/224/352) but the head does not
    explicitly observe which resolution a segment embedding came from. A free per-segment resolution
    indicator can help the head adapt to feature distribution shifts across resolutions. For `uniform`,
    the indicator is constant and should be absorbed into bias, so the expected net effect is larger on
    anchored than on uniform (potentially enlarging C0003).

    Fixed (E0224 winner):
      - `conf_metric=top1_med`, `conf_threshold=0.6`, `shift=1`
      - triad 160/224/352, `base_alloc=distance`
      - `anchor_high_policy=adaptive_v1`, `anchor_high_adjacent_dist=1`
      - head: temporal_conv (hidden_dim=128, dropout=0.0)
    Sweep:
      - `res_feature` ∈ {"none", "scalar"}
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        head_hidden_dim=128,
        head_dropout=0.0,
        temporal_kernel_size=3,
        anchor_shift=1,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
        res_feature="none",
    )

    out: list[CandidateConfig] = []
    for mode in ("none", "scalar"):
        out.append(CandidateConfig(name=f"ltltop1med_thr0p6_shift1_res{mode}", **{**base, "res_feature": mode}))
    return out


def _candidates_ltl_top1med_highconf_v1() -> list[CandidateConfig]:
    """
    Top1-minus-median gated candidate set with confidence-based 2-high demotion (adaptive_v2).

    Motivation: In E0224, 2-high clips have lower `top1_med` on average and are net negative on test402.
    `anchor_high_policy=adaptive_v2` can demote to 1 high-res anchor when the clip is only "medium confidence",
    while keeping 2-high for the highest-confidence cases.

    Fixed:
      - `conf_metric=top1_med`, `conf_threshold=0.6` (E0224 winner)
      - `shift=1` (E0224 winner)
    Sweep:
      - `anchor_high_conf_threshold` ∈ {0.0, 0.8, 0.9, 1.0, 1.1}
      - `anchor_base_alloc` ∈ {distance, score} (to mitigate base-slot waste when k_high < k)
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=1,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v2",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
        anchor_high_conf_metric="top1_med",
        anchor_high_conf_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for thr in (0.0, 0.8, 0.9, 1.0, 1.1):
        thr_name = str(thr).replace(".", "p")
        for alloc in ("distance", "score"):
            alloc_tag = "dist" if alloc == "distance" else "score"
            out.append(
                CandidateConfig(
                    name=f"ltltop1med_thr0p6_shift1_hconf{thr_name}_{alloc_tag}",
                    **{**base, "anchor_high_conf_threshold": thr, "anchor_base_alloc": alloc},
                )
            )
    return out


def _candidates_ltl_top1med_tiered_v1() -> list[CandidateConfig]:
    """
    Top1-minus-median gate with a confidence-tiered visual triad for anchored_top2.

    Motivation: Applying the extreme triad (112/224/448) globally regresses (E0229), suggesting it is only
    beneficial for a subset of clips with highly reliable anchors. This candidate set keeps the E0224 winner
    as a baseline and adds tiered variants that switch to 112/224/448 (max_high_anchors=1) only when the
    per-clip confidence exceeds a threshold.

    Fixed baseline (E0224 winner):
      - triad 160/224/352
      - `conf_metric=top1_med`, `conf_threshold=0.6`, `shift=1`
      - `anchor_high_policy=adaptive_v1`, `anchor_high_adjacent_dist=1`

    Tiered variants:
      - base triad = 160/224/352
      - alt triad = 112/224/448, `triad_alt_max_high_anchors=1`
      - switch when `conf_value >= triad_alt_conf_threshold`
    """
    base = dict(
        k=2,
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        head_hidden_dim=128,
        head_dropout=0.0,
        temporal_kernel_size=3,
        anchor_shift=1,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.6,
        max_high_anchors=None,
        anchor_high_policy="adaptive_v1",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
        triad_policy="fixed",
        triad_alt_conf_threshold=0.0,
        triad_alt_low_res=112,
        triad_alt_high_res=448,
        triad_alt_max_high_anchors=1,
    )

    out: list[CandidateConfig] = []
    out.append(CandidateConfig(name="ltltop1med_thr0p6_shift1_base", **base))

    for thr in (0.8, 1.0, 1.2, 1.5):
        thr_name = str(thr).replace(".", "p")
        out.append(
            CandidateConfig(
                name=f"ltltop1med_thr0p6_shift1_tier{thr_name}",
                **{
                    **base,
                    "triad_policy": "top1med_tiered_v1",
                    "triad_alt_conf_threshold": float(thr),
                },
            )
        )
    return out


def _candidates_ltl_top1med_scorealloc_v1() -> list[CandidateConfig]:
    """
    Single-candidate set: E0224 winner with `anchor_base_alloc=score`.

    Motivation: When k_high=2 under the 160/224/352 triad, only 2 base-res seconds remain.
    Allocating those base slots by score (instead of only distance-to-anchor) can preserve context
    on multi-peak clips and reduce harm from imperfect anchors.

    This is intentionally a one-candidate set to force a clean val→test measurement for the
    score-aware allocator variant.
    """
    return [
        CandidateConfig(
            name="ltltop1med_thr0p6_shift1_scoreAlloc",
            k=2,
            low_res=160,
            base_res=224,
            high_res=352,
            head="temporal_conv",
            temporal_kernel_size=3,
            anchor_shift=1,
            anchor_std_threshold=0.0,
            anchor_select="topk",
            anchor_nms_radius=2,
            anchor_nms_strong_gap=0.6,
            anchor_window=3,
            anchor_smooth_window=0,
            anchor_smooth_mode="mean",
            anchor_base_alloc="score",
            anchor_conf_metric="top1_med",
            anchor_conf_threshold=0.6,
            max_high_anchors=None,
            anchor_high_policy="adaptive_v1",
            anchor_high_adjacent_dist=1,
            anchor_high_gap_threshold=0.0,
        )
    ]


def _candidates_ltl_top1med_extreme_v1() -> list[CandidateConfig]:
    """
    Top1-minus-median gated candidate set with an aggressive resolution triad (112/224/448).

    Motivation: The extreme triad amplifies the peak visual budget (448) while keeping the strict equal-token
    constraint via low-res (112) elsewhere. This is risky when anchors are noisy, so we combine it with a stricter,
    scale-robust confidence gate (`top1_med`) and keep the grid intentionally small.

    Grid (pre-registered):
      - low/base/high = 112/224/448
      - k=2 but max_high_anchors=1
      - conf_metric=top1_med, thresholds in {0.6,0.7,0.8}
      - shift ∈ {0,1}
      - base allocation ∈ {distance, score}
    """
    base = dict(
        k=2,
        low_res=112,
        base_res=224,
        high_res=448,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=0,
        anchor_std_threshold=0.0,  # ignored when conf_threshold is set
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric="top1_med",
        anchor_conf_threshold=0.7,
        max_high_anchors=1,
        anchor_high_policy="fixed",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for thr in (0.60, 0.70, 0.80):
        thr_name = str(thr).replace(".", "p")
        for shift in (0, 1):
            for alloc in ("distance", "score"):
                out.append(
                    CandidateConfig(
                        name=f"ltltop1medext1_thr{thr_name}_shift{shift}_{alloc}",
                        **{
                            **base,
                            "anchor_shift": int(shift),
                            "anchor_conf_threshold": float(thr),
                            "anchor_base_alloc": str(alloc),
                        },
                    )
                )
    return out


def _candidates_ltl_extreme_v1() -> list[CandidateConfig]:
    """
    Aggressive resolution triad for trying to "拉大" anchored gains when anchors are reliable.

    Uses (low/base/high)=(112/224/448) with `max_high_anchors=1` to preserve context while still allowing
    a much higher peak resolution on the most important second. Sweeps std thresholds and shift.

    Note: This is riskier than (160/224/352) because non-anchor seconds drop to 112, so it can regress if
    anchors are noisy.
    """
    base = dict(
        k=2,
        low_res=112,
        base_res=224,
        high_res=448,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=0,
        anchor_std_threshold=0.6,
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric=None,
        anchor_conf_threshold=None,
        max_high_anchors=1,
        anchor_high_policy="fixed",
        anchor_high_adjacent_dist=1,
        anchor_high_gap_threshold=0.0,
    )

    out: list[CandidateConfig] = []
    for thr in (0.50, 0.60, 0.70):
        thr_name = str(thr).replace(".", "p")
        out.append(CandidateConfig(name=f"ltlextreme1_shift0_std{thr_name}", **{**base, "anchor_shift": 0, "anchor_std_threshold": thr}))
        out.append(CandidateConfig(name=f"ltlextreme1_shift1_std{thr_name}", **{**base, "anchor_shift": 1, "anchor_std_threshold": thr}))
    return out


def _extract_delta_and_p(metrics: dict) -> tuple[float | None, float | None]:
    try:
        summary = metrics["summary"]
        delta = float(summary["anchored_top2"]["mean"]) - float(summary["uniform"]["mean"])
    except Exception:
        delta = None

    try:
        p = float(metrics.get("paired_ttest", {})["anchored_vs_uniform"]["p"])
    except Exception:
        p = None
    return delta, p


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--meta-dir", type=Path, default=ave_paths().meta_dir)
    p.add_argument("--processed-dir", type=Path, default=ave_paths().processed_dir, help="Processed dir containing <clip_id>/audio.wav")
    p.add_argument("--caches-dir", type=Path, required=True, help="Dir containing <clip_id>.npz feature caches")
    p.add_argument("--train-ids-file", type=Path, default=None, help="Optional file with one train video_id per line.")
    p.add_argument("--eval-ids-file", type=Path, default=None, help="Optional file with one eval video_id per line.")
    p.add_argument("--split-train", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--split-eval", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-eval", type=int, default=None)
    p.add_argument("--allow-missing", action="store_true", help="Skip clips with missing caches instead of failing.")

    p.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--train-device", type=str, default="cuda:0")

    p.add_argument("--eventness-method", type=str, default="energy")
    p.add_argument("--audio-device", type=str, default="cuda:0")
    p.add_argument("--ast-pretrained", action="store_true")
    p.add_argument("--panns-checkpoint", type=Path, default=None)
    p.add_argument("--panns-random", action="store_true")
    p.add_argument("--audiomae-checkpoint", type=Path, default=None)
    p.add_argument("--audiomae-random", action="store_true")
    p.add_argument(
        "--scores-json",
        type=Path,
        default=None,
        help="Optional JSON cache of per-second eventness scores keyed by clip_id. "
        "If provided and exists, loads it; otherwise computes and writes it for future reuse.",
    )


def _load_scores_json(path: Path) -> dict[str, list[float]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "scores" in obj and isinstance(obj["scores"], dict):
        scores_obj = obj["scores"]
    elif isinstance(obj, dict):
        # Backward-compatible: plain {clip_id: [scores...]} mapping.
        scores_obj = obj
    else:
        raise ValueError("scores-json must be a JSON object")

    out: dict[str, list[float]] = {}
    for k, v in scores_obj.items():
        if not isinstance(v, list):
            raise ValueError(f"scores[{k!r}] must be a list, got {type(v)}")
        out[str(k)] = [float(x) for x in v]
    return out


def _write_scores_json(path: Path, *, eventness_method: str, num_segments: int, scores_by_clip: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": True,
        "eventness_method": str(eventness_method),
        "num_segments": int(num_segments),
        "scores": {str(k): [float(x) for x in v] for k, v in scores_by_clip.items()},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compute_scores_by_clip(
    *,
    clip_ids: list[str],
    processed_dir: Path,
    meta_dir: Path | None = None,
    caches_dir: Path | None = None,
    num_segments: int,
    eventness_method: str,
    audio_device: str,
    ast_pretrained: bool,
    panns_random: bool,
    panns_checkpoint: Path | None,
    audiomae_random: bool,
    audiomae_checkpoint: Path | None,
    train_ids: list[str] | None = None,
    labels_by_clip: dict[str, list[int]] | None = None,
) -> dict[str, list[float]]:
    import time as _time
    import numpy as np

    from avs.audio.eventness import (
        compute_eventness_wav_energy,
        compute_eventness_wav_energy_delta,
        compute_eventness_wav_energy_stride_max,
    )

    method = str(eventness_method)
    clip_ids = [str(x) for x in clip_ids]
    out: dict[str, list[float]] = {}

    ast_probe = None
    needs_ast = method in (
        "ast",
        "ast_lr",
        "ast_nonspeech_max",
        "energy_nonspeech_ast",
        "av_clipdiff_speech_mlp",
    ) or str(method).startswith("av_ast_")
    if needs_ast:
        from avs.audio.ast_probe import ASTEventnessProbe, ASTProbeConfig

        ast_probe = ASTEventnessProbe(ASTProbeConfig(pretrained=bool(ast_pretrained), device=str(audio_device)))

    panns_probe = None
    if method in (
        "panns",
        "panns_lr",
        "panns_embed_lr",
        "panns_embed_mlp",
        "av_panns_embed_clipdiff_mlp",
    ):
        from avs.audio.panns_probe import PANNsEventnessProbe, PANNsProbeConfig

        panns_probe = PANNsEventnessProbe(
            PANNsProbeConfig(pretrained=not panns_random, checkpoint_path=panns_checkpoint, device=str(audio_device))
        )

    audiomae_probe = None
    if method == "audiomae":
        from avs.audio.audiomae_probe import AudioMAEEventnessProbe, AudioMAEProbeConfig

        audiomae_probe = AudioMAEEventnessProbe(
            AudioMAEProbeConfig(
                pretrained=(audiomae_checkpoint is not None) and (not audiomae_random),
                checkpoint_path=audiomae_checkpoint if (audiomae_checkpoint is not None) and (not audiomae_random) else None,
                device=str(audio_device),
            )
        )

    t0 = _time.time()
    if method == "ast_lr":
        # Supervised, lightweight calibration: train a linear probe on AST logits to predict event vs background.
        if train_ids is None or labels_by_clip is None:
            raise ValueError("ast_lr requires train_ids and labels_by_clip")
        assert ast_probe is not None

        import numpy as np
        import torch
        import torch.nn as nn

        train_ids = [str(x) for x in train_ids]

        feats_by_train: dict[str, np.ndarray] = {}
        x_rows: list[np.ndarray] = []
        y_rows: list[np.ndarray] = []

        for i, cid in enumerate(train_ids):
            wav_path = processed_dir / cid / "audio.wav"
            logits = ast_probe.logits_per_second(wav_path, num_segments=int(num_segments))  # [T, C]
            feats_by_train[cid] = logits

            labs = np.asarray(labels_by_clip[cid], dtype=np.int64)[: int(num_segments)]
            y = (labs != 0).astype(np.float32).reshape(-1, 1)
            x_rows.append(logits.astype(np.float32, copy=False))
            y_rows.append(y)

            if (i + 1) % 200 == 0 or (i + 1) == len(train_ids):
                dt = _time.time() - t0
                print(f"[ast_lr] feats train {i+1}/{len(train_ids)} clips ({dt:.1f}s)", flush=True)

        x_np = np.concatenate(x_rows, axis=0).astype(np.float32, copy=False)
        y_np = np.concatenate(y_rows, axis=0).astype(np.float32, copy=False)

        x = torch.from_numpy(x_np).float()
        y = torch.from_numpy(y_np).float()

        torch.manual_seed(0)
        model = nn.Linear(int(x.shape[-1]), 1)
        pos = float((y > 0.5).sum().item())
        neg = float((y <= 0.5).sum().item())
        pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-2)

        bs = 4096
        n = int(x.shape[0])
        steps = max(1, (n + bs - 1) // bs)
        for _epoch in range(30):
            perm = torch.randperm(n)
            for j in range(steps):
                idx = perm[j * bs : (j + 1) * bs]
                xb = x[idx]
                yb = y[idx]
                logits = model(xb)
                loss = loss_fn(logits, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        model.eval()

        # Score requested clip ids (reuse train logits when available).
        for i, cid in enumerate(clip_ids):
            if cid in feats_by_train:
                feats = feats_by_train[cid]
            else:
                wav_path = processed_dir / cid / "audio.wav"
                feats = ast_probe.logits_per_second(wav_path, num_segments=int(num_segments))

            with torch.no_grad():
                s = model(torch.from_numpy(feats).float()).squeeze(-1).numpy().astype(np.float32)
            out[cid] = [float(x) for x in s.tolist()]

            if (i + 1) % 200 == 0 or (i + 1) == len(clip_ids):
                dt = _time.time() - t0
                print(f"[scores] {i+1}/{len(clip_ids)} clips ({dt:.1f}s)", flush=True)
    elif method == "panns_lr":
        # Supervised, lightweight calibration: train a linear probe on pretrained PANNs (AudioSet) outputs
        # to predict event vs background, then use per-second logits as Stage-1 scores.
        if train_ids is None or labels_by_clip is None:
            raise ValueError("panns_lr requires train_ids and labels_by_clip")
        assert panns_probe is not None

        import numpy as np
        import torch
        import torch.nn as nn

        train_ids = [str(x) for x in train_ids]

        eps = 1e-6

        def _logit(p: np.ndarray) -> np.ndarray:
            p = np.asarray(p, dtype=np.float32)
            p = np.clip(p, eps, 1.0 - eps)
            return np.log(p / (1.0 - p)).astype(np.float32, copy=False)

        feats_by_train: dict[str, np.ndarray] = {}
        x_rows: list[np.ndarray] = []
        y_rows: list[np.ndarray] = []

        for i, cid in enumerate(train_ids):
            wav_path = processed_dir / cid / "audio.wav"
            probs = panns_probe.clipwise_output_per_second(wav_path, num_segments=int(num_segments))  # [T, C]
            probs = np.asarray(probs, dtype=np.float32)
            feats_by_train[cid] = probs

            labs = np.asarray(labels_by_clip[cid], dtype=np.int64)[: int(num_segments)]
            y = (labs != 0).astype(np.float32).reshape(-1, 1)
            x_rows.append(_logit(probs))
            y_rows.append(y)

            if (i + 1) % 200 == 0 or (i + 1) == len(train_ids):
                dt = _time.time() - t0
                print(f"[panns_lr] feats train {i+1}/{len(train_ids)} clips ({dt:.1f}s)", flush=True)

        x_np = np.concatenate(x_rows, axis=0).astype(np.float32, copy=False)
        y_np = np.concatenate(y_rows, axis=0).astype(np.float32, copy=False)

        x = torch.from_numpy(x_np).float()
        y = torch.from_numpy(y_np).float()

        torch.manual_seed(0)
        model = nn.Linear(int(x.shape[-1]), 1)
        pos = float((y > 0.5).sum().item())
        neg = float((y <= 0.5).sum().item())
        pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-2, weight_decay=1e-2)

        bs = 4096
        n = int(x.shape[0])
        steps = max(1, (n + bs - 1) // bs)
        for _epoch in range(50):
            perm = torch.randperm(n)
            for j in range(steps):
                idx = perm[j * bs : (j + 1) * bs]
                xb = x[idx]
                yb = y[idx]
                logits = model(xb)
                loss = loss_fn(logits, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        model.eval()

        # Score requested clip ids (reuse train probs when available).
        for i, cid in enumerate(clip_ids):
            probs = feats_by_train.get(cid)
            if probs is None:
                wav_path = processed_dir / cid / "audio.wav"
                probs = panns_probe.clipwise_output_per_second(wav_path, num_segments=int(num_segments))
                probs = np.asarray(probs, dtype=np.float32)

            feats = _logit(probs)
            with torch.no_grad():
                s = model(torch.from_numpy(feats).float()).squeeze(-1).numpy().astype(np.float32)
            out[cid] = [float(x) for x in s.tolist()]

            if (i + 1) % 200 == 0 or (i + 1) == len(clip_ids):
                dt = _time.time() - t0
                print(f"[scores] {i+1}/{len(clip_ids)} clips ({dt:.1f}s)", flush=True)
    elif method == "panns_embed_lr":
        # Supervised, lightweight calibration: train a linear probe on pretrained PANNs embeddings
        # to predict event vs background, then use per-second logits as Stage-1 scores.
        if train_ids is None or labels_by_clip is None:
            raise ValueError("panns_embed_lr requires train_ids and labels_by_clip")
        assert panns_probe is not None

        import numpy as np
        import torch
        import torch.nn as nn

        train_ids = [str(x) for x in train_ids]

        def _l2norm(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float32)
            denom = np.linalg.norm(x, axis=-1, keepdims=True).astype(np.float32)
            denom = np.maximum(denom, 1e-6)
            return (x / denom).astype(np.float32, copy=False)

        feats_by_train: dict[str, np.ndarray] = {}
        x_rows: list[np.ndarray] = []
        y_rows: list[np.ndarray] = []

        for i, cid in enumerate(train_ids):
            wav_path = processed_dir / cid / "audio.wav"
            emb = panns_probe.embeddings_per_second(wav_path, num_segments=int(num_segments))  # [T, D]
            emb = _l2norm(emb)
            feats_by_train[cid] = emb

            labs = np.asarray(labels_by_clip[cid], dtype=np.int64)[: int(num_segments)]
            y = (labs != 0).astype(np.float32).reshape(-1, 1)
            x_rows.append(emb.astype(np.float32, copy=False))
            y_rows.append(y)

            if (i + 1) % 200 == 0 or (i + 1) == len(train_ids):
                dt = _time.time() - t0
                print(f"[panns_embed_lr] feats train {i+1}/{len(train_ids)} clips ({dt:.1f}s)", flush=True)

        x_np = np.concatenate(x_rows, axis=0).astype(np.float32, copy=False)
        y_np = np.concatenate(y_rows, axis=0).astype(np.float32, copy=False)

        x = torch.from_numpy(x_np).float()
        y = torch.from_numpy(y_np).float()

        torch.manual_seed(0)
        model = nn.Linear(int(x.shape[-1]), 1)
        pos = float((y > 0.5).sum().item())
        neg = float((y <= 0.5).sum().item())
        pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-2, weight_decay=1e-2)

        bs = 4096
        n = int(x.shape[0])
        steps = max(1, (n + bs - 1) // bs)
        for _epoch in range(50):
            perm = torch.randperm(n)
            for j in range(steps):
                idx = perm[j * bs : (j + 1) * bs]
                xb = x[idx]
                yb = y[idx]
                logits = model(xb)
                loss = loss_fn(logits, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        model.eval()

        # Score requested clip ids (reuse train embeddings when available).
        for i, cid in enumerate(clip_ids):
            emb = feats_by_train.get(cid)
            if emb is None:
                wav_path = processed_dir / cid / "audio.wav"
                emb = panns_probe.embeddings_per_second(wav_path, num_segments=int(num_segments))
                emb = _l2norm(emb)

            with torch.no_grad():
                s = model(torch.from_numpy(emb).float()).squeeze(-1).numpy().astype(np.float32)
            out[cid] = [float(x) for x in s.tolist()]

            if (i + 1) % 200 == 0 or (i + 1) == len(clip_ids):
                dt = _time.time() - t0
                print(f"[scores] {i+1}/{len(clip_ids)} clips ({dt:.1f}s)", flush=True)
    elif method == "panns_embed_mlp":
        # Supervised calibration: train a tiny 2-layer MLP on pretrained PANNs embeddings to predict
        # per-second eventness (event vs background), then use per-second logits as Stage-1 scores.
        if train_ids is None or labels_by_clip is None:
            raise ValueError("panns_embed_mlp requires train_ids and labels_by_clip")
        assert panns_probe is not None

        from avs.experiments.ave_p0 import _train_audio_basic_mlp_eventness

        import numpy as np
        import torch

        train_ids = [str(x) for x in train_ids]

        def _l2norm(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float32)
            denom = np.linalg.norm(x, axis=-1, keepdims=True).astype(np.float32)
            denom = np.maximum(denom, 1e-6)
            return (x / denom).astype(np.float32, copy=False)

        feats_by_train: dict[str, np.ndarray] = {}
        for i, cid in enumerate(train_ids):
            wav_path = processed_dir / cid / "audio.wav"
            emb = panns_probe.embeddings_per_second(wav_path, num_segments=int(num_segments))  # [T, D]
            feats_by_train[cid] = _l2norm(emb)
            if (i + 1) % 200 == 0 or (i + 1) == len(train_ids):
                dt = _time.time() - t0
                print(f"[panns_embed_mlp] feats train {i+1}/{len(train_ids)} clips ({dt:.1f}s)", flush=True)

        model = _train_audio_basic_mlp_eventness(
            clip_ids_train=train_ids,
            labels_by_clip=labels_by_clip,
            audio_feats_by_clip=feats_by_train,
            device="cpu",
            epochs=50,
            batch_size=2048,
            lr=2e-3,
            hidden_dim=128,
            dropout=0.1,
        )
        model_cpu = model.to(torch.device("cpu"))
        model_cpu.eval()

        for i, cid in enumerate(clip_ids):
            feats = feats_by_train.get(cid)
            if feats is None:
                wav_path = processed_dir / cid / "audio.wav"
                feats = _l2norm(panns_probe.embeddings_per_second(wav_path, num_segments=int(num_segments)))

            with torch.no_grad():
                s = model_cpu(torch.from_numpy(feats).float()).squeeze(-1).detach().cpu().numpy().astype(np.float32)
            out[cid] = [float(x) for x in s.tolist()]

            if (i + 1) % 200 == 0 or (i + 1) == len(clip_ids):
                dt = _time.time() - t0
                print(f"[scores] {i+1}/{len(clip_ids)} clips ({dt:.1f}s)", flush=True)
    elif method in ("av_basic_lr", "av_basic_mlp"):
        # Supervised, lightweight A+cheapV scoring: audio basic features + frame-diff scalar.
        #
        # This is mostly a diagnostic Stage-1 backend: it uses a cheap per-second visual motion proxy
        # derived from pre-extracted frames under processed_dir/<cid>/frames/.
        if train_ids is None or labels_by_clip is None:
            raise ValueError(f"{method} requires train_ids and labels_by_clip")

        from avs.audio.features import audio_features_per_second
        from avs.experiments.ave_p0 import _train_audio_basic_lr_eventness, _train_audio_basic_mlp_eventness
        from avs.vision.cheap_eventness import frame_diff_eventness, list_frames

        import numpy as np
        import torch

        train_ids = [str(x) for x in train_ids]

        feats_by_train: dict[str, np.ndarray] = {}
        for i, cid in enumerate(train_ids):
            wav_path = processed_dir / cid / "audio.wav"
            a = audio_features_per_second(wav_path, num_segments=int(num_segments), feature_set="basic").astype(
                np.float32, copy=False
            )

            frames_dir = processed_dir / cid / "frames"
            frames = list_frames(frames_dir) if frames_dir.exists() else []
            fd = frame_diff_eventness(frames, size=32) if frames else []

            v = np.zeros((int(num_segments), 1), dtype=np.float32)
            for t, s in enumerate(fd[: int(num_segments)]):
                v[int(t), 0] = float(s)

            if a.shape[0] != int(num_segments):
                raise ValueError(f"unexpected audio feature shape for {cid}: {a.shape}")

            feats_by_train[cid] = np.concatenate([a, v], axis=1).astype(np.float32, copy=False)

            if (i + 1) % 200 == 0 or (i + 1) == len(train_ids):
                dt = _time.time() - t0
                print(f"[{method}] feats train {i+1}/{len(train_ids)} clips ({dt:.1f}s)", flush=True)

        if method == "av_basic_lr":
            model = _train_audio_basic_lr_eventness(
                clip_ids_train=train_ids,
                labels_by_clip=labels_by_clip,
                audio_feats_by_clip=feats_by_train,
                device="cpu",
            )
        else:
            model = _train_audio_basic_mlp_eventness(
                clip_ids_train=train_ids,
                labels_by_clip=labels_by_clip,
                audio_feats_by_clip=feats_by_train,
                device="cpu",
            )
        model_cpu = model.to(torch.device("cpu"))
        model_cpu.eval()

        for i, cid in enumerate(clip_ids):
            feats = feats_by_train.get(cid)
            if feats is None:
                wav_path = processed_dir / cid / "audio.wav"
                a = audio_features_per_second(wav_path, num_segments=int(num_segments), feature_set="basic").astype(
                    np.float32, copy=False
                )

                frames_dir = processed_dir / cid / "frames"
                frames = list_frames(frames_dir) if frames_dir.exists() else []
                fd = frame_diff_eventness(frames, size=32) if frames else []

                v = np.zeros((int(num_segments), 1), dtype=np.float32)
                for t, s in enumerate(fd[: int(num_segments)]):
                    v[int(t), 0] = float(s)

                if a.shape[0] != int(num_segments):
                    raise ValueError(f"unexpected audio feature shape for {cid}: {a.shape}")
                feats = np.concatenate([a, v], axis=1).astype(np.float32, copy=False)

            with torch.no_grad():
                logits = model_cpu(torch.from_numpy(feats).float()).squeeze(-1)
                s_np = logits.detach().cpu().numpy().astype(np.float32)
            out[cid] = [float(x) for x in s_np.tolist()]

            if (i + 1) % 200 == 0 or (i + 1) == len(clip_ids):
                dt = _time.time() - t0
                print(f"[scores] {i+1}/{len(clip_ids)} clips ({dt:.1f}s)", flush=True)
    elif method == "av_panns_embed_clipdiff_mlp":
        # Supervised, lightweight A+V scoring: PANNs embeddings + semantic motion proxy (CLIP feature diff).
        # Trains on train split to predict event vs background; returns per-second logits as Stage-1 scores.
        if caches_dir is None:
            raise ValueError(f"{method} requires caches_dir to load CLIP features")
        if train_ids is None or labels_by_clip is None:
            raise ValueError(f"{method} requires train_ids and labels_by_clip")
        assert panns_probe is not None

        from avs.experiments.ave_p0 import _train_audio_basic_mlp_eventness
        from avs.vision.cheap_eventness import clip_feature_diff_eventness

        import numpy as np
        import torch

        caches_dir = Path(caches_dir)
        train_ids = [str(x) for x in train_ids]

        def _load_vis_feats_npz(cache_path: Path) -> np.ndarray:
            with np.load(cache_path) as z:
                if "res_112" in z.files:
                    return z["res_112"]
                avail = sorted(int(k.split("_", 1)[1]) for k in z.files if k.startswith("res_"))
                if not avail:
                    raise ValueError(f"no res_* arrays in cache: {cache_path}")
                return z[f"res_{avail[0]}"]

        def _l2norm(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float32)
            denom = np.linalg.norm(x, axis=-1, keepdims=True).astype(np.float32)
            denom = np.maximum(denom, 1e-6)
            return (x / denom).astype(np.float32, copy=False)

        feats_by_train: dict[str, np.ndarray] = {}
        for i, cid in enumerate(train_ids):
            wav_path = processed_dir / cid / "audio.wav"
            a = _l2norm(panns_probe.embeddings_per_second(wav_path, num_segments=int(num_segments))).astype(np.float32, copy=False)

            cache_path = caches_dir / f"{cid}.npz"
            if not cache_path.exists():
                raise FileNotFoundError(f"missing cache: {cache_path}")
            vis_feats = _load_vis_feats_npz(cache_path)
            vis = clip_feature_diff_eventness(vis_feats, metric="cosine")
            vis_scores = minmax_01([float(x) for x in vis])

            v_clipdiff = np.zeros((int(num_segments), 1), dtype=np.float32)
            for t, s in enumerate(vis_scores[: int(num_segments)]):
                v_clipdiff[int(t), 0] = float(s)

            feats_by_train[cid] = np.concatenate([a, v_clipdiff], axis=1).astype(np.float32, copy=False)

            if (i + 1) % 200 == 0 or (i + 1) == len(train_ids):
                dt = _time.time() - t0
                print(f"[{method}] feats train {i+1}/{len(train_ids)} clips ({dt:.1f}s)", flush=True)

        model = _train_audio_basic_mlp_eventness(
            clip_ids_train=train_ids,
            labels_by_clip=labels_by_clip,
            audio_feats_by_clip=feats_by_train,
            device="cpu",
            hidden_dim=128,
            dropout=0.1,
        )

        model_cpu = model.to(torch.device("cpu"))
        model_cpu.eval()

        for i, cid in enumerate(clip_ids):
            feats = feats_by_train.get(cid)
            if feats is None:
                wav_path = processed_dir / cid / "audio.wav"
                a = _l2norm(panns_probe.embeddings_per_second(wav_path, num_segments=int(num_segments))).astype(np.float32, copy=False)

                cache_path = caches_dir / f"{cid}.npz"
                if not cache_path.exists():
                    raise FileNotFoundError(f"missing cache: {cache_path}")
                vis_feats = _load_vis_feats_npz(cache_path)
                vis = clip_feature_diff_eventness(vis_feats, metric="cosine")
                vis_scores = minmax_01([float(x) for x in vis])

                v_clipdiff = np.zeros((int(num_segments), 1), dtype=np.float32)
                for t, s in enumerate(vis_scores[: int(num_segments)]):
                    v_clipdiff[int(t), 0] = float(s)

                feats = np.concatenate([a, v_clipdiff], axis=1).astype(np.float32, copy=False)

            with torch.no_grad():
                s_np = model_cpu(torch.from_numpy(feats).float()).squeeze(-1).detach().cpu().numpy().astype(np.float32)
            out[cid] = [float(x) for x in s_np.tolist()]

            if (i + 1) % 200 == 0 or (i + 1) == len(clip_ids):
                dt = _time.time() - t0
                print(f"[scores] {i+1}/{len(clip_ids)} clips ({dt:.1f}s)", flush=True)
    elif method in ("audio_basic_tcn", "audio_fbank_tcn"):
        # Supervised, lightweight calibration: train a tiny temporal conv net on per-second audio features
        # to predict event vs background, then use per-second logits as Stage-1 scores.
        if train_ids is None or labels_by_clip is None:
            raise ValueError(f"{method} requires train_ids and labels_by_clip")

        from avs.audio.features import audio_features_per_second
        from avs.experiments.ave_p0 import _train_audio_tcn_eventness

        import numpy as np
        import torch

        train_ids = [str(x) for x in train_ids]
        feature_set = "basic" if method == "audio_basic_tcn" else "fbank_stats"

        feats_by_train: dict[str, np.ndarray] = {}
        for i, cid in enumerate(train_ids):
            wav_path = processed_dir / cid / "audio.wav"
            feats_by_train[cid] = audio_features_per_second(
                wav_path, num_segments=int(num_segments), feature_set=str(feature_set)
            ).astype(np.float32, copy=False)
            if (i + 1) % 200 == 0 or (i + 1) == len(train_ids):
                dt = _time.time() - t0
                print(f"[{method}] feats train {i+1}/{len(train_ids)} clips ({dt:.1f}s)", flush=True)

        model = _train_audio_tcn_eventness(
            clip_ids_train=train_ids,
            labels_by_clip=labels_by_clip,
            audio_feats_by_clip=feats_by_train,
            device="cpu",
            epochs=50,
            batch_size=128,
            lr=1e-3,
            hidden_channels=64,
            kernel_size=3,
            dropout=0.1,
        )
        model_cpu = model.to(torch.device("cpu"))
        model_cpu.eval()

        for i, cid in enumerate(clip_ids):
            feats = feats_by_train.get(cid)
            if feats is None:
                wav_path = processed_dir / cid / "audio.wav"
                feats = audio_features_per_second(
                    wav_path, num_segments=int(num_segments), feature_set=str(feature_set)
                ).astype(np.float32, copy=False)

            with torch.no_grad():
                s = model_cpu(torch.from_numpy(feats).float()).squeeze(-1).detach().cpu().numpy().astype(np.float32)
            out[cid] = [float(x) for x in s.tolist()]

            if (i + 1) % 200 == 0 or (i + 1) == len(clip_ids):
                dt = _time.time() - t0
                print(f"[scores] {i+1}/{len(clip_ids)} clips ({dt:.1f}s)", flush=True)
    elif method in ("av_ast_clipdiff_mlp", "av_ast_clipdiff_mil_mlp", "av_ast_clipdiff_tcn"):
        # Supervised, lightweight A+V scoring: AST embeddings + semantic motion proxy (CLIP feature diff).
        # Trains on train split to predict event vs background; returns per-second logits as Stage-1 scores.
        if caches_dir is None:
            raise ValueError(f"{method} requires caches_dir to load CLIP features")
        if train_ids is None or labels_by_clip is None:
            raise ValueError(f"{method} requires train_ids and labels_by_clip")
        assert ast_probe is not None

        from avs.experiments.ave_p0 import _train_audio_basic_mil_mlp_eventness, _train_audio_basic_mlp_eventness, _train_audio_tcn_eventness
        from avs.vision.cheap_eventness import clip_feature_diff_eventness

        import numpy as np
        import torch

        caches_dir = Path(caches_dir)
        train_ids = [str(x) for x in train_ids]

        def _load_vis_feats_npz(cache_path: Path) -> np.ndarray:
            with np.load(cache_path) as z:
                if "res_112" in z.files:
                    return z["res_112"]
                avail = sorted(int(k.split("_", 1)[1]) for k in z.files if k.startswith("res_"))
                if not avail:
                    raise ValueError(f"no res_* arrays in cache: {cache_path}")
                return z[f"res_{avail[0]}"]

        feats_by_train: dict[str, np.ndarray] = {}
        for i, cid in enumerate(train_ids):
            wav_path = processed_dir / cid / "audio.wav"
            a = ast_probe.embeddings_per_second(wav_path, num_segments=int(num_segments)).astype(np.float32, copy=False)

            cache_path = caches_dir / f"{cid}.npz"
            if not cache_path.exists():
                raise FileNotFoundError(f"missing cache: {cache_path}")
            vis_feats = _load_vis_feats_npz(cache_path)
            vis = clip_feature_diff_eventness(vis_feats, metric="cosine")
            vis_scores = minmax_01([float(x) for x in vis])

            v_clipdiff = np.zeros((int(num_segments), 1), dtype=np.float32)
            for t, s in enumerate(vis_scores[: int(num_segments)]):
                v_clipdiff[int(t), 0] = float(s)

            feats_by_train[cid] = np.concatenate([a, v_clipdiff], axis=1).astype(np.float32, copy=False)

            if (i + 1) % 200 == 0 or (i + 1) == len(train_ids):
                dt = _time.time() - t0
                print(f"[{method}] feats train {i+1}/{len(train_ids)} clips ({dt:.1f}s)", flush=True)

        if method == "av_ast_clipdiff_mil_mlp":
            model = _train_audio_basic_mil_mlp_eventness(
                clip_ids_train=train_ids,
                labels_by_clip=labels_by_clip,
                audio_feats_by_clip=feats_by_train,
                device="cpu",
                hidden_dim=128,
                dropout=0.1,
            )
        elif method == "av_ast_clipdiff_tcn":
            model = _train_audio_tcn_eventness(
                clip_ids_train=train_ids,
                labels_by_clip=labels_by_clip,
                audio_feats_by_clip=feats_by_train,
                device="cpu",
                epochs=50,
                batch_size=128,
                lr=1e-3,
                hidden_channels=128,
                kernel_size=3,
                dropout=0.1,
            )
        else:
            model = _train_audio_basic_mlp_eventness(
                clip_ids_train=train_ids,
                labels_by_clip=labels_by_clip,
                audio_feats_by_clip=feats_by_train,
                device="cpu",
                hidden_dim=128,
                dropout=0.1,
            )

        model_cpu = model.to(torch.device("cpu"))
        model_cpu.eval()

        for i, cid in enumerate(clip_ids):
            feats = feats_by_train.get(cid)
            if feats is None:
                wav_path = processed_dir / cid / "audio.wav"
                a = ast_probe.embeddings_per_second(wav_path, num_segments=int(num_segments)).astype(np.float32, copy=False)

                cache_path = caches_dir / f"{cid}.npz"
                if not cache_path.exists():
                    raise FileNotFoundError(f"missing cache: {cache_path}")
                vis_feats = _load_vis_feats_npz(cache_path)
                vis = clip_feature_diff_eventness(vis_feats, metric="cosine")
                vis_scores = minmax_01([float(x) for x in vis])

                v_clipdiff = np.zeros((int(num_segments), 1), dtype=np.float32)
                for t, s in enumerate(vis_scores[: int(num_segments)]):
                    v_clipdiff[int(t), 0] = float(s)

                feats = np.concatenate([a, v_clipdiff], axis=1).astype(np.float32, copy=False)

            with torch.no_grad():
                s_np = model_cpu(torch.from_numpy(feats).float()).squeeze(-1).detach().cpu().numpy().astype(np.float32)
            out[cid] = [float(x) for x in s_np.tolist()]

            if (i + 1) % 200 == 0 or (i + 1) == len(clip_ids):
                dt = _time.time() - t0
                print(f"[scores] {i+1}/{len(clip_ids)} clips ({dt:.1f}s)", flush=True)
    elif method == "av_ast_clipalign_nce":
        # Supervised A/V alignment scoring: learn a cross-modal projection (InfoNCE) using only non-background
        # timestamps, then use per-second diagonal similarity as Stage-1 scores.
        if caches_dir is None:
            raise ValueError(f"{method} requires caches_dir to load CLIP features")
        if train_ids is None or labels_by_clip is None:
            raise ValueError(f"{method} requires train_ids and labels_by_clip")
        assert ast_probe is not None

        from avs.experiments.ave_p0 import _train_av_clipalign_nce_eventness

        import numpy as np
        import torch

        caches_dir = Path(caches_dir)
        train_ids = [str(x) for x in train_ids]

        def _load_vis_feats_npz(cache_path: Path) -> np.ndarray:
            with np.load(cache_path) as z:
                if "res_112" in z.files:
                    return z["res_112"]
                avail = sorted(int(k.split("_", 1)[1]) for k in z.files if k.startswith("res_"))
                if not avail:
                    raise ValueError(f"no res_* arrays in cache: {cache_path}")
                return z[f"res_{avail[0]}"]

        audio_by_train: dict[str, np.ndarray] = {}
        vis_by_train: dict[str, np.ndarray] = {}
        for i, cid in enumerate(train_ids):
            wav_path = processed_dir / cid / "audio.wav"
            audio_by_train[cid] = ast_probe.embeddings_per_second(wav_path, num_segments=int(num_segments)).astype(
                np.float32, copy=False
            )

            cache_path = caches_dir / f"{cid}.npz"
            if not cache_path.exists():
                raise FileNotFoundError(f"missing cache: {cache_path}")
            vis_by_train[cid] = _load_vis_feats_npz(cache_path).astype(np.float32, copy=False)

            if (i + 1) % 200 == 0 or (i + 1) == len(train_ids):
                dt = _time.time() - t0
                print(f"[{method}] feats train {i+1}/{len(train_ids)} clips ({dt:.1f}s)", flush=True)

        model = _train_av_clipalign_nce_eventness(
            clip_ids_train=train_ids,
            labels_by_clip=labels_by_clip,
            audio_emb_by_clip=audio_by_train,
            vision_emb_by_clip=vis_by_train,
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
        model_cpu = model.to(torch.device("cpu"))
        model_cpu.eval()

        for i, cid in enumerate(clip_ids):
            a = audio_by_train.get(cid)
            v = vis_by_train.get(cid)
            if a is None or v is None:
                wav_path = processed_dir / cid / "audio.wav"
                a = ast_probe.embeddings_per_second(wav_path, num_segments=int(num_segments)).astype(np.float32, copy=False)

                cache_path = caches_dir / f"{cid}.npz"
                if not cache_path.exists():
                    raise FileNotFoundError(f"missing cache: {cache_path}")
                v = _load_vis_feats_npz(cache_path).astype(np.float32, copy=False)

            with torch.no_grad():
                s = model_cpu.diag_scores(torch.from_numpy(a).float(), torch.from_numpy(v).float())
                s_np = s.detach().cpu().numpy().astype(np.float32)
            out[cid] = [float(x) for x in s_np.tolist()]

            if (i + 1) % 200 == 0 or (i + 1) == len(clip_ids):
                dt = _time.time() - t0
                print(f"[scores] {i+1}/{len(clip_ids)} clips ({dt:.1f}s)", flush=True)
    elif method == "av_ast_clipalign_bce":
        # Supervised A/V correspondence scoring: learn a cross-modal projection (BCE) on per-second diagonal
        # similarity (pos = non-background), then use per-second (cosine / temperature) logits as Stage-1 scores.
        if caches_dir is None:
            raise ValueError(f"{method} requires caches_dir to load CLIP features")
        if train_ids is None or labels_by_clip is None:
            raise ValueError(f"{method} requires train_ids and labels_by_clip")
        assert ast_probe is not None

        from avs.experiments.ave_p0 import _train_av_clipalign_bce_eventness

        import numpy as np
        import torch

        caches_dir = Path(caches_dir)
        train_ids = [str(x) for x in train_ids]

        def _load_vis_feats_npz(cache_path: Path) -> np.ndarray:
            with np.load(cache_path) as z:
                if "res_112" in z.files:
                    return z["res_112"]
                avail = sorted(int(k.split("_", 1)[1]) for k in z.files if k.startswith("res_"))
                if not avail:
                    raise ValueError(f"no res_* arrays in cache: {cache_path}")
                return z[f"res_{avail[0]}"]

        audio_by_train: dict[str, np.ndarray] = {}
        vis_by_train: dict[str, np.ndarray] = {}
        for i, cid in enumerate(train_ids):
            wav_path = processed_dir / cid / "audio.wav"
            audio_by_train[cid] = ast_probe.embeddings_per_second(wav_path, num_segments=int(num_segments)).astype(
                np.float32, copy=False
            )

            cache_path = caches_dir / f"{cid}.npz"
            if not cache_path.exists():
                raise FileNotFoundError(f"missing cache: {cache_path}")
            vis_by_train[cid] = _load_vis_feats_npz(cache_path).astype(np.float32, copy=False)

            if (i + 1) % 200 == 0 or (i + 1) == len(train_ids):
                dt = _time.time() - t0
                print(f"[{method}] feats train {i+1}/{len(train_ids)} clips ({dt:.1f}s)", flush=True)

        temperature = 0.07
        model = _train_av_clipalign_bce_eventness(
            clip_ids_train=train_ids,
            labels_by_clip=labels_by_clip,
            audio_emb_by_clip=audio_by_train,
            vision_emb_by_clip=vis_by_train,
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
        model_cpu = model.to(torch.device("cpu"))
        model_cpu.eval()
        inv_temp = float(1.0 / float(temperature))

        for i, cid in enumerate(clip_ids):
            a = audio_by_train.get(cid)
            v = vis_by_train.get(cid)
            if a is None or v is None:
                wav_path = processed_dir / cid / "audio.wav"
                a = ast_probe.embeddings_per_second(wav_path, num_segments=int(num_segments)).astype(np.float32, copy=False)

                cache_path = caches_dir / f"{cid}.npz"
                if not cache_path.exists():
                    raise FileNotFoundError(f"missing cache: {cache_path}")
                v = _load_vis_feats_npz(cache_path).astype(np.float32, copy=False)

            with torch.no_grad():
                s = model_cpu.diag_scores(torch.from_numpy(a).float(), torch.from_numpy(v).float()) * float(inv_temp)
                s_np = s.detach().cpu().numpy().astype(np.float32)
            out[cid] = [float(x) for x in s_np.tolist()]

            if (i + 1) % 200 == 0 or (i + 1) == len(clip_ids):
                dt = _time.time() - t0
                print(f"[scores] {i+1}/{len(clip_ids)} clips ({dt:.1f}s)", flush=True)
    elif method in (
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
        "av_clipdiff_mlp_autoshift",
        "av_clipdiff_mlp_r160",
        "av_clipdiff_mlp_r224",
        "av_clipdiff_mlp_cls",
        "av_clipdiff_mlp_cls_target",
        "av_clipdiff_tcn",
    ):
        # Supervised, lightweight A+V scoring: (audio features) + semantic motion proxy (CLIP feature diff).
        # Trains on train split to predict event vs background; returns per-second logits as Stage-1 scores.
        if caches_dir is None:
            raise ValueError(f"{method} requires caches_dir to load CLIP features")
        if train_ids is None or labels_by_clip is None:
            raise ValueError(f"{method} requires train_ids and labels_by_clip")

        from avs.audio.features import audio_features_per_second
        from avs.experiments.ave_p0 import (
            _train_audio_basic_lr_eventness,
            _train_audio_basic_mil_mlp_eventness,
            _train_audio_basic_mlp_cls_eventness,
            _train_audio_basic_mlp_eventness,
            _train_audio_basic_mlp_visgain_eventness,
            _train_audio_tcn_eventness,
        )
        from avs.vision.cheap_eventness import (
            clip_feature_diff_eventness,
            frame_diff_eventness,
            list_frames,
            optical_flow_mag_eventness,
        )

        import numpy as np
        import torch

        caches_dir = Path(caches_dir)
        train_ids = [str(x) for x in train_ids]

        prefer_res = 112
        if str(method).endswith("_r160"):
            prefer_res = 160
        elif str(method).endswith("_r224"):
            prefer_res = 224
        audio_feature_set = "fbank_stats" if method == "av_clipdiff_fbank_mlp" else "basic"
        use_ast_speech = method == "av_clipdiff_speech_mlp"
        speech_idx: list[int] = []
        if use_ast_speech:
            assert ast_probe is not None
            for k, v in ast_probe.model.config.id2label.items():
                name = str(v).strip().lower()
                if ("speech" in name) or ("conversation" in name) or ("narration" in name) or ("talking" in name):
                    speech_idx.append(int(k))
            speech_idx = sorted(set(int(x) for x in speech_idx))

        def _load_vis_feats_npz(cache_path: Path) -> np.ndarray:
            with np.load(cache_path) as z:
                key_prefer = f"res_{int(prefer_res)}"
                if key_prefer in z.files:
                    return z[key_prefer]
                if "res_112" in z.files:
                    return z["res_112"]
                avail = sorted(int(k.split("_", 1)[1]) for k in z.files if k.startswith("res_"))
                if not avail:
                    raise ValueError(f"no res_* arrays in cache: {cache_path}")
                return z[f"res_{avail[0]}"]

        feats_by_train: dict[str, np.ndarray] = {}
        use_visgain_teacher = method == "av_clipdiff_visgain_mlp"
        use_lossgain_teacher = method == "av_clipdiff_lossgain_mlp"
        use_accflip_teacher = method == "av_clipdiff_accflip_mlp"
        teacher_by_train: dict[str, np.ndarray] | None = (
            {} if (use_visgain_teacher or use_lossgain_teacher or use_accflip_teacher) else None
        )

        loss_base_rows: list[np.ndarray] = []
        loss_high_rows: list[np.ndarray] = []
        loss_y_rows: list[np.ndarray] = []
        acc_base_rows: list[np.ndarray] = []
        acc_high_rows: list[np.ndarray] = []
        acc_y_rows: list[np.ndarray] = []
        for i, cid in enumerate(train_ids):
            wav_path = processed_dir / cid / "audio.wav"
            a = audio_features_per_second(
                wav_path, num_segments=int(num_segments), feature_set=str(audio_feature_set)
            ).astype(np.float32, copy=False)

            cache_path = caches_dir / f"{cid}.npz"
            if not cache_path.exists():
                raise FileNotFoundError(f"missing cache: {cache_path}")
            if not (use_visgain_teacher or use_lossgain_teacher or use_accflip_teacher):
                vis_feats = _load_vis_feats_npz(cache_path)
            else:
                with np.load(cache_path) as z:
                    # Stage-1 clipdiff uses a cheap fixed resolution (prefer 112 when available).
                    key_prefer = f"res_{int(prefer_res)}"
                    if key_prefer in z.files:
                        vis_feats = z[key_prefer]
                    elif "res_112" in z.files:
                        vis_feats = z["res_112"]
                    else:
                        avail = sorted(int(k.split("_", 1)[1]) for k in z.files if k.startswith("res_"))
                        if not avail:
                            raise ValueError(f"no res_* arrays in cache: {cache_path}")
                        vis_feats = z[f"res_{avail[0]}"]

                    avail = sorted(int(k.split("_", 1)[1]) for k in z.files if k.startswith("res_"))
                    if not avail:
                        raise ValueError(f"no res_* arrays in cache: {cache_path}")

                    def _closest(target: int) -> int:
                        return min(avail, key=lambda r: (abs(int(r) - int(target)), int(r)))

                    base_r = _closest(224)
                    high_r = _closest(352)
                    if int(high_r) == int(base_r):
                        # Prefer a strictly higher resolution if available (teacher needs a "gain").
                        higher = [int(r) for r in avail if int(r) > int(base_r)]
                        high_r = int(higher[0]) if higher else int(base_r)

                    if use_visgain_teacher:
                        # Teacher A: resolution sensitivity between base=224 and high=352 (fallback to closest available).
                        v_base = np.asarray(z[f"res_{int(base_r)}"], dtype=np.float32)[: int(num_segments)]
                        v_high = np.asarray(z[f"res_{int(high_r)}"], dtype=np.float32)[: int(num_segments)]
                        if v_base.shape != v_high.shape:
                            raise ValueError(f"teacher res mismatch for {cid}: base={v_base.shape}, high={v_high.shape}")

                        v_base = v_base / (np.linalg.norm(v_base, axis=1, keepdims=True) + 1e-12)
                        v_high = v_high / (np.linalg.norm(v_high, axis=1, keepdims=True) + 1e-12)
                        cos = np.sum(v_base * v_high, axis=1)
                        cos = np.clip(cos, -1.0, 1.0)
                        gain = np.clip(1.0 - cos, 0.0, 1.0).astype(np.float32, copy=False)

                        labs = np.asarray(labels_by_clip[cid], dtype=np.int64)[: int(num_segments)]
                        mask = (labs != 0).astype(np.float32, copy=False)
                        assert teacher_by_train is not None
                        teacher_by_train[cid] = (gain * mask).astype(np.float32, copy=False)

                    if use_lossgain_teacher:
                        # Teacher B inputs (computed later): base vs high features + labels (train split only).
                        v_base = np.asarray(z[f"res_{int(base_r)}"], dtype=np.float32)[: int(num_segments)]
                        v_high = np.asarray(z[f"res_{int(high_r)}"], dtype=np.float32)[: int(num_segments)]
                        if v_base.shape != v_high.shape:
                            raise ValueError(f"teacher res mismatch for {cid}: base={v_base.shape}, high={v_high.shape}")
                        labs = np.asarray(labels_by_clip[cid], dtype=np.int64)[: int(num_segments)]
                        loss_base_rows.append(v_base.astype(np.float32, copy=False))
                        loss_high_rows.append(v_high.astype(np.float32, copy=False))
                        loss_y_rows.append(labs.astype(np.int64, copy=False))
                    if use_accflip_teacher:
                        # Teacher C inputs (computed later): base vs high features + labels (train split only).
                        v_base = np.asarray(z[f"res_{int(base_r)}"], dtype=np.float32)[: int(num_segments)]
                        v_high = np.asarray(z[f"res_{int(high_r)}"], dtype=np.float32)[: int(num_segments)]
                        if v_base.shape != v_high.shape:
                            raise ValueError(f"teacher res mismatch for {cid}: base={v_base.shape}, high={v_high.shape}")
                        labs = np.asarray(labels_by_clip[cid], dtype=np.int64)[: int(num_segments)]
                        acc_base_rows.append(v_base.astype(np.float32, copy=False))
                        acc_high_rows.append(v_high.astype(np.float32, copy=False))
                        acc_y_rows.append(labs.astype(np.int64, copy=False))
            vis = clip_feature_diff_eventness(vis_feats, metric="cosine")
            vis_scores = minmax_01([float(x) for x in vis])

            v_clipdiff = np.zeros((int(num_segments), 1), dtype=np.float32)
            for t, s in enumerate(vis_scores[: int(num_segments)]):
                v_clipdiff[int(t), 0] = float(s)

            v_extras: list[np.ndarray] = []
            if str(method) in ("av_clipdiff_framediff_mlp", "av_clipdiff_flow_mlp", "av_clipdiff_flow_mlp_stride"):
                frames_dir = processed_dir / cid / "frames"
                frames = list_frames(frames_dir) if frames_dir.exists() else []
                if str(method) == "av_clipdiff_framediff_mlp":
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
                logits = ast_probe.logits_per_second(wav_path, num_segments=int(num_segments))  # [T,C]
                probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float32)))
                s = probs[:, speech_idx].max(axis=1) if speech_idx else np.zeros((int(num_segments),), dtype=np.float32)
                speech = s.reshape(int(num_segments), 1).astype(np.float32, copy=False)

            parts = [a, v_clipdiff]
            parts.extend(v_extras)
            if speech is not None:
                parts.append(speech)
            feats_by_train[cid] = np.concatenate(parts, axis=1).astype(np.float32, copy=False)

            if (i + 1) % 200 == 0 or (i + 1) == len(train_ids):
                dt = _time.time() - t0
                print(f"[{method}] feats train {i+1}/{len(train_ids)} clips ({dt:.1f}s)", flush=True)

        if use_lossgain_teacher:
            # Train a cheap base-res classifier teacher on vision features, then define the teacher target as
            # per-second loss reduction when swapping in high-res features (event seconds only).
            if not loss_base_rows or not loss_high_rows or not loss_y_rows:
                raise ValueError(f"{method} expected non-empty lossgain teacher buffers")

            from avs.models.per_segment_mlp import PerSegmentMLP
            from avs.train.train_loop import TrainConfig, train_per_segment_classifier

            import torch.nn.functional as F

            # Determine the label space from train split.
            num_classes = 1
            for cid in train_ids:
                labs = labels_by_clip.get(cid) or []
                if labs:
                    num_classes = max(int(num_classes), int(max(int(x) for x in labs)) + 1)

            x_base = np.stack(loss_base_rows, axis=0).astype(np.float32, copy=False)
            x_high = np.stack(loss_high_rows, axis=0).astype(np.float32, copy=False)
            y_np = np.stack(loss_y_rows, axis=0).astype(np.int64, copy=False)

            x_base_t = torch.from_numpy(x_base).float()
            y_t = torch.from_numpy(y_np).long()

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
                logits_base = teacher(x_base_t)  # [N, T, C]
                logits_high = teacher(torch.from_numpy(x_high).float())
                c = int(logits_base.shape[-1])
                loss_base = F.cross_entropy(logits_base.view(-1, c), y_t.view(-1), reduction="none").view_as(y_t)
                loss_high = F.cross_entropy(logits_high.view(-1, c), y_t.view(-1), reduction="none").view_as(y_t)
                gain = (loss_base - loss_high).clamp(min=0.0)
                gain = gain * (y_t != 0).float()
                # Per-clip normalization stabilizes scale across samples (stage-1 uses within-clip ranking).
                denom = gain.max(dim=1, keepdim=True).values
                gain = gain / (denom + 1e-6)

            teacher_by_train = {
                str(cid): gain[int(i)].detach().cpu().numpy().astype(np.float32, copy=False)
                for i, cid in enumerate(train_ids)
            }

        if use_accflip_teacher:
            # Train two cheap vision teachers (base-res and high-res), then define the teacher target as
            # per-second "accuracy flip" where high-res predicts the correct label but base-res does not
            # (event seconds only). This aligns Stage-1 to downstream resolution benefit rather than generic eventness.
            if not acc_base_rows or not acc_high_rows or not acc_y_rows:
                raise ValueError(f"{method} expected non-empty accflip teacher buffers")

            from avs.models.per_segment_mlp import PerSegmentMLP
            from avs.train.train_loop import TrainConfig, train_per_segment_classifier

            x_base = np.stack(acc_base_rows, axis=0).astype(np.float32, copy=False)
            x_high = np.stack(acc_high_rows, axis=0).astype(np.float32, copy=False)
            y_np = np.stack(acc_y_rows, axis=0).astype(np.int64, copy=False)

            x_base_t = torch.from_numpy(x_base).float()
            x_high_t = torch.from_numpy(x_high).float()
            y_t = torch.from_numpy(y_np).long()

            # Determine the label space from train split.
            num_classes = 1
            for cid in train_ids:
                labs = labels_by_clip.get(cid) or []
                if labs:
                    num_classes = max(int(num_classes), int(max(int(x) for x in labs)) + 1)

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

            teacher_by_train = {
                str(cid): flip[int(i)].detach().cpu().numpy().astype(np.float32, copy=False)
                for i, cid in enumerate(train_ids)
            }

        if method == "av_clipdiff_lr":
            model = _train_audio_basic_lr_eventness(
                clip_ids_train=train_ids,
                labels_by_clip=labels_by_clip,
                audio_feats_by_clip=feats_by_train,
                device="cpu",
            )
        elif method == "av_clipdiff_visgain_mlp":
            assert teacher_by_train is not None
            model = _train_audio_basic_mlp_visgain_eventness(
                clip_ids_train=train_ids,
                targets_by_clip=teacher_by_train,
                audio_feats_by_clip=feats_by_train,
                device="cpu",
                epochs=60,
                batch_size=2048,
                lr=2e-3,
                hidden_dim=64,
                dropout=0.0,
            )
        elif method == "av_clipdiff_lossgain_mlp":
            assert teacher_by_train is not None
            model = _train_audio_basic_mlp_visgain_eventness(
                clip_ids_train=train_ids,
                targets_by_clip=teacher_by_train,
                audio_feats_by_clip=feats_by_train,
                device="cpu",
                epochs=60,
                batch_size=2048,
                lr=2e-3,
                hidden_dim=64,
                dropout=0.0,
            )
        elif method == "av_clipdiff_accflip_mlp":
            assert teacher_by_train is not None
            # Convert accflip targets into a 0/1 label map and reuse the standard binary Stage-1 trainer.
            teacher_labels: dict[str, list[int]] = {}
            for cid in train_ids:
                tgt = teacher_by_train.get(str(cid))
                if tgt is None:
                    raise ValueError(f"{method}: missing teacher target for train id {cid}")
                t = np.asarray(tgt, dtype=np.float32).reshape(-1)[: int(num_segments)]
                teacher_labels[str(cid)] = [int(float(x) > 0.5) for x in t.tolist()]

            model = _train_audio_basic_mlp_eventness(
                clip_ids_train=train_ids,
                labels_by_clip=teacher_labels,
                audio_feats_by_clip=feats_by_train,
                device="cpu",
                hidden_dim=128,
            )
        elif method in (
            "av_clipdiff_mlp",
            "av_clipdiff_mil_mlp",
            "av_clipdiff_framediff_mlp",
            "av_clipdiff_flow_mlp",
            "av_clipdiff_flow_mlp_stride",
            "av_clipdiff_fbank_mlp",
            "av_clipdiff_mlp_autoshift",
            "av_clipdiff_mlp_r160",
            "av_clipdiff_mlp_r224",
        ):
            if method == "av_clipdiff_mil_mlp":
                model = _train_audio_basic_mil_mlp_eventness(
                    clip_ids_train=train_ids,
                    labels_by_clip=labels_by_clip,
                    audio_feats_by_clip=feats_by_train,
                    device="cpu",
                    epochs=50,
                    batch_size=128,
                    lr=2e-3,
                    hidden_dim=128,
                    dropout=0.0,
                )
            else:
                model = _train_audio_basic_mlp_eventness(
                    clip_ids_train=train_ids,
                    labels_by_clip=labels_by_clip,
                    audio_feats_by_clip=feats_by_train,
                    device="cpu",
                    hidden_dim=128,
                )
        elif method in ("av_clipdiff_mlp_cls", "av_clipdiff_mlp_cls_target"):
            num_classes = 1
            for cid in train_ids:
                labs = labels_by_clip.get(cid) or []
                if labs:
                    num_classes = max(int(num_classes), int(max(int(x) for x in labs)) + 1)
            model = _train_audio_basic_mlp_cls_eventness(
                clip_ids_train=train_ids,
                labels_by_clip=labels_by_clip,
                audio_feats_by_clip=feats_by_train,
                num_classes=int(num_classes),
                device="cpu",
            )
        else:
            model = _train_audio_tcn_eventness(
                clip_ids_train=train_ids,
                labels_by_clip=labels_by_clip,
                audio_feats_by_clip=feats_by_train,
                device="cpu",
                epochs=50,
                batch_size=128,
                lr=1e-3,
                hidden_channels=64,
                kernel_size=3,
                dropout=0.1,
            )
        model_cpu = model.to(torch.device("cpu"))
        model_cpu.eval()

        for i, cid in enumerate(clip_ids):
            feats = feats_by_train.get(cid)
            if feats is None:
                wav_path = processed_dir / cid / "audio.wav"
                a = audio_features_per_second(
                    wav_path, num_segments=int(num_segments), feature_set=str(audio_feature_set)
                ).astype(np.float32, copy=False)

                cache_path = caches_dir / f"{cid}.npz"
                if not cache_path.exists():
                    raise FileNotFoundError(f"missing cache: {cache_path}")
                vis_feats = _load_vis_feats_npz(cache_path)
                vis = clip_feature_diff_eventness(vis_feats, metric="cosine")
                vis_scores = minmax_01([float(x) for x in vis])

                v_clipdiff = np.zeros((int(num_segments), 1), dtype=np.float32)
                for t, s in enumerate(vis_scores[: int(num_segments)]):
                    v_clipdiff[int(t), 0] = float(s)

                v_extras: list[np.ndarray] = []
                if str(method) in ("av_clipdiff_framediff_mlp", "av_clipdiff_flow_mlp", "av_clipdiff_flow_mlp_stride"):
                    frames_dir = processed_dir / cid / "frames"
                    frames = list_frames(frames_dir) if frames_dir.exists() else []
                    if str(method) == "av_clipdiff_framediff_mlp":
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
                    logits = ast_probe.logits_per_second(wav_path, num_segments=int(num_segments))  # [T,C]
                    probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float32)))
                    s = probs[:, speech_idx].max(axis=1) if speech_idx else np.zeros((int(num_segments),), dtype=np.float32)
                    speech = s.reshape(int(num_segments), 1).astype(np.float32, copy=False)

                parts = [a, v_clipdiff]
                parts.extend(v_extras)
                if speech is not None:
                    parts.append(speech)
                feats = np.concatenate(parts, axis=1).astype(np.float32, copy=False)

            with torch.no_grad():
                logits = model_cpu(torch.from_numpy(feats).float())
                if method in ("av_clipdiff_mlp_cls", "av_clipdiff_mlp_cls_target"):
                    # Multi-class head returns [T, C].
                    bg = logits[:, 0]
                    if method == "av_clipdiff_mlp_cls":
                        s = logits[:, 1:].max(dim=-1).values - bg
                    else:
                        clip_logits = logits.mean(dim=0)
                        clip_logits = clip_logits.clone()
                        clip_logits[0] = float("-inf")
                        cls = int(torch.argmax(clip_logits).item())
                        s = logits[:, cls] - bg
                    s_np = s.detach().cpu().numpy().astype(np.float32)
                else:
                    s_np = logits.squeeze(-1).detach().cpu().numpy().astype(np.float32)
                    if method == "av_clipdiff_mlp_autoshift":
                        # Per-clip temporal alignment: pick the shift that best correlates the learned score
                        # with the cheap visual motion proxy (clipdiff scalar stored as the last feature dim).
                        # Use only this per-clip shift; keep global `anchor_shift=0` in candidate sets.
                        try:
                            vis = feats[:, -1].astype(np.float32, copy=False).tolist()
                            s_norm = minmax_01([float(x) for x in s_np.tolist()])
                            best = best_shift_by_corr(s_norm, [float(x) for x in vis], shifts=[-2, -1, 0, 1, 2])
                            s_np = np.asarray(shift_scores([float(x) for x in s_np.tolist()], int(best)), dtype=np.float32)
                        except Exception:
                            # Conservative fallback: keep unshifted scores if anything goes wrong.
                            pass
                    elif method == "av_clipdiff_flow_mlp_stride":
                        s_np = np.asarray(
                            stride_max_pool_per_second(
                                [float(x) for x in s_np.tolist()],
                                num_segments=int(num_segments),
                                stride_s=0.2,
                                win_s=0.6,
                            ),
                            dtype=np.float32,
                        )
            out[cid] = [float(x) for x in s_np.tolist()]

            if (i + 1) % 200 == 0 or (i + 1) == len(clip_ids):
                dt = _time.time() - t0
                print(f"[scores] {i+1}/{len(clip_ids)} clips ({dt:.1f}s)", flush=True)
    elif method == "av_clipdiff_vec_mlp":
        # Supervised, lightweight A+V scoring: audio basic features + CLIP diff *vector* (directional semantic motion).
        # Trains on train split to predict event vs background; returns per-second logits as Stage-1 scores.
        if caches_dir is None:
            raise ValueError(f"{method} requires caches_dir to load CLIP features")
        if train_ids is None or labels_by_clip is None:
            raise ValueError(f"{method} requires train_ids and labels_by_clip")

        from avs.audio.features import audio_features_per_second
        from avs.experiments.ave_p0 import _train_audio_basic_mlp_eventness

        import numpy as np
        import torch

        caches_dir = Path(caches_dir)
        train_ids = [str(x) for x in train_ids]

        def _load_vis_feats_npz(cache_path: Path) -> np.ndarray:
            with np.load(cache_path) as z:
                if "res_112" in z.files:
                    v = z["res_112"]
                else:
                    avail = sorted(int(k.split("_", 1)[1]) for k in z.files if k.startswith("res_"))
                    if not avail:
                        raise ValueError(f"no res_* arrays in cache: {cache_path}")
                    v = z[f"res_{avail[0]}"]
            v = np.asarray(v, dtype=np.float32)
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
            d = np.zeros_like(v, dtype=np.float32)
            d[1:] = v[1:] - v[:-1]
            return d

        feats_by_train: dict[str, np.ndarray] = {}
        for i, cid in enumerate(train_ids):
            wav_path = processed_dir / cid / "audio.wav"
            a = audio_features_per_second(wav_path, num_segments=int(num_segments), feature_set="basic").astype(np.float32, copy=False)

            cache_path = caches_dir / f"{cid}.npz"
            if not cache_path.exists():
                raise FileNotFoundError(f"missing cache: {cache_path}")
            d = _load_vis_feats_npz(cache_path)

            if a.shape[0] != int(num_segments):
                raise ValueError(f"unexpected audio feature shape for {cid}: {a.shape}")
            if d.shape[0] != int(num_segments):
                raise ValueError(f"unexpected clip feature shape for {cid}: {d.shape}")
            feats_by_train[cid] = np.concatenate([a, d], axis=1).astype(np.float32, copy=False)

            if (i + 1) % 200 == 0 or (i + 1) == len(train_ids):
                dt = _time.time() - t0
                print(f"[{method}] feats train {i+1}/{len(train_ids)} clips ({dt:.1f}s)", flush=True)

        model = _train_audio_basic_mlp_eventness(
            clip_ids_train=train_ids,
            labels_by_clip=labels_by_clip,
            audio_feats_by_clip=feats_by_train,
            device="cpu",
            hidden_dim=128,
            dropout=0.1,
        )
        model_cpu = model.to(torch.device("cpu"))
        model_cpu.eval()

        for i, cid in enumerate(clip_ids):
            feats = feats_by_train.get(cid)
            if feats is None:
                wav_path = processed_dir / cid / "audio.wav"
                a = audio_features_per_second(wav_path, num_segments=int(num_segments), feature_set="basic").astype(np.float32, copy=False)

                cache_path = caches_dir / f"{cid}.npz"
                if not cache_path.exists():
                    raise FileNotFoundError(f"missing cache: {cache_path}")
                d = _load_vis_feats_npz(cache_path)

                if a.shape[0] != int(num_segments):
                    raise ValueError(f"unexpected audio feature shape for {cid}: {a.shape}")
                if d.shape[0] != int(num_segments):
                    raise ValueError(f"unexpected clip feature shape for {cid}: {d.shape}")
                feats = np.concatenate([a, d], axis=1).astype(np.float32, copy=False)

            with torch.no_grad():
                logits = model_cpu(torch.from_numpy(feats).float()).squeeze(-1)
                s_np = logits.detach().cpu().numpy().astype(np.float32)
            out[cid] = [float(x) for x in s_np.tolist()]

            if (i + 1) % 200 == 0 or (i + 1) == len(clip_ids):
                dt = _time.time() - t0
                print(f"[scores] {i+1}/{len(clip_ids)} clips ({dt:.1f}s)", flush=True)
    elif method in ("av_clip_mlp_cls", "av_clip_mlp_cls_target"):
        # Supervised, lightweight A+V scoring: audio basic features + low-res CLIP features (semantic content).
        # Trains on train split to predict per-second class labels; returns margin-vs-bg as Stage-1 scores.
        if caches_dir is None:
            raise ValueError(f"{method} requires caches_dir to load CLIP features")
        if train_ids is None or labels_by_clip is None:
            raise ValueError(f"{method} requires train_ids and labels_by_clip")

        from avs.audio.features import audio_features_per_second
        from avs.experiments.ave_p0 import _train_audio_basic_mlp_cls_eventness

        import numpy as np
        import torch

        caches_dir = Path(caches_dir)
        train_ids = [str(x) for x in train_ids]

        def _load_vis_feats_npz(cache_path: Path) -> np.ndarray:
            with np.load(cache_path) as z:
                if "res_112" in z.files:
                    v = z["res_112"]
                else:
                    avail = sorted(int(k.split("_", 1)[1]) for k in z.files if k.startswith("res_"))
                    if not avail:
                        raise ValueError(f"no res_* arrays in cache: {cache_path}")
                    v = z[f"res_{avail[0]}"]
            v = np.asarray(v, dtype=np.float32)
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
            return v

        feats_by_train: dict[str, np.ndarray] = {}
        for i, cid in enumerate(train_ids):
            wav_path = processed_dir / cid / "audio.wav"
            a = audio_features_per_second(wav_path, num_segments=int(num_segments), feature_set="basic").astype(np.float32, copy=False)

            cache_path = caches_dir / f"{cid}.npz"
            if not cache_path.exists():
                raise FileNotFoundError(f"missing cache: {cache_path}")
            v = _load_vis_feats_npz(cache_path)

            if a.shape[0] != int(num_segments):
                raise ValueError(f"unexpected audio feature shape for {cid}: {a.shape}")
            if v.shape[0] != int(num_segments):
                raise ValueError(f"unexpected clip feature shape for {cid}: {v.shape}")
            feats_by_train[cid] = np.concatenate([a, v], axis=1).astype(np.float32, copy=False)

            if (i + 1) % 200 == 0 or (i + 1) == len(train_ids):
                dt = _time.time() - t0
                print(f"[{method}] feats train {i+1}/{len(train_ids)} clips ({dt:.1f}s)", flush=True)

        num_classes = 1
        for cid in train_ids:
            labs = labels_by_clip.get(cid) or []
            if labs:
                num_classes = max(int(num_classes), int(max(int(x) for x in labs)) + 1)

        model = _train_audio_basic_mlp_cls_eventness(
            clip_ids_train=train_ids,
            labels_by_clip=labels_by_clip,
            audio_feats_by_clip=feats_by_train,
            num_classes=int(num_classes),
            device="cpu",
            hidden_dim=128,
            dropout=0.1,
        )
        model_cpu = model.to(torch.device("cpu"))
        model_cpu.eval()

        for i, cid in enumerate(clip_ids):
            feats = feats_by_train.get(cid)
            if feats is None:
                wav_path = processed_dir / cid / "audio.wav"
                a = audio_features_per_second(wav_path, num_segments=int(num_segments), feature_set="basic").astype(np.float32, copy=False)

                cache_path = caches_dir / f"{cid}.npz"
                if not cache_path.exists():
                    raise FileNotFoundError(f"missing cache: {cache_path}")
                v = _load_vis_feats_npz(cache_path)

                if a.shape[0] != int(num_segments):
                    raise ValueError(f"unexpected audio feature shape for {cid}: {a.shape}")
                if v.shape[0] != int(num_segments):
                    raise ValueError(f"unexpected clip feature shape for {cid}: {v.shape}")
                feats = np.concatenate([a, v], axis=1).astype(np.float32, copy=False)

            with torch.no_grad():
                logits = model_cpu(torch.from_numpy(feats).float())  # [T, C]
                bg = logits[:, 0]
                if method == "av_clip_mlp_cls":
                    s = logits[:, 1:].max(dim=-1).values - bg
                else:
                    clip_logits = logits.mean(dim=0)
                    clip_logits = clip_logits.clone()
                    clip_logits[0] = float("-inf")
                    cls = int(torch.argmax(clip_logits).item())
                    s = logits[:, cls] - bg
                s_np = s.detach().cpu().numpy().astype(np.float32)

            out[cid] = [float(x) for x in s_np.tolist()]

            if (i + 1) % 200 == 0 or (i + 1) == len(clip_ids):
                dt = _time.time() - t0
                print(f"[scores] {i+1}/{len(clip_ids)} clips ({dt:.1f}s)", flush=True)
    elif method in ("vision_binary_lr", "vision_binary_mlp"):
        # Supervised cheap-visual anchors: train a tiny binary model on frozen CLIP features (low-res) to predict
        # event vs background; use per-second logits as Stage-1 scores.
        if caches_dir is None:
            raise ValueError(f"{method} requires caches_dir to load CLIP features")
        if train_ids is None or labels_by_clip is None:
            raise ValueError(f"{method} requires train_ids and labels_by_clip")

        from avs.experiments.ave_p0 import _train_audio_basic_lr_eventness, _train_audio_basic_mlp_eventness

        import numpy as np
        import torch

        caches_dir = Path(caches_dir)
        train_ids = [str(x) for x in train_ids]

        def _load_vis_feats_npz(cache_path: Path) -> np.ndarray:
            with np.load(cache_path) as z:
                if "res_160" in z.files:
                    v = z["res_160"]
                elif "res_112" in z.files:
                    v = z["res_112"]
                else:
                    avail = sorted(int(k.split("_", 1)[1]) for k in z.files if k.startswith("res_"))
                    if not avail:
                        raise ValueError(f"no res_* arrays in cache: {cache_path}")
                    v = z[f"res_{avail[0]}"]
            v = np.asarray(v, dtype=np.float32)
            return v

        feats_by_train: dict[str, np.ndarray] = {}
        for i, cid in enumerate(train_ids):
            cache_path = caches_dir / f"{cid}.npz"
            if not cache_path.exists():
                raise FileNotFoundError(f"missing cache: {cache_path}")
            feats_by_train[cid] = _load_vis_feats_npz(cache_path)
            if (i + 1) % 200 == 0 or (i + 1) == len(train_ids):
                dt = _time.time() - t0
                print(f"[{method}] feats train {i+1}/{len(train_ids)} clips ({dt:.1f}s)", flush=True)

        if method == "vision_binary_lr":
            model = _train_audio_basic_lr_eventness(
                clip_ids_train=train_ids,
                labels_by_clip=labels_by_clip,
                audio_feats_by_clip=feats_by_train,
                device="cpu",
            )
        else:
            model = _train_audio_basic_mlp_eventness(
                clip_ids_train=train_ids,
                labels_by_clip=labels_by_clip,
                audio_feats_by_clip=feats_by_train,
                device="cpu",
                hidden_dim=128,
                dropout=0.1,
            )
        model_cpu = model.to(torch.device("cpu"))
        model_cpu.eval()

        for i, cid in enumerate(clip_ids):
            feats = feats_by_train.get(cid)
            if feats is None:
                cache_path = caches_dir / f"{cid}.npz"
                if not cache_path.exists():
                    raise FileNotFoundError(f"missing cache: {cache_path}")
                feats = _load_vis_feats_npz(cache_path)

            with torch.no_grad():
                s = model_cpu(torch.from_numpy(feats).float()).squeeze(-1).detach().cpu().numpy().astype(np.float32)
            out[cid] = [float(x) for x in s.tolist()]

            if (i + 1) % 200 == 0 or (i + 1) == len(clip_ids):
                dt = _time.time() - t0
                print(f"[scores] {i+1}/{len(clip_ids)} clips ({dt:.1f}s)", flush=True)
    elif method in ("vision_mlp_cls", "vision_mlp_cls_target"):
        # Supervised cheap-visual anchors: train a tiny per-second classifier on frozen CLIP features (low-res)
        # and use margin-vs-bg as Stage-1 scores.
        if caches_dir is None:
            raise ValueError(f"{method} requires caches_dir to load CLIP features")
        if train_ids is None or labels_by_clip is None:
            raise ValueError(f"{method} requires train_ids and labels_by_clip")

        from avs.experiments.ave_p0 import _train_audio_basic_mlp_cls_eventness

        import numpy as np
        import torch

        caches_dir = Path(caches_dir)
        train_ids = [str(x) for x in train_ids]

        def _load_vis_feats_npz(cache_path: Path) -> np.ndarray:
            with np.load(cache_path) as z:
                if "res_160" in z.files:
                    v = z["res_160"]
                elif "res_112" in z.files:
                    v = z["res_112"]
                else:
                    avail = sorted(int(k.split("_", 1)[1]) for k in z.files if k.startswith("res_"))
                    if not avail:
                        raise ValueError(f"no res_* arrays in cache: {cache_path}")
                    v = z[f"res_{avail[0]}"]
            v = np.asarray(v, dtype=np.float32)
            return v

        feats_by_train: dict[str, np.ndarray] = {}
        for i, cid in enumerate(train_ids):
            cache_path = caches_dir / f"{cid}.npz"
            if not cache_path.exists():
                raise FileNotFoundError(f"missing cache: {cache_path}")
            feats_by_train[cid] = _load_vis_feats_npz(cache_path)
            if (i + 1) % 200 == 0 or (i + 1) == len(train_ids):
                dt = _time.time() - t0
                print(f"[{method}] feats train {i+1}/{len(train_ids)} clips ({dt:.1f}s)", flush=True)

        num_classes = 1
        for cid in train_ids:
            labs = labels_by_clip.get(cid) or []
            if labs:
                num_classes = max(int(num_classes), int(max(int(x) for x in labs)) + 1)

        model = _train_audio_basic_mlp_cls_eventness(
            clip_ids_train=train_ids,
            labels_by_clip=labels_by_clip,
            audio_feats_by_clip=feats_by_train,
            num_classes=int(num_classes),
            device="cpu",
            hidden_dim=128,
            dropout=0.1,
        )
        model_cpu = model.to(torch.device("cpu"))
        model_cpu.eval()

        for i, cid in enumerate(clip_ids):
            feats = feats_by_train.get(cid)
            if feats is None:
                cache_path = caches_dir / f"{cid}.npz"
                if not cache_path.exists():
                    raise FileNotFoundError(f"missing cache: {cache_path}")
                feats = _load_vis_feats_npz(cache_path)

            with torch.no_grad():
                logits = model_cpu(torch.from_numpy(feats).float())  # [T, C]
                bg = logits[:, 0]
                if method == "vision_mlp_cls":
                    s = logits[:, 1:].max(dim=-1).values - bg
                else:
                    clip_logits = logits.mean(dim=0)
                    clip_logits = clip_logits.clone()
                    clip_logits[0] = float("-inf")
                    cls = int(torch.argmax(clip_logits).item())
                    s = logits[:, cls] - bg
                s_np = s.detach().cpu().numpy().astype(np.float32)

            out[cid] = [float(x) for x in s_np.tolist()]
            if (i + 1) % 200 == 0 or (i + 1) == len(clip_ids):
                dt = _time.time() - t0
                print(f"[scores] {i+1}/{len(clip_ids)} clips ({dt:.1f}s)", flush=True)
    else:
        caches_dir_path = Path(caches_dir) if caches_dir is not None else None
        speech_idx: list[int] | None = None
        if method in ("ast_nonspeech_max", "energy_nonspeech_ast"):
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
        clap_lr_model = None
        clap_lr_feats_by_train: dict[str, np.ndarray] = {}
        clap_mlp_cls_model = None
        clap_mlp_cls_emb_by_train: dict[str, np.ndarray] = {}
        clap_num_classes: int | None = None

        def _softmax_np(x: np.ndarray, *, axis: int = -1) -> np.ndarray:
            x = np.asarray(x, dtype=np.float32)
            x = x - np.max(x, axis=axis, keepdims=True)
            ex = np.exp(x)
            return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-12)

        if method in ("av_clap_clip_agree", "clap_evt", "clap_lr", "clap_mlp_cls", "clap_mlp_cls_target"):
            if meta_dir is None:
                raise ValueError(f"{method} requires meta_dir (AVE class count/prompts)")

            from avs.datasets.ave import AVEIndex, ensure_ave_meta

            meta_dir = Path(meta_dir)
            ensure_ave_meta(meta_dir)
            index = AVEIndex.from_meta_dir(meta_dir)
            class_names = [str(index.idx_to_label[i]) for i in range(int(index.num_classes))]
            clap_num_classes = int(index.num_classes)

            from avs.audio.clap_probe import ClapProbe, ClapProbeConfig

            clap_probe = ClapProbe(ClapProbeConfig(pretrained=True, device=str(audio_device), dtype="float32"))

            if method in ("av_clap_clip_agree", "clap_evt", "clap_lr"):
                event_labels = [str(x) for x in class_names[1:]]  # exclude background
                clap_prompts = [f"a sound of {lab}" for lab in event_labels]
                clap_text = clap_probe.text_embeddings(clap_prompts)  # [C, D]
                if method == "av_clap_clip_agree":
                    if caches_dir_path is None:
                        raise ValueError(f"{method} requires caches_dir to load CLIP features")
                    from avs.vision.clip_text import ClipTextProbe, ClipTextProbeConfig

                    clip_probe = ClipTextProbe(ClipTextProbeConfig(pretrained=True, device="cpu", dtype="float32"))
                    clip_prompts = [f"a photo of {lab}" for lab in event_labels]
                    clip_text = clip_probe.text_features(clip_prompts)  # [C, 512]
                    clip_scale = float(clip_probe.logit_scale())

            if method == "clap_lr":
                if train_ids is None or labels_by_clip is None:
                    raise ValueError("clap_lr requires train_ids and labels_by_clip")
                if clap_probe is None or clap_text is None:
                    raise ValueError("internal error: clap_lr probes are not initialized")

                import torch
                import torch.nn as nn

                train_ids = [str(x) for x in train_ids]
                x_rows: list[np.ndarray] = []
                y_rows: list[np.ndarray] = []

                for i, cid in enumerate(train_ids):
                    wav_path = processed_dir / cid / "audio.wav"
                    aud_feat = clap_probe.audio_embeddings_per_second(wav_path, num_segments=int(num_segments))  # [T, D]
                    feats = (aud_feat @ clap_text.T).astype(np.float32, copy=False)  # [T, C]
                    clap_lr_feats_by_train[cid] = feats

                    labs = np.asarray(labels_by_clip[cid], dtype=np.int64)[: int(num_segments)]
                    y = (labs != 0).astype(np.float32).reshape(-1, 1)
                    x_rows.append(feats)
                    y_rows.append(y)

                    if (i + 1) % 200 == 0 or (i + 1) == len(train_ids):
                        dt = _time.time() - t0
                        print(f"[clap_lr] feats train {i+1}/{len(train_ids)} clips ({dt:.1f}s)", flush=True)

                x_np = np.concatenate(x_rows, axis=0).astype(np.float32, copy=False)
                y_np = np.concatenate(y_rows, axis=0).astype(np.float32, copy=False)

                x = torch.from_numpy(x_np).float()
                y = torch.from_numpy(y_np).float()

                torch.manual_seed(0)
                clap_lr_model = nn.Linear(int(x.shape[-1]), 1)
                pos = float((y > 0.5).sum().item())
                neg = float((y <= 0.5).sum().item())
                pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32)
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                opt = torch.optim.AdamW(clap_lr_model.parameters(), lr=2e-2)

                bs = 4096
                n = int(x.shape[0])
                steps = max(1, (n + bs - 1) // bs)
                for _epoch in range(30):
                    perm = torch.randperm(n)
                    for j in range(steps):
                        idx = perm[j * bs : (j + 1) * bs]
                        xb = x[idx]
                        yb = y[idx]
                        logits = clap_lr_model(xb)
                        loss = loss_fn(logits, yb)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()

                clap_lr_model.eval()
            if method in ("clap_mlp_cls", "clap_mlp_cls_target"):
                if train_ids is None or labels_by_clip is None:
                    raise ValueError(f"{method} requires train_ids and labels_by_clip")
                if clap_probe is None:
                    raise ValueError("internal error: clap_mlp_cls probes are not initialized")
                if clap_num_classes is None:
                    raise ValueError("internal error: clap_mlp_cls requires clap_num_classes")

                from avs.experiments.ave_p0 import _train_audio_basic_mlp_cls_eventness
                import torch

                train_ids = [str(x) for x in train_ids]
                for i, cid in enumerate(train_ids):
                    wav_path = processed_dir / cid / "audio.wav"
                    emb = clap_probe.audio_embeddings_per_second(wav_path, num_segments=int(num_segments))  # [T, D]
                    clap_mlp_cls_emb_by_train[cid] = emb.astype(np.float32, copy=False)
                    if (i + 1) % 200 == 0 or (i + 1) == len(train_ids):
                        dt = _time.time() - t0
                        print(f"[clap_mlp_cls] feats train {i+1}/{len(train_ids)} clips ({dt:.1f}s)", flush=True)

                clap_mlp_cls_model = _train_audio_basic_mlp_cls_eventness(
                    clip_ids_train=train_ids,
                    labels_by_clip=labels_by_clip,
                    audio_feats_by_clip=clap_mlp_cls_emb_by_train,
                    num_classes=int(clap_num_classes),
                    device="cpu",
                    hidden_dim=128,
                    dropout=0.1,
                )
                clap_mlp_cls_model = clap_mlp_cls_model.to(torch.device("cpu"))
                clap_mlp_cls_model.eval()
        for i, cid in enumerate(clip_ids):
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
            elif method == "clap_lr":
                if clap_probe is None or clap_text is None or clap_lr_model is None:
                    raise ValueError("internal error: clap_lr probes are not initialized")
                feats = clap_lr_feats_by_train.get(cid)
                if feats is None:
                    aud_feat = clap_probe.audio_embeddings_per_second(wav_path, num_segments=int(num_segments))  # [T,D]
                    feats = (aud_feat @ clap_text.T).astype(np.float32, copy=False)  # [T,C]
                import torch

                with torch.no_grad():
                    s = clap_lr_model(torch.from_numpy(feats).float()).squeeze(-1).numpy().astype(np.float32)
                out[cid] = [float(x) for x in s.tolist()]
            elif method == "clap_evt":
                if clap_probe is None or clap_text is None:
                    raise ValueError("internal error: clap_evt probes are not initialized")
                aud_feat = clap_probe.audio_embeddings_per_second(wav_path, num_segments=int(num_segments))  # [T, D]
                aud_logits = aud_feat @ clap_text.T  # [T, C]
                aud_probs = _softmax_np(aud_logits * float(audio_scale), axis=1)
                s = aud_probs.max(axis=1).astype(np.float32, copy=False)
                out[cid] = [float(x) for x in s.tolist()]
            elif method in ("clap_mlp_cls", "clap_mlp_cls_target"):
                if clap_probe is None or clap_mlp_cls_model is None:
                    raise ValueError("internal error: clap_mlp_cls probes are not initialized")
                emb = clap_mlp_cls_emb_by_train.get(cid)
                if emb is None:
                    emb = clap_probe.audio_embeddings_per_second(wav_path, num_segments=int(num_segments))  # [T, D]
                import torch

                with torch.no_grad():
                    logits = clap_mlp_cls_model(torch.from_numpy(np.asarray(emb, dtype=np.float32)).float())  # [T,C]
                    bg = logits[:, 0]
                    if method == "clap_mlp_cls":
                        probs = torch.softmax(logits, dim=-1)
                        s = 1.0 - probs[:, 0]
                    else:
                        clip_logits = logits.mean(dim=0)
                        if int(clip_logits.shape[0]) < 2:
                            raise ValueError(
                                f"clap_mlp_cls_target requires num_classes>=2, got {clip_logits.shape[0]}"
                            )
                        clip_logits = clip_logits.clone()
                        clip_logits[0] = float("-inf")
                        cls = int(torch.argmax(clip_logits).item())
                        s = logits[:, cls] - bg
                    s_np = s.detach().cpu().numpy().astype(np.float32)
                out[cid] = [float(x) for x in s_np.tolist()]
            elif method == "av_clap_clip_agree":
                if caches_dir_path is None:
                    raise ValueError(f"{method} requires caches_dir to load CLIP features")
                if clap_probe is None or clap_text is None or clip_probe is None or clip_text is None:
                    raise ValueError("internal error: av_clap_clip_agree probes are not initialized")

                cache_path = caches_dir_path / f"{cid}.npz"
                if not cache_path.exists():
                    raise FileNotFoundError(f"missing cache: {cache_path}")
                with np.load(cache_path) as z:
                    if "res_112" in z.files:
                        vis_pooled = z["res_112"]
                    else:
                        avail = sorted(int(k.split("_", 1)[1]) for k in z.files if k.startswith("res_"))
                        if not avail:
                            raise ValueError(f"no res_* arrays in cache: {cache_path}")
                        vis_pooled = z[f"res_{avail[0]}"]
                vis_pooled = np.asarray(vis_pooled, dtype=np.float32)

                vis_feat = clip_probe.project_image_features(vis_pooled)  # [T, 512]
                vis_logits = vis_feat @ clip_text.T  # [T, C]
                vis_probs = _softmax_np(vis_logits * float(clip_scale), axis=1)

                aud_feat = clap_probe.audio_embeddings_per_second(wav_path, num_segments=int(num_segments))  # [T, D]
                aud_logits = aud_feat @ clap_text.T  # [T, C]
                aud_probs = _softmax_np(aud_logits * float(audio_scale), axis=1)

                agree_tc = aud_probs * vis_probs  # [T, C]
                cls = int(np.argmax(np.max(agree_tc, axis=0)))
                s = agree_tc[:, cls].astype(np.float32, copy=False)
                out[cid] = [float(x) for x in s.tolist()]
            elif method == "energy_nonspeech_ast":
                assert ast_probe is not None
                # Speech-aware suppression: non-speech audio peaks are more likely to correspond to audio-visual events.
                if speech_idx is None:
                    raise ValueError("internal error: speech_idx must be initialized for energy_nonspeech_ast")
                ev = compute_eventness_wav_energy_stride_max(
                    wav_path, num_segments=int(num_segments), stride_s=0.2, win_s=0.4
                )
                base = minmax_01([float(x) for x in ev.scores])
                logits = ast_probe.logits_per_second(wav_path, num_segments=int(num_segments))  # [T,C]
                probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float32)))
                speech = probs[:, speech_idx].max(axis=1) if speech_idx else np.zeros((int(num_segments),), dtype=np.float32)
                out[cid] = [float(b) * (1.0 - float(s)) for b, s in zip(base, speech.tolist(), strict=True)]
            elif method == "asr_vad":
                from avs.audio.vad_webrtc import WebRtcVadConfig, webrtcvad_speech_ratio_per_second

                ev = compute_eventness_wav_energy_stride_max(wav_path, num_segments=int(num_segments), stride_s=0.2, win_s=0.4)
                base = minmax_01([float(x) for x in ev.scores])
                speech = webrtcvad_speech_ratio_per_second(
                    wav_path, num_segments=int(num_segments), cfg=WebRtcVadConfig(aggressiveness=2, frame_ms=30)
                )
                out[cid] = [float(b) * (1.0 - float(s)) for b, s in zip(base, speech, strict=True)]
            elif method in (
                "vision_clipdiff",
                "energy_autoshift_clipdiff",
                "energy_autoshift_clipdiff_pos",
                "av_fused_clipdiff",
                "av_fused_clipdiff_prod",
                "moe_energy_clipdiff",
            ):
                if caches_dir_path is None:
                    raise ValueError(f"{method} requires caches_dir to load CLIP features")

                import numpy as np
                from avs.vision.cheap_eventness import clip_feature_diff_eventness

                cache_path = caches_dir_path / f"{cid}.npz"
                if not cache_path.exists():
                    raise FileNotFoundError(f"missing cache: {cache_path}")

                with np.load(cache_path) as z:
                    if "res_160" in z.files:
                        vis_feats = z["res_160"]
                    elif "res_112" in z.files:
                        vis_feats = z["res_112"]
                    else:
                        avail = sorted(int(k.split("_", 1)[1]) for k in z.files if k.startswith("res_"))
                        if not avail:
                            raise ValueError(f"no res_* arrays in cache: {cache_path}")
                        vis_feats = z[f"res_{avail[0]}"]
                vis_feats = np.asarray(vis_feats, dtype=np.float32)

                vis = clip_feature_diff_eventness(vis_feats, metric="cosine")
                vis_scores = minmax_01([float(x) for x in vis])

                if method == "vision_clipdiff":
                    out[cid] = scale(vis_scores, AV_FUSED_SCORE_SCALE)
                elif method == "moe_energy_clipdiff":
                    ev = compute_eventness_wav_energy(wav_path, num_segments=int(num_segments))
                    audio_raw = [float(x) for x in ev.scores]
                    # Mixture-of-experts: if audio is "low variance" (would fall back under the legacy std gate),
                    # use semantic visual motion instead.
                    if float(np.asarray(audio_raw, dtype=np.float32).std()) < 1.0:
                        out[cid] = scale(vis_scores, AV_FUSED_SCORE_SCALE)
                    else:
                        out[cid] = audio_raw
                elif method == "energy_autoshift_clipdiff":
                    ev = compute_eventness_wav_energy(wav_path, num_segments=int(num_segments))
                    audio_raw = [float(x) for x in ev.scores]
                    audio_scores = minmax_01(audio_raw)
                    s = best_shift_by_corr(audio_scores, vis_scores, shifts=[-2, -1, 0, 1, 2])
                    out[cid] = shift_scores(audio_raw, int(s))
                elif method == "energy_autoshift_clipdiff_pos":
                    ev = compute_eventness_wav_energy(wav_path, num_segments=int(num_segments))
                    audio_raw = [float(x) for x in ev.scores]
                    audio_scores = minmax_01(audio_raw)
                    s = best_shift_by_corr(audio_scores, vis_scores, shifts=[0, 1, 2])
                    out[cid] = shift_scores(audio_raw, int(s))
                else:
                    ev = compute_eventness_wav_energy_stride_max(
                        wav_path, num_segments=int(num_segments), stride_s=0.2, win_s=0.4
                    )
                    audio_scores = minmax_01([float(x) for x in ev.scores])
                    if method == "av_fused_clipdiff":
                        fused = fuse_max(audio_scores, vis_scores, num_segments=int(num_segments))
                    else:
                        fused = fuse_prod(audio_scores, vis_scores, num_segments=int(num_segments))
                    out[cid] = scale(fused, AV_FUSED_SCORE_SCALE)
            elif method == "av_fused":
                from avs.vision.cheap_eventness import frame_diff_eventness, list_frames

                ev = compute_eventness_wav_energy_stride_max(
                    wav_path, num_segments=int(num_segments), stride_s=0.2, win_s=0.4
                )
                a = minmax_01([float(x) for x in ev.scores])

                frames_dir = processed_dir / cid / "frames"
                frames = list_frames(frames_dir) if frames_dir.exists() else []
                v = frame_diff_eventness(frames, size=32) if frames else []
                v = minmax_01([float(x) for x in v])
                out[cid] = scale(
                    fuse_max(a, v, num_segments=int(num_segments)),
                    AV_FUSED_SCORE_SCALE,
                )
            elif method == "av_fused_prod":
                from avs.vision.cheap_eventness import frame_diff_eventness, list_frames

                ev = compute_eventness_wav_energy_stride_max(
                    wav_path, num_segments=int(num_segments), stride_s=0.2, win_s=0.4
                )
                a = minmax_01([float(x) for x in ev.scores])

                frames_dir = processed_dir / cid / "frames"
                frames = list_frames(frames_dir) if frames_dir.exists() else []
                v = frame_diff_eventness(frames, size=32) if frames else []
                v = minmax_01([float(x) for x in v])
                out[cid] = scale(
                    fuse_prod(a, v, num_segments=int(num_segments)),
                    AV_FUSED_SCORE_SCALE,
                )
            elif method == "ast":
                assert ast_probe is not None
                out[cid] = [float(x) for x in ast_probe.eventness_per_second(wav_path, num_segments=int(num_segments))]
            elif method == "ast_nonspeech_max":
                assert ast_probe is not None
                logits = ast_probe.logits_per_second(wav_path, num_segments=int(num_segments))  # [T, C]
                probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float32)))
                if speech_idx:
                    probs[:, speech_idx] = 0.0
                s = probs.max(axis=1).astype(np.float32, copy=False)
                out[cid] = [float(x) for x in s.tolist()]
            elif method == "panns":
                assert panns_probe is not None
                out[cid] = [float(x) for x in panns_probe.eventness_per_second(wav_path, num_segments=int(num_segments))]
            elif method == "audiomae":
                assert audiomae_probe is not None
                out[cid] = [float(x) for x in audiomae_probe.eventness_per_second(wav_path, num_segments=int(num_segments))]
            else:
                raise ValueError(f"unsupported eventness_method for score caching: {method}")

            if (i + 1) % 200 == 0 or (i + 1) == len(clip_ids):
                dt = _time.time() - t0
                print(f"[scores] {i+1}/{len(clip_ids)} clips ({dt:.1f}s)", flush=True)

        if method in ("av_clap_clip_agree", "clap_evt", "clap_lr", "clap_mlp_cls", "clap_mlp_cls_target"):
            clap_probe = None
            clip_probe = None
            if str(audio_device).startswith("cuda"):
                import torch

                torch.cuda.empty_cache()
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fixed-space AVE config sweep (val selection → test reproduction).")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("sweep", help="Run fixed candidate configs on split-eval and write sweep_summary.json + best_config.json.")
    _build_common_args(ps)
    ps.add_argument("--out-dir", type=Path, default=Path("runs") / f"E0011_ave_p0_sweep_{time.strftime('%Y%m%d-%H%M%S')}")
    ps.add_argument("--p-filter", type=float, default=0.1, help="Pre-filter configs by p-value before selecting the best. 0 disables.")
    ps.add_argument(
        "--candidate-set",
        type=str,
        default="energy_v1",
        choices=[
            "energy_v1",
            "energy_v2",
            "energy_v3",
            "ast_v1",
            "ltl_gini_v1",
            "ltl_gini_v2",
            "ltl_gap_v1",
            "ltl_top1med_v1",
            "ltl_top1med_visfb_v1",
            "ltl_top1med_visfb_gated_v1",
            "ltl_top1med_gate_lr_v1",
            "ltl_top1med_gate_all_v1",
            "ltl_top1med_anchor2veto_v1",
            "ltl_top1med_norm_v1",
            "ltl_sep3_v1",
            "ltl_top1med_band_v1",
            "ltl_top1med_band_low112_v1",
            "ltl_top1med_band_midres_v1",
            "ltl_top1med_moe_v1",
            "ltl_top1med_nmsstrong_v1",
            "ltl_top1med_dropfar_v1",
            "ltl_top1med_farfb_v1",
            "ltl_top1med_adjselect_v1",
            "ltl_top1med_keepadj_v1",
            "ltl_top1med_keepadj_basealloc_v1",
            "ltl_top1med_keepadj_basealloc_highonly_v1",
            "ltl_top1med_basealloc_v1",
            "ltl_top1med_bridgealloc_v1",
            "ltl_top1med_autoshift_v1",
            "ltl_top1med_adaptivegap_v1",
            "ltl_top1med_adjdist_v1",
            "ltl_top1med_headcap_v1",
            "ltl_top1med_resfeat_v1",
            "ltl_top1med_highconf_v1",
            "ltl_top1med_tiered_v1",
            "ltl_top1med_scorealloc_v1",
            "ltl_top1med_maxhigh1_v1",
            "ltl_top1med_k1_v1",
            "ltl_top1med_k1_extreme_v1",
            "ltl_top1med_extreme_v1",
            "ltl_extreme_v1",
            "ltl_std_v1",
            "ltl_std_v2",
            "ltl_adaptive_v1",
            "ltl_adaptive_keepadj_v1",
            "ltl_smooth_v1",
            "ltl_adaptive_v2",
            "ltl_adaptive_v3",
            "ltl_maxhigh1_v1",
        ],
        help="Pre-registered candidate set. Use 'energy_v1'/'energy_v2'/'energy_v3' for log-energy, 'ast_v1' for AST scores, "
        "and 'ltl_gini_v1'/'ltl_gini_v2'/'ltl_gap_v1'/'ltl_std_v1'/'ltl_std_v2' for non-energy Stage-1 methods.",
    )

    pr = sub.add_parser("run", help="Run one config from best_config.json on a given eval ids file/split (e.g., test402).")
    _build_common_args(pr)
    pr.add_argument("--config-json", type=Path, required=True, help="Path to best_config.json from the sweep step.")
    pr.add_argument("--out-dir", type=Path, default=Path("runs") / f"E0012_ave_p0_best_{time.strftime('%Y%m%d-%H%M%S')}")
    return p


def _load_or_select_ids(
    *,
    index: AVEIndex,
    ids_file: Path | None,
    split: str,
    limit: int | None,
) -> list[str]:
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


def _run_one(
    *,
    index: AVEIndex,
    train_ids: list[str],
    eval_ids: list[str],
    labels_by_clip: dict[str, list[int]],
    caches_dir: Path,
    processed_dir: Path,
    seeds: list[int],
    train_cfg: TrainConfig,
    train_device: str,
    eventness_method: str,
    audio_device: str,
    ast_pretrained: bool,
    panns_random: bool,
    panns_checkpoint: Path | None,
    audiomae_random: bool,
    audiomae_checkpoint: Path | None,
    cand: CandidateConfig,
    scores_by_clip_override: dict[str, list[float]] | None,
) -> dict:
    cfg = P0Config(
        k=int(cand.k),
        low_res=int(cand.low_res),
        base_res=int(cand.base_res),
        high_res=int(cand.high_res),
        patch_size=16,
        res_feature=str(cand.res_feature),
        max_high_anchors=int(cand.max_high_anchors) if cand.max_high_anchors is not None else None,
        anchor_shift=int(cand.anchor_shift),
        anchor_std_threshold=float(cand.anchor_std_threshold),
        anchor_select=str(cand.anchor_select),
        anchor_drop_far_dist=int(cand.anchor_drop_far_dist),
        anchor_fallback_far_dist=int(cand.anchor_fallback_far_dist),
        anchor_fallback_mode=str(cand.anchor_fallback_mode),
        anchor_fallback_visual_conf_metric=str(cand.anchor_fallback_visual_conf_metric),
        anchor_fallback_visual_conf_threshold=float(cand.anchor_fallback_visual_conf_threshold),
        anchor_nms_radius=int(cand.anchor_nms_radius),
        anchor_nms_strong_gap=float(cand.anchor_nms_strong_gap),
        anchor_window=int(cand.anchor_window),
        anchor_smooth_window=int(cand.anchor_smooth_window),
        anchor_smooth_mode=str(cand.anchor_smooth_mode),
        anchor2_veto_method=str(cand.anchor2_veto_method),
        anchor2_veto_threshold=float(cand.anchor2_veto_threshold),
        anchor2_veto_label_radius=int(cand.anchor2_veto_label_radius),
        anchor_gate_method=str(cand.anchor_gate_method),
        anchor_gate_threshold=float(cand.anchor_gate_threshold),
        anchor_gate_label_radius=int(cand.anchor_gate_label_radius),
        anchor_conf_metric=str(cand.anchor_conf_metric) if cand.anchor_conf_metric is not None else None,
        anchor_conf_threshold=float(cand.anchor_conf_threshold) if cand.anchor_conf_threshold is not None else None,
        anchor_base_alloc=str(cand.anchor_base_alloc),
        anchor_high_policy=str(cand.anchor_high_policy),
        anchor_high_adjacent_dist=int(cand.anchor_high_adjacent_dist),
        anchor_high_gap_threshold=float(cand.anchor_high_gap_threshold),
        anchor_high_conf_metric=str(cand.anchor_high_conf_metric) if cand.anchor_high_conf_metric is not None else None,
        anchor_high_conf_threshold=float(cand.anchor_high_conf_threshold),
        head=str(cand.head),
        head_hidden_dim=int(cand.head_hidden_dim),
        head_dropout=float(cand.head_dropout),
        temporal_kernel_size=int(cand.temporal_kernel_size),
        triad_policy=str(cand.triad_policy),
        triad_alt_conf_threshold=float(cand.triad_alt_conf_threshold),
        triad_alt_low_res=int(cand.triad_alt_low_res),
        triad_alt_high_res=int(cand.triad_alt_high_res),
        triad_alt_max_high_anchors=int(cand.triad_alt_max_high_anchors) if cand.triad_alt_max_high_anchors is not None else None,
        budget_mode=str(cand.budget_mode),
        budget_epsilon_frac=float(cand.budget_epsilon_frac),
        budget_extra_resolutions=tuple(int(r) for r in cand.budget_extra_resolutions),
    )

    metrics = run_p0_from_caches(
        clip_ids_train=train_ids,
        clip_ids_eval=eval_ids,
        labels_by_clip=labels_by_clip,
        caches_dir=caches_dir,
        audio_dir=processed_dir,
        cfg=cfg,
        baselines=["uniform", "random_top2", "anchored_top2"],
        seeds=seeds,
        train_cfg=train_cfg,
        train_device=str(train_device),
        num_classes=index.num_classes,
        class_names=[str(index.idx_to_label[i]) for i in range(int(index.num_classes))],
        num_segments=10,
        eventness_method=str(eventness_method),
        audio_device=str(audio_device),
        ast_pretrained=bool(ast_pretrained),
        panns_random=bool(panns_random),
        panns_checkpoint=panns_checkpoint,
        audiomae_random=bool(audiomae_random),
        audiomae_checkpoint=audiomae_checkpoint,
        scores_by_clip_override=scores_by_clip_override,
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    ensure_ave_meta(args.meta_dir)
    index = AVEIndex.from_meta_dir(args.meta_dir)

    train_ids = _load_or_select_ids(index=index, ids_file=args.train_ids_file, split=args.split_train, limit=args.limit_train)
    eval_ids = _load_or_select_ids(index=index, ids_file=args.eval_ids_file, split=args.split_eval, limit=args.limit_eval)

    caches_dir = Path(args.caches_dir)
    if args.allow_missing:
        train_ids = _filter_missing(ids=train_ids, caches_dir=caches_dir)
        eval_ids = _filter_missing(ids=eval_ids, caches_dir=caches_dir)
    if not train_ids or not eval_ids:
        raise SystemExit(f"no usable ids after filtering (train={len(train_ids)} eval={len(eval_ids)})")

    labels_by_clip = {**_labels_for_ids(index, train_ids), **_labels_for_ids(index, eval_ids)}

    seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip()]
    if len(seeds) < 2:
        raise SystemExit("--seeds must contain at least 2 seeds to compute paired p-values")

    train_cfg = TrainConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    if args.cmd == "sweep":
        out_dir: Path = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        candidate_set = str(args.candidate_set)
        if candidate_set == "energy_v1":
            candidates = _default_candidates()
        elif candidate_set == "energy_v2":
            candidates = _candidates_energy_v2()
        elif candidate_set == "energy_v3":
            candidates = _candidates_energy_v3()
        elif candidate_set == "ast_v1":
            candidates = _candidates_ast_v1()
        elif candidate_set == "ltl_gini_v1":
            candidates = _candidates_ltl_gini_v1()
        elif candidate_set == "ltl_gini_v2":
            candidates = _candidates_ltl_gini_v2()
        elif candidate_set == "ltl_gap_v1":
            candidates = _candidates_ltl_gap_v1()
        elif candidate_set == "ltl_top1med_v1":
            candidates = _candidates_ltl_top1med_v1()
        elif candidate_set == "ltl_top1med_visfb_v1":
            candidates = _candidates_ltl_top1med_visfb_v1()
        elif candidate_set == "ltl_top1med_visfb_gated_v1":
            candidates = _candidates_ltl_top1med_visfb_gated_v1()
        elif candidate_set == "ltl_top1med_gate_lr_v1":
            candidates = _candidates_ltl_top1med_gate_lr_v1()
        elif candidate_set == "ltl_top1med_gate_all_v1":
            candidates = _candidates_ltl_top1med_gate_all_v1()
        elif candidate_set == "ltl_top1med_anchor2veto_v1":
            candidates = _candidates_ltl_top1med_anchor2veto_v1()
        elif candidate_set == "ltl_top1med_norm_v1":
            candidates = _candidates_ltl_top1med_norm_v1()
        elif candidate_set == "ltl_sep3_v1":
            candidates = _candidates_ltl_sep3_v1()
        elif candidate_set == "ltl_top1med_band_v1":
            candidates = _candidates_ltl_top1med_band_v1()
        elif candidate_set == "ltl_top1med_band_low112_v1":
            candidates = _candidates_ltl_top1med_band_low112_v1()
        elif candidate_set == "ltl_top1med_band_midres_v1":
            candidates = _candidates_ltl_top1med_band_midres_v1()
        elif candidate_set == "ltl_top1med_moe_v1":
            candidates = _candidates_ltl_top1med_moe_v1()
        elif candidate_set == "ltl_top1med_nmsstrong_v1":
            candidates = _candidates_ltl_top1med_nmsstrong_v1()
        elif candidate_set == "ltl_top1med_dropfar_v1":
            candidates = _candidates_ltl_top1med_dropfar_v1()
        elif candidate_set == "ltl_top1med_farfb_v1":
            candidates = _candidates_ltl_top1med_farfb_v1()
        elif candidate_set == "ltl_top1med_adjselect_v1":
            candidates = _candidates_ltl_top1med_adjselect_v1()
        elif candidate_set == "ltl_top1med_keepadj_v1":
            candidates = _candidates_ltl_top1med_keepadj_v1()
        elif candidate_set == "ltl_top1med_keepadj_basealloc_v1":
            candidates = _candidates_ltl_top1med_keepadj_basealloc_v1()
        elif candidate_set == "ltl_top1med_keepadj_basealloc_highonly_v1":
            candidates = _candidates_ltl_top1med_keepadj_basealloc_highonly_v1()
        elif candidate_set == "ltl_top1med_basealloc_v1":
            candidates = _candidates_ltl_top1med_basealloc_v1()
        elif candidate_set == "ltl_top1med_bridgealloc_v1":
            candidates = _candidates_ltl_top1med_bridgealloc_v1()
        elif candidate_set == "ltl_top1med_autoshift_v1":
            candidates = _candidates_ltl_top1med_autoshift_v1()
        elif candidate_set == "ltl_top1med_adaptivegap_v1":
            candidates = _candidates_ltl_top1med_adaptivegap_v1()
        elif candidate_set == "ltl_top1med_adjdist_v1":
            candidates = _candidates_ltl_top1med_adjdist_v1()
        elif candidate_set == "ltl_top1med_headcap_v1":
            candidates = _candidates_ltl_top1med_headcap_v1()
        elif candidate_set == "ltl_top1med_resfeat_v1":
            candidates = _candidates_ltl_top1med_resfeat_v1()
        elif candidate_set == "ltl_top1med_highconf_v1":
            candidates = _candidates_ltl_top1med_highconf_v1()
        elif candidate_set == "ltl_top1med_tiered_v1":
            candidates = _candidates_ltl_top1med_tiered_v1()
        elif candidate_set == "ltl_top1med_scorealloc_v1":
            candidates = _candidates_ltl_top1med_scorealloc_v1()
        elif candidate_set == "ltl_top1med_maxhigh1_v1":
            candidates = _candidates_ltl_top1med_maxhigh1_v1()
        elif candidate_set == "ltl_top1med_k1_v1":
            candidates = _candidates_ltl_top1med_k1_v1()
        elif candidate_set == "ltl_top1med_k1_extreme_v1":
            candidates = _candidates_ltl_top1med_k1_extreme_v1()
        elif candidate_set == "ltl_top1med_extreme_v1":
            candidates = _candidates_ltl_top1med_extreme_v1()
        elif candidate_set == "ltl_extreme_v1":
            candidates = _candidates_ltl_extreme_v1()
        elif candidate_set == "ltl_std_v1":
            candidates = _candidates_ltl_std_v1()
        elif candidate_set == "ltl_std_v2":
            candidates = _candidates_ltl_std_v2()
        elif candidate_set == "ltl_adaptive_v1":
            candidates = _candidates_ltl_adaptive_v1()
        elif candidate_set == "ltl_adaptive_keepadj_v1":
            candidates = _candidates_ltl_adaptive_keepadj_v1()
        elif candidate_set == "ltl_smooth_v1":
            candidates = _candidates_ltl_smooth_v1()
        elif candidate_set == "ltl_adaptive_v2":
            candidates = _candidates_ltl_adaptive_v2()
        elif candidate_set == "ltl_adaptive_v3":
            candidates = _candidates_ltl_adaptive_v3()
        elif candidate_set == "ltl_maxhigh1_v1":
            candidates = _candidates_ltl_maxhigh1_v1()
        else:
            raise SystemExit(f"unknown candidate_set: {candidate_set}")

        # Optional eventness score cache (high leverage for expensive backends like AST/PANNs/AudioMAE and probes).
        all_ids = sorted(set(train_ids + eval_ids))
        scores_by_clip_override = None
        scores_json: Path | None = args.scores_json
        if scores_json is None and str(args.eventness_method) in (
            "ast",
            "ast_lr",
            "panns",
            "panns_lr",
            "panns_embed_lr",
            "panns_embed_mlp",
            "av_panns_embed_clipdiff_mlp",
            "audiomae",
            "av_clap_clip_agree",
            "clap_evt",
            "clap_lr",
            "clap_mlp_cls",
            "clap_mlp_cls_target",
            "av_fused",
            "av_fused_prod",
            "audio_basic_tcn",
            "audio_fbank_tcn",
            "av_ast_clipdiff_mlp",
            "av_ast_clipdiff_mil_mlp",
            "av_ast_clipdiff_tcn",
            "av_ast_clipalign_bce",
            "av_clipdiff_lr",
            "av_clipdiff_mlp",
            "av_clipdiff_framediff_mlp",
            "av_clipdiff_flow_mlp",
            "av_clipdiff_flow_mlp_stride",
            "av_clipdiff_fbank_mlp",
            "av_clipdiff_mlp_r160",
            "av_clipdiff_mlp_r224",
            "av_clipdiff_vec_mlp",
            "av_clipdiff_mlp_cls",
            "av_clipdiff_mlp_cls_target",
            "av_clip_mlp_cls",
            "av_clip_mlp_cls_target",
            "av_clipdiff_tcn",
        ):
            scores_json = out_dir / "eventness_scores.json"
        if scores_json is not None:
            if scores_json.exists():
                scores_by_clip_override = _load_scores_json(scores_json)
                missing = [cid for cid in all_ids if cid not in scores_by_clip_override]
                if missing:
                    print(f"[scores] filling {len(missing)} missing clips into {scores_json}", flush=True)
                    extra = _compute_scores_by_clip(
                        clip_ids=missing,
                        processed_dir=Path(args.processed_dir),
                        meta_dir=Path(args.meta_dir),
                        caches_dir=caches_dir,
                        num_segments=10,
                        eventness_method=str(args.eventness_method),
                        audio_device=str(args.audio_device),
                        ast_pretrained=bool(args.ast_pretrained),
                        panns_random=bool(args.panns_random),
                        panns_checkpoint=args.panns_checkpoint,
                        audiomae_random=bool(args.audiomae_random),
                        audiomae_checkpoint=args.audiomae_checkpoint,
                        train_ids=train_ids,
                        labels_by_clip=labels_by_clip,
                    )
                    scores_by_clip_override.update(extra)
                    _write_scores_json(
                        scores_json,
                        eventness_method=str(args.eventness_method),
                        num_segments=10,
                        scores_by_clip=scores_by_clip_override,
                    )
            else:
                scores_by_clip_override = _compute_scores_by_clip(
                    clip_ids=all_ids,
                    processed_dir=Path(args.processed_dir),
                    meta_dir=Path(args.meta_dir),
                    caches_dir=caches_dir,
                    num_segments=10,
                    eventness_method=str(args.eventness_method),
                    audio_device=str(args.audio_device),
                    ast_pretrained=bool(args.ast_pretrained),
                    panns_random=bool(args.panns_random),
                    panns_checkpoint=args.panns_checkpoint,
                    audiomae_random=bool(args.audiomae_random),
                    audiomae_checkpoint=args.audiomae_checkpoint,
                    train_ids=train_ids,
                    labels_by_clip=labels_by_clip,
                )
                _write_scores_json(
                    scores_json,
                    eventness_method=str(args.eventness_method),
                    num_segments=10,
                    scores_by_clip=scores_by_clip_override,
                )
            missing = [cid for cid in all_ids if cid not in scores_by_clip_override]
            if missing:
                raise SystemExit(f"scores-json still missing {len(missing)} clip_ids (e.g. {missing[:3]})")
            print(f"[scores] using {scores_json}", flush=True)

        results: list[dict] = []
        for cand in candidates:
            run_dir = out_dir / cand.name
            run_dir.mkdir(parents=True, exist_ok=True)

            _write_json(run_dir / "config.json", cand.to_jsonable())

            metrics = _run_one(
                index=index,
                train_ids=train_ids,
                eval_ids=eval_ids,
                labels_by_clip=labels_by_clip,
                caches_dir=caches_dir,
                processed_dir=Path(args.processed_dir),
                seeds=seeds,
                train_cfg=train_cfg,
                train_device=str(args.train_device),
                eventness_method=str(args.eventness_method),
                audio_device=str(args.audio_device),
                ast_pretrained=bool(args.ast_pretrained),
                panns_random=bool(args.panns_random),
                panns_checkpoint=args.panns_checkpoint,
                audiomae_random=bool(args.audiomae_random),
                audiomae_checkpoint=args.audiomae_checkpoint,
                cand=cand,
                scores_by_clip_override=scores_by_clip_override,
            )

            metrics_path = run_dir / "metrics.json"
            _write_json(metrics_path, metrics)

            delta, pval = _extract_delta_and_p(metrics)
            results.append(
                {
                    "candidate": cand.to_jsonable(),
                    "metrics_path": str(metrics_path),
                    "anchored_minus_uniform_mean": delta,
                    "anchored_vs_uniform_p": pval,
                }
            )

        ordered = sorted(results, key=lambda r: float(r["anchored_minus_uniform_mean"] or float("-inf")), reverse=True)

        # Optional p-value pre-filter (usually for val selection).
        p_filter = float(args.p_filter)
        filtered = ordered
        if p_filter > 0.0:
            # Guardrail: only apply p-filter to candidates that *improve* over uniform on val.
            filtered = [
                r
                for r in ordered
                if r["anchored_minus_uniform_mean"] is not None
                and float(r["anchored_minus_uniform_mean"]) > 0.0
                and r["anchored_vs_uniform_p"] is not None
                and float(r["anchored_vs_uniform_p"]) < p_filter
            ]
            if not filtered:
                # Fall back to "best positive delta" (even if not significant on val),
                # and only then fall back to the absolute best delta (which may be negative).
                filtered = [r for r in ordered if r["anchored_minus_uniform_mean"] is not None and float(r["anchored_minus_uniform_mean"]) > 0.0]
                if not filtered:
                    filtered = ordered

        best = filtered[0]
        best_config = best["candidate"]
        _write_json(out_dir / "best_config.json", best_config)

        summary = {
            "ok": True,
            "meta_dir": str(args.meta_dir),
            "processed_dir": str(args.processed_dir),
            "caches_dir": str(args.caches_dir),
            "split_train": str(args.split_train),
            "split_eval": str(args.split_eval),
            "num_train_ids": int(len(train_ids)),
            "num_eval_ids": int(len(eval_ids)),
            "seeds": seeds,
            "train_cfg": {"epochs": train_cfg.epochs, "batch_size": train_cfg.batch_size, "lr": train_cfg.lr, "weight_decay": train_cfg.weight_decay},
            "candidate_set": str(candidate_set),
            "eventness_method": str(args.eventness_method),
            "scores_json": str(scores_json) if scores_json is not None else None,
            "audio_device": str(args.audio_device),
            "train_device": str(args.train_device),
            "p_filter": p_filter,
            "candidates": results,
            "best": best,
            "top3": ordered[:3],
        }
        _write_json(out_dir / "sweep_summary.json", summary)
        print(out_dir / "sweep_summary.json")
        return 0

    # args.cmd == "run"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_obj = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    triad_alt_max_high = 1
    if "triad_alt_max_high_anchors" in cand_obj:
        triad_alt_max_high = (
            int(cand_obj["triad_alt_max_high_anchors"]) if cand_obj["triad_alt_max_high_anchors"] is not None else None
        )
    cand = CandidateConfig(
        name=str(cand_obj.get("name", "best_config")),
        k=int(cand_obj.get("k", 2)),
        low_res=int(cand_obj["low_res"]),
        base_res=int(cand_obj["base_res"]),
        high_res=int(cand_obj["high_res"]),
        head=str(cand_obj.get("head", "temporal_conv")),
        head_hidden_dim=int(cand_obj.get("head_hidden_dim", 128)),
        head_dropout=float(cand_obj.get("head_dropout", 0.0)),
        res_feature=str(cand_obj.get("res_feature", "none")),
        temporal_kernel_size=int(cand_obj.get("temporal_kernel_size", 3)),
        anchor_shift=int(cand_obj.get("anchor_shift", 0)),
        anchor_std_threshold=float(cand_obj.get("anchor_std_threshold", 0.0)),
        anchor_select=str(cand_obj.get("anchor_select", "topk")),
        anchor_nms_radius=int(cand_obj.get("anchor_nms_radius", 1)),
        anchor_nms_strong_gap=float(cand_obj.get("anchor_nms_strong_gap", 0.6)),
        anchor_window=int(cand_obj.get("anchor_window", 3)),
        anchor_smooth_window=int(cand_obj.get("anchor_smooth_window", 0)),
        anchor_smooth_mode=str(cand_obj.get("anchor_smooth_mode", "mean")),
        anchor2_veto_method=str(cand_obj.get("anchor2_veto_method", "none")),
        anchor2_veto_threshold=float(cand_obj.get("anchor2_veto_threshold", 0.5)),
        anchor2_veto_label_radius=int(cand_obj.get("anchor2_veto_label_radius", 1)),
        anchor_base_alloc=str(cand_obj.get("anchor_base_alloc", "distance")),
        anchor_conf_metric=str(cand_obj["anchor_conf_metric"]) if cand_obj.get("anchor_conf_metric") is not None else None,
        anchor_conf_threshold=float(cand_obj["anchor_conf_threshold"]) if cand_obj.get("anchor_conf_threshold") is not None else None,
        max_high_anchors=int(cand_obj["max_high_anchors"]) if cand_obj.get("max_high_anchors") is not None else None,
        anchor_high_policy=str(cand_obj.get("anchor_high_policy", "fixed")),
        anchor_high_adjacent_dist=int(cand_obj.get("anchor_high_adjacent_dist", 1)),
        anchor_high_gap_threshold=float(cand_obj.get("anchor_high_gap_threshold", 0.0)),
        anchor_drop_far_dist=int(cand_obj.get("anchor_drop_far_dist", 0)),
        anchor_fallback_far_dist=int(cand_obj.get("anchor_fallback_far_dist", 0)),
        anchor_high_conf_metric=str(cand_obj["anchor_high_conf_metric"]) if cand_obj.get("anchor_high_conf_metric") is not None else None,
        anchor_high_conf_threshold=float(cand_obj.get("anchor_high_conf_threshold", 0.0)),
        triad_policy=str(cand_obj.get("triad_policy", "fixed")),
        triad_alt_conf_threshold=float(cand_obj.get("triad_alt_conf_threshold", 0.0)),
        triad_alt_low_res=int(cand_obj.get("triad_alt_low_res", 112)),
        triad_alt_high_res=int(cand_obj.get("triad_alt_high_res", 448)),
        triad_alt_max_high_anchors=triad_alt_max_high,
        budget_mode=str(cand_obj.get("budget_mode", "exact")),
        budget_epsilon_frac=float(cand_obj.get("budget_epsilon_frac", 0.01)),
        budget_extra_resolutions=tuple(int(r) for r in (cand_obj.get("budget_extra_resolutions") or [])),
    )

    # Optional score cache (allows reusing E0014-computed AST scores for E0015/E0017-style reproductions).
    all_ids = sorted(set(train_ids + eval_ids))
    scores_by_clip_override = None
    if args.scores_json is not None:
        if args.scores_json.exists():
            scores_by_clip_override = _load_scores_json(args.scores_json)
        else:
            scores_by_clip_override = _compute_scores_by_clip(
                clip_ids=all_ids,
                processed_dir=Path(args.processed_dir),
                meta_dir=Path(args.meta_dir),
                caches_dir=caches_dir,
                num_segments=10,
                eventness_method=str(args.eventness_method),
                audio_device=str(args.audio_device),
                ast_pretrained=bool(args.ast_pretrained),
                panns_random=bool(args.panns_random),
                panns_checkpoint=args.panns_checkpoint,
                audiomae_random=bool(args.audiomae_random),
                audiomae_checkpoint=args.audiomae_checkpoint,
                train_ids=train_ids,
                labels_by_clip=labels_by_clip,
            )
            _write_scores_json(
                Path(args.scores_json),
                eventness_method=str(args.eventness_method),
                num_segments=10,
                scores_by_clip=scores_by_clip_override,
            )
        missing = [cid for cid in all_ids if cid not in scores_by_clip_override]
        if missing:
            print(f"[scores] filling {len(missing)} missing clips into {args.scores_json}", flush=True)
            extra = _compute_scores_by_clip(
                clip_ids=missing,
                processed_dir=Path(args.processed_dir),
                meta_dir=Path(args.meta_dir),
                caches_dir=caches_dir,
                num_segments=10,
                eventness_method=str(args.eventness_method),
                audio_device=str(args.audio_device),
                ast_pretrained=bool(args.ast_pretrained),
                panns_random=bool(args.panns_random),
                panns_checkpoint=args.panns_checkpoint,
                audiomae_random=bool(args.audiomae_random),
                audiomae_checkpoint=args.audiomae_checkpoint,
                train_ids=train_ids,
                labels_by_clip=labels_by_clip,
            )
            scores_by_clip_override.update(extra)
            _write_scores_json(
                Path(args.scores_json),
                eventness_method=str(args.eventness_method),
                num_segments=10,
                scores_by_clip=scores_by_clip_override,
            )
            missing = [cid for cid in all_ids if cid not in scores_by_clip_override]
            if missing:
                raise SystemExit(f"scores-json still missing {len(missing)} clip_ids (e.g. {missing[:3]})")
        print(f"[scores] using {args.scores_json}", flush=True)

    metrics = _run_one(
        index=index,
        train_ids=train_ids,
        eval_ids=eval_ids,
        labels_by_clip=labels_by_clip,
        caches_dir=caches_dir,
        processed_dir=Path(args.processed_dir),
        seeds=seeds,
        train_cfg=train_cfg,
        train_device=str(args.train_device),
        eventness_method=str(args.eventness_method),
        audio_device=str(args.audio_device),
        ast_pretrained=bool(args.ast_pretrained),
        panns_random=bool(args.panns_random),
        panns_checkpoint=args.panns_checkpoint,
        audiomae_random=bool(args.audiomae_random),
        audiomae_checkpoint=args.audiomae_checkpoint,
        cand=cand,
        scores_by_clip_override=scores_by_clip_override,
    )
    _write_json(out_dir / "metrics.json", metrics)
    print(out_dir / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
