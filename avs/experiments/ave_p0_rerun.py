from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from avs.datasets.ave import AVEIndex, ensure_ave_meta
from avs.experiments.ave_p0 import P0Config, run_p0_from_caches
from avs.train.train_loop import TrainConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Re-run AVE-P0 using an existing end-to-end run's caches/audio/ids.")
    p.add_argument("--base-metrics", type=Path, required=True, help="Path to ave_p0_end2end metrics.json")
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
        choices=["topk", "nms", "nms_strong"],
        help="Anchor selection strategy on per-second eventness scores.",
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
        "--anchor-base-alloc",
        type=str,
        default="distance",
        choices=["distance", "score", "farthest", "mixed"],
        help="How to allocate base-res segments in the equal-budget anchored plan.",
    )
    p.add_argument(
        "--anchor-high-policy",
        type=str,
        default="fixed",
        choices=["fixed", "adaptive_v1"],
        help="How many anchors get high-res allocation.",
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
    p.add_argument("--out-dir", type=Path, default=Path("runs") / f"AVE_P0_RERUN_{time.strftime('%Y%m%d-%H%M%S')}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base = json.loads(args.base_metrics.read_text(encoding="utf-8"))

    caches_dir = Path(base["caches_dir"])
    processed_dir = Path(base["processed_dir"])
    meta_dir = Path(base["meta_dir"])
    train_ids = [str(x) for x in base["train_ids"]]
    eval_ids = [str(x) for x in base["eval_ids"]]

    ensure_ave_meta(meta_dir)
    index = AVEIndex.from_meta_dir(meta_dir)
    clip_by_id = {c.video_id: c for c in index.clips}

    labels_by_clip: dict[str, list[int]] = {}
    for cid in sorted(set(train_ids + eval_ids)):
        clip = clip_by_id[cid]
        labels_by_clip[cid] = [int(x) for x in index.segment_labels(clip)]

    seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip()]
    baselines = ["uniform", "uniform_low", "audio_concat_uniform", "random_top2", "anchored_top2", "oracle_top2"]
    metrics = run_p0_from_caches(
        clip_ids_train=train_ids,
        clip_ids_eval=eval_ids,
        labels_by_clip=labels_by_clip,
        caches_dir=caches_dir,
        audio_dir=processed_dir,
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
        train_cfg=TrainConfig(epochs=5, batch_size=16, lr=2e-3),
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
