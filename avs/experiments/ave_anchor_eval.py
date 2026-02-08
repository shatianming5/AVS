from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from avs.audio.ast_probe import ASTEventnessProbe, ASTProbeConfig
from avs.audio.eventness import (
    anchors_from_scores,
    compute_eventness_wav_energy,
    compute_eventness_wav_energy_delta,
    compute_eventness_wav_energy_stride_max,
)
from avs.datasets.ave import AVEIndex, ensure_ave_meta
from avs.datasets.layout import ave_paths
from avs.metrics.anchors import recall_at_k
from avs.utils.scores import AV_FUSED_SCORE_SCALE, fuse_max, fuse_prod, minmax_01, scale
from avs.utils.paths import data_dir


@dataclass(frozen=True)
class AnchorEvalClip:
    clip_id: str
    wav_path: Path
    gt_segments: list[int]
    frames_dir: Path | None = None


def _gt_segments_from_labels(segment_labels: list[int]) -> list[int]:
    return [i for i, lab in enumerate(segment_labels) if int(lab) != 0]


def evaluate_anchor_quality(
    clips: list[AnchorEvalClip],
    *,
    method: str = "energy",
    k: int = 2,
    deltas: list[int] = [0, 1, 2],
    seed: int = 0,
    anchor_shift: int = 0,
    anchor_std_threshold: float = 0.0,
    anchor_select: str = "topk",
    anchor_window: int = 3,
    anchor_smooth_window: int = 0,
    anchor_smooth_mode: str = "mean",
    anchor_nms_radius: int = 1,
    anchor_nms_strong_gap: float = 0.6,
    anchor_conf_metric: str | None = None,
    anchor_conf_threshold: float | None = None,
    audio_device: str = "cpu",
    ast_pretrained: bool = False,
    panns_random: bool = False,
    panns_checkpoint: Path | None = None,
    audiomae_random: bool = False,
    audiomae_checkpoint: Path | None = None,
) -> dict:
    rng = random.Random(seed)

    ast_probe = None
    if method == "ast":
        ast_probe = ASTEventnessProbe(ASTProbeConfig(pretrained=ast_pretrained, device=str(audio_device)))

    panns_probe = None
    if method == "panns":
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

    valid_clips = [c for c in clips if c.gt_segments]
    ours_anchors_by_clip: dict[str, list[int]] = {}
    rand_anchors_by_clip: dict[str, list[int]] = {}
    for clip in valid_clips:
        if method == "energy":
            ev = compute_eventness_wav_energy(clip.wav_path, num_segments=10)
            scores = ev.scores
        elif method == "energy_delta":
            ev = compute_eventness_wav_energy_delta(clip.wav_path, num_segments=10)
            scores = ev.scores
        elif method == "energy_stride_max":
            ev = compute_eventness_wav_energy_stride_max(clip.wav_path, num_segments=10, stride_s=0.2, win_s=0.4)
            scores = ev.scores
        elif method == "cheap_visual":
            from avs.vision.cheap_eventness import frame_diff_eventness, list_frames

            if clip.frames_dir is None:
                continue
            frames = list_frames(clip.frames_dir)
            if len(frames) != 10:
                continue
            scores = frame_diff_eventness(frames)
        elif method == "av_fused":
            from avs.vision.cheap_eventness import frame_diff_eventness, list_frames

            ev = compute_eventness_wav_energy_stride_max(clip.wav_path, num_segments=10, stride_s=0.2, win_s=0.4)
            a = minmax_01([float(x) for x in ev.scores])

            frames: list[Path] = []
            if clip.frames_dir is not None and clip.frames_dir.exists():
                frames = list_frames(clip.frames_dir)
            v = frame_diff_eventness(frames, size=32) if frames else []
            v = minmax_01([float(x) for x in v])
            scores = scale(fuse_max(a, v, num_segments=10), AV_FUSED_SCORE_SCALE)
        elif method == "av_fused_prod":
            from avs.vision.cheap_eventness import frame_diff_eventness, list_frames

            ev = compute_eventness_wav_energy_stride_max(clip.wav_path, num_segments=10, stride_s=0.2, win_s=0.4)
            a = minmax_01([float(x) for x in ev.scores])

            frames: list[Path] = []
            if clip.frames_dir is not None and clip.frames_dir.exists():
                frames = list_frames(clip.frames_dir)
            v = frame_diff_eventness(frames, size=32) if frames else []
            v = minmax_01([float(x) for x in v])
            scores = scale(fuse_prod(a, v, num_segments=10), AV_FUSED_SCORE_SCALE)
        elif method == "ast":
            assert ast_probe is not None
            scores = ast_probe.eventness_per_second(clip.wav_path, num_segments=10)
        elif method == "panns":
            assert panns_probe is not None
            scores = panns_probe.eventness_per_second(clip.wav_path, num_segments=10)
        elif method == "audiomae":
            assert audiomae_probe is not None
            scores = audiomae_probe.eventness_per_second(clip.wav_path, num_segments=10)
        else:
            raise ValueError(f"unknown method: {method}")

        ours_anchors_by_clip[clip.clip_id] = anchors_from_scores(
            [float(x) for x in scores],
            k=int(k),
            num_segments=10,
            shift=int(anchor_shift),
            std_threshold=float(anchor_std_threshold),
            select=str(anchor_select),
            anchor_window=int(anchor_window),
            smooth_window=int(anchor_smooth_window),
            smooth_mode=str(anchor_smooth_mode),
            nms_radius=int(anchor_nms_radius),
            nms_strong_gap=float(anchor_nms_strong_gap),
            conf_metric=str(anchor_conf_metric) if anchor_conf_metric is not None else None,
            conf_threshold=float(anchor_conf_threshold) if anchor_conf_threshold is not None else None,
        )
        rand_anchors_by_clip[clip.clip_id] = rng.sample(range(10), k=min(k, 10))

    per_delta: dict[int, dict[str, float]] = {}
    for delta in deltas:
        ours_recalls: list[float] = []
        rand_recalls: list[float] = []

        for clip in valid_clips:
            ours = recall_at_k(clip.gt_segments, ours_anchors_by_clip[clip.clip_id], num_segments=10, delta=delta).recall
            ours_recalls.append(float(ours))

            rand = recall_at_k(clip.gt_segments, rand_anchors_by_clip[clip.clip_id], num_segments=10, delta=delta).recall
            rand_recalls.append(float(rand))

        per_delta[int(delta)] = {
            "ours_mean_recall": float(sum(ours_recalls) / max(1, len(ours_recalls))),
            "random_mean_recall": float(sum(rand_recalls) / max(1, len(rand_recalls))),
            "num_clips": int(len(ours_recalls)),
        }

    return {"method": method, "k": int(k), "deltas": deltas, "by_delta": per_delta}


def _load_custom_clips(jsonl_path: Path) -> list[AnchorEvalClip]:
    clips: list[AnchorEvalClip] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        clips.append(
            AnchorEvalClip(
                clip_id=str(obj["clip_id"]),
                wav_path=Path(obj["wav_path"]),
                gt_segments=[int(x) for x in obj["gt_segments"]],
                frames_dir=Path(obj["frames_dir"]) if "frames_dir" in obj and obj["frames_dir"] is not None else None,
            )
        )
    return clips


def _build_ave_clips(meta_dir: Path, processed_dir: Path, split: str, limit: int | None) -> list[AnchorEvalClip]:
    ensure_ave_meta(meta_dir)
    index = AVEIndex.from_meta_dir(meta_dir)
    ids = index.splits[split]
    if limit is not None:
        ids = ids[:limit]

    clips: list[AnchorEvalClip] = []
    for idx in ids:
        clip = index.clips[int(idx)]
        wav_path = processed_dir / clip.video_id / "audio.wav"
        frames_dir = processed_dir / clip.video_id / "frames"
        if not wav_path.exists():
            continue
        seg_labels = index.segment_labels(clip)
        gt = _gt_segments_from_labels(seg_labels)
        clips.append(AnchorEvalClip(clip_id=clip.video_id, wav_path=wav_path, gt_segments=gt, frames_dir=frames_dir))
    return clips


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate anchor quality (Recall@K / Recall@K,Δ).")
    p.add_argument(
        "--method",
        type=str,
        default="energy",
        choices=[
            "energy",
            "energy_delta",
            "energy_stride_max",
            "cheap_visual",
            "av_fused",
            "av_fused_prod",
            "ast",
            "panns",
            "audiomae",
        ],
    )
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--deltas", type=str, default="0,1,2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--anchor-shift", type=int, default=0, help="Shift anchor indices by this many segments (A/V misalignment).")
    p.add_argument(
        "--anchor-std-threshold",
        type=float,
        default=0.0,
        help="If std(scores) < threshold, return [] (fallback). 0 disables.",
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
        help="Anchor confidence metric. If set, uses --anchor-conf-threshold to decide fallback to uniform.",
    )
    p.add_argument(
        "--anchor-conf-threshold",
        type=float,
        default=None,
        help="For --anchor-conf-metric: if confidence < threshold, fall back to uniform (return empty anchors).",
    )
    p.add_argument("--out-dir", type=Path, default=Path("runs") / f"anchors_{time.strftime('%Y%m%d-%H%M%S')}")

    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--clips-jsonl", type=Path, help="Custom clips JSONL with clip_id,wav_path,gt_segments")
    src.add_argument("--ave", action="store_true", help="Use AVE metadata + processed dir")

    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--meta-dir", type=Path, default=ave_paths().meta_dir)
    p.add_argument("--processed-dir", type=Path, default=ave_paths().processed_dir)
    p.add_argument("--audio-device", type=str, default="cpu", help="Device for audio probe inference (e.g., cuda:0).")
    p.add_argument("--ast-pretrained", action="store_true", help="Use pretrained AST weights (downloads from HF)")
    p.add_argument("--panns-checkpoint", type=Path, default=None, help="Path to PANNs Cnn14 checkpoint (.pth)")
    p.add_argument("--panns-random", action="store_true", help="Use random PANNs weights (no checkpoint; smoke/debug only)")
    p.add_argument("--audiomae-checkpoint", type=Path, default=None, help="Path to AudioMAE(-style) checkpoint (optional)")
    p.add_argument("--audiomae-random", action="store_true", help="Use random AudioMAE(-style) weights (no checkpoint; smoke/debug only)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    deltas = [int(x) for x in str(args.deltas).split(",") if str(x).strip()]

    if args.clips_jsonl:
        clips = _load_custom_clips(args.clips_jsonl)
    elif args.ave:
        clips = _build_ave_clips(args.meta_dir, args.processed_dir, args.split, args.limit)
    else:
        raise SystemExit("must pass --clips-jsonl or --ave")

    metrics = evaluate_anchor_quality(
        clips,
        method=args.method,
        k=args.k,
        deltas=deltas,
        seed=args.seed,
        anchor_shift=int(args.anchor_shift),
        anchor_std_threshold=float(args.anchor_std_threshold),
        anchor_select=str(args.anchor_select),
        anchor_window=int(args.anchor_window),
        anchor_smooth_window=int(args.anchor_smooth_window),
        anchor_smooth_mode=str(args.anchor_smooth_mode),
        anchor_nms_radius=int(args.anchor_nms_radius),
        anchor_nms_strong_gap=float(args.anchor_nms_strong_gap),
        anchor_conf_metric=str(args.anchor_conf_metric) if args.anchor_conf_metric is not None else None,
        anchor_conf_threshold=float(args.anchor_conf_threshold) if args.anchor_conf_threshold is not None else None,
        audio_device=str(args.audio_device),
        ast_pretrained=bool(args.ast_pretrained),
        panns_random=bool(args.panns_random),
        panns_checkpoint=args.panns_checkpoint,
        audiomae_random=bool(args.audiomae_random),
        audiomae_checkpoint=args.audiomae_checkpoint,
    )

    payload = {
        "data_dir": str(data_dir()),
        "num_input_clips": len(clips),
        "metrics": metrics,
    }
    out_path = out_dir / "anchors_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
