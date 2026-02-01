from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

from avs.audio.ast_probe import ASTEventnessProbe, ASTProbeConfig
from avs.audio.eventness import eventness_energy_delta_per_second, eventness_energy_per_second, load_wav_mono, topk_anchors
from avs.datasets.epic_sounds import EpicSoundsIndex
from avs.datasets.layout import epic_sounds_paths
from avs.metrics.anchors import recall_at_k


@dataclass(frozen=True)
class EpicSoundsAnchorEvalClip:
    clip_id: str
    wav_path: Path
    gt_segments: list[int]
    num_segments: int | None = None


def _seconds_from_interval(start_sec: float, stop_sec: float) -> set[int]:
    """
    Convert a continuous interval into the set of integer seconds that overlap with it.
    """
    if stop_sec <= start_sec:
        return set()
    start = int(math.floor(start_sec))
    end = int(math.ceil(stop_sec))
    out: set[int] = set()
    for t in range(start, end):
        if (t + 1.0) > start_sec and float(t) < stop_sec:
            out.add(int(t))
    return out


def _infer_num_segments(wav_path: Path, *, max_seconds: int | None) -> int:
    audio, sr = load_wav_mono(wav_path)
    dur = int(audio.shape[0] // int(sr))
    if max_seconds is not None:
        dur = min(dur, int(max_seconds))
    return max(0, int(dur))


def evaluate_anchor_quality(
    clips: list[EpicSoundsAnchorEvalClip],
    *,
    method: str = "energy",
    k: int = 5,
    deltas: list[int] = [0, 1, 2],
    seed: int = 0,
    max_seconds: int | None = None,
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

    valid = [c for c in clips if c.gt_segments]

    ours_anchors_by_clip: dict[str, list[int]] = {}
    rand_anchors_by_clip: dict[str, list[int]] = {}
    num_segments_by_clip: dict[str, int] = {}

    for clip in valid:
        num_segments = int(clip.num_segments) if clip.num_segments is not None else _infer_num_segments(clip.wav_path, max_seconds=max_seconds)
        if num_segments <= 0:
            continue

        if method == "energy":
            audio, sr = load_wav_mono(clip.wav_path)
            audio = audio[: int(sr) * num_segments]
            scores = eventness_energy_per_second(audio, sr, num_segments=num_segments)
        elif method == "energy_delta":
            audio, sr = load_wav_mono(clip.wav_path)
            audio = audio[: int(sr) * num_segments]
            scores = eventness_energy_delta_per_second(audio, sr, num_segments=num_segments)
        elif method == "ast":
            assert ast_probe is not None
            scores = ast_probe.eventness_per_second(clip.wav_path, num_segments=num_segments)
        elif method == "panns":
            assert panns_probe is not None
            scores = panns_probe.eventness_per_second(clip.wav_path, num_segments=num_segments)
        elif method == "audiomae":
            assert audiomae_probe is not None
            scores = audiomae_probe.eventness_per_second(clip.wav_path, num_segments=num_segments)
        else:
            raise ValueError(f"unknown method: {method}")

        ours_anchors_by_clip[clip.clip_id] = topk_anchors([float(x) for x in scores], k=min(int(k), num_segments))
        rand_anchors_by_clip[clip.clip_id] = rng.sample(range(num_segments), k=min(int(k), num_segments))
        num_segments_by_clip[clip.clip_id] = num_segments

    per_delta: dict[int, dict[str, float]] = {}
    for delta in deltas:
        ours_recalls: list[float] = []
        rand_recalls: list[float] = []

        for clip in valid:
            if clip.clip_id not in num_segments_by_clip:
                continue
            num_segments = num_segments_by_clip[clip.clip_id]
            # Clip gt segments to our evaluated duration.
            gt = [int(x) for x in clip.gt_segments if 0 <= int(x) < num_segments]
            if not gt:
                continue

            ours = recall_at_k(gt, ours_anchors_by_clip[clip.clip_id], num_segments=num_segments, delta=int(delta)).recall
            rand = recall_at_k(gt, rand_anchors_by_clip[clip.clip_id], num_segments=num_segments, delta=int(delta)).recall
            ours_recalls.append(float(ours))
            rand_recalls.append(float(rand))

        per_delta[int(delta)] = {
            "ours_mean_recall": float(sum(ours_recalls) / max(1, len(ours_recalls))),
            "random_mean_recall": float(sum(rand_recalls) / max(1, len(rand_recalls))),
            "num_clips": int(len(ours_recalls)),
        }

    return {"method": method, "k": int(k), "deltas": [int(x) for x in deltas], "by_delta": per_delta}


def _load_custom_clips(jsonl_path: Path) -> list[EpicSoundsAnchorEvalClip]:
    clips: list[EpicSoundsAnchorEvalClip] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        clips.append(
            EpicSoundsAnchorEvalClip(
                clip_id=str(obj["clip_id"]),
                wav_path=Path(obj["wav_path"]),
                gt_segments=[int(x) for x in obj["gt_segments"]],
                num_segments=int(obj["num_segments"]) if "num_segments" in obj and obj["num_segments"] is not None else None,
            )
        )
    return clips


def _build_epic_sounds_clips(
    meta_dir: Path,
    *,
    audio_dir: Path,
    split: str,
    limit_videos: int | None,
    include_not_categorised: bool,
    max_seconds: int | None,
) -> list[EpicSoundsAnchorEvalClip]:
    index = EpicSoundsIndex.from_meta_dir(meta_dir)

    if split == "train":
        segs = index.train
    elif split == "val":
        segs = index.val
    elif split == "test":
        segs = index.test
    else:
        raise ValueError(f"unknown split: {split}")

    by_video: dict[str, set[int]] = {}
    for s in segs:
        by_video.setdefault(s.video_id, set()).update(_seconds_from_interval(s.start_sec, s.stop_sec))

    if include_not_categorised:
        for s in index.not_categorised:
            by_video.setdefault(s.video_id, set()).update(_seconds_from_interval(s.start_sec, s.stop_sec))

    video_ids = sorted(by_video.keys())
    if limit_videos is not None:
        video_ids = video_ids[: int(limit_videos)]

    clips: list[EpicSoundsAnchorEvalClip] = []
    for vid in video_ids:
        wav_path = audio_dir / f"{vid}.wav"
        if not wav_path.exists():
            continue
        gt = sorted(by_video.get(vid, set()))
        if max_seconds is not None:
            gt = [t for t in gt if t < int(max_seconds)]
        clips.append(EpicSoundsAnchorEvalClip(clip_id=vid, wav_path=wav_path, gt_segments=gt))
    return clips


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EPIC-SOUNDS anchor quality eval (Recall@K / Recall@K,Δ).")
    p.add_argument("--method", type=str, default="energy", choices=["energy", "energy_delta", "ast", "panns", "audiomae"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--deltas", type=str, default="0,1,2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-seconds", type=int, default=None, help="Optionally cap evaluated audio length (seconds).")
    p.add_argument("--out-dir", type=Path, default=Path("runs") / f"epic_sounds_anchors_{time.strftime('%Y%m%d-%H%M%S')}")

    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--clips-jsonl", type=Path, help="Custom clips JSONL with clip_id,wav_path,gt_segments,(optional)num_segments")
    src.add_argument("--epic-sounds", action="store_true", help="Use EPIC-SOUNDS annotations + local audio dir")

    p.add_argument("--meta-dir", type=Path, default=epic_sounds_paths().meta_dir)
    p.add_argument("--audio-dir", type=Path, default=epic_sounds_paths().root / "audio", help="Dir containing <video_id>.wav")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--limit-videos", type=int, default=None)
    p.add_argument("--include-not-categorised", action="store_true")

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
    elif args.epic_sounds:
        clips = _build_epic_sounds_clips(
            args.meta_dir,
            audio_dir=args.audio_dir,
            split=str(args.split),
            limit_videos=args.limit_videos,
            include_not_categorised=bool(args.include_not_categorised),
            max_seconds=args.max_seconds,
        )
    else:
        raise SystemExit("must pass --clips-jsonl or --epic-sounds")

    metrics = evaluate_anchor_quality(
        clips,
        method=str(args.method),
        k=int(args.k),
        deltas=deltas,
        seed=int(args.seed),
        max_seconds=args.max_seconds,
        audio_device=str(args.audio_device),
        ast_pretrained=bool(args.ast_pretrained),
        panns_random=bool(args.panns_random),
        panns_checkpoint=args.panns_checkpoint,
        audiomae_random=bool(args.audiomae_random),
        audiomae_checkpoint=args.audiomae_checkpoint,
    )

    payload = {
        "num_input_clips": len(clips),
        "metrics": metrics,
    }
    out_path = out_dir / "epic_sounds_anchor_eval.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
