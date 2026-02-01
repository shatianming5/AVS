from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from avs.datasets.layout import epic_sounds_paths
from avs.pipeline.long_plan_apply import materialize_selected_frames, write_frame_manifest_jsonl
from avs.pipeline.long_plan_generation import long_plan_from_wav_hybrid
from avs.preprocess.epic_sounds_audio import extract_epic_sounds_audio
from avs.preprocess.epic_sounds_frames import extract_epic_sounds_frames
from avs.vision.clip_vit import ClipVisionEncoder, ClipVisionEncoderConfig
from avs.vision.feature_cache import build_clip_feature_cache_from_seconds


def run_epic_sounds_long_pack(
    *,
    videos_dir: Path,
    out_dir: Path,
    video_ids: list[str],
    max_seconds: int | None,
    max_steps: int,
    method: str,
    k: int,
    anchor_radius: int,
    background_stride: int,
    encode: bool,
    clip_pretrained: bool,
    clip_model_name: str,
    clip_device: str,
    clip_dtype: str,
    start_offset_sec: float,
    jpg_quality: int,
    anchor_shift: int,
    anchor_std_threshold: float,
    ast_pretrained: bool,
    panns_random: bool,
    panns_checkpoint: Path | None,
    audiomae_random: bool,
    audiomae_checkpoint: Path | None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = out_dir / "audio"
    frames_root = out_dir / "frames"
    plans_dir = out_dir / "plans"
    selected_frames_root = out_dir / "selected_frames"
    caches_dir = out_dir / "caches"
    manifest_path = out_dir / "manifest.jsonl"

    audio_dir.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)
    plans_dir.mkdir(parents=True, exist_ok=True)
    selected_frames_root.mkdir(parents=True, exist_ok=True)
    caches_dir.mkdir(parents=True, exist_ok=True)

    done_audio = extract_epic_sounds_audio(videos_dir=videos_dir, out_audio_dir=audio_dir, video_ids=video_ids)
    counts_frames = extract_epic_sounds_frames(
        videos_dir=videos_dir,
        out_frames_dir=frames_root,
        video_ids=video_ids,
        start_offset_sec=float(start_offset_sec),
        max_seconds=int(max_seconds) if max_seconds is not None else None,
        jpg_quality=int(jpg_quality),
    )

    encoder = None
    if encode:
        encoder = ClipVisionEncoder(
            ClipVisionEncoderConfig(
                model_name=str(clip_model_name),
                device=str(clip_device),
                dtype=str(clip_dtype),
                pretrained=bool(clip_pretrained),
            )
        )

    manifest: list[dict] = []
    per_video: dict[str, dict] = {}

    for vid in video_ids:
        wav_path = audio_dir / f"{vid}.wav"
        frames_dir = frames_root / vid / "frames"

        record = long_plan_from_wav_hybrid(
            clip_id=str(vid),
            wav_path=wav_path,
            max_seconds=int(max_seconds) if max_seconds is not None else None,
            eventness_method=str(method),
            k=int(k),
            anchor_radius=int(anchor_radius),
            background_stride=int(background_stride),
            max_steps=int(max_steps),
            anchor_shift=int(anchor_shift),
            anchor_std_threshold=float(anchor_std_threshold),
            ast_pretrained=bool(ast_pretrained),
            panns_random=bool(panns_random),
            panns_checkpoint=panns_checkpoint,
            audiomae_random=bool(audiomae_random),
            audiomae_checkpoint=audiomae_checkpoint,
        )

        plan_path = plans_dir / f"{vid}.long_plan.json"
        record.save_json(plan_path)

        out_selected_dir = selected_frames_root / vid
        materialize_selected_frames(frames_dir=frames_dir, out_dir=out_selected_dir, seconds=record.selected_seconds)

        cache_path = caches_dir / f"{vid}.npz"
        if encode:
            assert encoder is not None
            cache = build_clip_feature_cache_from_seconds(
                frames_dir=frames_dir,
                seconds=record.selected_seconds,
                resolutions=[112, 224, 448],
                encoder=encoder,
            )
            cache.save_npz(cache_path)
        else:
            cache_path = None

        rec = {
            "video_id": str(vid),
            "wav_path": str(wav_path),
            "frames_dir": str(frames_dir),
            "selected_seconds": [int(x) for x in record.selected_seconds],
            "resolutions": [int(x) for x in record.plan.resolutions],
            "anchors_seconds": [int(x) for x in record.anchors_seconds],
            "plan_total_tokens": int(record.plan.total_tokens()),
            "long_plan_path": str(plan_path),
            "selected_frames_dir": str(out_selected_dir),
            "feature_cache_path": str(cache_path) if cache_path is not None else None,
        }
        manifest.append(rec)
        per_video[str(vid)] = {
            "duration_seconds": int(record.duration_seconds),
            "num_selected": int(len(record.selected_seconds)),
            "num_frames_extracted": int(counts_frames.get(vid, 0)),
        }

    write_frame_manifest_jsonl(manifest_path, manifest)
    payload = {
        "ok": True,
        "videos_dir": str(videos_dir),
        "out_dir": str(out_dir),
        "video_ids": [str(x) for x in video_ids],
        "done_audio": [str(x) for x in done_audio],
        "counts_frames": {str(k): int(v) for k, v in counts_frames.items()},
        "manifest_path": str(manifest_path),
        "encode": bool(encode),
        "method": str(method),
        "k": int(k),
        "anchor_radius": int(anchor_radius),
        "background_stride": int(background_stride),
        "max_seconds": int(max_seconds) if max_seconds is not None else None,
        "max_steps": int(max_steps),
        "per_video": per_video,
    }
    (out_dir / "epic_sounds_long_pack.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EPIC-SOUNDS long-video pack: audio+frames → hybrid plan → manifest(+cache).")

    default_videos_dir = epic_sounds_paths().raw_videos_dir
    default_out_dir = Path("runs") / f"epic_sounds_long_pack_{time.strftime('%Y%m%d-%H%M%S')}"

    p.add_argument("--videos-dir", type=Path, default=default_videos_dir, help="Dir containing <video_id>.mp4")
    p.add_argument("--out-dir", type=Path, default=default_out_dir)
    p.add_argument("--video-id", action="append", default=[], help="Video id (repeatable)")

    p.add_argument("--max-seconds", type=int, default=None, help="Optionally cap extracted duration (seconds).")
    p.add_argument("--max-steps", type=int, default=120, help="Fixed number of selected steps (seconds).")

    p.add_argument("--method", type=str, default="energy", choices=["energy", "energy_delta", "ast", "panns", "audiomae"])
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--anchor-radius", type=int, default=2)
    p.add_argument("--background-stride", type=int, default=5)

    p.add_argument("--anchor-shift", type=int, default=0)
    p.add_argument("--anchor-std-threshold", type=float, default=0.0)

    p.add_argument("--encode", action=argparse.BooleanOptionalAction, default=True, help="Build feature caches (npz).")
    p.add_argument("--clip-pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--clip-model-name", type=str, default="openai/clip-vit-base-patch16")
    p.add_argument("--clip-device", type=str, default="cpu")
    p.add_argument("--clip-dtype", type=str, default="float32")

    p.add_argument("--start-offset-sec", type=float, default=0.5)
    p.add_argument("--jpg-quality", type=int, default=2, help="ffmpeg -q:v (lower is higher quality)")

    p.add_argument("--ast-pretrained", action="store_true", help="Use pretrained AST weights (downloads from HF)")
    p.add_argument("--panns-checkpoint", type=Path, default=None, help="Path to PANNs Cnn14 checkpoint (.pth)")
    p.add_argument("--panns-random", action="store_true", help="Use random PANNs weights (no checkpoint; smoke/debug only)")
    p.add_argument("--audiomae-checkpoint", type=Path, default=None, help="Path to AudioMAE(-style) checkpoint (optional)")
    p.add_argument("--audiomae-random", action="store_true", help="Use random AudioMAE(-style) weights (no checkpoint; smoke/debug only)")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.video_id:
        raise SystemExit("at least one --video-id is required")

    run_epic_sounds_long_pack(
        videos_dir=args.videos_dir,
        out_dir=args.out_dir,
        video_ids=[str(x) for x in args.video_id],
        max_seconds=args.max_seconds,
        max_steps=args.max_steps,
        method=args.method,
        k=args.k,
        anchor_radius=args.anchor_radius,
        background_stride=args.background_stride,
        encode=bool(args.encode),
        clip_pretrained=bool(args.clip_pretrained),
        clip_model_name=str(args.clip_model_name),
        clip_device=str(args.clip_device),
        clip_dtype=str(args.clip_dtype),
        start_offset_sec=float(args.start_offset_sec),
        jpg_quality=int(args.jpg_quality),
        anchor_shift=int(args.anchor_shift),
        anchor_std_threshold=float(args.anchor_std_threshold),
        ast_pretrained=bool(args.ast_pretrained),
        panns_random=bool(args.panns_random),
        panns_checkpoint=args.panns_checkpoint,
        audiomae_random=bool(args.audiomae_random),
        audiomae_checkpoint=args.audiomae_checkpoint,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

