from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _extract_fps1_frames(
    video_path: Path,
    out_dir: Path,
    *,
    start_offset_sec: float = 0.5,
    max_frames: int | None = None,
    jpg_quality: int = 2,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Avoid rename collisions from previous runs.
    for p in out_dir.glob("*.jpg"):
        p.unlink()

    tmp_pattern = out_dir / "frame_%06d.jpg"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(float(start_offset_sec)),
        "-i",
        str(video_path),
        "-vf",
        "fps=1",
        "-q:v",
        str(int(jpg_quality)),
    ]
    if max_frames is not None:
        cmd += ["-frames:v", str(int(max_frames))]
    cmd.append(str(tmp_pattern))
    subprocess.run(cmd, check=True)  # noqa: S603,S607 - controlled args

    tmp = sorted(out_dir.glob("frame_*.jpg"))
    frames: list[Path] = []
    for i, src in enumerate(tmp):
        dst = out_dir / f"{i}.jpg"
        src.replace(dst)
        frames.append(dst)
    if not frames:
        raise RuntimeError(f"no frames extracted from: {video_path}")
    return frames


def extract_epic_sounds_frames(
    *,
    videos_dir: Path,
    out_frames_dir: Path,
    video_ids: list[str],
    start_offset_sec: float = 0.5,
    max_seconds: int | None = None,
    jpg_quality: int = 2,
) -> dict[str, int]:
    """
    Extract untrimmed per-second frames for EPIC-SOUNDS-style processing.

    Expected input layout:
      <videos_dir>/<video_id>.mp4

    Output:
      <out_frames_dir>/<video_id>/frames/{0..T-1}.jpg  (fps=1)
    """
    out_frames_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for vid in video_ids:
        in_path = videos_dir / f"{vid}.mp4"
        if not in_path.exists():
            raise FileNotFoundError(f"missing video: {in_path}")
        frames_dir = out_frames_dir / vid / "frames"
        frames = _extract_fps1_frames(
            in_path,
            frames_dir,
            start_offset_sec=float(start_offset_sec),
            max_frames=int(max_seconds) if max_seconds is not None else None,
            jpg_quality=int(jpg_quality),
        )
        counts[str(vid)] = int(len(frames))
    return counts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract EPIC-SOUNDS untrimmed fps=1 frames from EPIC-KITCHENS videos.")
    p.add_argument("--videos-dir", type=Path, required=True, help="Dir containing <video_id>.mp4")
    p.add_argument("--out-frames-dir", type=Path, required=True, help="Output root dir for <video_id>/frames/{t}.jpg")
    p.add_argument("--video-id", action="append", default=[], help="Video id (repeatable)")
    p.add_argument("--start-offset-sec", type=float, default=0.5)
    p.add_argument("--max-seconds", type=int, default=None, help="Optionally cap extracted duration (seconds/frames).")
    p.add_argument("--jpg-quality", type=int, default=2, help="ffmpeg -q:v (lower is higher quality)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.video_id:
        raise SystemExit("at least one --video-id is required")
    extract_epic_sounds_frames(
        videos_dir=args.videos_dir,
        out_frames_dir=args.out_frames_dir,
        video_ids=args.video_id,
        start_offset_sec=float(args.start_offset_sec),
        max_seconds=args.max_seconds,
        jpg_quality=int(args.jpg_quality),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

