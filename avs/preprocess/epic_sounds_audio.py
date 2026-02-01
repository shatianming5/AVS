from __future__ import annotations

import argparse
from pathlib import Path

from avs.preprocess.ave_extract import extract_wav


def extract_epic_sounds_audio(*, videos_dir: Path, out_audio_dir: Path, video_ids: list[str]) -> list[str]:
    """
    Extract untrimmed audio wavs for EPIC-SOUNDS-style processing.

    Expected input layout (common EPIC-KITCHENS download output):
      <videos_dir>/<video_id>.mp4

    Output:
      <out_audio_dir>/<video_id>.wav  (mono, 16kHz)
    """
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    done: list[str] = []
    for vid in video_ids:
        in_path = videos_dir / f"{vid}.mp4"
        if not in_path.exists():
            raise FileNotFoundError(f"missing video: {in_path}")
        out_path = out_audio_dir / f"{vid}.wav"
        extract_wav(in_path, out_path, sample_rate=16000)
        done.append(vid)
    return done


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract EPIC-SOUNDS untrimmed audio wavs from EPIC-KITCHENS videos.")
    p.add_argument("--videos-dir", type=Path, required=True, help="Dir containing <video_id>.mp4")
    p.add_argument("--out-audio-dir", type=Path, required=True, help="Output dir for <video_id>.wav")
    p.add_argument("--video-id", action="append", default=[], help="Video id (repeatable)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.video_id:
        raise SystemExit("at least one --video-id is required")
    extract_epic_sounds_audio(videos_dir=args.videos_dir, out_audio_dir=args.out_audio_dir, video_ids=args.video_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

