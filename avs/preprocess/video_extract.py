from __future__ import annotations

import json
import subprocess
from pathlib import Path

from avs.utils.ffmpeg import ffmpeg_bin, ffprobe_bin


def probe_has_audio_stream(video_path: Path) -> bool:
    """
    Return True if the video has at least one audio stream, False otherwise.

    Best-effort: on ffprobe failure, return True and let ffmpeg surface the error later.
    """
    cmd = [
        ffprobe_bin(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True)  # noqa: S603,S607 - controlled args
    except Exception:
        return True
    return bool(str(out).strip())


def probe_duration_seconds(video_path: Path, *, default: float | None = None) -> float | None:
    """
    Return duration in seconds (float) or `default` if unknown.
    """
    cmd = [
        ffprobe_bin(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(video_path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True)  # noqa: S603,S607 - controlled args
        v = float(str(out).strip())
        if v > 0.0:
            return float(v)
    except Exception:
        return default
    return default


def extract_audio_wav(
    video_path: Path,
    out_wav: Path,
    *,
    sample_rate: int = 16000,
    duration_sec: float | None = None,
) -> Path:
    """
    Extract mono wav at `sample_rate` from `video_path`.

    If the video has no audio stream, produce a deterministic silent wav with the requested duration.
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    if not probe_has_audio_stream(video_path):
        if duration_sec is None:
            duration = probe_duration_seconds(video_path, default=10.0) or 10.0
        else:
            duration = float(duration_sec)
        cmd = [
            ffmpeg_bin(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=channel_layout=mono:sample_rate={int(sample_rate)}",
            "-t",
            str(float(duration)),
            str(out_wav),
        ]
        subprocess.run(cmd, check=True)  # noqa: S603,S607 - controlled args
        return out_wav

    cmd = [
        ffmpeg_bin(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
    ]
    if duration_sec is not None:
        cmd += ["-t", str(float(duration_sec))]
    cmd.append(str(out_wav))
    subprocess.run(cmd, check=True)  # noqa: S603,S607 - controlled args
    return out_wav


def extract_fps1_frames(
    video_path: Path,
    out_dir: Path,
    *,
    start_offset_sec: float = 0.5,
    max_seconds: int | None = None,
    jpg_quality: int = 2,
) -> list[Path]:
    """
    Extract fps=1 frames from `video_path` into:
      <out_dir>/{0..T-1}.jpg

    This is deterministic and matches the repo's existing "per-second" protocol.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Avoid collisions from previous runs.
    for p in out_dir.glob("*.jpg"):
        p.unlink()

    tmp_pattern = out_dir / "frame_%06d.jpg"
    cmd = [
        ffmpeg_bin(),
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
    if max_seconds is not None:
        cmd += ["-frames:v", str(int(max_seconds))]
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


def ensure_processed_fps1(
    *,
    video_path: Path,
    out_dir: Path,
    sample_rate: int = 16000,
    start_offset_sec: float = 0.5,
    max_seconds: int | None = None,
    force: bool = False,
) -> dict:
    """
    Cache helper: materialize `{out_dir}/audio.wav` and `{out_dir}/frames/{t}.jpg`.
    """
    out_dir = Path(out_dir)
    audio_path = out_dir / "audio.wav"
    frames_dir = out_dir / "frames"

    meta_path = out_dir / "meta.json"
    if (not force) and audio_path.exists() and frames_dir.exists() and any(frames_dir.glob("*.jpg")) and meta_path.exists():
        return {"ok": True, "cached": True, "out_dir": str(out_dir), "audio": str(audio_path), "frames_dir": str(frames_dir)}

    out_dir.mkdir(parents=True, exist_ok=True)
    extract_audio_wav(video_path, audio_path, sample_rate=int(sample_rate), duration_sec=float(max_seconds) if max_seconds is not None else None)
    frames = extract_fps1_frames(
        video_path,
        frames_dir,
        start_offset_sec=float(start_offset_sec),
        max_seconds=int(max_seconds) if max_seconds is not None else None,
    )

    meta = {
        "ok": True,
        "cached": False,
        "video_path": str(video_path),
        "out_dir": str(out_dir),
        "audio": str(audio_path),
        "frames_dir": str(frames_dir),
        "num_frames": int(len(frames)),
        "start_offset_sec": float(start_offset_sec),
        "max_seconds": int(max_seconds) if max_seconds is not None else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta
