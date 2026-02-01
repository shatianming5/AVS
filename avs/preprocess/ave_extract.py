from __future__ import annotations

import subprocess
import shutil
from pathlib import Path


def _probe_has_audio_stream(video_path: Path) -> bool:
    cmd = [
        "ffprobe",
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
        # Best-effort: if probing fails, assume audio exists and let ffmpeg surface the error.
        return True
    return bool(str(out).strip())


def _probe_duration_seconds(video_path: Path, *, default: float = 10.0) -> float:
    cmd = [
        "ffprobe",
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
        return float(default)
    return float(default)


def extract_wav(
    video_path: Path,
    out_wav: Path,
    *,
    sample_rate: int = 16000,
    duration_sec: float | None = 10.0,
) -> Path:
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    if not _probe_has_audio_stream(video_path):
        # Some YouTube downloads may not contain an audio stream. For AVE-style pipelines, we
        # fall back to a silent wav to keep the processing deterministic and avoid hard failures.
        duration = float(duration_sec) if duration_sec is not None else _probe_duration_seconds(video_path, default=10.0)
        cmd = [
            "ffmpeg",
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
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-t",
        str(float(duration_sec)) if duration_sec is not None else str(float(_probe_duration_seconds(video_path, default=10.0))),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)  # noqa: S603,S607 - controlled args
    return out_wav


def extract_middle_frames(
    video_path: Path,
    out_dir: Path,
    *,
    num_frames: int = 10,
    start_offset_sec: float = 0.5,
    jpg_quality: int = 2,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_pattern = out_dir / "frame_%02d.jpg"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(start_offset_sec),
        "-i",
        str(video_path),
        "-vf",
        "fps=1",
        "-frames:v",
        str(num_frames),
        "-q:v",
        str(jpg_quality),
        str(tmp_pattern),
    ]
    subprocess.run(cmd, check=True)  # noqa: S603,S607 - controlled args

    # Some videos have slightly-shorter video streams than their container duration, which can
    # result in <num_frames> outputs. Pad missing tail frames by repeating the last available.
    for i in range(1, int(num_frames) + 1):
        src = out_dir / f"frame_{i:02d}.jpg"
        if src.exists():
            continue
        if i == 1:
            raise FileNotFoundError(f"expected frame not found: {src}")
        prev = out_dir / f"frame_{i-1:02d}.jpg"
        if not prev.exists():
            raise FileNotFoundError(f"expected frame not found: {src} (and cannot pad; missing {prev})")
        shutil.copy2(prev, src)

    frames: list[Path] = []
    for i in range(num_frames):
        src = out_dir / f"frame_{i+1:02d}.jpg"
        dst = out_dir / f"{i}.jpg"
        if not src.exists():
            raise FileNotFoundError(f"expected frame not found: {src}")
        src.replace(dst)
        frames.append(dst)
    return frames


def preprocess_one(video_path: Path, out_root: Path, *, clip_id: str) -> dict[str, Path | list[Path]]:
    clip_root = out_root / clip_id
    audio_path = extract_wav(video_path, clip_root / "audio.wav")
    frames = extract_middle_frames(video_path, clip_root / "frames")
    return {"audio": audio_path, "frames": frames}
