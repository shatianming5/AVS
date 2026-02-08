from __future__ import annotations

import os
import shutil
from pathlib import Path

from avs.utils.paths import repo_root


def _candidate_bin_dirs() -> list[Path]:
    # 1) Explicit override (dir that contains ffmpeg/ffprobe)
    d = os.environ.get("AVS_FFMPEG_DIR")
    if d:
        return [Path(d)]

    # 2) Repo-local install location (gitignored)
    # Layout:
    #   data/tools/ffmpeg/bin/ffmpeg
    #   data/tools/ffmpeg/bin/ffprobe
    return [repo_root() / "data" / "tools" / "ffmpeg" / "bin"]


def ensure_ffmpeg_in_path() -> None:
    """
    Best-effort: if ffmpeg is not on PATH but a repo-local bundle exists, prepend it to PATH.

    This makes all subprocess calls that use "ffmpeg"/"ffprobe" work without requiring sudo/apt.
    """
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return

    for bin_dir in _candidate_bin_dirs():
        ffmpeg = bin_dir / "ffmpeg"
        ffprobe = bin_dir / "ffprobe"
        if ffmpeg.exists() and ffprobe.exists():
            old = os.environ.get("PATH", "")
            os.environ["PATH"] = str(bin_dir) + (os.pathsep + old if old else "")
            return


def ffmpeg_bin() -> str:
    ensure_ffmpeg_in_path()
    return shutil.which("ffmpeg") or "ffmpeg"


def ffprobe_bin() -> str:
    ensure_ffmpeg_in_path()
    return shutil.which("ffprobe") or "ffprobe"

