from __future__ import annotations

import json
import shutil
from pathlib import Path


def materialize_selected_frames(*, frames_dir: Path, out_dir: Path, seconds: list[int]) -> list[Path]:
    """
    Copy selected per-second frames into a contiguous directory for downstream VLM usage.

    Input:  <frames_dir>/{sec}.jpg
    Output: <out_dir>/{i:04d}_t{sec}.jpg
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in out_dir.glob("*.jpg"):
        p.unlink()

    out_paths: list[Path] = []
    for i, sec in enumerate(seconds):
        src = frames_dir / f"{int(sec)}.jpg"
        if not src.exists():
            raise FileNotFoundError(f"missing frame: {src}")
        dst = out_dir / f"{i:04d}_t{int(sec)}.jpg"
        shutil.copy2(src, dst)
        out_paths.append(dst)
    return out_paths


def write_frame_manifest_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

