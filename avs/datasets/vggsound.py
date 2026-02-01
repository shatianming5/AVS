from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from avs.utils.download import download_url


VGG_SOUND_META_URL = "https://raw.githubusercontent.com/hche11/VGGSound/master/data/vggsound.csv"


def ensure_vggsound_meta(meta_dir: Path) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)
    download_url(VGG_SOUND_META_URL, meta_dir / "vggsound.csv")


@dataclass(frozen=True)
class VGGSoundClip:
    youtube_id: str
    start_sec: int
    label: str
    split: str


def _read_vggsound_csv(path: Path) -> list[VGGSoundClip]:
    """
    VGGSound official metadata is a headerless CSV with 4 columns:
    youtube_id,start_sec,label,split
    """
    clips: list[VGGSoundClip] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) != 4:
                raise ValueError(f"expected 4 columns in {path}, got {len(row)}: {row}")
            youtube_id, start_sec, label, split = row
            clips.append(
                VGGSoundClip(
                    youtube_id=str(youtube_id).strip(),
                    start_sec=int(str(start_sec).strip()),
                    label=str(label).strip(),
                    split=str(split).strip(),
                )
            )
    return clips


class VGGSoundIndex:
    def __init__(self, clips: list[VGGSoundClip]):
        self.clips = clips
        by_split: dict[str, list[VGGSoundClip]] = {}
        for c in clips:
            by_split.setdefault(c.split, []).append(c)
        self.by_split = by_split

    @classmethod
    def from_meta_dir(cls, meta_dir: Path) -> "VGGSoundIndex":
        clips = _read_vggsound_csv(meta_dir / "vggsound.csv")
        return cls(clips)

