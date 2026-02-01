from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py

from avs.utils.download import download_url


AVE_META_BASE_URL = "https://raw.githubusercontent.com/YapengTian/AVE-ECCV18/master/data"


@dataclass(frozen=True)
class AVEClip:
    label: str
    video_id: str
    quality: str
    start_sec: float
    end_sec: float


def ensure_ave_meta(meta_dir: Path) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)
    download_url(f"{AVE_META_BASE_URL}/Annotations.txt", meta_dir / "Annotations.txt")
    for split in ("train", "val", "test"):
        download_url(f"{AVE_META_BASE_URL}/{split}_order.h5", meta_dir / f"{split}_order.h5")


def _load_order_h5(path: Path) -> list[int]:
    with h5py.File(path, "r") as f:
        if "order" not in f:
            raise KeyError(f"missing 'order' dataset in {path}")
        order = f["order"][:].tolist()
    return [int(x) for x in order]


def _parse_annotations(path: Path) -> list[AVEClip]:
    clips: list[AVEClip] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("&")
        if len(parts) != 5:
            raise ValueError(f"unexpected annotation line format: {line!r}")
        label, video_id, quality, start, end = parts
        clips.append(
            AVEClip(
                label=label.strip(),
                video_id=video_id.strip(),
                quality=quality.strip(),
                start_sec=float(start),
                end_sec=float(end),
            )
        )
    return clips


class AVEIndex:
    def __init__(self, clips: list[AVEClip], splits: dict[str, list[int]]):
        self.clips = clips
        self.splits = splits

        event_labels = sorted({c.label for c in clips})
        self.label_to_idx = {"background": 0, **{lab: i + 1 for i, lab in enumerate(event_labels)}}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

    @property
    def num_classes(self) -> int:
        return len(self.label_to_idx)

    def segment_labels(self, clip: AVEClip, *, num_segments: int = 10) -> list[int]:
        start = float(clip.start_sec)
        end = float(clip.end_sec)
        labels: list[int] = []
        for t in range(num_segments):
            seg_start = float(t)
            seg_end = float(t + 1)
            overlap = max(0.0, min(seg_end, end) - max(seg_start, start))
            labels.append(self.label_to_idx[clip.label] if overlap > 0 else self.label_to_idx["background"])
        return labels

    @classmethod
    def from_meta_dir(cls, meta_dir: Path) -> "AVEIndex":
        annotations_path = meta_dir / "Annotations.txt"
        clips = _parse_annotations(annotations_path)
        splits = {split: _load_order_h5(meta_dir / f"{split}_order.h5") for split in ("train", "val", "test")}
        return cls(clips=clips, splits=splits)

