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
    # If the official dataset split files are present, prefer them and avoid overwriting.
    # They are installed by `scripts/ave_install_official.sh` and are more consistent with
    # the official zip than the GitHub `*_order.h5` lists.
    has_official_splits = all((meta_dir / f"{s}Set.txt").exists() for s in ("train", "val", "test"))
    if has_official_splits and (meta_dir / "Annotations.txt").exists():
        return

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
        try:
            start_f = float(start)
            end_f = float(end)
        except Exception:
            # Official AVE zip includes a header row; skip any non-numeric start/end lines.
            continue
        clips.append(
            AVEClip(
                label=label.strip(),
                video_id=video_id.strip(),
                quality=quality.strip(),
                start_sec=float(start_f),
                end_sec=float(end_f),
            )
        )
    return clips


def _read_split_txt(path: Path) -> list[str]:
    """
    Read `<split>Set.txt` from the official AVE zip.

    File format: one `video_id` per line.
    """
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = str(line).strip()
        if not s:
            continue
        ids.append(s)
    # Stable unique: some environments may have accidental duplicates after manual edits.
    out: list[str] = []
    seen: set[str] = set()
    for x in ids:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


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
        # Preferred official format: trainSet/valSet/testSet contain per-split annotation rows
        # (same `label&video_id&quality&start&end` format as Annotations.txt).
        has_official_splits = all((meta_dir / f"{s}Set.txt").exists() for s in ("train", "val", "test"))
        if has_official_splits:
            train = _parse_annotations(meta_dir / "trainSet.txt")
            val = _parse_annotations(meta_dir / "valSet.txt")
            test = _parse_annotations(meta_dir / "testSet.txt")

            clips = train + val + test
            splits = {
                "train": list(range(0, len(train))),
                "val": list(range(len(train), len(train) + len(val))),
                "test": list(range(len(train) + len(val), len(train) + len(val) + len(test))),
            }
            return cls(clips=clips, splits=splits)

        # Backward-compatible GitHub meta format: Annotations.txt + *_order.h5 indices into raw annotation rows.
        annotations_path = meta_dir / "Annotations.txt"
        clips = _parse_annotations(annotations_path)
        splits = {split: _load_order_h5(meta_dir / f"{split}_order.h5") for split in ("train", "val", "test")}
        return cls(clips=clips, splits=splits)
