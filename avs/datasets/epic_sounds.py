from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from avs.utils.download import download_url


EPIC_SOUNDS_META_BASE_URL = "https://raw.githubusercontent.com/epic-kitchens/epic-sounds-annotations/master"


def ensure_epic_sounds_meta(meta_dir: Path) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)
    download_url(f"{EPIC_SOUNDS_META_BASE_URL}/EPIC_Sounds_train.csv", meta_dir / "EPIC_Sounds_train.csv")
    download_url(f"{EPIC_SOUNDS_META_BASE_URL}/EPIC_Sounds_validation.csv", meta_dir / "EPIC_Sounds_validation.csv")
    download_url(
        f"{EPIC_SOUNDS_META_BASE_URL}/EPIC_Sounds_recognition_test_timestamps.csv",
        meta_dir / "EPIC_Sounds_recognition_test_timestamps.csv",
    )
    download_url(f"{EPIC_SOUNDS_META_BASE_URL}/sound_events_not_categorised.csv", meta_dir / "sound_events_not_categorised.csv")


def _parse_timestamp_seconds(ts: str) -> float:
    """
    Parse EPIC-SOUNDS timestamps formatted as `HH:mm:ss.SSS` into seconds.
    """
    ts = str(ts).strip()
    parts = ts.split(":")
    if len(parts) != 3:
        raise ValueError(f"invalid timestamp: {ts!r}")
    hh = int(parts[0])
    mm = int(parts[1])
    ss = float(parts[2])
    return float(hh * 3600 + mm * 60) + ss


@dataclass(frozen=True)
class EpicSoundsSegment:
    annotation_id: str
    participant_id: str
    video_id: str
    start_timestamp: str
    stop_timestamp: str
    start_sec: float
    stop_sec: float
    start_sample: int
    stop_sample: int
    description: str | None
    label: str | None
    class_id: int | None
    split: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            rows.append({str(k): str(v) for k, v in row.items()})
    return rows


def _parse_segments(path: Path, *, split: str) -> list[EpicSoundsSegment]:
    rows = _read_csv(path)
    out: list[EpicSoundsSegment] = []
    for row in rows:
        annotation_id = row.get("annotation_id", "").strip()
        participant_id = row.get("participant_id", "").strip()
        video_id = row.get("video_id", "").strip()
        start_timestamp = row.get("start_timestamp", "").strip()
        stop_timestamp = row.get("stop_timestamp", "").strip()
        if not (annotation_id and participant_id and video_id and start_timestamp and stop_timestamp):
            raise ValueError(f"missing required fields in {path}: {row}")

        start_sample = int(row.get("start_sample", "0"))
        stop_sample = int(row.get("stop_sample", "0"))
        description = row.get("description")
        label = row.get("class")
        class_id = row.get("class_id")

        out.append(
            EpicSoundsSegment(
                annotation_id=annotation_id,
                participant_id=participant_id,
                video_id=video_id,
                start_timestamp=start_timestamp,
                stop_timestamp=stop_timestamp,
                start_sec=_parse_timestamp_seconds(start_timestamp),
                stop_sec=_parse_timestamp_seconds(stop_timestamp),
                start_sample=start_sample,
                stop_sample=stop_sample,
                description=str(description).strip() if description is not None and str(description).strip() else None,
                label=str(label).strip() if label is not None and str(label).strip() else None,
                class_id=int(class_id) if class_id is not None and str(class_id).strip() else None,
                split=str(split),
            )
        )
    return out


class EpicSoundsIndex:
    def __init__(
        self,
        *,
        train: list[EpicSoundsSegment],
        val: list[EpicSoundsSegment],
        test: list[EpicSoundsSegment],
        not_categorised: list[EpicSoundsSegment],
    ):
        self.train = train
        self.val = val
        self.test = test
        self.not_categorised = not_categorised

        by_video_id: dict[str, list[EpicSoundsSegment]] = {}
        for seg in [*train, *val, *test, *not_categorised]:
            by_video_id.setdefault(seg.video_id, []).append(seg)
        for vid in list(by_video_id.keys()):
            by_video_id[vid] = sorted(by_video_id[vid], key=lambda s: (s.start_sec, s.stop_sec, s.annotation_id))
        self.by_video_id = by_video_id

    def segments(self, video_id: str) -> list[EpicSoundsSegment]:
        return list(self.by_video_id.get(str(video_id), []))

    @classmethod
    def from_meta_dir(cls, meta_dir: Path) -> "EpicSoundsIndex":
        train = _parse_segments(meta_dir / "EPIC_Sounds_train.csv", split="train")
        val = _parse_segments(meta_dir / "EPIC_Sounds_validation.csv", split="val")
        test = _parse_segments(meta_dir / "EPIC_Sounds_recognition_test_timestamps.csv", split="test")
        nc = _parse_segments(meta_dir / "sound_events_not_categorised.csv", split="not_categorised")
        return cls(train=train, val=val, test=test, not_categorised=nc)

