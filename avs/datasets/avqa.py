from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from avs.utils.download import download_url


AVQA_META_BASE_URL = "https://raw.githubusercontent.com/AlyssaYoung/AVQA/main/data/annotation"


def ensure_avqa_meta(meta_dir: Path) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)
    download_url(f"{AVQA_META_BASE_URL}/train_qa.json", meta_dir / "train_qa.json")
    download_url(f"{AVQA_META_BASE_URL}/val_qa.json", meta_dir / "val_qa.json")


@dataclass(frozen=True)
class AVQAItem:
    id: int
    video_name: str
    video_id: int
    question_text: str
    multi_choice: list[str]
    answer: int
    question_relation: str
    question_type: str


def _parse_items(path: Path) -> list[AVQAItem]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"expected list in {path}, got {type(raw).__name__}")

    items: list[AVQAItem] = []
    for obj in raw:
        if not isinstance(obj, dict):
            raise ValueError(f"expected dict item in {path}, got {type(obj).__name__}")
        items.append(
            AVQAItem(
                id=int(obj["id"]),
                video_name=str(obj["video_name"]),
                video_id=int(obj["video_id"]),
                question_text=str(obj["question_text"]),
                multi_choice=[str(x) for x in obj["multi_choice"]],
                answer=int(obj["answer"]),
                question_relation=str(obj["question_relation"]),
                question_type=str(obj["question_type"]),
            )
        )
    return items


class AVQAIndex:
    def __init__(self, *, train: list[AVQAItem], val: list[AVQAItem]):
        self.train = train
        self.val = val

    @classmethod
    def from_meta_dir(cls, meta_dir: Path) -> "AVQAIndex":
        train = _parse_items(meta_dir / "train_qa.json")
        val = _parse_items(meta_dir / "val_qa.json")
        return cls(train=train, val=val)

