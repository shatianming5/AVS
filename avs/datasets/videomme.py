from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

from avs.datasets.layout import videomme_paths


@dataclass(frozen=True)
class VideoMMEItem:
    """
    Video-MME (2024) item for multiple-choice video QA.

    Source dataset (HF): `VLM2Vec/Video-MME` (split: `test`).

    Note: The HF split is named `test` but includes labels (`answer`) in the released version.
    """

    question_id: str
    video_id: str
    youtube_id: str
    url: str
    question: str
    options: list[str]
    answer: str
    answer_idx: int | None
    task_type: str
    duration: str
    domain: str
    sub_category: str

    def to_jsonable(self) -> dict:
        return {
            "question_id": str(self.question_id),
            "video_id": str(self.video_id),
            "youtube_id": str(self.youtube_id),
            "url": str(self.url),
            "question": str(self.question),
            "options": [str(x) for x in self.options],
            "answer": str(self.answer),
            "answer_idx": None if self.answer_idx is None else int(self.answer_idx),
            "task_type": str(self.task_type),
            "duration": str(self.duration),
            "domain": str(self.domain),
            "sub_category": str(self.sub_category),
        }


def _answer_idx(answer: str, *, num_options: int) -> int | None:
    """
    Convert answers like "A"/"B"/"C"/"D" to 0-based index.
    Returns None if parsing fails or if index is out of range.
    """
    s = str(answer).strip().upper()
    if not s:
        return None
    ch = s[0]
    if "A" <= ch <= "Z":
        idx = ord(ch) - ord("A")
        if 0 <= idx < int(num_options):
            return int(idx)
    return None


def _row_to_item(row: dict) -> VideoMMEItem:
    options = [str(x) for x in (row.get("options") or [])]
    ans = str(row.get("answer") or "").strip()
    return VideoMMEItem(
        question_id=str(row.get("question_id") or "").strip(),
        video_id=str(row.get("video_id") or "").strip(),
        youtube_id=str(row.get("videoID") or row.get("videoId") or row.get("youtube_id") or "").strip(),
        url=str(row.get("url") or "").strip(),
        question=str(row.get("question") or "").strip(),
        options=options,
        answer=ans,
        answer_idx=_answer_idx(ans, num_options=len(options)),
        task_type=str(row.get("task_type") or "").strip(),
        duration=str(row.get("duration") or "").strip(),
        domain=str(row.get("domain") or "").strip(),
        sub_category=str(row.get("sub_category") or "").strip(),
    )


def ensure_videomme_meta(meta_dir: Path | None = None, *, split: str = "test", force: bool = False) -> Path:
    """
    Materialize a JSONL snapshot under `data/VideoMME/meta/` for reproducibility.
    """
    p = videomme_paths()
    meta_dir = Path(meta_dir) if meta_dir is not None else p.meta_dir
    meta_dir.mkdir(parents=True, exist_ok=True)

    out = meta_dir / f"{str(split)}.jsonl"
    info = meta_dir / "dataset_info.json"
    if out.exists() and out.stat().st_size > 0 and (not force):
        return out

    ds = load_dataset("VLM2Vec/Video-MME", split=str(split))
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            item = _row_to_item(dict(row))
            f.write(json.dumps(item.to_jsonable(), ensure_ascii=False) + "\n")

    payload = {
        "repo_id": "VLM2Vec/Video-MME",
        "split": str(split),
        "num_rows": int(len(ds)),
        "generated_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "meta_jsonl": str(out),
    }
    info.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def load_videomme_split(
    *,
    split: str = "test",
    limit: int | None = None,
    seed: int = 0,
    order: str = "hash",
    meta_dir: Path | None = None,
) -> list[VideoMMEItem]:
    """
    Load a deterministic subset of Video-MME.

    Ordering:
    - `order="hash"`: stable pseudo-random order using sha1(seed|question_id).
    - `order="original"`: keep source JSONL order.
    """
    meta_path = ensure_videomme_meta(meta_dir=meta_dir, split=str(split))
    items: list[VideoMMEItem] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            items.append(
                VideoMMEItem(
                    question_id=str(row["question_id"]),
                    video_id=str(row["video_id"]),
                    youtube_id=str(row["youtube_id"]),
                    url=str(row["url"]),
                    question=str(row["question"]),
                    options=[str(x) for x in row["options"]],
                    answer=str(row["answer"]),
                    answer_idx=None if row.get("answer_idx") is None else int(row["answer_idx"]),
                    task_type=str(row.get("task_type", "")),
                    duration=str(row.get("duration", "")),
                    domain=str(row.get("domain", "")),
                    sub_category=str(row.get("sub_category", "")),
                )
            )

    if str(order) == "hash":
        def _key(it: VideoMMEItem) -> str:
            h = hashlib.sha1(f"{int(seed)}|{it.question_id}".encode("utf-8")).hexdigest()
            return h

        items = sorted(items, key=_key)
    elif str(order) == "original":
        pass
    else:
        raise ValueError(f"unknown order={order!r}; expected 'hash' or 'original'")

    if limit is not None and int(limit) > 0:
        items = items[: int(limit)]
    return items

