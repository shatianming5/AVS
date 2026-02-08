from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from avs.datasets.layout import intentqa_paths


@dataclass(frozen=True)
class IntentQAItem:
    qid: str
    video_id: str
    question: str
    options: list[str]
    answer_idx: int
    qtype: str | None = None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing IntentQA split csv: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        out: list[dict[str, str]] = []
        for row in reader:
            if not row:
                continue
            out.append({str(k): str(v) for k, v in row.items()})
    return out


def load_intentqa_split(split: str) -> list[IntentQAItem]:
    """
    Load IntentQA items from local CSVs.

    Expected layout:
      - data/IntentQA/IntentQA/{train,val,test}.csv
      - data/IntentQA/IntentQA/videos/<video_id>.mp4
    """
    p = intentqa_paths()
    csv_path = p.split_csv_path(str(split))
    rows = _read_csv_rows(csv_path)

    out: list[IntentQAItem] = []
    for r in rows:
        vid = str(r.get("video_id", "")).strip()
        q = str(r.get("question", "")).strip()
        ans_raw = str(r.get("answer", "")).strip()
        qid = str(r.get("qid", "")).strip()
        qtype = str(r.get("type", "")).strip() or None

        opts = [str(r.get(f"a{i}", "")).strip() for i in range(5)]
        if any(o == "" for o in opts):
            raise ValueError(f"invalid IntentQA row: empty option in qid={qid} video_id={vid}")

        try:
            ans = int(ans_raw)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"invalid IntentQA answer={ans_raw!r} for qid={qid} video_id={vid}") from e
        if not (0 <= ans < 5):
            raise ValueError(f"invalid IntentQA answer idx={ans} for qid={qid} video_id={vid}")

        if not vid or not q:
            continue
        out.append(IntentQAItem(qid=qid, video_id=vid, question=q, options=opts, answer_idx=int(ans), qtype=qtype))
    return out


def intentqa_video_path(video_id: str) -> Path:
    p = intentqa_paths()
    return p.raw_video_path(str(video_id))

