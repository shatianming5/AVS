from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from avs.datasets.layout import egoschema_paths


@dataclass(frozen=True)
class EgoSchemaItem:
    question_idx: str
    video_idx: str
    question: str
    options: list[str]
    answer_idx: int | None


def _parse_answer_idx(*, answer: object, options: list[str]) -> int | None:
    """
    Best-effort answer parsing across common EgoSchema formats.
    """
    if answer is None:
        return None
    s = str(answer).strip()
    if not s:
        return None

    # Common encodings: "0".."4"
    if s.isdigit():
        v = int(s)
        if 0 <= v < len(options):
            return int(v)

    # Common encodings: "A".."E"
    letters = ["A", "B", "C", "D", "E"]
    su = s.upper()
    if su in letters[: len(options)]:
        return int(letters.index(su))

    # Sometimes the answer is the option text.
    for i, opt in enumerate(options):
        if str(opt).strip() == s:
            return int(i)
    return None


def load_egoschema_split(*, config: str = "MC", split: str = "test", limit: int | None = None) -> list[EgoSchemaItem]:
    """
    Load EgoSchema from a local HF dataset clone under `data/hf_repos/egoschema/`.

    Notes:
      - The public dataset commonly provides only a `test` split.
      - Answer labels may be missing depending on the config/source.
    """
    p = egoschema_paths()
    repo_dir = Path(p.hf_repo_dir)
    if not repo_dir.exists():
        raise FileNotFoundError(f"missing EgoSchema HF repo clone: {repo_dir}")

    from datasets import load_dataset  # local import: heavy dependency

    ds = load_dataset(str(repo_dir), name=str(config), split=str(split))

    out: list[EgoSchemaItem] = []
    n = int(len(ds)) if hasattr(ds, "__len__") else None
    for i, row in enumerate(ds):
        if limit is not None and i >= int(limit):
            break

        qidx = str(row.get("question_idx", "")).strip()
        vidx = str(row.get("video_idx", "")).strip()
        q = str(row.get("question", "")).strip()
        opts = [str(x) for x in (row.get("option") or [])]
        ans = row.get("answer")
        ans_idx = _parse_answer_idx(answer=ans, options=opts)

        if not qidx:
            qidx = str(i)
        out.append(EgoSchemaItem(question_idx=qidx, video_idx=vidx, question=q, options=opts, answer_idx=ans_idx))

    if n is not None and limit is None and len(out) != int(n):
        raise AssertionError("bug: truncated EgoSchema unexpectedly")
    return out


def egoschema_video_path(video_idx: str) -> Path:
    p = egoschema_paths()
    return p.video_path(str(video_idx))

