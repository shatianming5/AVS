from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SkillItem(BaseModel):
    topic: str = ""
    title: str = ""
    text: str = ""
    tags: list[str] = Field(default_factory=list)
    source: str | None = None


@dataclass(frozen=True)
class SkillLib:
    path: Path
    items: list[SkillItem]
    fingerprint: str
    topics_count: dict[str, int]
    max_items: int


_CACHE: dict[tuple[str, int, int, int], SkillLib] = {}


def _normalize_skill_obj(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    topic = obj.get("topic") or obj.get("category") or obj.get("tag") or ""
    title = obj.get("title") or obj.get("name") or obj.get("id") or ""
    text = obj.get("text") or obj.get("content") or obj.get("body") or obj.get("description") or ""
    tags = obj.get("tags") or obj.get("labels") or []
    source = obj.get("source") or obj.get("url") or None
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    if not isinstance(tags, list):
        tags = []
    tags = [str(t) for t in tags if str(t).strip()]
    return {
        "topic": str(topic or "").strip(),
        "title": str(title or "").strip(),
        "text": str(text or "").strip(),
        "tags": tags,
        "source": (str(source).strip() if source else None),
    }


def load_skill_lib(*, path: Path, max_items: int) -> SkillLib:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Skill library not found: {p}")
    st = p.stat()
    key = (str(p.resolve()), int(max_items), int(st.st_size), int(st.st_mtime_ns))
    if key in _CACHE:
        return _CACHE[key]

    h = hashlib.sha256()
    items: list[SkillItem] = []
    topics_count: dict[str, int] = {}

    with p.open("rb") as f:
        for line in f:
            if len(items) >= max_items:
                break
            if not line.strip():
                continue
            h.update(line)
            try:
                obj = json.loads(line.decode("utf-8", errors="replace"))
            except Exception:  # noqa: BLE001
                continue
            norm = _normalize_skill_obj(obj)
            if not norm.get("text") and not norm.get("title"):
                continue
            try:
                it = SkillItem.model_validate(norm)
            except Exception:  # noqa: BLE001
                continue
            items.append(it)
            topic = (it.topic or "").strip() or "unknown"
            topics_count[topic] = topics_count.get(topic, 0) + 1

    lib = SkillLib(path=p, items=items, fingerprint=h.hexdigest(), topics_count=topics_count, max_items=int(max_items))
    _CACHE[key] = lib
    return lib


def load_skill_lib_from_env() -> SkillLib | None:
    raw = (os.environ.get("PAPER_SKILL_SKILL_LIB_PATH") or "").strip()
    if not raw:
        return None
    max_items = int((os.environ.get("PAPER_SKILL_SKILL_LIB_MAX_ITEMS") or "20000").strip() or "20000")
    return load_skill_lib(path=Path(raw), max_items=max(1, max_items))

