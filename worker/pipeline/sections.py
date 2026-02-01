from __future__ import annotations

import json
import re
from pathlib import Path

from shared.schemas import TextBlock
from worker.pipeline.cache import load_stage_payload, save_stage_payload, sha256_text, stage_cache_path


_RE_SECTION_HEADING = re.compile(r"^\s*(\d+)(?:\.\d+)*\s+(.+)$")

SECTIONS_CACHE_VERSION = 1


def assign_section_paths(*, blocks: list[TextBlock], toc: list[dict] | None) -> None:
    if toc:
        _assign_from_toc(blocks=blocks, toc=toc)
        return
    _assign_from_headings(blocks=blocks)


def _assign_from_toc(*, blocks: list[TextBlock], toc: list[dict]) -> None:
    entries = []
    for e in toc:
        try:
            page_index = int(e.get("page_index") or 0)
            level = int(e.get("level") or 0)
            title = str(e.get("title") or "").strip()
        except Exception:  # noqa: BLE001
            continue
        if page_index <= 0 or level <= 0 or not title:
            continue
        entries.append((page_index, level, title))
    entries.sort(key=lambda x: (x[0], x[1]))

    stack: list[str] = []
    snapshots: list[tuple[int, list[str]]] = []
    for page_index, level, title in entries:
        stack = stack[: max(0, level - 1)]
        stack.append(title)
        snapshots.append((page_index, list(stack)))

    if not snapshots:
        _assign_from_headings(blocks=blocks)
        return

    snap_i = 0
    current_path: list[str] = []
    for b in blocks:
        while snap_i < len(snapshots) and snapshots[snap_i][0] <= b.page_index:
            current_path = snapshots[snap_i][1]
            snap_i += 1
        b.section_path = list(current_path)


def _assign_from_headings(*, blocks: list[TextBlock]) -> None:
    current: list[str] = []
    for b in blocks:
        if b.block_type == "heading":
            t = re.sub(r"\s+", " ", b.text).strip()
            m = _RE_SECTION_HEADING.match(t)
            if m:
                title = m.group(2).strip()
                current = [title] if title else []
            elif t.isupper() and len(t) <= 80:
                current = [t.title()]
        b.section_path = list(current)


def assign_section_paths_cached(*, pdf_id: str, blocks: list[TextBlock], toc: list[dict] | None, data_dir: Path) -> bool:
    heading_fingerprint = [
        {"block_id": b.block_id, "page_index": b.page_index, "text": (b.text or "").strip()}
        for b in blocks
        if b.block_type == "heading"
    ]
    input_hash = sha256_text(json.dumps({"toc": toc, "headings": heading_fingerprint}, ensure_ascii=False, sort_keys=True))
    path = stage_cache_path(data_dir=data_dir, stage="sections", key=pdf_id)
    payload = load_stage_payload(path=path, version=SECTIONS_CACHE_VERSION, input_hash=input_hash)

    if isinstance(payload, dict):
        for b in blocks:
            sp = payload.get(b.block_id)
            b.section_path = list(sp) if isinstance(sp, list) else []
        return True

    assign_section_paths(blocks=blocks, toc=toc)
    mapping = {b.block_id: list(b.section_path or []) for b in blocks}
    save_stage_payload(path=path, version=SECTIONS_CACHE_VERSION, input_hash=input_hash, payload=mapping)
    return False
