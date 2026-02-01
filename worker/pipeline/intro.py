from __future__ import annotations

import json
import re
from pathlib import Path

from shared.schemas import TextBlock
from worker.pipeline.cache import load_stage_payload, save_stage_payload, sha256_text, stage_cache_path


_RE_INTRO = re.compile(r"\bintroduction\b", re.IGNORECASE)
_RE_SECTION_HEADING = re.compile(r"^\s*(\d+)(?:\.\d+)*\s+([A-Za-z].*)$")

INTRO_CACHE_VERSION = 1


def _heading_text(block: TextBlock) -> str:
    return re.sub(r"\s+", " ", block.text).strip()


def _looks_like_top_heading(block: TextBlock) -> bool:
    if block.block_type != "heading":
        return False
    t = _heading_text(block)
    return bool(_RE_SECTION_HEADING.match(t)) or (len(t) <= 80 and t.isupper())


def _parse_section_number(heading: str) -> int | None:
    m = _RE_SECTION_HEADING.match(heading)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def locate_intro_blocks(pdf_id: str, blocks: list[TextBlock]) -> list[TextBlock]:
    return locate_intro_blocks_with_toc(pdf_id=pdf_id, blocks=blocks, toc=None)


def locate_intro_blocks_with_toc(*, pdf_id: str, blocks: list[TextBlock], toc: list[dict] | None) -> list[TextBlock]:
    # Prefer TOC if available.
    if toc:
        intro_entry = None
        for e in toc:
            title = str(e.get("title") or "")
            if _RE_INTRO.search(title):
                intro_entry = e
                break
        if intro_entry is not None:
            start = int(intro_entry.get("page_index") or 1)
            level = int(intro_entry.get("level") or 1)
            end = start + 5
            for e in toc:
                try:
                    p = int(e.get("page_index") or 0)
                    lv = int(e.get("level") or 99)
                except Exception:  # noqa: BLE001
                    continue
                if p > start and lv <= level:
                    end = max(start, p - 1)
                    break
            intro_blocks = [b for b in blocks if start <= b.page_index <= end and b.block_type == "paragraph"]
            if intro_blocks:
                for b in intro_blocks:
                    b.section_path = ["Introduction"]
                return intro_blocks[:30]

    # Find an "Introduction" heading
    intro_heading_idx: int | None = None
    intro_heading_text: str | None = None
    for i, b in enumerate(blocks):
        if b.block_type == "heading" and _RE_INTRO.search(b.text):
            intro_heading_idx = i
            intro_heading_text = _heading_text(b)
            break

    # Fallback: search all blocks for a line that looks like "1. Introduction"
    if intro_heading_idx is None:
        for i, b in enumerate(blocks):
            if _RE_INTRO.search(b.text) and _looks_like_top_heading(b):
                intro_heading_idx = i
                intro_heading_text = _heading_text(b)
                break

    # Last resort: take the first page's paragraphs (keeps demo usable).
    if intro_heading_idx is None:
        intro_blocks = [b for b in blocks if b.page_index == 1 and b.block_type == "paragraph"][:12]
        for b in intro_blocks:
            b.section_path = ["Introduction"]
        return intro_blocks

    intro_num = _parse_section_number(intro_heading_text or "")
    start_page = blocks[intro_heading_idx].page_index

    intro_blocks: list[TextBlock] = []
    for b in blocks[intro_heading_idx + 1 :]:
        if b.page_index < start_page:
            continue
        # Stop if we hit the next top-level section.
        if _looks_like_top_heading(b):
            hn = _parse_section_number(_heading_text(b))
            if intro_num is not None and hn is not None and hn > intro_num:
                break
            if intro_num is None and b.page_index > start_page:
                # Without a section number, assume next heading on later page ends intro.
                break

        if b.page_index > start_page + 5:
            break
        if b.block_type == "paragraph":
            b.section_path = ["Introduction"]
            intro_blocks.append(b)
            if len(intro_blocks) >= 30:
                break

    # If nothing captured, fallback to same-page paragraphs after the heading
    if not intro_blocks:
        for b in blocks[intro_heading_idx + 1 :]:
            if b.page_index != start_page:
                break
            if b.block_type == "paragraph":
                b.section_path = ["Introduction"]
                intro_blocks.append(b)

    return intro_blocks


def locate_intro_blocks_cached(
    *,
    pdf_id: str,
    blocks: list[TextBlock],
    toc: list[dict] | None,
    data_dir: Path,
) -> tuple[list[TextBlock], bool]:
    # Fingerprint only early pages to keep small and stable; intro heuristics rely on headings + early paragraphs.
    early = [
        {"block_id": b.block_id, "page_index": b.page_index, "block_type": b.block_type, "text": (b.text or "").strip()}
        for b in blocks
        if b.page_index <= 2 and b.block_type in {"heading", "paragraph"}
    ]
    input_hash = sha256_text(json.dumps({"toc": toc, "early": early}, ensure_ascii=False, sort_keys=True))
    path = stage_cache_path(data_dir=data_dir, stage="intro", key=pdf_id)
    payload = load_stage_payload(path=path, version=INTRO_CACHE_VERSION, input_hash=input_hash)
    if isinstance(payload, list) and all(isinstance(x, str) for x in payload):
        wanted = set(payload)
        intro_blocks = [b for b in blocks if b.block_id in wanted and b.block_type == "paragraph"]
        for b in intro_blocks:
            b.section_path = ["Introduction"]
        return intro_blocks, True

    intro_blocks = locate_intro_blocks_with_toc(pdf_id=pdf_id, blocks=blocks, toc=toc)
    save_stage_payload(path=path, version=INTRO_CACHE_VERSION, input_hash=input_hash, payload=[b.block_id for b in intro_blocks])
    return intro_blocks, False
