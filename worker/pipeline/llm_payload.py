from __future__ import annotations

from shared.schemas import TextBlock


def blocks_for_llm(
    blocks: list[TextBlock],
    *,
    max_blocks: int,
    max_chars: int,
) -> list[dict]:
    out: list[dict] = []
    for b in blocks:
        if len(out) >= max_blocks:
            break
        text = (b.clean_text or b.text or "").strip().replace("\n", " ")
        if max_chars > 0 and len(text) > max_chars:
            text = text[: max_chars - 1] + "…"
        out.append(
            {
                "block_id": b.block_id,
                "pdf_id": b.pdf_id,
                "page_index": int(b.page_index),
                "block_type": b.block_type,
                "text": text,
            }
        )
    return out

