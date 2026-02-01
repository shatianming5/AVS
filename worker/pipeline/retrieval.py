from __future__ import annotations

import re
from dataclasses import dataclass

from shared.schemas import RhetoricalMove, TextBlock


@dataclass(frozen=True)
class EvidenceCandidate:
    pdf_id: str
    block: TextBlock
    score: float


def select_blocks_for_labels(
    *,
    intro_blocks_by_pdf: dict[str, list[TextBlock]],
    moves_by_pdf: dict[str, list[RhetoricalMove]],
    labels: list[str],
    keyword_pattern: re.Pattern | None,
    limit_total: int,
    limit_per_pdf: int = 1,
) -> list[TextBlock]:
    move_by_block: dict[str, dict[str, str]] = {}
    for pdf_id, moves in moves_by_pdf.items():
        move_by_block[pdf_id] = {m.block_id: m.label for m in moves}

    candidates: list[EvidenceCandidate] = []
    label_set = set(labels)
    for pdf_id, blocks in intro_blocks_by_pdf.items():
        for idx, b in enumerate(blocks):
            lab = move_by_block.get(pdf_id, {}).get(b.block_id)
            if lab not in label_set:
                continue
            t = (b.clean_text or b.text or "").strip()
            score = 1.0
            if keyword_pattern is not None and keyword_pattern.search(t):
                score += 1.0
            # early/late preference for certain labels
            if lab in {"Context", "Problem"}:
                score += max(0.0, 0.4 - (idx * 0.05))
            if lab in {"Roadmap", "Contribution"}:
                score += max(0.0, 0.4 - ((len(blocks) - 1 - idx) * 0.05))
            candidates.append(EvidenceCandidate(pdf_id=pdf_id, block=b, score=score))

    # pick best per pdf then global
    by_pdf: dict[str, list[EvidenceCandidate]] = {}
    for c in candidates:
        by_pdf.setdefault(c.pdf_id, []).append(c)
    picked: list[EvidenceCandidate] = []
    for pdf_id, cs in by_pdf.items():
        cs.sort(key=lambda x: (-x.score, x.block.page_index, x.block.block_id))
        picked.extend(cs[:limit_per_pdf])
    picked.sort(key=lambda x: (-x.score, x.pdf_id, x.block.page_index, x.block.block_id))
    return [p.block for p in picked[:limit_total]]

