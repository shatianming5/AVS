from __future__ import annotations

import re
from dataclasses import dataclass

from shared.schemas import Template, TextBlock


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\{[^}]+\}", " ", s)  # strip slots
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _char_ngrams(s: str, n: int = 5) -> set[str]:
    s = _normalize(s)
    if len(s) < n:
        return set()
    return {s[i : i + n] for i in range(0, len(s) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@dataclass(frozen=True)
class PlagiarismHit:
    template_id: str
    similarity: float
    pdf_id: str
    block_id: str


def check_templates_against_sources(
    templates: list[Template],
    intro_blocks_by_pdf: dict[str, list[TextBlock]],
    *,
    threshold: float = 0.35,
) -> tuple[float, list[PlagiarismHit]]:
    sources: list[tuple[str, TextBlock, set[str]]] = []
    for pdf_id, blocks in intro_blocks_by_pdf.items():
        for b in blocks:
            sources.append((pdf_id, b, _char_ngrams(b.text)))

    max_sim = 0.0
    hits: list[PlagiarismHit] = []

    for t in templates:
        t_grams = _char_ngrams(t.text_with_slots)
        best = (0.0, None, None)
        for pdf_id, b, b_grams in sources:
            sim = _jaccard(t_grams, b_grams)
            if sim > best[0]:
                best = (sim, pdf_id, b)
        sim, pdf_id, b = best
        max_sim = max(max_sim, sim)
        if sim >= threshold and pdf_id is not None and b is not None:
            hits.append(PlagiarismHit(template_id=t.template_id, similarity=float(sim), pdf_id=pdf_id, block_id=b.block_id))

    return float(max_sim), hits
