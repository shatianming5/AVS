from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from shared.schemas import EvidencePointer, TextBlock
from worker.pipeline.extract_blocks import get_page_size


def _norm_bbox(bbox: list[float], page_w: float, page_h: float) -> list[float]:
    x0, y0, x1, y1 = bbox
    if page_w <= 0 or page_h <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [x0 / page_w, y0 / page_h, x1 / page_w, y1 / page_h]


@dataclass(frozen=True)
class EvidenceBuilder:
    data_dir: Path

    def from_block(
        self,
        *,
        block: TextBlock,
        reason: str,
        confidence: float,
        kind: str = "other",
        excerpt_chars: int = 220,
    ) -> EvidencePointer:
        evidence_id = uuid.uuid4().hex
        bbox = block.bbox
        bbox_norm = None
        bbox_norm_list = None
        if bbox is not None:
            w, h = get_page_size(block.pdf_id, block.page_index, self.data_dir)
            bbox_norm = _norm_bbox(bbox, w, h)
        if block.bbox_list:
            w, h = get_page_size(block.pdf_id, block.page_index, self.data_dir)
            bbox_norm_list = [_norm_bbox(bb, w, h) for bb in block.bbox_list]
        excerpt = (block.text or "").strip().replace("\n", " ")
        if len(excerpt) > excerpt_chars:
            excerpt = excerpt[: excerpt_chars - 1] + "…"
        return EvidencePointer(
            evidence_id=evidence_id,
            pdf_id=block.pdf_id,
            page_index=block.page_index,
            bbox=bbox,
            bbox_norm=bbox_norm,
            bbox_norm_list=bbox_norm_list,
            block_id=block.block_id,
            excerpt=excerpt or None,
            reason=reason,
            kind=kind,  # type: ignore[arg-type]
            confidence=max(0.0, min(1.0, float(confidence))),
        )
