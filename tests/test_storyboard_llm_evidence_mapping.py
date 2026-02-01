from __future__ import annotations

from pathlib import Path

import pytest

from shared.schemas import TextBlock
from worker.context import BuildContext


def test_storyboard_llm_maps_caption_bbox_list_to_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from worker.pipeline import evidence as evidence_mod
    from worker.pipeline.captions import build_storyboard_llm

    monkeypatch.setattr(evidence_mod, "get_page_size", lambda _pdf_id, _page_index, _data_dir: (600.0, 800.0))

    data_dir = tmp_path / "data"
    ctx = BuildContext(
        job_id="job1",
        pack_id="pack1",
        pdf_ids=["pdf1", "pdf2", "pdf3"],
        pack_name="Test",
        field_hint=None,
        target_venue_hint=None,
        language="English",
        data_dir=data_dir,
        owner_token="owner",
    )

    caption = TextBlock(
        block_id="pdf1:1:cap0",
        pdf_id="pdf1",
        page_index=1,
        text="Figure 1: Overview of the proposed pipeline. Continued caption line.",
        clean_text="Figure 1: Overview of the proposed pipeline. Continued caption line.",
        bbox=[50, 380, 246.2, 418.7],
        bbox_list=[[50, 380, 246.2, 400.0], [50, 401.0, 230.0, 418.7]],
        block_type="caption",
        section_path=[],
    )

    blocks_by_pdf = {"pdf1": [caption], "pdf2": [], "pdf3": []}
    intro_blocks_by_pdf = {"pdf1": [], "pdf2": [], "pdf3": []}

    storyboard, evidence, _info = build_storyboard_llm(ctx=ctx, blocks_by_pdf=blocks_by_pdf, intro_blocks_by_pdf=intro_blocks_by_pdf)

    assert storyboard
    assert evidence
    assert any(ev.bbox_norm_list and len(ev.bbox_norm_list) == 2 for ev in evidence)
    assert any(item.supporting_evidence for item in storyboard)

