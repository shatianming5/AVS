from __future__ import annotations

from pathlib import Path

import pytest

from shared.schemas import RhetoricalMove, StyleFeatures, TextBlock
from worker.context import BuildContext


def test_blueprint_llm_downgrades_rule_without_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from shared import llm_client
    from worker.pipeline import evidence as evidence_mod
    from worker.pipeline.blueprint import build_intro_blueprint_llm

    # Avoid requiring extracted page size caches in this unit test.
    monkeypatch.setattr(evidence_mod, "get_page_size", lambda _pdf_id, _page_index, _data_dir: (600.0, 800.0))

    data_dir = tmp_path / "data"
    ctx = BuildContext(
        job_id="job1",
        pack_id="pack1",
        pdf_ids=["pdf1", "pdf2", "pdf3"],
        pack_name="Test",
        field_hint="NLP",
        target_venue_hint=None,
        language="English",
        data_dir=data_dir,
        owner_token="owner",
    )

    intro_blocks_by_pdf = {}
    moves_by_pdf = {}
    for pdf_id in ctx.pdf_ids:
        b1 = TextBlock(
            block_id=f"{pdf_id}:1:0",
            pdf_id=pdf_id,
            page_index=1,
            text="However, existing approaches still lack robustness.",
            clean_text="However, existing approaches still lack robustness.",
            bbox=[50, 170, 545, 220],
            block_type="paragraph",
            section_path=["Introduction"],
        )
        b2 = TextBlock(
            block_id=f"{pdf_id}:1:1",
            pdf_id=pdf_id,
            page_index=1,
            text="In this paper, we propose MethodX, which may improve stability.",
            clean_text="In this paper, we propose MethodX, which may improve stability.",
            bbox=[50, 230, 545, 280],
            block_type="paragraph",
            section_path=["Introduction"],
        )
        intro_blocks_by_pdf[pdf_id] = [b1, b2]
        moves_by_pdf[pdf_id] = [
            RhetoricalMove(move_id="m1", label="Gap", block_id=b1.block_id, confidence=0.9),
            RhetoricalMove(move_id="m2", label="Approach", block_id=b2.block_id, confidence=0.8),
        ]

    alignment = {"consensus_sequence": ["Context", "Gap", "Approach"]}
    style_by_pdf = {pdf_id: StyleFeatures() for pdf_id in ctx.pdf_ids}

    original = llm_client.chat_completions_json

    def fake_chat(*, cfg, messages, schema_name):  # noqa: ANN001
        if schema_name != "blueprint":
            return original(cfg=cfg, messages=messages, schema_name=schema_name)
        # Force one story rule to reference a missing block_id, so it gets downgraded.
        return {
            "paragraph_plan": [
                {"paragraph_index": 1, "label": "Gap", "description": "Gap paragraph.", "supporting_block_ids": [f"pdf1:1:0"]},
            ],
            "story_rules": [
                {"title": "Missing evidence rule", "description": "Should be downgraded.", "supporting_block_ids": ["does-not-exist"]},
            ],
            "claim_rules": [
                {"title": "Hedging", "description": "Use may/might.", "supporting_block_ids": [f"pdf1:1:1"]},
            ],
            "checklist": [],
        }

    monkeypatch.setattr(llm_client, "chat_completions_json", fake_chat)

    blueprint, evidence = build_intro_blueprint_llm(
        ctx=ctx,
        intro_blocks_by_pdf=intro_blocks_by_pdf,
        moves_by_pdf=moves_by_pdf,
        alignment=alignment,
        style_by_pdf=style_by_pdf,
    )

    assert evidence
    assert blueprint.claim_rules and blueprint.claim_rules[0].supporting_evidence
    assert any(r.title.startswith("[Suggestion] Missing evidence rule") for r in blueprint.checklist)

