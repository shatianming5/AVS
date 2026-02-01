from __future__ import annotations

import json
from pathlib import Path

from shared.schemas import RhetoricalMove, StyleFeatures, TextBlock
from worker.context import BuildContext


def test_blueprint_includes_rule_candidates(tmp_path: Path, monkeypatch) -> None:
    skills = tmp_path / "skills.jsonl"
    skills.write_text(
        "\n".join(
            [
                json.dumps(
                    {"topic": "writing", "title": "Intro: problem → gap", "text": "After defining the problem, use a contrastive connector to state the gap.", "tags": ["however", "gap"]},
                    ensure_ascii=False,
                ),
                json.dumps(
                    {"topic": "writing", "title": "Contributions list", "text": "State contributions as a short enumerated list.", "tags": ["contribution", "threefold"]},
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PAPER_SKILL_SKILL_LIB_PATH", str(skills))
    monkeypatch.setenv("PAPER_SKILL_SKILL_LIB_MAX_ITEMS", "10")

    captured: dict = {}

    from shared import llm_client

    def capture_chat_completions_json(*, cfg, messages, schema_name):  # noqa: ANN001
        payload = json.loads((messages[-1] or {}).get("content") or "{}")
        captured["payload"] = payload
        return {
            "paragraph_plan": [
                {"paragraph_index": 1, "label": "Context", "description": "Context paragraph.", "supporting_block_ids": [payload["pdfs"][0]["intro_blocks"][0]["block_id"]]}
            ],
            "story_rules": [{"title": "Rule A", "description": "Desc", "supporting_block_ids": [payload["pdfs"][0]["intro_blocks"][0]["block_id"]]}],
            "claim_rules": [{"title": "Rule B", "description": "Desc", "supporting_block_ids": [payload["pdfs"][0]["intro_blocks"][0]["block_id"]]}],
            "checklist": [{"title": "Check", "description": "Desc", "supporting_block_ids": []}],
        }

    monkeypatch.setattr(llm_client, "chat_completions_json", capture_chat_completions_json)

    ctx = BuildContext(
        job_id="j",
        pack_id="p",
        pdf_ids=["pdf1", "pdf2", "pdf3"],
        pack_name="Pack",
        field_hint="NLP",
        target_venue_hint=None,
        language="English",
        data_dir=tmp_path / "data",
        owner_token="tok",
    )
    intro_blocks_by_pdf = {
        pdf_id: [TextBlock(block_id=f"{pdf_id}:1:0", pdf_id=pdf_id, page_index=1, text="However, we propose X.", clean_text="However, we propose X.", block_type="paragraph", section_path=["Introduction"])]
        for pdf_id in ctx.pdf_ids
    }
    moves_by_pdf = {pdf_id: [RhetoricalMove(move_id="m0", label="Gap", block_id=f"{pdf_id}:1:0", confidence=0.9)] for pdf_id in ctx.pdf_ids}
    style_by_pdf = {pdf_id: StyleFeatures() for pdf_id in ctx.pdf_ids}

    from worker.pipeline.blueprint import build_intro_blueprint_llm

    _bp, _ev = build_intro_blueprint_llm(ctx=ctx, intro_blocks_by_pdf=intro_blocks_by_pdf, moves_by_pdf=moves_by_pdf, alignment={"consensus_sequence": ["Context"]}, style_by_pdf=style_by_pdf)

    payload = captured.get("payload") or {}
    assert isinstance(payload.get("rule_candidates"), list)
    assert len(payload["rule_candidates"]) >= 1
    assert isinstance(payload.get("rule_candidates_fingerprint"), str)

