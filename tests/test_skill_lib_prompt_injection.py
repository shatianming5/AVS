from __future__ import annotations

import json
from pathlib import Path

from shared.schemas import TextBlock


def test_moves_injects_skill_prior(tmp_path: Path, monkeypatch) -> None:
    skills = tmp_path / "skills.jsonl"
    skills.write_text(
        "\n".join(
            [
                json.dumps(
                    {"topic": "writing", "title": "Gap connector", "text": "Use However to introduce the gap.", "tags": ["however", "gap"]},
                    ensure_ascii=False,
                ),
                json.dumps(
                    {"topic": "writing", "title": "Roadmap", "text": "Use 'organized as follows' to preview sections.", "tags": ["organized", "section"]},
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PAPER_SKILL_SKILL_LIB_PATH", str(skills))
    monkeypatch.setenv("PAPER_SKILL_SKILL_LIB_MAX_ITEMS", "10")

    captured = {"system": None}

    from shared import llm_client

    def capture_chat_completions_json(*, cfg, messages, schema_name):  # noqa: ANN001
        captured["system"] = (messages[0] or {}).get("content")
        payload = json.loads((messages[-1] or {}).get("content") or "{}")
        blocks = payload.get("blocks") or []
        moves = [{"block_id": (b or {}).get("block_id"), "label": "Context", "confidence": 0.9} for b in blocks]
        return {"moves": moves}

    monkeypatch.setattr(llm_client, "chat_completions_json", capture_chat_completions_json)

    from worker.pipeline.moves import label_moves_llm

    blocks = [
        TextBlock(block_id="b0", pdf_id="p", page_index=1, text="Recent progress has attracted attention.", clean_text="Recent progress has attracted attention.", block_type="paragraph", section_path=["Introduction"])
    ]
    _ = label_moves_llm(intro_blocks=blocks, field_hint="NLP")
    assert isinstance(captured["system"], str)
    assert "Reference skills" in captured["system"]

