from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _stub_local_llm(monkeypatch):  # noqa: ANN001
    # Default test mode: require LLM, but stub out network calls.
    monkeypatch.setenv("PAPER_SKILL_REQUIRE_LLM", "1")
    monkeypatch.setenv("PAPER_SKILL_LLM_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("PAPER_SKILL_LLM_MODEL", "test-model")

    from shared import llm_client

    def fake_chat_completions_json(*, cfg, messages, schema_name):  # noqa: ANN001
        payload = {}
        try:
            payload = json.loads((messages[-1] or {}).get("content") or "{}")
        except Exception:  # noqa: BLE001
            payload = {}

        def pick_label(text: str) -> str:
            t = (text or "").lower()
            if "organized as follows" in t or "section 2" in t:
                return "Roadmap"
            if "contribution" in t or "threefold" in t:
                return "Contribution"
            if "in this paper" in t or "we propose" in t or "we present" in t:
                return "Approach"
            if "however" in t or "nevertheless" in t or "still lack" in t:
                return "Gap"
            if "we study" in t or "problem" in t or "task" in t:
                return "Problem"
            return "Context"

        if schema_name == "moves":
            blocks = payload.get("blocks") or []
            moves = []
            for b in blocks:
                bid = (b or {}).get("block_id")
                label = pick_label((b or {}).get("text") or "")
                moves.append({"block_id": bid, "label": label, "confidence": 0.85})
            return {"moves": moves}

        if schema_name == "blueprint":
            consensus = payload.get("consensus_sequence") or ["Context", "Problem", "Gap", "Approach", "Contribution", "Roadmap"]
            pdfs = payload.get("pdfs") or []
            all_blocks = []
            move_map = {}
            for p in pdfs:
                for b in (p.get("intro_blocks") or []):
                    all_blocks.append(b)
                for m in (p.get("moves") or []):
                    move_map[m.get("block_id")] = m.get("label")

            def first_block_for(label: str) -> str | None:
                for b in all_blocks:
                    bid = b.get("block_id")
                    if bid and move_map.get(bid) == label:
                        return bid
                return (all_blocks[0].get("block_id") if all_blocks else None)

            paragraph_plan = []
            for i, lab in enumerate(consensus, start=1):
                bid = first_block_for(lab)
                paragraph_plan.append(
                    {
                        "paragraph_index": i,
                        "label": lab,
                        "description": f"{lab} paragraph in mentor style.",
                        "supporting_block_ids": [bid] if bid else [],
                    }
                )

            gap_bid = None
            for b in all_blocks:
                if "however" in (b.get("text") or "").lower():
                    gap_bid = b.get("block_id")
                    break
            if not gap_bid:
                gap_bid = first_block_for("Gap")

            claim_bid = None
            for b in all_blocks:
                if " may " in f" {(b.get('text') or '').lower()} ":
                    claim_bid = b.get("block_id")
                    break
            if not claim_bid:
                claim_bid = first_block_for("Approach")

            story_rules = [
                {
                    "title": "Use a contrastive connector for the gap",
                    "description": "Introduce limitations with a contrastive connector like However.",
                    "supporting_block_ids": [gap_bid] if gap_bid else [],
                }
            ]
            claim_rules = [
                {
                    "title": "Use hedging for broad claims",
                    "description": "Use may/might/likely to avoid absolute claims.",
                    "supporting_block_ids": [claim_bid] if claim_bid else [],
                }
            ]
            checklist = [
                {"title": f"Include a {lab} paragraph", "description": f"Ensure a {lab} move is present.", "supporting_block_ids": []}
                for lab in consensus
            ]
            return {"paragraph_plan": paragraph_plan, "story_rules": story_rules, "claim_rules": claim_rules, "checklist": checklist}

        if schema_name == "storyboard":
            pdfs = payload.get("pdfs") or []
            caption_blocks = []
            for p in pdfs:
                caption_blocks.extend(p.get("captions") or [])

            cap_bid = (caption_blocks[0].get("block_id") if caption_blocks else None)
            items = [
                {"figure_role": "Overview", "recommended_position": "After intro paragraph 3", "caption_formula": "Figure {N}. Overview of {SYSTEM/METHOD}.", "supporting_block_ids": [cap_bid] if cap_bid else []},
                {"figure_role": "MethodPipeline", "recommended_position": "After intro paragraph 4", "caption_formula": "Figure {N}. Pipeline of {METHOD}.", "supporting_block_ids": [cap_bid] if cap_bid else []},
                {"figure_role": "Results", "recommended_position": "After intro paragraph 5", "caption_formula": "Figure {N}. Main results on {BENCHMARK}.", "supporting_block_ids": [cap_bid] if cap_bid else []},
            ]
            return {"items": items}

        raise AssertionError(f"Unexpected schema_name in fake LLM: {schema_name}")

    monkeypatch.setattr(llm_client, "chat_completions_json", fake_chat_completions_json)
    return None
