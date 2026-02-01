from __future__ import annotations

import json
import os
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def _now() -> int:
    return int(time.time())


def _pick_label(text: str) -> str:
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


def _wrap(content_obj: dict) -> JSONResponse:
    return JSONResponse(
        {
            "id": "chatcmpl_fake",
            "object": "chat.completion",
            "created": _now(),
            "model": "fake-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(content_obj, ensure_ascii=False)}}],
        }
    )


app = FastAPI(title="fake_llm_server")


@app.get("/v1/models")
async def models() -> dict:
    # Minimal OpenAI-compatible shape.
    return {"object": "list", "data": [{"id": "fake-model", "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    body: dict[str, Any] = await request.json()
    messages = body.get("messages") or []
    user = (messages[-1] or {}).get("content") if isinstance(messages, list) and messages else ""
    try:
        payload = json.loads(user) if isinstance(user, str) else {}
    except Exception:  # noqa: BLE001
        payload = {}

    task = payload.get("task")

    if task == "label_moves":
        blocks = payload.get("blocks") or []
        moves = []
        for b in blocks:
            bid = (b or {}).get("block_id")
            label = _pick_label((b or {}).get("text") or "")
            moves.append({"block_id": bid, "label": label, "confidence": 0.85})
        return _wrap({"moves": moves})

    if task == "build_intro_blueprint":
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
        return _wrap({"paragraph_plan": paragraph_plan, "story_rules": story_rules, "claim_rules": claim_rules, "checklist": checklist})

    if task == "build_storyboard":
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
        return _wrap({"items": items})

    return _wrap({"error": f"Unknown task: {task}"})


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("FAKE_LLM_PORT", "18001"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

