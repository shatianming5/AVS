from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from shared.schemas import EvidencePointer, StoryboardItem, TextBlock
from shared.llm_client import load_llm_config, llm_required, require_llm_config
from shared.llm_schemas import LlmStoryboardOutput
from worker.context import BuildContext
from worker.pipeline.cache import load_stage_payload, save_stage_payload, sha256_text, stage_cache_path
from worker.pipeline.evidence import EvidenceBuilder
from worker.pipeline.llm_payload import blocks_for_llm
from worker.pipeline.llm_state import add_missing, add_total, mark_repair_used


_RE_CAPTION = re.compile(r"^(figure|fig\.?|table|tab\.?)\s*(\d+)", re.IGNORECASE)
_RE_FIGREF = re.compile(r"(fig\.?|figure|table|tab\.?)\s*(\d+)", re.IGNORECASE)

STORYBOARD_CACHE_VERSION = 2
STORYBOARD_PROMPT_VERSION = 1


def _figure_role(caption: str) -> str:
    c = caption.lower()
    if "architecture" in c or "overview" in c:
        return "Overview"
    if "pipeline" in c or "framework" in c:
        return "MethodPipeline"
    if "ablation" in c:
        return "Ablation"
    if "comparison" in c or "sota" in c or "state-of-the-art" in c:
        return "SOTAComparison"
    if "qualitative" in c or "visual" in c:
        return "Qualitative"
    if "results" in c or "performance" in c:
        return "Results"
    return "Unknown"


@dataclass(frozen=True)
class Caption:
    pdf_id: str
    page_index: int
    bbox: list[float] | None
    figure_kind: str
    figure_number: str
    text: str
    role: str
    block_id: str | None


def _extract_captions(blocks_by_pdf: dict[str, list[TextBlock]]) -> list[Caption]:
    out: list[Caption] = []
    for pdf_id, blocks in blocks_by_pdf.items():
        for b in blocks:
            m = _RE_CAPTION.match(b.text.strip())
            if not m:
                continue
            kind = m.group(1)
            num = m.group(2)
            cap = Caption(
                pdf_id=pdf_id,
                page_index=b.page_index,
                bbox=b.bbox,
                figure_kind=kind.lower(),
                figure_number=num,
                text=b.text.strip(),
                role=_figure_role(b.text),
                block_id=b.block_id,
            )
            out.append(cap)
    return out


def _intro_refs(intro_blocks_by_pdf: dict[str, list[TextBlock]]) -> dict[str, set[str]]:
    refs: dict[str, set[str]] = {}
    for pdf_id, blocks in intro_blocks_by_pdf.items():
        s: set[str] = set()
        for b in blocks:
            for _k, num in _RE_FIGREF.findall(b.text):
                s.add(str(num))
        refs[pdf_id] = s
    return refs


def build_storyboard_llm(
    *,
    ctx: BuildContext,
    blocks_by_pdf: dict[str, list[TextBlock]],
    intro_blocks_by_pdf: dict[str, list[TextBlock]],
) -> tuple[list[StoryboardItem], list[EvidencePointer], dict]:
    cfg = require_llm_config()
    refs = _intro_refs(intro_blocks_by_pdf)

    caption_blocks_by_pdf: dict[str, list[TextBlock]] = {}
    for pdf_id, blocks in blocks_by_pdf.items():
        caption_blocks_by_pdf[pdf_id] = [b for b in blocks if b.block_type == "caption"]

    payload = {
        "task": "build_storyboard",
        "pdfs": [
            {
                "pdf_id": pdf_id,
                "captions": blocks_for_llm(caption_blocks_by_pdf.get(pdf_id) or [], max_blocks=24, max_chars=900),
                "intro_figure_numbers": sorted(list(refs.get(pdf_id, set()))),
            }
            for pdf_id in ctx.pdf_ids
        ],
        "evidence_first": True,
    }

    system = (
        "You are a precise academic writing analyst.\n"
        "Goal: propose a figure storyboard (figure roles, placement, and caption formulas) based on mentor papers.\n"
        "Evidence-first: each storyboard item should cite supporting_block_ids that refer to caption blocks.\n"
        "Return JSON only. Do not include markdown.\n\n"
        "Output schema:\n"
        "{\n"
        '  "items": [\n'
        '    {"figure_role": "...", "recommended_position": "...", "caption_formula": "...", "supporting_block_ids": ["..."]}\n'
        "  ]\n"
        "}\n"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    from shared.llm_client import chat_completions_json

    try:
        out = chat_completions_json(cfg=cfg, messages=messages, schema_name="storyboard")
        parsed = LlmStoryboardOutput.model_validate(out)
    except Exception as e:  # noqa: BLE001
        mark_repair_used("storyboard")
        repair = {"role": "user", "content": f"Fix your output. Return ONLY a JSON object matching the schema. Error: {e}"}
        out = chat_completions_json(cfg=cfg, messages=messages + [repair], schema_name="storyboard")
        parsed = LlmStoryboardOutput.model_validate(out)

    eb = EvidenceBuilder(data_dir=ctx.data_dir)
    evidence: list[EvidencePointer] = []
    ev_cache: dict[str, str] = {}
    blocks_by_id: dict[str, TextBlock] = {}
    for pdf_id, blocks in caption_blocks_by_pdf.items():
        for b in blocks:
            blocks_by_id[b.block_id] = b

    def evidence_ids(block_ids: list[str], *, reason: str) -> list[str]:
        ids: list[str] = []
        missing_block_ids = 0
        for bid in block_ids or []:
            b = blocks_by_id.get(bid)
            if b is None:
                missing_block_ids += 1
                continue
            if bid in ev_cache:
                ids.append(ev_cache[bid])
            else:
                ev = eb.from_block(block=b, reason=reason, confidence=0.75, kind="caption")
                evidence.append(ev)
                ev_cache[bid] = ev.evidence_id
                ids.append(ev.evidence_id)
            if len(ids) >= 2:
                break
        if missing_block_ids:
            add_missing("storyboard_missing_block_ids", missing_block_ids)
        return ids

    items: list[StoryboardItem] = []
    role_counts: dict[str, int] = {}
    add_total("storyboard_items", len(parsed.items))
    missing_items = 0
    for it in parsed.items:
        role = str(it.figure_role)
        role_counts[role] = role_counts.get(role, 0) + 1
        ev = evidence_ids(it.supporting_block_ids, reason=f"Example caption for role={role}.")
        if not ev:
            missing_items += 1
        items.append(
            StoryboardItem(
                item_id=uuid.uuid4().hex,
                figure_role=role,
                recommended_position=str(it.recommended_position),
                caption_formula=str(it.caption_formula) if it.caption_formula is not None else None,
                supporting_evidence=ev,
            )
        )
    if missing_items:
        add_missing("storyboard_items", missing_items)

    caption_count_by_pdf = {pdf_id: len(caption_blocks_by_pdf.get(pdf_id) or []) for pdf_id in ctx.pdf_ids}
    info = {
        "caption_count_by_pdf": caption_count_by_pdf,
        "figure_refs_by_pdf": {k: sorted(list(v)) for k, v in refs.items()},
        "role_counts": role_counts,
    }
    return items, evidence, info


def build_storyboard(
    *,
    ctx: BuildContext,
    blocks_by_pdf: dict[str, list[TextBlock]],
    intro_blocks_by_pdf: dict[str, list[TextBlock]],
) -> tuple[list[StoryboardItem], list, dict]:
    cfg = load_llm_config()
    if llm_required() and cfg is None:
        raise RuntimeError("PAPER_SKILL_LLM_MODEL is required (PAPER_SKILL_REQUIRE_LLM=1).")
    if cfg is not None:
        return build_storyboard_llm(ctx=ctx, blocks_by_pdf=blocks_by_pdf, intro_blocks_by_pdf=intro_blocks_by_pdf)

    eb = EvidenceBuilder(data_dir=ctx.data_dir)
    captions = _extract_captions(blocks_by_pdf)
    refs = _intro_refs(intro_blocks_by_pdf)

    # Prefer captions referenced in intro, otherwise any caption.
    evidence: list = []

    caption_count_by_pdf: dict[str, int] = {}
    for c in captions:
        caption_count_by_pdf[c.pdf_id] = caption_count_by_pdf.get(c.pdf_id, 0) + 1

    # Role distribution from samples
    role_counts: dict[str, int] = {}
    for c in captions:
        if c.role == "Unknown":
            continue
        role_counts[c.role] = role_counts.get(c.role, 0) + 1

    roles_sorted = [k for k, _v in sorted(role_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    # Keep some reasonable defaults
    for must in ["Overview", "MethodPipeline", "Results"]:
        if must not in roles_sorted:
            roles_sorted.append(must)
    suggested_roles = roles_sorted[:6] if roles_sorted else ["Overview", "MethodPipeline", "Results"]
    items: list[StoryboardItem] = []
    early_par = 1 if any("1" in s for s in refs.values()) else 3

    for i, role in enumerate(suggested_roles, start=1):
        # pick a caption example, prefer ones referenced in intro
        candidates = [c for c in captions if c.role == role]
        cap = None
        if candidates:
            candidates.sort(
                key=lambda c: (
                    0 if (c.figure_number in refs.get(c.pdf_id, set())) else 1,
                    c.pdf_id,
                    c.page_index,
                )
            )
            cap = candidates[0]
        ev_ids: list[str] = []
        if cap is not None:
            # Find corresponding block for bbox/excerpt. Use original block if possible.
            block = None
            if cap.block_id is not None:
                for b in blocks_by_pdf[cap.pdf_id]:
                    if b.block_id == cap.block_id:
                        block = b
                        break
            if block is not None:
                ev = eb.from_block(block=block, reason=f"Example caption for role={role}.", confidence=0.75, kind="caption")
                evidence.append(ev)
                ev_ids.append(ev.evidence_id)
        formula = {
            "Overview": "Figure {N}. Overview of {SYSTEM/METHOD} highlighting {KEY IDEA}.",
            "MethodPipeline": "Figure {N}. Pipeline of {METHOD}, showing {STEPS/MODULES} and data flow.",
            "Results": "Figure {N}. Main results on {BENCHMARK}, comparing {BASELINES} and {METHOD}.",
            "Ablation": "Figure {N}. Ablation study showing the impact of {COMPONENTS} on {METRIC}.",
            "SOTAComparison": "Figure {N}. Comparison with state-of-the-art methods on {TASK}.",
            "Qualitative": "Figure {N}. Qualitative examples illustrating {BEHAVIOR/FAILURE MODES}.",
        }.get(role, "Figure {N}. {CAPTION}.")

        items.append(
            StoryboardItem(
                item_id=uuid.uuid4().hex,
                figure_role=role,
                recommended_position=f"After intro paragraph {min(early_par + i - 1, 5)}",
                caption_formula=formula,
                supporting_evidence=ev_ids,
            )
        )

    info = {
        "caption_count_by_pdf": caption_count_by_pdf,
        "figure_refs_by_pdf": {k: sorted(list(v)) for k, v in refs.items()},
        "role_counts": role_counts,
    }
    return items, evidence, info


def build_storyboard_cached(
    *,
    ctx: BuildContext,
    blocks_by_pdf: dict[str, list[TextBlock]],
    intro_blocks_by_pdf: dict[str, list[TextBlock]],
    cache_key: str,
) -> tuple[list[StoryboardItem], list[EvidencePointer], dict, bool]:
    # Fingerprint: captions + intro figure refs.
    cfg = load_llm_config()
    if llm_required() and cfg is None:
        raise RuntimeError("PAPER_SKILL_LLM_MODEL is required (PAPER_SKILL_REQUIRE_LLM=1).")
    mode = "llm" if cfg is not None else "rules"

    captions = _extract_captions(blocks_by_pdf)
    refs = _intro_refs(intro_blocks_by_pdf)
    cap_fp = [
        {"pdf_id": c.pdf_id, "page_index": c.page_index, "block_id": c.block_id, "text": c.text, "role": c.role}
        for c in captions
    ]
    refs_fp = {pdf_id: sorted(list(nums)) for pdf_id, nums in refs.items()}
    llm_fp = None
    if cfg is not None:
        llm_fp = {"model": cfg.model, "prompt_version": STORYBOARD_PROMPT_VERSION}
    input_hash = sha256_text(json.dumps({"mode": mode, "llm": llm_fp, "captions": cap_fp, "refs": refs_fp}, ensure_ascii=False, sort_keys=True))

    path = stage_cache_path(data_dir=ctx.data_dir, stage="storyboard", key=cache_key)
    payload = load_stage_payload(path=path, version=STORYBOARD_CACHE_VERSION, input_hash=input_hash)
    if isinstance(payload, dict):
        try:
            s_raw = payload.get("storyboard") or []
            e_raw = payload.get("evidence") or []
            info = payload.get("info") or {}
            storyboard = [StoryboardItem.model_validate(s) for s in s_raw]
            evidence = [EvidencePointer.model_validate(e) for e in e_raw]
            return storyboard, evidence, info, True
        except Exception:  # noqa: BLE001
            pass

    storyboard, evidence, info = build_storyboard(ctx=ctx, blocks_by_pdf=blocks_by_pdf, intro_blocks_by_pdf=intro_blocks_by_pdf)
    save_stage_payload(
        path=path,
        version=STORYBOARD_CACHE_VERSION,
        input_hash=input_hash,
        payload={
            "storyboard": [s.model_dump(mode="json") for s in storyboard],
            "evidence": [e.model_dump(mode="json") for e in evidence],
            "info": info,
        },
    )
    return storyboard, evidence, info, False
