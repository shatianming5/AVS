from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from pathlib import Path

from shared.schemas import BlueprintRule, EvidencePointer, IntroBlueprint, RhetoricalMove, StyleFeatures, TextBlock
from shared.llm_client import load_llm_config, llm_required, require_llm_config
from shared.llm_schemas import LlmIntroBlueprintOutput
from worker.context import BuildContext
from worker.pipeline.evidence import EvidenceBuilder
from worker.pipeline.cache import load_stage_payload, save_stage_payload, sha256_text, stage_cache_path
from worker.pipeline.retrieval import select_blocks_for_labels
from worker.pipeline.llm_payload import blocks_for_llm
from worker.pipeline.llm_state import add_missing, add_total, mark_repair_used
from worker.pipeline.skill_hints import blueprint_rule_candidates, evidence_keyword_pattern, skill_lib_fingerprint


_LABEL_DESC = {
    "Context": "Set context and motivate the research area with recent progress and why it matters.",
    "Problem": "Define the core problem/task precisely and why it is important.",
    "Gap": "Explain what is missing in prior work; use a clear contrast and be specific.",
    "Approach": "Introduce the key idea/method at a high level (no deep details yet).",
    "Contribution": "State contributions clearly (often as a short enumerated list).",
    "Roadmap": "Provide a short paper roadmap (optional but common).",
    "RelatedWorkHook": "Briefly anchor the gap against prior work without turning into a survey.",
    "Claim": "Summarize the main takeaways with appropriate claim strength.",
    "Limitation": "Mention limitations or open problems (optional).",
    "Other": "Optional supportive paragraph (definitions, examples, etc.).",
}

BLUEPRINT_CACHE_VERSION = 2
BLUEPRINT_PROMPT_VERSION = 1


def _downgrade_rules_missing_evidence(rules: list[BlueprintRule]) -> tuple[list[BlueprintRule], list[BlueprintRule]]:
    kept: list[BlueprintRule] = []
    downgraded: list[BlueprintRule] = []
    for r in rules:
        if r.supporting_evidence:
            kept.append(r)
            continue
        downgraded.append(
            BlueprintRule(
                rule_id=r.rule_id,
                title=f"[Suggestion] {r.title}",
                description=f"{r.description}\n\n(No direct evidence found; treat as a suggestion.)",
                supporting_evidence=[],
            )
        )
    return kept, downgraded


def _fallback_evidence_rule(
    *,
    ctx: BuildContext,
    intro_blocks_by_pdf: dict[str, list[TextBlock]],
    moves_by_pdf: dict[str, list[RhetoricalMove]],
    style_by_pdf: dict[str, StyleFeatures],
    evidence_out: list[EvidencePointer],
    kind: str,
) -> tuple[BlueprintRule | None, BlueprintRule | None]:
    eb = EvidenceBuilder(data_dir=ctx.data_dir)

    # Story fallback: gap connector evidence
    connector_counter = Counter()
    for s in style_by_pdf.values():
        counts = (s.connector_profile or {}).get("counts") or {}
        for k, v in counts.items():
            if k in {"however", "nevertheless", "yet", "but"}:
                connector_counter[k] += int(v)
    best_connector = (connector_counter.most_common(1)[0][0] if connector_counter else "however").capitalize()
    gap_kw = re.compile(rf"(^|\n)\s*{re.escape(best_connector)}\b", re.IGNORECASE)
    gap_blocks = select_blocks_for_labels(
        intro_blocks_by_pdf=intro_blocks_by_pdf,
        moves_by_pdf=moves_by_pdf,
        labels=["Gap", "Problem", "Context"],
        keyword_pattern=gap_kw,
        limit_total=3,
        limit_per_pdf=1,
    )
    story_rule = None
    if gap_blocks:
        ev_ids: list[str] = []
        for b in gap_blocks:
            ev = eb.from_block(block=b, reason=f"Shows a contrastive connector ('{best_connector}') introducing the gap.", confidence=0.8, kind=kind)
            evidence_out.append(ev)
            ev_ids.append(ev.evidence_id)
        story_rule = BlueprintRule(
            rule_id=uuid.uuid4().hex,
            title=f"Introduce the gap with a contrastive connector (e.g., {best_connector})",
            description="When transitioning from prior progress to limitations, use a contrastive connector and state the gap explicitly.",
            supporting_evidence=ev_ids,
        )

    claim_blocks = select_blocks_for_labels(
        intro_blocks_by_pdf=intro_blocks_by_pdf,
        moves_by_pdf=moves_by_pdf,
        labels=["Contribution", "Claim", "Approach"],
        keyword_pattern=evidence_keyword_pattern(field_hint=ctx.field_hint),
        limit_total=3,
        limit_per_pdf=1,
    )
    claim_rule = None
    if claim_blocks:
        ev_ids = []
        for b in claim_blocks:
            ev = eb.from_block(block=b, reason="Example of claim/contribution phrasing in the mentor papers.", confidence=0.7, kind=kind)
            evidence_out.append(ev)
            ev_ids.append(ev.evidence_id)
        claim_rule = BlueprintRule(
            rule_id=uuid.uuid4().hex,
            title="Use hedging for broad claims",
            description="Broad claims are often phrased with hedging (may/might/likely) to avoid absolute statements while still being confident.",
            supporting_evidence=ev_ids,
        )

    return story_rule, claim_rule


def build_intro_blueprint_llm(
    *,
    ctx: BuildContext,
    intro_blocks_by_pdf: dict[str, list[TextBlock]],
    moves_by_pdf: dict[str, list[RhetoricalMove]],
    alignment: dict,
    style_by_pdf: dict[str, StyleFeatures],
) -> tuple[IntroBlueprint, list[EvidencePointer]]:
    cfg = require_llm_config()
    consensus: list[str] = list(alignment.get("consensus_sequence") or [])
    if not consensus:
        consensus = ["Context", "Problem", "Gap", "Approach", "Contribution", "Roadmap"]

    pdfs = []
    for pdf_id in ctx.pdf_ids:
        intro_blocks = intro_blocks_by_pdf.get(pdf_id) or []
        moves = moves_by_pdf.get(pdf_id) or []
        pdfs.append(
            {
                "pdf_id": pdf_id,
                "intro_blocks": blocks_for_llm(intro_blocks, max_blocks=36, max_chars=900),
                "moves": [{"block_id": m.block_id, "label": m.label, "confidence": float(m.confidence)} for m in moves],
            }
        )

    payload = {
        "task": "build_intro_blueprint",
        "consensus_sequence": consensus,
        "pdfs": pdfs,
        "style_summary": {pdf_id: style_by_pdf[pdf_id].model_dump(mode="json") for pdf_id in style_by_pdf},
        "evidence_first": True,
    }
    candidates = blueprint_rule_candidates(field_hint=ctx.field_hint)
    if candidates:
        payload["rule_candidates"] = candidates
        payload["rule_candidates_fingerprint"] = skill_lib_fingerprint()

    system = (
        "You are a precise academic writing analyst.\n"
        "Goal: produce an Introduction blueprint (paragraph plan + rules + checklist) for the mentor style.\n"
        "Evidence-first: every rule must cite supporting_block_ids from the provided intro_blocks.\n"
        "If rule_candidates are provided, prefer selecting/adapting from them instead of inventing new rules.\n"
        "Return JSON only. Do not include markdown.\n\n"
        "Output schema:\n"
        "{\n"
        '  "paragraph_plan": [\n'
        '    {"paragraph_index": 1, "label": "Context|Problem|Gap|Approach|Contribution|Roadmap|RelatedWorkHook|Claim|Limitation|Other", "description": "...", "supporting_block_ids": ["..."]}\n'
        "  ],\n"
        '  "story_rules": [{"title":"...","description":"...","supporting_block_ids":["..."]}],\n'
        '  "claim_rules": [{"title":"...","description":"...","supporting_block_ids":["..."]}],\n'
        '  "checklist": [{"title":"...","description":"...","supporting_block_ids":["..."]}]\n'
        "}\n"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    from shared.llm_client import chat_completions_json

    try:
        out = chat_completions_json(cfg=cfg, messages=messages, schema_name="blueprint")
        parsed = LlmIntroBlueprintOutput.model_validate(out)
    except Exception as e:  # noqa: BLE001
        mark_repair_used("blueprint")
        repair = {"role": "user", "content": f"Fix your output. Return ONLY a JSON object matching the schema. Error: {e}"}
        out = chat_completions_json(cfg=cfg, messages=messages + [repair], schema_name="blueprint")
        parsed = LlmIntroBlueprintOutput.model_validate(out)

    eb = EvidenceBuilder(data_dir=ctx.data_dir)
    evidence: list[EvidencePointer] = []
    ev_cache: dict[tuple[str, str, str], str] = {}
    blocks_by_id: dict[str, TextBlock] = {}
    for pdf_id, blocks in intro_blocks_by_pdf.items():
        for b in blocks:
            blocks_by_id[b.block_id] = b

    def evidence_ids(block_ids: list[str], *, reason: str, kind: str, confidence: float, limit: int = 3) -> list[str]:
        ids: list[str] = []
        missing_block_ids = 0
        for bid in block_ids or []:
            b = blocks_by_id.get(bid)
            if b is None:
                missing_block_ids += 1
                continue
            k = (bid, reason, kind)
            if k in ev_cache:
                ids.append(ev_cache[k])
            else:
                ev = eb.from_block(block=b, reason=reason, confidence=confidence, kind=kind)
                evidence.append(ev)
                ev_cache[k] = ev.evidence_id
                ids.append(ev.evidence_id)
            if len(ids) >= limit:
                break
        if missing_block_ids:
            add_missing("blueprint_missing_block_ids", missing_block_ids)
        return ids

    paragraph_plan: list[dict] = []
    for p in sorted(parsed.paragraph_plan, key=lambda x: int(x.paragraph_index)):
        paragraph_plan.append(
            {
                "paragraph_index": int(p.paragraph_index),
                "label": p.label,
                "description": str(p.description),
                "supporting_evidence": evidence_ids(
                    p.supporting_block_ids,
                    reason=f"Example {p.label} paragraph in the mentor papers.",
                    kind="intro_paragraph",
                    confidence=0.75,
                    limit=3,
                ),
            }
        )

    story_rules: list[BlueprintRule] = []
    story_claim_total = len(parsed.story_rules) + len(parsed.claim_rules)
    add_total("blueprint_rules", story_claim_total)
    missing_rules = 0
    for r in parsed.story_rules:
        ev = evidence_ids(
            r.supporting_block_ids,
            reason=f"Supports story rule: {r.title}",
            kind="intro_paragraph",
            confidence=0.7,
            limit=3,
        )
        if not ev:
            missing_rules += 1
        story_rules.append(
            BlueprintRule(
                rule_id=uuid.uuid4().hex,
                title=str(r.title),
                description=str(r.description),
                supporting_evidence=ev,
            )
        )

    claim_rules: list[BlueprintRule] = []
    for r in parsed.claim_rules:
        ev = evidence_ids(
            r.supporting_block_ids,
            reason=f"Supports claim rule: {r.title}",
            kind="intro_paragraph",
            confidence=0.7,
            limit=3,
        )
        if not ev:
            missing_rules += 1
        claim_rules.append(
            BlueprintRule(
                rule_id=uuid.uuid4().hex,
                title=str(r.title),
                description=str(r.description),
                supporting_evidence=ev,
            )
        )
    if missing_rules:
        add_missing("blueprint_rules", missing_rules)

    checklist: list[BlueprintRule] = []
    for r in parsed.checklist:
        checklist.append(
            BlueprintRule(
                rule_id=uuid.uuid4().hex,
                title=str(r.title),
                description=str(r.description),
                supporting_evidence=evidence_ids(
                    r.supporting_block_ids,
                    reason=f"Checklist example: {r.title}",
                    kind="intro_paragraph",
                    confidence=0.6,
                    limit=2,
                ),
            )
        )

    # Evidence-first enforcement for rules: rules without evidence are downgraded into checklist suggestions.
    story_rules, downgraded_story = _downgrade_rules_missing_evidence(story_rules)
    claim_rules, downgraded_claim = _downgrade_rules_missing_evidence(claim_rules)

    checklist.extend(downgraded_story)
    checklist.extend(downgraded_claim)

    # Ensure at least some evidence-backed rules exist, otherwise inject safe fallbacks.
    if not story_rules or not claim_rules:
        fb_story, fb_claim = _fallback_evidence_rule(
            ctx=ctx,
            intro_blocks_by_pdf=intro_blocks_by_pdf,
            moves_by_pdf=moves_by_pdf,
            style_by_pdf=style_by_pdf,
            evidence_out=evidence,
            kind="intro_paragraph",
        )
        if not story_rules and fb_story is not None:
            story_rules = [fb_story]
        if not claim_rules and fb_claim is not None:
            claim_rules = [fb_claim]

    # Ensure checklist at least covers consensus labels.
    have = {c.title for c in checklist}
    for label in consensus:
        title = f"Include a {label} paragraph"
        if title in have:
            continue
        checklist.append(BlueprintRule(rule_id=uuid.uuid4().hex, title=title, description=_LABEL_DESC.get(label, _LABEL_DESC["Other"]), supporting_evidence=[]))

    return IntroBlueprint(paragraph_plan=paragraph_plan, story_rules=story_rules, claim_rules=claim_rules, checklist=checklist), evidence


def build_intro_blueprint(
    *,
    ctx: BuildContext,
    intro_blocks_by_pdf: dict[str, list[TextBlock]],
    moves_by_pdf: dict[str, list[RhetoricalMove]],
    alignment: dict,
    style_by_pdf: dict[str, StyleFeatures],
) -> tuple[IntroBlueprint, list]:
    cfg = load_llm_config()
    if llm_required() and cfg is None:
        raise RuntimeError("PAPER_SKILL_LLM_MODEL is required (PAPER_SKILL_REQUIRE_LLM=1).")
    if cfg is not None:
        return build_intro_blueprint_llm(
            ctx=ctx,
            intro_blocks_by_pdf=intro_blocks_by_pdf,
            moves_by_pdf=moves_by_pdf,
            alignment=alignment,
            style_by_pdf=style_by_pdf,
        )

    eb = EvidenceBuilder(data_dir=ctx.data_dir)
    consensus: list[str] = list(alignment.get("consensus_sequence") or [])
    if not consensus:
        consensus = ["Context", "Problem", "Gap", "Approach", "Contribution", "Roadmap"]

    evidence: list = []

    # Paragraph plan: for each label pick one example per pdf if possible
    paragraph_plan: list[dict] = []
    for i, label in enumerate(consensus, start=1):
        ev_ids: list[str] = []
        blocks = select_blocks_for_labels(
            intro_blocks_by_pdf=intro_blocks_by_pdf,
            moves_by_pdf=moves_by_pdf,
            labels=[label],
            keyword_pattern=None,
            limit_total=len(ctx.pdf_ids),
            limit_per_pdf=1,
        )
        for b in blocks:
            ev = eb.from_block(block=b, reason=f"Example {label} paragraph in the mentor papers.", confidence=0.75, kind="intro_paragraph")
            evidence.append(ev)
            ev_ids.append(ev.evidence_id)

        paragraph_plan.append(
            {
                "paragraph_index": i,
                "label": label,
                "description": _LABEL_DESC.get(label, _LABEL_DESC["Other"]),
                "supporting_evidence": ev_ids,
            }
        )

    story_rules: list[BlueprintRule] = []
    claim_rules: list[BlueprintRule] = []
    checklist: list[BlueprintRule] = []

    # Story rule: gap connectors
    connector_counter = Counter()
    for s in style_by_pdf.values():
        counts = (s.connector_profile or {}).get("counts") or {}
        for k, v in counts.items():
            if k in {"however", "nevertheless", "yet", "but"}:
                connector_counter[k] += int(v)
    best_connector = (connector_counter.most_common(1)[0][0] if connector_counter else "however").capitalize()
    gap_kw = re.compile(rf"(^|\n)\s*{re.escape(best_connector)}\b", re.IGNORECASE)
    gap_blocks = select_blocks_for_labels(
        intro_blocks_by_pdf=intro_blocks_by_pdf,
        moves_by_pdf=moves_by_pdf,
        labels=["Gap"],
        keyword_pattern=gap_kw,
        limit_total=3,
        limit_per_pdf=1,
    )
    ev_ids: list[str] = []
    for b in gap_blocks:
        ev = eb.from_block(block=b, reason=f"Shows a contrastive connector ('{best_connector}') introducing the gap.", confidence=0.8, kind="intro_paragraph")
        evidence.append(ev)
        ev_ids.append(ev.evidence_id)
    story_rules.append(
        BlueprintRule(
            rule_id=uuid.uuid4().hex,
            title=f"Introduce the gap with a contrastive connector (e.g., {best_connector})",
            description="When transitioning from prior progress to limitations, use a contrastive connector and state the gap explicitly.",
            supporting_evidence=ev_ids,
        )
    )

    # Story rule: keep sentences in a typical length range
    p50s = []
    p90s = []
    for s in style_by_pdf.values():
        sl = s.sentence_len_stats or {}
        if sl.get("p50"):
            p50s.append(float(sl["p50"]))
        if sl.get("p90"):
            p90s.append(float(sl["p90"]))
    p50 = int(round(sum(p50s) / len(p50s))) if p50s else 20
    p90 = int(round(sum(p90s) / len(p90s))) if p90s else 35
    story_rules.append(
        BlueprintRule(
            rule_id=uuid.uuid4().hex,
            title="Match typical sentence length",
            description=f"Most intro sentences are around {p50} words; avoid overly long sentences (try to keep most under ~{p90} words).",
            supporting_evidence=[],
        )
    )
    # Attach a couple of representative paragraph examples as anchors.
    length_blocks = select_blocks_for_labels(
        intro_blocks_by_pdf=intro_blocks_by_pdf,
        moves_by_pdf=moves_by_pdf,
        labels=["Context", "Problem", "Approach"],
        keyword_pattern=None,
        limit_total=2,
        limit_per_pdf=1,
    )
    if length_blocks:
        ids: list[str] = []
        for b in length_blocks:
            ev = eb.from_block(block=b, reason="Representative intro paragraph for sentence-length style.", confidence=0.6, kind="intro_paragraph")
            evidence.append(ev)
            ids.append(ev.evidence_id)
        story_rules[-1].supporting_evidence = ids

    # Claim strength rule: hedging usage
    hedge_total = 0.0
    for s in style_by_pdf.values():
        hedge_total += float((s.hedging_profile or {}).get("total_per_1k_words") or 0.0)
    hedge_avg = hedge_total / max(1, len(style_by_pdf))
    if hedge_avg >= 2.0:
        title = "Use hedging for broad claims"
        desc = "Broad claims are often phrased with hedging (may/might/likely) to avoid absolute statements while still being confident."
    else:
        title = "Be assertive but avoid absolutes"
        desc = "Claims tend to be clear and assertive, but avoid absolute words (always/never) unless strictly proven."

    claim_blocks = select_blocks_for_labels(
        intro_blocks_by_pdf=intro_blocks_by_pdf,
        moves_by_pdf=moves_by_pdf,
        labels=["Contribution", "Claim"],
        keyword_pattern=None,
        limit_total=3,
        limit_per_pdf=1,
    )
    ev_ids = []
    for b in claim_blocks:
        ev = eb.from_block(block=b, reason="Example of claim/contribution phrasing in the mentor papers.", confidence=0.7, kind="intro_paragraph")
        evidence.append(ev)
        ev_ids.append(ev.evidence_id)
    claim_rules.append(
        BlueprintRule(
            rule_id=uuid.uuid4().hex,
            title=title,
            description=desc,
            supporting_evidence=ev_ids,
        )
    )

    # Evidence-first enforcement for rules: rules without evidence are downgraded into checklist suggestions.
    story_rules, downgraded_story = _downgrade_rules_missing_evidence(story_rules)
    claim_rules, downgraded_claim = _downgrade_rules_missing_evidence(claim_rules)

    # Checklist: one item per consensus label (evidence optional), plus downgraded suggestions.
    for label in consensus:
        checklist.append(
            BlueprintRule(
                rule_id=uuid.uuid4().hex,
                title=f"Include a {label} paragraph",
                description=_LABEL_DESC.get(label, _LABEL_DESC["Other"]),
                supporting_evidence=[],
            )
        )
    checklist.extend(downgraded_story)
    checklist.extend(downgraded_claim)

    blueprint = IntroBlueprint(paragraph_plan=paragraph_plan, story_rules=story_rules, claim_rules=claim_rules, checklist=checklist)
    return blueprint, evidence


def build_intro_blueprint_cached(
    *,
    ctx: BuildContext,
    intro_blocks_by_pdf: dict[str, list[TextBlock]],
    moves_by_pdf: dict[str, list[RhetoricalMove]],
    alignment: dict,
    style_by_pdf: dict[str, StyleFeatures],
    cache_key: str,
) -> tuple[IntroBlueprint, list[EvidencePointer], bool]:
    # Fingerprint: style + moves + alignment (all deterministic inputs).
    cfg = load_llm_config()
    if llm_required() and cfg is None:
        raise RuntimeError("PAPER_SKILL_LLM_MODEL is required (PAPER_SKILL_REQUIRE_LLM=1).")
    mode = "llm" if cfg is not None else "rules"

    moves_dump: dict[str, list[dict]] = {}
    for pdf_id, moves in moves_by_pdf.items():
        moves_dump[pdf_id] = [{"label": m.label, "block_id": m.block_id, "confidence": float(m.confidence)} for m in moves]
    style_dump = {pdf_id: style_by_pdf[pdf_id].model_dump(mode="json") for pdf_id in style_by_pdf}
    llm_fp = None
    if cfg is not None:
        llm_fp = {"model": cfg.model, "prompt_version": BLUEPRINT_PROMPT_VERSION}
    fingerprint = json.dumps(
        {
            "mode": mode,
            "llm": llm_fp,
            "alignment": alignment,
            "moves": moves_dump,
            "style": style_dump,
            "skills_fp": skill_lib_fingerprint(),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    input_hash = sha256_text(fingerprint)

    path = stage_cache_path(data_dir=ctx.data_dir, stage="blueprint", key=cache_key)
    payload = load_stage_payload(path=path, version=BLUEPRINT_CACHE_VERSION, input_hash=input_hash)
    if isinstance(payload, dict):
        try:
            bp_raw = payload.get("blueprint")
            ev_raw = payload.get("evidence")
            bp = IntroBlueprint.model_validate(bp_raw)
            ev_list = [EvidencePointer.model_validate(e) for e in (ev_raw or [])]
            return bp, ev_list, True
        except Exception:  # noqa: BLE001
            pass

    blueprint, evidence = build_intro_blueprint(
        ctx=ctx,
        intro_blocks_by_pdf=intro_blocks_by_pdf,
        moves_by_pdf=moves_by_pdf,
        alignment=alignment,
        style_by_pdf=style_by_pdf,
    )
    save_stage_payload(
        path=path,
        version=BLUEPRINT_CACHE_VERSION,
        input_hash=input_hash,
        payload={"blueprint": blueprint.model_dump(mode="json"), "evidence": [e.model_dump(mode="json") for e in evidence]},
    )
    return blueprint, evidence, False
