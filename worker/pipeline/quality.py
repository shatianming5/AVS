from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from shared.schemas import IntroBlueprint, QualityReport, StoryboardItem, Template
from worker.context import BuildContext
from worker.pipeline.cache import load_stage_payload, save_stage_payload, sha256_text, stage_cache_path
from worker.pipeline.skill_hints import blueprint_rule_candidates, estimate_skill_rules_adopted, selected_skill_topics, skill_lib_fingerprint


@dataclass(frozen=True)
class _Counts:
    total: int
    with_evidence: int


QUALITY_CACHE_VERSION = 1


def _count_rule_evidence(blueprint: IntroBlueprint) -> _Counts:
    # Checklist items are executable reminders; evidence is optional there.
    rules = (blueprint.story_rules or []) + (blueprint.claim_rules or [])
    total = len(rules)
    with_ev = sum(1 for r in rules if (r.supporting_evidence or []))
    return _Counts(total=total, with_evidence=with_ev)


def _count_template_evidence(templates: list[Template]) -> _Counts:
    total = len(templates)
    with_ev = sum(1 for t in templates if (t.supporting_evidence or []))
    return _Counts(total=total, with_evidence=with_ev)


def _count_storyboard_evidence(items: list[StoryboardItem]) -> _Counts:
    total = len(items)
    with_ev = sum(1 for s in items if (s.supporting_evidence or []))
    return _Counts(total=total, with_evidence=with_ev)


def _template_slot_score(templates: list[Template]) -> float:
    if not templates:
        return 0.0
    ok = 0
    for t in templates:
        if isinstance(t.slot_schema, dict) and len(t.slot_schema.keys()) >= 3 and t.text_with_slots and "{" in t.text_with_slots:
            ok += 1
    return ok / len(templates)


def build_quality_report(
    *,
    ctx: BuildContext,
    blueprint: IntroBlueprint,
    templates: list[Template],
    storyboard: list[StoryboardItem],
    evidence: list,
    sequences: dict[str, list[str]],
    alignment_strength: float = 0.0,
    ocr_used: bool = False,
    intro_blocks_count_by_pdf: dict | None = None,
    caption_count_by_pdf: dict | None = None,
    plagiarism_max_similarity: float = 0.0,
    plagiarism_hits: list[dict] | None = None,
) -> QualityReport:
    b = _count_rule_evidence(blueprint)
    t = _count_template_evidence(templates)
    s = _count_storyboard_evidence(storyboard)

    total = b.total + t.total + s.total
    with_ev = b.with_evidence + t.with_evidence + s.with_evidence
    evidence_coverage = (with_ev / total) if total else 0.0

    weak_items: list[dict] = []
    for r in (blueprint.story_rules or []) + (blueprint.claim_rules or []):
        if not (r.supporting_evidence or []):
            weak_items.append({"kind": "blueprint_rule", "id": r.rule_id, "title": r.title, "reason": "missing_evidence"})
    # Evidence-first downgrade marker: checklist suggestions without evidence
    for r in (blueprint.checklist or []):
        if (r.title or "").startswith("[Suggestion]") and not (r.supporting_evidence or []):
            weak_items.append({"kind": "blueprint_rule", "id": r.rule_id, "title": r.title, "reason": "missing_evidence_downgraded"})
    for tplt in templates:
        if not (tplt.supporting_evidence or []):
            weak_items.append({"kind": "template", "id": tplt.template_id, "title": tplt.template_type, "reason": "missing_evidence"})
    for item in storyboard:
        if not (item.supporting_evidence or []):
            weak_items.append({"kind": "storyboard", "id": item.item_id, "title": item.figure_role, "reason": "missing_evidence"})

    notes: list[str] = []
    if evidence_coverage < 0.8:
        notes.append("Evidence coverage below 0.8; some rules/items lack clickable anchors.")
    if len(evidence) == 0:
        notes.append("No evidence extracted; check PDF text extraction quality.")
    if plagiarism_max_similarity >= 0.35:
        notes.append("Potential plagiarism risk detected for one or more templates; see plagiarism_flagged.")
    if ocr_used:
        notes.append("OCR was used for at least one PDF (likely scanned PDF).")

    candidates = blueprint_rule_candidates(field_hint=ctx.field_hint)
    rule_titles = [r.title for r in (blueprint.story_rules or []) + (blueprint.claim_rules or [])]
    topics_used = selected_skill_topics(field_hint=ctx.field_hint)
    skills_fp = skill_lib_fingerprint()

    return QualityReport(
        evidence_coverage=float(evidence_coverage),
        structure_strength=float(max(0.0, min(1.0, alignment_strength))),
        template_slot_score=float(_template_slot_score(templates)),
        plagiarism_max_similarity=float(plagiarism_max_similarity),
        plagiarism_flagged=(plagiarism_hits or []),
        ocr_used=bool(ocr_used),
        intro_blocks_count_by_pdf=(intro_blocks_count_by_pdf or {}),
        caption_count_by_pdf=(caption_count_by_pdf or {}),
        weak_items=weak_items,
        notes=notes,
        skill_lib_used=bool(skills_fp),
        skill_lib_fingerprint=skills_fp,
        skill_topics_used=topics_used,
        skill_rules_adopted=estimate_skill_rules_adopted(rule_titles=rule_titles, candidates=candidates) if skills_fp else 0,
    )


def build_quality_report_cached(
    *,
    ctx: BuildContext,
    blueprint: IntroBlueprint,
    templates: list[Template],
    storyboard: list[StoryboardItem],
    evidence: list,
    sequences: dict[str, list[str]],
    cache_key: str,
    alignment_strength: float = 0.0,
    ocr_used: bool = False,
    intro_blocks_count_by_pdf: dict | None = None,
    caption_count_by_pdf: dict | None = None,
    plagiarism_max_similarity: float = 0.0,
    plagiarism_hits: list[dict] | None = None,
) -> tuple[QualityReport, bool]:
    fingerprint = json.dumps(
        {
            "blueprint": blueprint.model_dump(mode="json"),
            "templates": [t.model_dump(mode="json") for t in templates],
            "storyboard": [s.model_dump(mode="json") for s in storyboard],
            "evidence_count": len(evidence),
            "sequences": sequences,
            "alignment_strength": float(alignment_strength),
            "ocr_used": bool(ocr_used),
            "intro_blocks_count_by_pdf": intro_blocks_count_by_pdf or {},
            "caption_count_by_pdf": caption_count_by_pdf or {},
            "plagiarism_max_similarity": float(plagiarism_max_similarity),
            "plagiarism_hits": plagiarism_hits or [],
            "skills_fp": skill_lib_fingerprint(),
            "skills_topics": selected_skill_topics(field_hint=ctx.field_hint),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    input_hash = sha256_text(fingerprint)
    path = stage_cache_path(data_dir=ctx.data_dir, stage="quality", key=cache_key)
    payload = load_stage_payload(path=path, version=QUALITY_CACHE_VERSION, input_hash=input_hash)
    if isinstance(payload, dict):
        try:
            return QualityReport.model_validate(payload), True
        except Exception:  # noqa: BLE001
            pass

    quality = build_quality_report(
        ctx=ctx,
        blueprint=blueprint,
        templates=templates,
        storyboard=storyboard,
        evidence=evidence,
        sequences=sequences,
        alignment_strength=alignment_strength,
        ocr_used=ocr_used,
        intro_blocks_count_by_pdf=intro_blocks_count_by_pdf,
        caption_count_by_pdf=caption_count_by_pdf,
        plagiarism_max_similarity=plagiarism_max_similarity,
        plagiarism_hits=plagiarism_hits,
    )
    save_stage_payload(path=path, version=QUALITY_CACHE_VERSION, input_hash=input_hash, payload=quality.model_dump(mode="json"))
    return quality, False
