from __future__ import annotations

import json
import uuid
from pathlib import Path

from shared.schemas import EvidencePointer, RhetoricalMove, StyleFeatures, Template, TextBlock
from worker.context import BuildContext
from worker.pipeline.cache import load_stage_payload, save_stage_payload, sha256_text, stage_cache_path
from worker.pipeline.evidence import EvidenceBuilder
from worker.pipeline.retrieval import select_blocks_for_labels


TEMPLATES_CACHE_VERSION = 1


def build_templates(
    *,
    ctx: BuildContext,
    intro_blocks_by_pdf: dict[str, list[TextBlock]],
    moves_by_pdf: dict[str, list[RhetoricalMove]],
    style_by_pdf: dict[str, StyleFeatures],
    phrase_bank: dict,
) -> tuple[list[Template], list]:
    eb = EvidenceBuilder(data_dir=ctx.data_dir)
    evidence: list = []

    connector = _pick_connector(phrase_bank)
    hedge = _pick_hedge(phrase_bank)
    slot_len = _slot_len(style_by_pdf)

    templates: list[Template] = []

    # Template A: opening/context/problem
    t1_id = uuid.uuid4().hex
    t1_text = (
        "{CONTEXT_SENTENCE} "
        "In particular, {PROBLEM_STATEMENT}. "
        "{CONNECTOR}, existing approaches {LIMITATION_STATEMENT}, which {HEDGE} hinder(s) {IMPACT_STATEMENT}."
    )
    t1_text = t1_text.replace("{CONNECTOR}", connector).replace("{HEDGE}", hedge)
    t1_slot_schema = {
        "CONTEXT_SENTENCE": {"what": "1 sentence background context", "len_words": [max(6, slot_len - 6), slot_len + 6], "voice": "neutral"},
        "PROBLEM_STATEMENT": {"what": "1 sentence describing the task/problem", "len_words": [max(6, slot_len - 6), slot_len + 6], "voice": "neutral"},
        "LIMITATION_STATEMENT": {"what": "1 sentence about what is missing/insufficient", "len_words": [max(6, slot_len - 6), slot_len + 8], "voice": "neutral"},
        "IMPACT_STATEMENT": {"what": "short phrase describing why the limitation matters", "len_words": [3, 12], "voice": "neutral"},
        "citation_density_hint": _citation_hint(style_by_pdf),
    }
    ev_ids = _template_evidence_ids(
        eb,
        evidence,
        intro_blocks_by_pdf=intro_blocks_by_pdf,
        moves_by_pdf=moves_by_pdf,
        labels=["Context", "Problem"],
        reason="Example of opening/context/problem style.",
    )
    templates.append(
        Template(
            template_id=t1_id,
            template_type="IntroOpening",
            text_with_slots=t1_text,
            slot_schema=t1_slot_schema,
            do_rules=[f"Use a contrastive connector like '{connector}' to introduce the gap."],
            dont_rules=["Do not copy original sentences verbatim."],
            supporting_evidence=ev_ids,
        )
    )

    # Template B: gap -> approach
    t2_id = uuid.uuid4().hex
    t2_text = (
        "{CONNECTOR}, prior work {PRIOR_WORK_SUMMARY}. "
        "To address this gap, we propose {METHOD_NAME}, which {METHOD_KEY_IDEA}. "
        "Concretely, {METHOD_DETAILS}."
    )
    t2_text = t2_text.replace("{CONNECTOR}", connector)
    t2_slot_schema = {
        "PRIOR_WORK_SUMMARY": {"what": "short summary of what prior work does well", "len_words": [6, slot_len + 6]},
        "METHOD_NAME": {"what": "method name/one-line label", "len_words": [1, 10]},
        "METHOD_KEY_IDEA": {"what": "main idea of the approach in one sentence", "len_words": [max(6, slot_len - 6), slot_len + 8]},
        "METHOD_DETAILS": {"what": "1 sentence concrete instantiation (components/steps)", "len_words": [max(6, slot_len - 6), slot_len + 10]},
        "citation_density_hint": _citation_hint(style_by_pdf),
    }
    ev_ids = _template_evidence_ids(
        eb,
        evidence,
        intro_blocks_by_pdf=intro_blocks_by_pdf,
        moves_by_pdf=moves_by_pdf,
        labels=["Gap", "Approach"],
        reason="Example of gap-to-approach phrasing.",
    )
    templates.append(
        Template(
            template_id=t2_id,
            template_type="GapApproach",
            text_with_slots=t2_text,
            slot_schema=t2_slot_schema,
            do_rules=["Name the gap explicitly before introducing the method."],
            dont_rules=["Avoid overly absolute claims (always/never) unless proven."],
            supporting_evidence=ev_ids,
        )
    )

    # Template C: contributions + roadmap
    t3_id = uuid.uuid4().hex
    t3_text = (
        "Our main contributions are: (1) {CONTRIB_1}; (2) {CONTRIB_2}; (3) {CONTRIB_3}. "
        "The rest of this paper is organized as follows: {ROADMAP}."
    )
    t3_slot_schema = {
        "CONTRIB_1": {"what": "first contribution (1 clause)", "len_words": [4, 16]},
        "CONTRIB_2": {"what": "second contribution (1 clause)", "len_words": [4, 16]},
        "CONTRIB_3": {"what": "third contribution (1 clause)", "len_words": [4, 16]},
        "ROADMAP": {"what": "1 sentence roadmap (Sec.2...)", "len_words": [max(10, slot_len - 4), slot_len + 10]},
    }
    ev_ids = _template_evidence_ids(
        eb,
        evidence,
        intro_blocks_by_pdf=intro_blocks_by_pdf,
        moves_by_pdf=moves_by_pdf,
        labels=["Contribution", "Roadmap"],
        reason="Example of contributions/roadmap style.",
    )
    templates.append(
        Template(
            template_id=t3_id,
            template_type="ContributionsRoadmap",
            text_with_slots=t3_text,
            slot_schema=t3_slot_schema,
            do_rules=["Use a short enumerated list for contributions."],
            dont_rules=["Do not include dataset/model specifics in the roadmap sentence."],
            supporting_evidence=ev_ids,
        )
    )

    return templates, evidence


def build_templates_cached(
    *,
    ctx: BuildContext,
    intro_blocks_by_pdf: dict[str, list[TextBlock]],
    moves_by_pdf: dict[str, list[RhetoricalMove]],
    style_by_pdf: dict[str, StyleFeatures],
    phrase_bank: dict,
    cache_key: str,
) -> tuple[list[Template], list[EvidencePointer], bool]:
    moves_dump: dict[str, list[dict]] = {}
    for pdf_id, moves in moves_by_pdf.items():
        moves_dump[pdf_id] = [{"label": m.label, "block_id": m.block_id, "confidence": float(m.confidence)} for m in moves]
    style_dump = {pdf_id: style_by_pdf[pdf_id].model_dump(mode="json") for pdf_id in style_by_pdf}
    fingerprint = json.dumps({"moves": moves_dump, "style": style_dump, "phrase_bank": phrase_bank}, ensure_ascii=False, sort_keys=True)
    input_hash = sha256_text(fingerprint)

    path = stage_cache_path(data_dir=ctx.data_dir, stage="templates", key=cache_key)
    payload = load_stage_payload(path=path, version=TEMPLATES_CACHE_VERSION, input_hash=input_hash)
    if isinstance(payload, dict):
        try:
            t_raw = payload.get("templates") or []
            e_raw = payload.get("evidence") or []
            templates = [Template.model_validate(t) for t in t_raw]
            evidence = [EvidencePointer.model_validate(e) for e in e_raw]
            return templates, evidence, True
        except Exception:  # noqa: BLE001
            pass

    templates, evidence = build_templates(
        ctx=ctx,
        intro_blocks_by_pdf=intro_blocks_by_pdf,
        moves_by_pdf=moves_by_pdf,
        style_by_pdf=style_by_pdf,
        phrase_bank=phrase_bank,
    )
    save_stage_payload(
        path=path,
        version=TEMPLATES_CACHE_VERSION,
        input_hash=input_hash,
        payload={"templates": [t.model_dump(mode="json") for t in templates], "evidence": [e.model_dump(mode="json") for e in evidence]},
    )
    return templates, evidence, False


def _pick_connector(phrase_bank: dict) -> str:
    candidates = phrase_bank.get("connectors") or []
    for c in candidates:
        if c.lower() in {"however", "nevertheless", "yet", "but"}:
            return c.capitalize() if len(c) <= 10 else "However"
    return "However"


def _pick_hedge(phrase_bank: dict) -> str:
    candidates = phrase_bank.get("hedges") or []
    for h in candidates:
        if h.lower() in {"may", "might", "likely", "could"}:
            return h.lower()
    return "may"


def _slot_len(style_by_pdf: dict[str, StyleFeatures]) -> int:
    p50s = []
    for s in style_by_pdf.values():
        sl = s.sentence_len_stats or {}
        if sl.get("p50"):
            p50s.append(int(sl["p50"]))
    if not p50s:
        return 18
    return int(round(sum(p50s) / len(p50s)))


def _citation_hint(style_by_pdf: dict[str, StyleFeatures]) -> dict:
    means = []
    for s in style_by_pdf.values():
        cd = s.citation_density or {}
        if cd.get("mean_per_paragraph") is not None:
            means.append(float(cd["mean_per_paragraph"]))
    mean = sum(means) / len(means) if means else 0.0
    return {"mean_citations_per_paragraph": mean}


def _template_evidence_ids(
    eb: EvidenceBuilder,
    evidence_out: list,
    *,
    intro_blocks_by_pdf: dict[str, list[TextBlock]],
    moves_by_pdf: dict[str, list[RhetoricalMove]],
    labels: list[str],
    reason: str,
) -> list[str]:
    blocks = select_blocks_for_labels(
        intro_blocks_by_pdf=intro_blocks_by_pdf,
        moves_by_pdf=moves_by_pdf,
        labels=labels,
        keyword_pattern=None,
        limit_total=3,
        limit_per_pdf=1,
    )
    ids: list[str] = []
    for b in blocks:
        ev = eb.from_block(block=b, reason=reason, confidence=0.7, kind="intro_paragraph")
        evidence_out.append(ev)
        ids.append(ev.evidence_id)
    return ids
