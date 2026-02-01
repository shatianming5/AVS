from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from shared.schemas import EvidenceIndex, SkillPack
from worker.context import BuildContext
from worker.pipeline.blueprint import build_intro_blueprint_cached
from worker.pipeline.captions import build_storyboard_cached
from worker.pipeline.cache import sha256_text
from worker.pipeline.extract_blocks import extract_blocks_cached, was_ocr_used
from worker.pipeline.intro import locate_intro_blocks_cached
from worker.pipeline.metadata import cache_pdf_metadata, extract_pdf_metadata
from worker.pipeline.alignment import align_move_sequences
from worker.pipeline.cleaning import clean_text_preserve
from worker.pipeline.moves import compress_moves, label_moves_cached
from worker.pipeline.phrase_bank import build_phrase_bank
from worker.pipeline.plagiarism import check_templates_against_sources
from worker.pipeline.quality import build_quality_report_cached
from worker.pipeline.style import extract_style_features_cached
from worker.pipeline.templates import build_templates_cached


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_skillpack(
    *,
    job_id: str,
    pdf_ids: list[str],
    pack_name: str,
    field_hint: str | None,
    target_venue_hint: str | None,
    language: str,
    owner_token: str,
    data_dir: Path,
    on_progress,
) -> dict:
    pack_id = uuid.uuid4().hex
    ctx = BuildContext(
        job_id=job_id,
        pack_id=pack_id,
        pdf_ids=pdf_ids,
        pack_name=pack_name,
        field_hint=field_hint,
        target_venue_hint=target_venue_hint,
        language=language,
        data_dir=data_dir,
        owner_token=owner_token,
    )
    from worker.pipeline.llm_state import reset_llm_stats

    reset_llm_stats()

    timings_ms: dict[str, int] = {}
    cache_hits: dict[str, dict[str, bool]] = {
        "sections": {},
        "locate_intro": {},
        "style_features": {},
        "moves": {},
        "blueprint": {},
        "templates": {},
        "storyboard": {},
        "quality": {},
    }
    pack_cache_key = sha256_text(
        json.dumps(
            {
                "pdf_ids": sorted(list(pdf_ids)),
                "field_hint": (field_hint or "").strip().lower(),
                "target_venue_hint": (target_venue_hint or "").strip().lower(),
                "language": (language or "").strip().lower(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    t_stage = time.perf_counter()
    on_progress("pdf_metadata", 0.02)
    meta_by_pdf: dict[str, dict] = {}
    from backend.services.pdfs import update_pdf_metadata

    for pdf_id in pdf_ids:
        meta = extract_pdf_metadata(pdf_id=pdf_id, data_dir=data_dir)
        cache_pdf_metadata(pdf_id=pdf_id, data_dir=data_dir, meta=meta)
        update_pdf_metadata(pdf_id=pdf_id, num_pages=meta["num_pages"], title=meta.get("title"), toc=meta.get("toc"))
        meta_by_pdf[pdf_id] = meta
    timings_ms["pdf_metadata"] = int((time.perf_counter() - t_stage) * 1000)

    t_stage = time.perf_counter()
    on_progress("extract_blocks", 0.05)
    blocks_by_pdf = {pdf_id: extract_blocks_cached(pdf_id, data_dir) for pdf_id in pdf_ids}
    timings_ms["extract_blocks"] = int((time.perf_counter() - t_stage) * 1000)

    t_stage = time.perf_counter()
    on_progress("sections", 0.12)
    from worker.pipeline.sections import assign_section_paths_cached

    for pdf_id in pdf_ids:
        hit = assign_section_paths_cached(
            pdf_id=pdf_id,
            blocks=blocks_by_pdf[pdf_id],
            toc=meta_by_pdf.get(pdf_id, {}).get("toc"),
            data_dir=data_dir,
        )
        cache_hits["sections"][pdf_id] = bool(hit)
    timings_ms["sections"] = int((time.perf_counter() - t_stage) * 1000)

    t_stage = time.perf_counter()
    on_progress("locate_intro", 0.20)
    intro_by_pdf = {}
    for pdf_id in pdf_ids:
        intro_blocks, hit = locate_intro_blocks_cached(
            pdf_id=pdf_id,
            blocks=blocks_by_pdf[pdf_id],
            toc=meta_by_pdf.get(pdf_id, {}).get("toc"),
            data_dir=data_dir,
        )
        intro_by_pdf[pdf_id] = intro_blocks
        cache_hits["locate_intro"][pdf_id] = bool(hit)
    timings_ms["locate_intro"] = int((time.perf_counter() - t_stage) * 1000)

    t_stage = time.perf_counter()
    on_progress("clean_intro", 0.28)
    for pdf_id in pdf_ids:
        for b in intro_by_pdf[pdf_id]:
            raw = b.raw_text or b.text or ""
            cleaned = clean_text_preserve(raw)
            b.clean_text = cleaned
            b.text = cleaned
    timings_ms["clean_intro"] = int((time.perf_counter() - t_stage) * 1000)

    t_stage = time.perf_counter()
    on_progress("style_features", 0.32)
    style_by_pdf = {}
    for pdf_id in pdf_ids:
        features, hit = extract_style_features_cached(pdf_id=pdf_id, intro_blocks=intro_by_pdf[pdf_id], data_dir=data_dir)
        style_by_pdf[pdf_id] = features
        cache_hits["style_features"][pdf_id] = bool(hit)
    phrase_bank = build_phrase_bank(style_by_pdf)
    timings_ms["style_features"] = int((time.perf_counter() - t_stage) * 1000)

    t_stage = time.perf_counter()
    on_progress("label_moves", 0.35)
    moves_by_pdf = {}
    for pdf_id in pdf_ids:
        moves, hit = label_moves_cached(pdf_id, intro_by_pdf[pdf_id], field_hint=field_hint, data_dir=data_dir)
        moves_by_pdf[pdf_id] = moves
        cache_hits["moves"][pdf_id] = bool(hit)
    compressed_by_pdf = {pdf_id: compress_moves(moves_by_pdf[pdf_id]) for pdf_id in pdf_ids}
    seq_by_pdf = {pdf_id: compressed_by_pdf[pdf_id]["sequence"] for pdf_id in pdf_ids}
    alignment = align_move_sequences(seq_by_pdf)
    timings_ms["label_moves"] = int((time.perf_counter() - t_stage) * 1000)

    t_stage = time.perf_counter()
    on_progress("build_blueprint", 0.55)
    blueprint, evidence_for_blueprint, hit_blueprint = build_intro_blueprint_cached(
        ctx=ctx,
        intro_blocks_by_pdf=intro_by_pdf,
        moves_by_pdf=moves_by_pdf,
        alignment=alignment,
        style_by_pdf=style_by_pdf,
        cache_key=pack_cache_key,
    )
    cache_hits["blueprint"][pack_cache_key] = bool(hit_blueprint)
    timings_ms["build_blueprint"] = int((time.perf_counter() - t_stage) * 1000)

    t_stage = time.perf_counter()
    on_progress("templates", 0.70)
    templates, evidence_for_templates, hit_templates = build_templates_cached(
        ctx=ctx,
        intro_blocks_by_pdf=intro_by_pdf,
        moves_by_pdf=moves_by_pdf,
        style_by_pdf=style_by_pdf,
        phrase_bank=phrase_bank,
        cache_key=pack_cache_key,
    )
    cache_hits["templates"][pack_cache_key] = bool(hit_templates)
    timings_ms["templates"] = int((time.perf_counter() - t_stage) * 1000)

    # Plagiarism self-check and deterministic rewrite fallback (no-LLM).
    plagiarism_max, plagiarism_hits = check_templates_against_sources(templates, intro_by_pdf)
    if plagiarism_hits and plagiarism_max >= 0.35:
        for hit in plagiarism_hits:
            for t in templates:
                if t.template_id == hit.template_id:
                    t.text_with_slots = _rewrite_template_text(t.text_with_slots)
        plagiarism_max, plagiarism_hits = check_templates_against_sources(templates, intro_by_pdf)

    t_stage = time.perf_counter()
    on_progress("storyboard", 0.82)
    storyboard, evidence_for_storyboard, storyboard_info, hit_storyboard = build_storyboard_cached(
        ctx=ctx,
        blocks_by_pdf=blocks_by_pdf,
        intro_blocks_by_pdf=intro_by_pdf,
        cache_key=pack_cache_key,
    )
    cache_hits["storyboard"][pack_cache_key] = bool(hit_storyboard)
    timings_ms["storyboard"] = int((time.perf_counter() - t_stage) * 1000)

    evidence_all = evidence_for_blueprint + evidence_for_templates + evidence_for_storyboard
    evidence_index = EvidenceIndex(pack_id=pack_id, evidence=evidence_all)

    t_stage = time.perf_counter()
    on_progress("quality", 0.90)
    quality, hit_quality = build_quality_report_cached(
        ctx=ctx,
        blueprint=blueprint,
        templates=templates,
        storyboard=storyboard,
        evidence=evidence_all,
        sequences=seq_by_pdf,
        cache_key=pack_cache_key,
        alignment_strength=float(alignment.get("strength_score") or 0.0),
        ocr_used=any(was_ocr_used(pdf_id, data_dir) for pdf_id in pdf_ids),
        intro_blocks_count_by_pdf={pdf_id: len(intro_by_pdf[pdf_id]) for pdf_id in pdf_ids},
        caption_count_by_pdf=(storyboard_info.get("caption_count_by_pdf") or {}),
        plagiarism_max_similarity=plagiarism_max,
        plagiarism_hits=[h.__dict__ for h in plagiarism_hits],
    )
    cache_hits["quality"][pack_cache_key] = bool(hit_quality)
    timings_ms["quality"] = int((time.perf_counter() - t_stage) * 1000)

    pack = SkillPack(
        pack_id=pack_id,
        pack_name=pack_name,
        pdf_ids=pdf_ids,
        intro_blueprint=blueprint,
        templates=templates,
        storyboard=storyboard,
        patterns=[
            {
                "pattern_id": uuid.uuid4().hex,
                "pattern_type": "MoveSequence",
                "description": "Aligned intro move sequence across the 3 mentor papers.",
                "supporting_evidence": [],
                "strength_score": float(alignment.get("strength_score") or 0.0),
                "data": alignment,
            },
            {
                "pattern_id": uuid.uuid4().hex,
                "pattern_type": "FigureRefs",
                "description": "Figure/table references mentioned in introductions.",
                "supporting_evidence": [],
                "strength_score": 1.0,
                "data": storyboard_info.get("figure_refs_by_pdf") or {},
            },
            {
                "pattern_id": uuid.uuid4().hex,
                "pattern_type": "FigureRoleDistribution",
                "description": "Role distribution inferred from figure captions.",
                "supporting_evidence": [],
                "strength_score": 1.0,
                "data": storyboard_info.get("role_counts") or {},
            }
        ],
        version="v0",
        build_metadata={
            "job_id": job_id,
            "built_at": now_iso(),
            "field_hint": field_hint,
            "target_venue_hint": target_venue_hint,
            "language": language,
        },
        quality_report=quality,
    )

    # Persist artifacts
    out_skillpack = data_dir / "skillpacks" / f"{pack_id}.json"
    out_yaml = data_dir / "skillpacks" / f"{pack_id}.yaml"
    out_ev = data_dir / "evidence" / f"{pack_id}.json"
    out_skillpack.parent.mkdir(parents=True, exist_ok=True)
    out_ev.parent.mkdir(parents=True, exist_ok=True)

    out_skillpack.write_text(pack.model_dump_json(indent=2), encoding="utf-8")
    out_yaml.write_text(yaml.safe_dump(pack.model_dump(mode="json"), sort_keys=False, allow_unicode=True), encoding="utf-8")
    out_ev.write_text(evidence_index.model_dump_json(indent=2), encoding="utf-8")

    # Persist index row (for access control / discovery)
    from backend.services.skillpacks import upsert_skillpack_row

    upsert_skillpack_row(pack_id=pack_id, owner_token=owner_token, pack_name=pack_name, pdf_ids=pdf_ids)

    # Persist job metrics (best-effort)
    from backend.services.jobs import update_job_metrics

    from shared.llm_client import load_llm_config, redact_base_url
    from worker.pipeline.blueprint import BLUEPRINT_PROMPT_VERSION
    from worker.pipeline.captions import STORYBOARD_PROMPT_VERSION
    from worker.pipeline.moves import MOVES_PROMPT_VERSION

    llm_cfg = load_llm_config()
    llm_meta = None
    if llm_cfg is not None:
        llm_meta = {
            "provider": "local_openai_compat",
            "model": llm_cfg.model,
            "base_url": redact_base_url(llm_cfg.base_url),
            "prompt_versions": {
                "moves": MOVES_PROMPT_VERSION,
                "blueprint": BLUEPRINT_PROMPT_VERSION,
                "storyboard": STORYBOARD_PROMPT_VERSION,
            },
        }
        timings_ms.setdefault("llm_moves", int(timings_ms.get("label_moves") or 0))
        timings_ms.setdefault("llm_blueprint", int(timings_ms.get("build_blueprint") or 0))
        timings_ms.setdefault("llm_storyboard", int(timings_ms.get("storyboard") or 0))

    from worker.pipeline.llm_state import snapshot_llm_stats
    from worker.pipeline.metrics import llm_health_metrics
    from worker.pipeline.skill_hints import skill_lib_info

    llm_stats = snapshot_llm_stats()
    llm_health = llm_health_metrics(llm_stats=llm_stats) if llm_cfg is not None else None
    skills = skill_lib_info()

    update_job_metrics(
        job_id,
        metrics={
            "timings_ms": timings_ms,
            "llm": llm_meta,
            "llm_health": llm_health,
            "skill_lib": {
                "used": skills.used,
                "fingerprint": skills.fingerprint,
                "items_loaded": skills.items_loaded,
                "topics_loaded": skills.topics_loaded,
                "path_hint": skills.path_hint,
            },
            "cache_hits": cache_hits,
            "ocr_used": bool(quality.ocr_used),
            "intro_blocks_count_by_pdf": quality.intro_blocks_count_by_pdf,
            "caption_count_by_pdf": quality.caption_count_by_pdf,
            "evidence_coverage": quality.evidence_coverage,
            "structure_strength": quality.structure_strength,
        },
    )

    on_progress("done", 1.0)
    return {"pack_id": pack_id}


def _rewrite_template_text(text: str) -> str:
    # Deterministic, low-risk rewrites to reduce n-gram overlap without changing meaning too much.
    out = text
    out = out.replace("However,", "Nevertheless,", 1)
    out = out.replace("Nevertheless,", "However,", 1)
    out = out.replace(" may ", " can ", 1)
    parts = out.split(". ")
    if len(parts) >= 2:
        parts[0], parts[1] = parts[1], parts[0]
        out = ". ".join(parts)
    return out
