from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from shared.schemas import RhetoricalMove, TextBlock
from shared.llm_client import load_llm_config, llm_required, require_llm_config
from shared.llm_schemas import LlmMovesOutput
from worker.pipeline.cache import load_stage_payload, save_stage_payload, sha256_text, stage_cache_path
from worker.pipeline.llm_payload import blocks_for_llm
from worker.pipeline.llm_state import add_missing, add_total, mark_repair_used
from worker.pipeline.skill_hints import moves_label_prior_text, skill_lib_fingerprint


_LABELS: list[str] = [
    "Context",
    "Problem",
    "Gap",
    "Approach",
    "Contribution",
    "Roadmap",
    "RelatedWorkHook",
    "Claim",
    "Limitation",
    "Other",
]

MOVES_CACHE_VERSION = 2
MOVES_PROMPT_VERSION = 1


_PATTERNS: dict[str, list[re.Pattern]] = {
    "Roadmap": [
        re.compile(r"(the rest of (this|the) paper|is organized as follows|we organize this paper)", re.IGNORECASE),
        re.compile(r"section\s+\d+", re.IGNORECASE),
    ],
    "Contribution": [
        re.compile(r"(our contributions|we make (the )?following contributions|main contributions|we contribute)", re.IGNORECASE),
        re.compile(r"\b(three|two|four)\s+fold\b", re.IGNORECASE),
    ],
    "Approach": [
        re.compile(r"(we propose|we present|we introduce|we develop|we design|we build|we leverage)", re.IGNORECASE),
        re.compile(r"\bin this paper\b", re.IGNORECASE),
    ],
    "Gap": [
        re.compile(r"\b(however|nevertheless|yet|but)\b", re.IGNORECASE),
        re.compile(r"(remains (a )?(challenge|open)|still (lacks|missing)|few (studies|methods)|limited)", re.IGNORECASE),
    ],
    "Problem": [
        re.compile(r"(problem|task|goal|aim|objective)", re.IGNORECASE),
        re.compile(r"\bwe study\b", re.IGNORECASE),
    ],
    "Context": [
        re.compile(r"(recent(ly)?|in recent years|has attracted|widely used|has been (studied|explored))", re.IGNORECASE),
        re.compile(r"\bimportant\b", re.IGNORECASE),
    ],
    "RelatedWorkHook": [
        re.compile(r"(prior work|previous work|existing approaches|related work|recent studies)", re.IGNORECASE),
    ],
    "Claim": [
        re.compile(r"(to the best of our knowledge|we show|we demonstrate|we find that|we achieve)", re.IGNORECASE),
    ],
    "Limitation": [
        re.compile(r"(limitation|future work|remain(s)? to be explored)", re.IGNORECASE),
    ],
}


@dataclass(frozen=True)
class MoveSpan:
    label: str
    block_ids: list[str]
    mean_confidence: float


def label_moves(pdf_id: str, intro_blocks: list[TextBlock], *, field_hint: str | None) -> list[RhetoricalMove]:
    moves: list[RhetoricalMove] = []
    n = max(1, len(intro_blocks))
    for i, b in enumerate(intro_blocks):
        t = (b.clean_text or b.text or "").strip()
        lower = t.lower()

        scores: dict[str, float] = {lab: 0.0 for lab in _LABELS}
        scores["Other"] = 0.2

        # Pattern matches
        for lab, pats in _PATTERNS.items():
            for p in pats:
                if p.search(t):
                    scores[lab] += 1.0

        # Position priors
        if i <= 1:
            scores["Context"] += 0.6
            scores["Problem"] += 0.35
        if i >= n - 2:
            scores["Roadmap"] += 0.5
            scores["Contribution"] += 0.25

        # Special phrases
        if "in this paper" in lower:
            scores["Approach"] += 0.7
        if lower.startswith(("however", "nevertheless", "yet", "but")):
            scores["Gap"] += 0.6

        # Field hint (lightweight)
        if field_hint:
            fh = field_hint.lower()
            if "systems" in fh or "system" in fh:
                if "latency" in lower or "throughput" in lower:
                    scores["Problem"] += 0.25
            if "nlp" in fh:
                if "language" in lower or "text" in lower:
                    scores["Context"] += 0.2

        label, confidence = _pick_label(scores)
        moves.append(
            RhetoricalMove(
                move_id=uuid.uuid4().hex,
                label=label,  # type: ignore[arg-type]
                block_id=b.block_id,
                confidence=confidence,
            )
        )
    return moves


def label_moves_llm(*, intro_blocks: list[TextBlock], field_hint: str | None) -> list[RhetoricalMove]:
    cfg = require_llm_config()
    add_total("moves_blocks", len(intro_blocks))

    payload = {
        "task": "label_moves",
        "field_hint": (field_hint or "").strip(),
        "allowed_labels": _LABELS,
        "blocks": blocks_for_llm(intro_blocks, max_blocks=48, max_chars=800),
    }
    system = (
        "You are a precise academic writing analyst.\n"
        "Label each input block with exactly one rhetorical move label.\n"
        "Return JSON only. Do not include markdown.\n\n"
        "Output schema:\n"
        "{\n"
        '  "moves": [\n'
        '    {"block_id": "...", "label": "Context|Problem|Gap|Approach|Contribution|Roadmap|RelatedWorkHook|Claim|Limitation|Other", "confidence": 0.0..1.0}\n'
        "  ]\n"
        "}\n"
    )
    prior = moves_label_prior_text(field_hint=field_hint)
    if prior:
        system += "\n\nReference skills (do not quote verbatim):\n" + prior + "\n"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    raw = cfg  # keep cfg in local for mypy friendliness
    from shared.llm_client import chat_completions_json

    def normalize_label(label: object) -> tuple[str, bool]:
        if not isinstance(label, str):
            return "Other", True
        lab = label.strip()
        if not lab:
            return "Other", True
        for allowed in _LABELS:
            if allowed.lower() == lab.lower():
                return allowed, (allowed != lab)
        aliases = {
            "abstract": "Other",
            "title": "Other",
            "background": "Context",
            "motivation": "Context",
            "method": "Approach",
            "results": "Claim",
        }
        mapped = aliases.get(lab.lower())
        if mapped:
            return mapped, True
        return "Other", True

    def normalize_llm_output(obj: object) -> tuple[dict, int]:
        if not isinstance(obj, dict):
            return {}, 0
        moves = obj.get("moves")
        if not isinstance(moves, list):
            return obj, 0
        coerced = 0
        out_moves: list[dict] = []
        for m in moves:
            if not isinstance(m, dict):
                continue
            block_id = m.get("block_id")
            if not isinstance(block_id, str):
                block_id = str(block_id) if block_id is not None else ""
            label, changed = normalize_label(m.get("label"))
            if changed:
                coerced += 1
            conf = m.get("confidence")
            try:
                conf_f = float(conf)
            except Exception:  # noqa: BLE001
                conf_f = 0.7
            out_moves.append({"block_id": block_id, "label": label, "confidence": max(0.0, min(1.0, conf_f))})
        return {**obj, "moves": out_moves}, coerced

    try:
        out = chat_completions_json(cfg=raw, messages=messages, schema_name="moves")
        norm, coerced = normalize_llm_output(out)
        parsed = LlmMovesOutput.model_validate(norm)
    except Exception as e:  # noqa: BLE001
        mark_repair_used("moves")
        repair = {"role": "user", "content": f"Fix your output. Return ONLY a JSON object matching the schema. Error: {e}"}
        out = chat_completions_json(cfg=raw, messages=messages + [repair], schema_name="moves")
        norm, coerced = normalize_llm_output(out)
        parsed = LlmMovesOutput.model_validate(norm)

    if coerced:
        add_missing("moves_labels_coerced", coerced)

    by_block: dict[str, tuple[str, float]] = {}
    for m in parsed.moves:
        by_block[str(m.block_id)] = (str(m.label), float(m.confidence))

    moves: list[RhetoricalMove] = []
    missing = 0
    for b in intro_blocks:
        if b.block_id in by_block:
            label, conf = by_block[b.block_id]
        else:
            missing += 1
            label, conf = ("Other", 0.2)
        moves.append(
            RhetoricalMove(
                move_id=uuid.uuid4().hex,
                label=label,  # type: ignore[arg-type]
                block_id=b.block_id,
                confidence=max(0.0, min(1.0, float(conf))),
            )
        )
    add_missing("moves_blocks", missing)
    return moves


def label_moves_cached(
    pdf_id: str,
    intro_blocks: list[TextBlock],
    *,
    field_hint: str | None,
    data_dir: Path,
) -> tuple[list[RhetoricalMove], bool]:
    fh = (field_hint or "").strip().lower()
    cfg = load_llm_config()
    if llm_required() and cfg is None:
        raise RuntimeError("PAPER_SKILL_LLM_MODEL is required (PAPER_SKILL_REQUIRE_LLM=1).")
    mode = "llm" if cfg is not None else "rules"

    fingerprint = "\n".join(f"{b.block_id}\t{(b.clean_text or b.text or '').strip()}" for b in intro_blocks)
    llm_fp = ""
    if cfg is not None:
        llm_fp = json.dumps(
            {"model": cfg.model, "prompt_version": MOVES_PROMPT_VERSION},
            ensure_ascii=False,
            sort_keys=True,
        )
    skills_fp = skill_lib_fingerprint() or ""
    input_hash = sha256_text(f"{mode}\n{fh}\n{llm_fp}\nskills={skills_fp}\n{fingerprint}")
    path = stage_cache_path(data_dir=data_dir, stage="moves", key=pdf_id)
    payload = load_stage_payload(path=path, version=MOVES_CACHE_VERSION, input_hash=input_hash)
    if isinstance(payload, list):
        try:
            return [RhetoricalMove.model_validate(m) for m in payload], True
        except Exception:  # noqa: BLE001
            pass

    moves = label_moves_llm(intro_blocks=intro_blocks, field_hint=field_hint) if cfg is not None else label_moves(pdf_id, intro_blocks, field_hint=field_hint)
    save_stage_payload(path=path, version=MOVES_CACHE_VERSION, input_hash=input_hash, payload=[m.model_dump(mode="json") for m in moves])
    return moves, False


def _pick_label(scores: dict[str, float]) -> tuple[str, float]:
    # Softmax over scores for a calibrated-ish confidence.
    mx = max(scores.values()) if scores else 0.0
    exps = {k: math.exp(v - mx) for k, v in scores.items()}
    s = sum(exps.values()) or 1.0
    probs = {k: v / s for k, v in exps.items()}
    label = max(probs.items(), key=lambda kv: kv[1])[0]
    return label, float(probs[label])


def move_sequence(moves: list[RhetoricalMove]) -> list[str]:
    seq: list[str] = []
    for m in moves:
        if not seq or seq[-1] != m.label:
            seq.append(m.label)
    return seq


def compress_moves(moves: list[RhetoricalMove]) -> dict:
    spans: list[MoveSpan] = []
    current_label: str | None = None
    current_ids: list[str] = []
    current_confs: list[float] = []

    def flush() -> None:
        nonlocal current_label, current_ids, current_confs
        if current_label is None or not current_ids:
            return
        mean = sum(current_confs) / max(1, len(current_confs))
        spans.append(MoveSpan(label=current_label, block_ids=list(current_ids), mean_confidence=float(mean)))
        current_label = None
        current_ids = []
        current_confs = []

    for m in moves:
        if current_label is None:
            current_label = m.label
        if m.label != current_label:
            flush()
            current_label = m.label
        current_ids.append(m.block_id)
        current_confs.append(float(m.confidence))

    flush()
    return {
        "sequence": [s.label for s in spans],
        "spans": [{"label": s.label, "block_ids": s.block_ids, "mean_confidence": s.mean_confidence} for s in spans],
    }
