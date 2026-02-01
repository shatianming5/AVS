from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass

from shared.skill_lib import SkillItem, load_skill_lib_from_env


@dataclass(frozen=True)
class SkillLibInfo:
    used: bool
    fingerprint: str | None
    items_loaded: int
    topics_loaded: int
    path_hint: str | None


def _tokenize(text: str) -> list[str]:
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    toks = [x for x in t.split() if len(x) >= 3]
    uniq: list[str] = []
    seen = set()
    for x in toks:
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
    return uniq


def _select_skills(*, items: list[SkillItem], field_hint: str | None, limit: int) -> list[SkillItem]:
    if not items or limit <= 0:
        return []
    toks = _tokenize(field_hint or "")
    if not toks:
        return items[:limit]

    def score(it: SkillItem) -> tuple[int, str, str]:
        hay = f"{it.topic} {it.title} {it.text}".lower()
        s = sum(1 for tok in toks if tok in hay)
        return (s, (it.topic or "").lower(), (it.title or "").lower())

    scored = [(score(it), it) for it in items]
    scored.sort(key=lambda x: (-x[0][0], x[0][1], x[0][2]))
    return [it for _, it in scored[:limit]]


def skill_lib_info() -> SkillLibInfo:
    try:
        lib = load_skill_lib_from_env()
    except Exception:  # noqa: BLE001
        lib = None
    if lib is None:
        return SkillLibInfo(used=False, fingerprint=None, items_loaded=0, topics_loaded=0, path_hint=None)
    raw = (os.environ.get("PAPER_SKILL_SKILL_LIB_PATH") or "").strip()
    hint = None
    if raw:
        hint = os.path.basename(raw)
    return SkillLibInfo(
        used=True,
        fingerprint=lib.fingerprint,
        items_loaded=len(lib.items),
        topics_loaded=len(lib.topics_count),
        path_hint=hint,
    )


def skill_lib_fingerprint() -> str | None:
    try:
        lib = load_skill_lib_from_env()
    except Exception:  # noqa: BLE001
        lib = None
    return (lib.fingerprint if lib is not None else None)


def selected_skill_topics(*, field_hint: str | None, limit: int = 24) -> dict[str, int]:
    try:
        lib = load_skill_lib_from_env()
    except Exception:  # noqa: BLE001
        lib = None
    if lib is None:
        return {}
    picked = _select_skills(items=lib.items, field_hint=field_hint, limit=limit)
    counts: dict[str, int] = {}
    for it in picked:
        topic = (it.topic or "").strip() or "unknown"
        counts[topic] = counts.get(topic, 0) + 1
    return counts


def moves_label_prior_text(*, field_hint: str | None, limit: int = 20, max_chars: int = 1400) -> str | None:
    try:
        lib = load_skill_lib_from_env()
    except Exception:  # noqa: BLE001
        lib = None
    if lib is None:
        return None

    picked = _select_skills(items=lib.items, field_hint=field_hint, limit=limit)
    if not picked:
        return None
    lines: list[str] = []
    total = 0
    for it in picked:
        head = f"[{(it.topic or 'unknown').strip()}] {(it.title or '').strip()}".strip()
        body = (it.text or "").strip().replace("\n", " ")
        body = re.sub(r"\s+", " ", body).strip()
        if len(body) > 220:
            body = body[:220].rstrip() + "…"
        line = f"- {head}: {body}" if body else f"- {head}"
        if total + len(line) + 1 > max_chars:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines) if lines else None


def blueprint_rule_candidates(*, field_hint: str | None, limit: int = 60) -> list[dict]:
    try:
        lib = load_skill_lib_from_env()
    except Exception:  # noqa: BLE001
        lib = None
    if lib is None or not lib.items:
        return []

    toks = set(_tokenize(field_hint or ""))
    must_terms = {"intro", "introduction", "paper", "writing", "academic", "contribution", "roadmap", "gap", "problem"}

    def score(it: SkillItem) -> tuple[int, str, str]:
        hay = f"{it.topic} {it.title} {it.text}".lower()
        s = 0
        s += sum(1 for tok in toks if tok in hay)
        s += sum(1 for t in must_terms if t in hay)
        return (s, (it.topic or "").lower(), (it.title or "").lower())

    scored = [(score(it), it) for it in lib.items]
    scored.sort(key=lambda x: (-x[0][0], x[0][1], x[0][2]))
    out: list[dict] = []
    for _, it in scored[: max(200, limit * 3)]:
        title = (it.title or "").strip()
        text = (it.text or "").strip()
        if not title:
            title = (text[:80].split("\n")[0].strip() if text else "")
        if not title:
            continue
        desc = text.replace("\n", " ")
        desc = re.sub(r"\s+", " ", desc).strip()
        if len(desc) > 240:
            desc = desc[:240].rstrip() + "…"
        cid = hashlib.sha256(f"{it.topic}|{title}|{desc}".encode("utf-8")).hexdigest()[:12]
        out.append(
            {
                "candidate_id": cid,
                "topic": (it.topic or "unknown").strip(),
                "title": title,
                "description": desc,
                "tags": list(it.tags or []),
            }
        )
        if len(out) >= limit:
            break
    return out


def evidence_keyword_pattern(*, field_hint: str | None, limit_terms: int = 12) -> re.Pattern | None:
    try:
        lib = load_skill_lib_from_env()
    except Exception:  # noqa: BLE001
        lib = None
    if lib is None or not lib.items:
        return None

    picked = _select_skills(items=lib.items, field_hint=field_hint, limit=18)
    terms: list[str] = []
    for it in picked:
        for tag in (it.tags or []):
            tt = str(tag).strip().lower()
            if 3 <= len(tt) <= 24 and tt not in terms:
                terms.append(tt)
        for tok in _tokenize(it.title or "")[:6]:
            if tok not in terms:
                terms.append(tok)
        if len(terms) >= limit_terms:
            break
    if not terms:
        return None
    parts = [re.escape(t) for t in terms[:limit_terms]]
    return re.compile(r"(" + "|".join(parts) + r")", re.IGNORECASE)


def _norm_title(t: str) -> str:
    s = (t or "").lower()
    s = s.replace("[suggestion]", "")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def estimate_skill_rules_adopted(*, rule_titles: list[str], candidates: list[dict]) -> int:
    cand_titles = [_norm_title(str(c.get("title") or "")) for c in candidates]
    cand_set = [set(ct.split()) for ct in cand_titles if ct]
    adopted = 0
    for rt in rule_titles:
        nt = _norm_title(rt)
        if not nt:
            continue
        tokens = set(nt.split())
        best = 0.0
        for c_tokens in cand_set:
            if not c_tokens:
                continue
            inter = len(tokens & c_tokens)
            union = len(tokens | c_tokens) or 1
            j = inter / union
            if j > best:
                best = j
        if best >= 0.85:
            adopted += 1
    return adopted

