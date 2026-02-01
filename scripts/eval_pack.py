from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from shared.schemas import SkillPack


def _data_dir(default: str = "data") -> Path:
    return Path(os.environ.get("PAPER_SKILL_DATA_DIR", default))


def _db_path(data_dir: Path) -> Path:
    return data_dir / "paper_skill.sqlite3"


def _load_pack(*, data_dir: Path, pack_id: str) -> SkillPack:
    path = data_dir / "skillpacks" / f"{pack_id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return SkillPack.model_validate(raw)


def _pack_id_from_job(*, data_dir: Path, job_id: str) -> str:
    db = _db_path(data_dir)
    if not db.exists():
        raise FileNotFoundError(f"DB not found: {db}")
    with sqlite3.connect(str(db)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT result_json FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if row is None:
        raise RuntimeError(f"Job not found: {job_id}")
    result_json = row["result_json"]
    if not result_json:
        raise RuntimeError("Job has no result_json yet.")
    result = json.loads(result_json)
    pack_id = (result or {}).get("pack_id")
    if not isinstance(pack_id, str) or not pack_id.strip():
        raise RuntimeError("Job result missing pack_id.")
    return pack_id


def _lcs_ratio(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    n = len(a)
    m = len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = dp[i - 1][j] if dp[i - 1][j] >= dp[i][j - 1] else dp[i][j - 1]
    return float(dp[n][m]) / float(max(n, m))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b) or 1
    return inter / union


def _norm_title(t: str) -> str:
    s = (t or "").strip().lower()
    if s.startswith("[suggestion]"):
        s = s[len("[suggestion]") :].strip()
    return " ".join(s.split())


@dataclass(frozen=True)
class EvalResult:
    ok: bool
    metrics: dict


def eval_pack(
    *,
    data_dir: Path,
    pack_id: str,
    min_evidence_coverage: float,
    min_structure_strength: float,
    min_stability: float,
) -> EvalResult:
    pack = _load_pack(data_dir=data_dir, pack_id=pack_id)

    rules = (pack.intro_blueprint.story_rules or []) + (pack.intro_blueprint.claim_rules or [])
    rules_total = len(rules)
    rules_with_ev = sum(1 for r in rules if (r.supporting_evidence or []))
    evidence_coverage_rules = (rules_with_ev / rules_total) if rules_total else 0.0

    templates = pack.templates or []
    templates_total = len(templates)
    templates_ok = 0
    for t in templates:
        if isinstance(t.slot_schema, dict) and len(t.slot_schema.keys()) >= 3 and t.text_with_slots and "{" in t.text_with_slots:
            templates_ok += 1
    template_slot_usability = (templates_ok / templates_total) if templates_total else 0.0

    structure_strength = float(getattr(pack.quality_report, "structure_strength", 0.0) or 0.0)

    para = pack.intro_blueprint.paragraph_plan or []
    para_sorted = sorted(para, key=lambda x: int((x or {}).get("paragraph_index") or 0))
    seq = [str((p or {}).get("label") or "") for p in para_sorted if str((p or {}).get("label") or "").strip()]

    rule_titles = {_norm_title(r.title) for r in rules if _norm_title(r.title)}

    stability = None
    stability_pair = None
    skillpacks_dir = data_dir / "skillpacks"
    built_at = str((pack.build_metadata or {}).get("built_at") or "")
    field_hint = (pack.build_metadata or {}).get("field_hint")
    venue_hint = (pack.build_metadata or {}).get("target_venue_hint")
    lang = (pack.build_metadata or {}).get("language")
    if skillpacks_dir.exists():
        candidates: list[tuple[str, str]] = []
        for p in sorted(skillpacks_dir.glob("*.json")):
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("pack_id") == pack.pack_id:
                continue
            if sorted(list(obj.get("pdf_ids") or [])) != sorted(list(pack.pdf_ids or [])):
                continue
            bm = obj.get("build_metadata") or {}
            if (bm.get("field_hint") or None) != field_hint:
                continue
            if (bm.get("target_venue_hint") or None) != venue_hint:
                continue
            if (bm.get("language") or None) != lang:
                continue
            other_built_at = str((bm.get("built_at") or ""))
            if not other_built_at:
                continue
            candidates.append((other_built_at, str(obj.get("pack_id") or "")))

        candidates.append((built_at, pack.pack_id))
        candidates = [(ts, pid) for ts, pid in candidates if pid and ts]
        candidates.sort(key=lambda x: x[0])
        if len(candidates) >= 2:
            ts1, pid1 = candidates[-2]
            ts2, pid2 = candidates[-1]
            p1 = _load_pack(data_dir=data_dir, pack_id=pid1)
            p2 = _load_pack(data_dir=data_dir, pack_id=pid2)

            def seq_for(p: SkillPack) -> list[str]:
                pp = p.intro_blueprint.paragraph_plan or []
                pp = sorted(pp, key=lambda x: int((x or {}).get("paragraph_index") or 0))
                return [str((x or {}).get("label") or "") for x in pp if str((x or {}).get("label") or "").strip()]

            def titles_for(p: SkillPack) -> set[str]:
                rs = (p.intro_blueprint.story_rules or []) + (p.intro_blueprint.claim_rules or [])
                return {_norm_title(r.title) for r in rs if _norm_title(r.title)}

            seq_sim = _lcs_ratio(seq_for(p1), seq_for(p2))
            rules_sim = _jaccard(titles_for(p1), titles_for(p2))
            stability = (seq_sim + rules_sim) / 2.0
            stability_pair = {"pack_id_a": pid1, "pack_id_b": pid2, "built_at_a": ts1, "built_at_b": ts2, "seq_sim": seq_sim, "rules_sim": rules_sim}

    metrics = {
        "pack_id": pack.pack_id,
        "pdf_ids": pack.pdf_ids,
        "built_at": built_at,
        "evidence_coverage_rules": float(evidence_coverage_rules),
        "template_slot_usability": float(template_slot_usability),
        "structure_strength": float(structure_strength),
        "output_stability": (float(stability) if stability is not None else None),
        "output_stability_pair": stability_pair,
        "rule_titles_count": len(rule_titles),
        "rules_total": int(rules_total),
        "rules_with_evidence": int(rules_with_ev),
    }
    ok = True
    ok = ok and (metrics["evidence_coverage_rules"] >= float(min_evidence_coverage))
    ok = ok and (metrics["structure_strength"] >= float(min_structure_strength))
    if stability is not None:
        ok = ok and (float(stability) >= float(min_stability))
    return EvalResult(ok=bool(ok), metrics=metrics)


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a built SkillPack against plan.md success metrics")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--pack-id", default=None)
    ap.add_argument("--job-id", default=None)
    ap.add_argument("--out", default=None, help="Optional path to write JSON metrics")
    ap.add_argument("--min-evidence-coverage", type=float, default=0.80)
    ap.add_argument("--min-structure-strength", type=float, default=0.60)
    ap.add_argument("--min-stability", type=float, default=0.90)
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else _data_dir()
    if args.pack_id:
        pack_id = str(args.pack_id)
    elif args.job_id:
        pack_id = _pack_id_from_job(data_dir=data_dir, job_id=str(args.job_id))
    else:
        raise SystemExit("Need --pack-id or --job-id")

    res = eval_pack(
        data_dir=data_dir,
        pack_id=pack_id,
        min_evidence_coverage=float(args.min_evidence_coverage),
        min_structure_strength=float(args.min_structure_strength),
        min_stability=float(args.min_stability),
    )
    out = {"ok": res.ok, "metrics": res.metrics}
    js = json.dumps(out, ensure_ascii=False, indent=2)
    print(js)
    if args.out:
        Path(args.out).write_text(js, encoding="utf-8")
    return 0 if res.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

