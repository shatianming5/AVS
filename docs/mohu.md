# mohu (Missing / Ambiguous tracking)

This file tracks what is still missing/unclear compared to `plan.md`, focusing on items that matter for a repeatable demo and iteration velocity.

## Implemented (recent)

### Done-001: Skill library hooks (plan.md §9)
- Summary: Added optional skill library loader (`PAPER_SKILL_SKILL_LIB_PATH`) and used it for (a) move-label prompt priors, (b) blueprint rule candidates, (c) lightweight evidence keyword hint in fallback, (d) quality/metrics reporting.
- Evidence: `shared/skill_lib.py`, `worker/pipeline/skill_hints.py`, `worker/pipeline/moves.py`, `worker/pipeline/blueprint.py`, `worker/pipeline/quality.py`, `worker/build.py`.

### Done-002: Quantitative pack evaluation (plan.md §10)
- Summary: Added `scripts/eval_pack.py` to compute evidence coverage (rules), template slot usability, structure strength, and output stability across repeated builds.

### Done-003: Full E2E regression (UI)
- Summary: Added `scripts/e2e_playwright.py` to cover all user-facing flows end-to-end (upload/build/job/pack/viewer/download/share/jobs/events/cache hits).

## Ambiguous

### Amb-001: Skill library source-of-truth format
- Current assumption: JSONL where each line is a JSON object. Normalized fields: `topic`, `title`, `text`, optional `tags`, optional `source` (or `url`).
- Open question: do you want a different schema (e.g. nested sections, per-skill IDs, multiple examples) and do you want to support directory-of-markdown skills?
- Acceptance criteria: provide one canonical fixture file and document it in `README.md`.

## Missing

### Missing-001: Evidence retrieval hints for MIU-13 beyond fallbacks
- Gap: Most evidence is currently bound via LLM-selected `supporting_block_ids`; skill-derived keywords are only used in a fallback selection path.
- Acceptance criteria: when LLM returns a rule with empty/invalid `supporting_block_ids`, attempt a retrieval-based evidence binding using label + skill keyword hints and record this in `QualityReport.notes`.

### Missing-002: Shareable “Style Fingerprint Card” (plan.md §10.2)
- Gap: No UI artifact for quick sharing (screenshot-friendly card).
- Acceptance criteria: add a section on the pack page showing a compact “style fingerprint” (sentence length p50/p90, hedging density, connector top-5, citation density) with a stable layout.

### Missing-003: CI-friendly E2E runner
- Gap: E2E requires Playwright browser install; not wired into any CI task.
- Acceptance criteria: add a single command (e.g. `make e2e` or `python -m scripts.e2e_playwright`) and document required system deps; optionally add a GitHub Actions workflow.

