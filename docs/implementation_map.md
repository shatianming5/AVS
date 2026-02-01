# Implementation map (MIU → code)

This repo is a local demo implementation of `plan.md`. The MIU IDs below refer to sections in `plan.md`.

## Web UI

- Home build form: `frontend/index.html`, `frontend/app.js`
- Job status page: `frontend/job.html`, `frontend/job.js`
- Job events page: `frontend/job_events.html`, `frontend/job_events.js`
- Jobs list page: `frontend/jobs.html`, `frontend/jobs.js`
- SkillPack report page: `frontend/pack.html`, `frontend/pack.js`
- PDF viewer (page render + bbox highlights): `frontend/viewer.html`, `frontend/viewer.js`, `frontend/viewer.css`

## Backend API

- Server entrypoint + static: `backend/app.py`
- Pages router: `backend/routes/pages.py`
- API router: `backend/routes/api.py`

## Storage / DB / auth

- SQLite + migrations: `backend/services/db.py`, `backend/services/migrations.py`, `backend/services/init_db.py`
- PDFs store + access control: `backend/services/storage.py`, `backend/services/authz.py`, `backend/services/pdfs.py`
- SkillPack index rows + share tokens: `backend/services/skillpacks.py`
- Job lifecycle + events: `backend/services/jobs.py`, `backend/services/telemetry.py`

## Worker pipeline (MIU chain)

- MIU-03 PDF metadata: `worker/pipeline/metadata.py` (persisted via `backend/services/pdfs.update_pdf_metadata`)
- MIU-04 text blocks (+bbox): `worker/pipeline/extract_blocks.py`
- MIU-05 block type: `worker/pipeline/extract_blocks.py`
- MIU-06 section paths + intro locate: `worker/pipeline/sections.py`, `worker/pipeline/intro.py`
- MIU-07 intro cleaning: `worker/pipeline/cleaning.py`
- MIU-08 style features: `worker/pipeline/style.py`
- MIU-09 rhetorical moves: `worker/pipeline/moves.py`
- MIU-10 move compression: `worker/pipeline/moves.compress_moves`
- MIU-11 alignment: `worker/pipeline/alignment.py`
- MIU-12 blueprint: `worker/pipeline/blueprint.py`
- MIU-13 evidence pointers: `worker/pipeline/evidence.py`, `worker/pipeline/retrieval.py`
- MIU-14/15 templates: `worker/pipeline/templates.py`, `worker/pipeline/phrase_bank.py`
- MIU-16 plagiarism check: `worker/pipeline/plagiarism.py`
- MIU-17 captions: `worker/pipeline/captions.py`
- MIU-18 intro figure refs: `worker/pipeline/captions.py` (figure ref extraction in build output `patterns`)
- MIU-19 figure role inference: `worker/pipeline/captions.py`
- MIU-20 storyboard: `worker/pipeline/captions.py`
- MIU-21 pack artifacts: `worker/build.py` (writes `skillpacks/*.json|yaml` + `evidence/*.json`)
- MIU-22 quality report: `worker/pipeline/quality.py`
- MIU-26 job orchestration: `backend/services/jobs.py`, `worker/runner.py`
- MIU-27 cache: `worker/pipeline/cache.py` + per-stage `*_cached` functions
- MIU-28 telemetry: `backend/services/telemetry.py`, `backend/services/jobs.py` events
- MIU-29 privacy/sharing: `backend/services/authz.py`, `backend/routes/api.py` share/unshare endpoints

## LLM integration

- Config + JSON-mode fallback + model discovery: `shared/llm_client.py`
- Prompt payload helpers: `worker/pipeline/llm_payload.py`
- Per-stage stats + health: `worker/pipeline/llm_state.py`, `worker/pipeline/metrics.py`
- Offline deterministic LLM for testing: `scripts/fake_llm_server.py`

## E2E + evaluation

- Full headless UI regression: `scripts/e2e_playwright.py`
- Manual/MCP checklist: `docs/e2e_checklist.md`
- Pack metric evaluation (plan.md §10): `scripts/eval_pack.py`

