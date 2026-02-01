# paper_skill

Local demo implementation of `plan.md`.

## Quickstart

### 1) Install deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run the server

```bash
uvicorn backend.app:app --reload --port 8000
```

### 3) Run the worker (required)

In another terminal:

```bash
source .venv/bin/activate
python -m worker.runner
```

Open `http://127.0.0.1:8000/`.

Observability:

- Jobs list: `http://127.0.0.1:8000/jobs`
- Per-job events: `http://127.0.0.1:8000/job/<job_id>/events`

## Config

The app automatically loads a repo-root `.env` file at startup (if present).

- `PAPER_SKILL_DATA_DIR`: data root directory (default: `./data`)
 - `PAPER_SKILL_INLINE_WORKER`: set to `1` to run jobs inside the API process (dev only)
- `PAPER_SKILL_REQUIRE_LLM`: default `1` (must use LLM). Set to `0` to allow fallback to rule-based extraction.

### Local LLM (OpenAI-compatible)

The pipeline uses a local OpenAI-compatible Chat Completions API (default base URL points to localhost).

- `PAPER_SKILL_LLM_BASE_URL`: default `http://127.0.0.1:8001/v1`
- `PAPER_SKILL_LLM_MODEL`: **required** when `PAPER_SKILL_REQUIRE_LLM=1`
- `PAPER_SKILL_LLM_TIMEOUT_S`: default `60`
- `PAPER_SKILL_LLM_MAX_RETRIES`: default `2`
- `PAPER_SKILL_LLM_TEMPERATURE`: default `0.2`
- `PAPER_SKILL_LLM_API_KEY`: optional bearer token if your local gateway requires auth

Compatibility:

- If you already have `OPENAI_API_BASE` / `OPENAI_API_KEY` in your `.env`, the app will reuse them.
- If no model is configured, the app will try to auto-discover one via `GET /models`.

### Optional skill library (plan.md §9)

Use an external skill corpus to bias labels/rules and to surface usage metrics:

- `PAPER_SKILL_SKILL_LIB_PATH`: path to a JSONL file (1 JSON object per line). Supported keys include `topic`, `title`, `text`, optional `tags`, optional `source`/`url`.
- `PAPER_SKILL_SKILL_LIB_MAX_ITEMS`: max number of lines to load (default: `20000`).

## What you get

- Upload 3 PDFs
- Build a SkillPack (Intro blueprint + 3 templates + figure storyboard)
- Every rule/template/storyboard item links back to PDF page evidence with bbox highlights

## Data directory

Runtime data is stored under `./data/` (ignored by git):

- `data/pdfs/` uploaded PDFs
- `data/pages/` rendered page images (cached)
- `data/blocks/` extracted text blocks
- `data/cache/` cached intermediate results (style/moves)
- `data/skillpacks/` generated `skillpack.json` (+ optional YAML)
- `data/evidence/` `evidence_index.json`
- `data/paper_skill.sqlite3` SQLite DB for jobs/pdfs/packs

## Dev / tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

### Full UI E2E regression (headless Playwright)

```bash
pip install -r requirements-dev.txt
python -m playwright install chromium
python scripts/e2e_playwright.py
```

### LLM smoke run

Run an in-process build using your local LLM settings from `.env`:

```bash
python scripts/llm_smoke.py
```

### Evaluate a built pack (plan.md §10)

```bash
python scripts/eval_pack.py --job-id <job_id>
```
