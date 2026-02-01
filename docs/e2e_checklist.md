# E2E checklist (manual / Playwright MCP)

Use this list when doing a full end-to-end regression on the local demo.

Preferred (repeatable): run `python scripts/e2e_playwright.py` (headless Playwright). Use this checklist mainly when debugging UI manually or when Playwright MCP is available.

## Index → Build → Pack

- Open `/` and verify the form loads.
- Select **exactly 3 PDFs**, fill a pack name, click **Build SkillPack**.
- Confirm navigation `/job/<job_id>?token=...` and eventual redirect to `/pack/<pack_id>?token=...`.

## Pack → Viewer evidence

- Verify sections exist: Intro Blueprint / Templates / Figure storyboard / Quality report.
- Verify at least 1 evidence link exists (`a.evidence`).
- Click evidence link → Viewer loads image and shows `.hl` highlight(s).
- Multi-bbox evidence: pick a caption evidence with `bbox_norm_list` and confirm multiple `.hl` boxes render.

## Downloads

- Download JSON/YAML from Pack page → HTTP 200 and content parses / non-empty.

## Jobs + Events

- Open `/jobs?token=...` and ensure the latest job is listed and links work.
- Open `/job/<id>/events?token=...` and confirm metrics + events render.

## Share / Unshare / Access control

- Click Share on a pack; open the share URL in a fresh browser context → pack loads.
- Click Unshare; verify the old share token can no longer access `/api/skillpacks/<pack_id>` (403).
- With no token/localStorage, open `/jobs` and `/pack/<pack_id>` → should show “Missing token”.

## Cache hits

- Trigger a second build with the same `pdf_ids` and hints.
- Verify `/jobs` shows `metrics.cache_hits` has pack-level hits (`blueprint/templates/storyboard/quality` true).
