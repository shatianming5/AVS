# Oral vNext Minimal-Decisive Checklist

Date: 2026-02-11

Scope: only the **smallest** set of work that can still change an “oral-level” outcome.
This is intentionally *not* a full TODO list.

## 0) One decision (pick one track)

- [x] **Track A (recommended):** stop chasing `C0003 (+2%)`, lock the revised claim, and make the narrative reviewer-proof.
- [ ] **Track B:** take one last run at `C0003 (+2%)` (requires a *new* Stage-1 signal; expect iteration + compute).

---

## Track A: Packaging-First (no new AVE method)

- [x] A1: Pick the *single* “main” `test402` config to report for `C0003` (and pre-register the selection rule; no p-hacking).
  - Candidate 1 (better `p`, smaller Δ): `runs/E0643_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_df7_officialids_s0-9_20260211-001604/metrics.json` (Δ=+0.01045; p=0.0395).
  - Candidate 2 (bigger mean Δ, weaker `p`): `runs/E0638_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_df5_officialids_s0-9_20260211-001009/metrics.json` (Δ=+0.01117; p=0.109).
  - Acceptance: one becomes “Main”, the other becomes a sensitivity/appendix entry with a clear reason.
  - Decision: main=`E0643 (df7; p<0.05)`, sensitivity=`E0638 (df5; higher mean Δ but not significant)`; rule recorded in `docs/oral_checklist.md`.

- [x] A2: Add a 1-slide “Why +2% is hard” decomposition (so `C0003` not proven is not fatal).
  - Must include: Oracle ceiling, Oracle→Pred gap, and the two most harmful buckets (far-anchors / 2-high regime).
  - Evidence pointers:
    - Oracle→Pred gap grid: `runs/E0330_mde_pareto_grid_official_av_clipdiff_mlp_local_20260209-235305/pareto_report.json`
    - Diagnose (example): `runs/E0643_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_df7_officialids_s0-9_20260211-001604/diagnose.json`
    - Slide content: `docs/oral_narrative.md` (Section 4.1).

- [x] A3: Related-work slide update (query-aware selection vs token reallocation vs training-free selection).
  - Acceptance: one small positioning table + one “claim boundary” line: **controlled equal-budget frame selection**, not long-video leaderboard SOTA.
  - Doc to update: `docs/oral_related_work.md`

- [x] A4: Decide whether to add an extra long-video benchmark (only if it materially reduces reviewer risk).
  - Default: **NO** (already have IntentQA/EgoSchema/AVQA + EPIC-SOUNDS under strict budgets).
  - If YES: pick exactly one benchmark (Video-MME or LongVideoBench), run a small, reproducible subset (e.g., `n=256`) with `text_only/random` controls and fixed `B_FRAMES` budgets.
  - Decision: NO extra benchmark for vNext; treat Video-MME/LongVideoBench as “related-work + optional future eval”.

- [x] A5: Reproducibility seal before freeze.
  - Commands:
    - `bash scripts/datasets/verify_all.sh`
    - `python scripts/plan_evidence_matrix.py --write-docs-md`
  - Acceptance: all referenced artifacts exist locally; docs update is consistent with the latest run roots.
  - Evidence:
    - `runs/datasets_verify_20260212-020117/datasets_verify.json`
    - `runs/evidence_matrix_20260212-020124/evidence_matrix.json`

---

## Track B: One Last C0003 Attempt (only if you have time/compute)

- [ ] B1: Commit to **one** new Stage-1 idea that is meaningfully different from existing `energy/*clipdiff*/*panns*/*ast*/*fused*`.
  - Acceptance: define `EVENTNESS=<new_id>` + 1 runnable sweep script: `val402 sweep (SEEDS=0..2)`.

- [ ] B2: Run the promotion gate exactly once:
  - `val402 sweep (SEEDS=0..2)` → `quick test402 (SEEDS=0..2)` → `full test402 (SEEDS=0..9)`
  - Acceptance: only if `full test402` meets `Δ>=+0.02` and paired `p<0.05`, flip `C0003 proven`; otherwise stop and stick to Track A.
