# Oral vNext Minimal-Decisive Checklist

Date: 2026-02-12

Scope: only the **smallest** set of work that can still change an “oral-level” outcome.
This is intentionally *not* a full TODO list.

## 0) One decision (pick one track)

- [x] **Track A (recommended):** stop chasing `C0003 (+2%)`, lock the revised claim, and make the narrative reviewer-proof.
- [x] **Track B:** attempted and failed the promotion gate (ImageBind + WavLM); stop chasing `C0003 (+2%)`.

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

- [x] A6: Slide assembly helpers (so packaging is truly “done”).
  - Exports:
    - `docs/oral_assets/fig4_qa_budget_curve_{intentqa,avqa}.png`
  - Slide-by-slide outline:
    - `docs/oral_deck_outline.md`

---

## Track B: One Last C0003 Attempt (only if you have time/compute)

- [x] B1: Commit to new Stage-1 ideas (bounded search: 2 ideas max).
  - Tried:
    - `imagebind_av_sim` (ImageBind AV-consistency)
    - `wavlm_evt_mlp` (WavLM supervised eventness)

- [x] B2: Run the fixed promotion gate once per idea.
  - ImageBind:
    - val402: `runs/E0801_val402_imagebind_keepadjv2_20260212-035956/sweep_summary.json` (best Δ≈-0.00008)
    - quick test402: `runs/E0802_quick_test402_imagebind_20260212-040440/metrics.json` (Δ≈-0.00265; p≈0.754) → not promoted
  - WavLM:
    - val402: `runs/E0810_val402_wavlm_20260212-041931/sweep_summary.json` (best Δ≈-0.00424)
    - quick test402: `runs/E0811_quick_test402_wavlm_20260212-042425/metrics.json` (Δ≈+0.00124; p≈0.918) → not promoted
