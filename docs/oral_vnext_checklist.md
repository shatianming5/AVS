# Oral vNext Minimal-Decisive Checklist

Date: 2026-02-14

Scope: only the **smallest** set of work that can still change an “oral-level” outcome.
This is intentionally *not* a full TODO list.

## 0) One decision (pick one track)

- [x] **Track C (current):** `C0003 (+2%)` is proven via PSP/CPSP Stage-1 teacher + keepadj+hconf Stage-2; freeze this as the main result and ship the oral pack.
- [ ] **Track A (contingency):** if C0003 cannot be proven, lock the revised claim and make the narrative reviewer-proof.
- [x] **Track B:** attempted and failed the bounded promotion gate (ImageBind + WavLM); do not spend more budget there.

---

## Track C: C0003-Proven Packaging (minimal updates)

- [x] C1: Freeze the single “main” `test402` config to report for `C0003` (selection rule sealed; no p-hacking).
  - Main (C0003-proven): `runs/E0980_full_test402_psp_evt_gini_keepadj_hconf_best_s0-9_20260214-031741/metrics.json` (Δ=+0.02169; p=0.00149).
  - Selected config JSON: `runs/E0978_val402_psp_evt_gini_keepadj_hconf_v1_20260214-030933/best_config.json` (name=`ltlgini_keepadj_gini0p45_hconf0p5`).
  - Prior best (for context/sensitivity): `runs/E0643_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_df7_officialids_s0-9_20260211-001604/metrics.json` (Δ=+0.01045; p=0.0395).

- [x] C2: Regenerate the 1-slide “C0003 decomposition” with the winning run.
  - Must include: Oracle ceiling, Oracle→Pred gap, and the two most harmful buckets (far-anchors / 2-high regime).
  - Evidence pointers:
    - Oracle→Pred gap grid: `runs/E0330_mde_pareto_grid_official_psp_avel_evt_20260214-155549/pareto_report.json`
    - Diagnose (main): `runs/E0980_full_test402_psp_evt_gini_keepadj_hconf_best_s0-9_20260214-031741/diagnose.json`
    - Slide content: `docs/oral_narrative.md` (Section 4.1).

- [x] C3: Related-work slide update (query-aware selection vs token reallocation vs training-free selection).
  - Acceptance: one small positioning table + one “claim boundary” line: **controlled equal-budget frame selection**, not long-video leaderboard SOTA.
  - Doc to update: `docs/oral_related_work.md`

- [x] C4: Decide whether to add an extra long-video benchmark (only if it materially reduces reviewer risk).
  - Default: **NO** (already have IntentQA/EgoSchema/AVQA + EPIC-SOUNDS under strict budgets).
  - If YES: pick exactly one benchmark (Video-MME or LongVideoBench), run a small, reproducible subset (e.g., `n=256`) with `text_only/random` controls and fixed `B_FRAMES` budgets.
  - Decision: NO extra benchmark for vNext; treat Video-MME/LongVideoBench as “related-work + optional future eval”.

- [x] C5: Reproducibility seal before freeze.
  - Commands:
    - `bash scripts/datasets/verify_all.sh`
    - `python scripts/plan_evidence_matrix.py --write-docs-md`
  - Acceptance: all referenced artifacts exist locally; docs update is consistent with the latest run roots.
  - Evidence:
    - `runs/datasets_verify_20260214-033737/datasets_verify.json`
    - `runs/evidence_matrix_20260214-033747/evidence_matrix.json`

- [x] C6: Slide assembly helpers (so packaging is truly “done”).
  - Exports:
    - `docs/oral_assets/fig4_qa_budget_curve_{intentqa,avqa}.png`
  - Slide-by-slide outline:
    - `docs/oral_deck_outline.md`

---

## Track A: Packaging-First (when C0003 is not proven)

- [x] A1: Pick the *single* “main” `test402` config to report for `C0003` (and pre-register the selection rule; no p-hacking).
  - Candidate 1 (better `p`, smaller Δ): `runs/E0643_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_df7_officialids_s0-9_20260211-001604/metrics.json` (Δ=+0.01045; p=0.0395).
  - Candidate 2 (bigger mean Δ, weaker `p`): `runs/E0638_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_df5_officialids_s0-9_20260211-001009/metrics.json` (Δ=+0.01117; p=0.109).
  - Acceptance: one becomes “Main”, the other becomes a sensitivity/appendix entry with a clear reason.
  - Decision (historical): main=`E0643 (df7; p<0.05)`, sensitivity=`E0638 (df5; higher mean Δ but not significant)`; rule recorded in `docs/oral_checklist.md`.

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

- [x] B3: Extra “qualitatively different” Stage-1 probes (override stop rule; still failed on val402).
  - XAttn MIL Stage-1 (`av_wavlm_clip_xattn_mil`): `E0902~E0905` (near-zero/negative on val; not promotable).
  - High-res vision binary Stage-1 (`vision_binary_mlp_r352`): `E0906~E0907` (negative on val; not promotable).
  - See: `docs/oral_competitive_queue.md` (Track M/N) + `docs/experiment.md` (E0902~E0907).

- [x] B4: Vision backbone swap attempt (timm EVA02 caches; Stage-2 swap + Stage-1-only override) — still not competitive.
  - Cache build (112/160/224/352): `runs/E0915_build_cache_eva02_clip_p16_112_160_224_352_20260212-225043/cache_build.json` + `runs/E0915_build_cache_eva02_clip_p16_112_160_224_352_test_20260212-230913/cache_build.json`
  - Val402 sweeps:
    - EVA02 Stage-2: `runs/E0916_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_adaptive_keepadj_v1_eva02_20260212-231218/sweep_summary.json` (best Δ≈+0.00241)
    - EVA02 Stage-1 only: `runs/E0917_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_adaptive_keepadj_v1_stage1eva02_20260212-231759/sweep_summary.json` (best Δ≈+0.00324), `runs/E0918_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_top1med_norm_v1_stage1eva02_20260212-232240/sweep_summary.json` (best Δ≈+0.00183)
  - Decision: not promoted to test402.
