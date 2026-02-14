# Oral Narrative Pack (Listen-then-Look / AVE-P0)

This document is the **oral-ready story skeleton** that maps directly to the experiment ledger (`docs/experiment.md`) and the actionable claims (`docs/plan.md`).

## 1) What Is New (One Sentence)

**Audio is a cheap temporal index**: we “listen” to propose a small set of candidate moments, then **allocate the visual token budget only where evidence is likely**, instead of uniformly spending compute across time.

## 2) Budget Definition Is Sealed (No Wiggle Room)

We evaluate under **strict equal visual-token budgets**.

- Unit: **ViT patch tokens** fed to the visual encoder.
- For AVE (10s clips): fixed budget points (see the multi-budget grid below).
- For EPIC-SOUNDS (up to 120s): budget defined as `max_steps × base_res` (token-equivalent) with a deterministic equal-budget low/base/high plan for anchored methods.
- Accounting artifacts:
  - Budgeted Pareto report + plot: `runs/E0330_mde_pareto_grid_official_psp_avel_evt_20260214-155549/pareto_report.json`, `runs/E0330_mde_pareto_grid_official_psp_avel_evt_20260214-155549/pareto.png`
  - Slide export: `docs/oral_assets/fig1_pareto.png`
  - Vision efficiency calibration (tokens↔latency reference): `runs/E0408_vision_efficiency_20260209-233151/vision_efficiency.json`

## 3) Pre-Registered MDE Protocol (Oracle → Predicted + Controls)

For each fixed budget, we always compare the same method family:

1. **Oracle anchors** (upper bound at the same budget)
2. **Predicted anchors** (deployable Stage-1)
3. Controls:
   - **Uniform** (spend budget evenly)
   - **Random anchors** (tests “any window works?”)
   - **Cheap-visual anchors** (tests “audio is not special?”)

Deliverable that packages this into one “生死图”:
- MDE Pareto grid: `runs/E0330_mde_pareto_grid_official_psp_avel_evt_20260214-155549/pareto.png`
- Oracle→Predicted gap (read `oracle_minus_predicted` deltas from `pareto_report.json`): `runs/E0330_mde_pareto_grid_official_psp_avel_evt_20260214-155549/pareto_report.json`

## 4) Main Mechanism Evidence (What Reviewers Must Believe)

**(i) There is a strong ceiling at fixed budget (Oracle).**  
**(ii) Predicted remains behind Oracle (Stage-1 reliability gap) and can regress vs Uniform.**  
**(iii) Random/cheap-visual do not trivially match predicted (controls).**

Key evidence pointers:
- Oracle vs Predicted gap (multi-budget; includes token_budget=1960 under `triad=160_224_352`): `runs/E0330_mde_pareto_grid_official_psp_avel_evt_20260214-155549/pareto_report.json`

### 4.1 C0003 (+2%) Decomposition (Why It’s Hard)

This is the “one-slide” explanation for why the `+2%` hard gate is hard, and what would need to change.

- Slide export: `docs/oral_assets/fig2_c0003_decomposition.png`

- Ceiling exists at fixed budget: in the multi-budget grid at `token_budget=1960` (`triad=160_224_352`), `oracle - uniform ≈ +0.02944` abs (`pareto_report.json`).
- Best deployable C0003 run meets the hard gate: `runs/E0980_full_test402_psp_evt_gini_keepadj_hconf_best_s0-9_20260214-031741/metrics.json` reports `anchored_top2 - uniform = +0.02169` (paired `p=0.00149`, `SEEDS=0..9`).
  - Prior best (pre-PSP): `runs/E0643_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_df7_officialids_s0-9_20260211-001604/metrics.json` (Δ=+0.01045; `p≈0.0395`).
- Key decomposition signal is still “dilution vs anchor-used buckets”, but the keepadj+hconf plan shifts the mixture toward higher-gain regimes (diagnose for E0980):
  - Fallback dilution: `fallback_used_frac≈0.709` (285/402 clips fall back to uniform because `conf_below_threshold`).
  - Mean Δ by high-count: `high_count=0 mean_delta≈+0.01547` (`n=285`), `high_count=1 mean_delta≈+0.02016` (`n=62`), `high_count=2 mean_delta≈+0.05564` (`n=55`).
  - Only adjacent anchor pairs are used (`anchor_dist_hist` contains only dist=1).
  - Artifact: `runs/E0980_full_test402_psp_evt_gini_keepadj_hconf_best_s0-9_20260214-031741/diagnose.json`.

## 5) Why It Works / When It Fails (Bucketed Diagnosis + Evidence Alignment)

We explicitly show **where gains come from** and **where they disappear**.

- Bucketed diagnosis identifies failure modes:
  - fallback-heavy clips (reliability gate forces uniform)
  - far-anchor plans (anchors too far from evidence)
  - “2-high harms context” regime (too many high-res anchors can remove context)
- Evidence Alignment (Coverage@τ) is reported as a **diagnostic**:
  - It is **weakly correlated** with downstream accuracy deltas in our current setup (negative-but-clean evidence).

Evidence pointers:
- Evidence Alignment report (energy baseline; test402): `runs/E0202_evidence_alignment_energy_test402_20260209-061145/evidence_alignment.json`
- Evidence Alignment report (C0003-proven PSP+hconf; positive corr): `runs/E0981_evidence_alignment_psp_keepadj_hconf_best_test402_20260214-033440/evidence_alignment.json`

## 6) Robustness (Degradation Curves + Alpha Lower Bound)

We pre-register and execute a **degradation protocol** (shift/noise/silence × α) and require:

- Anchor quality degrades smoothly under perturbations.
- Downstream anchored accuracy stays **no worse than the computable α-floor** (uniform fallback baseline).

Evidence pointers:
- Anchor-quality degradation suite (Recall@K,Δ grid): `runs/E0203_degradation_energy_20260209-061156/degradation_suite.json`
- Downstream degradation-accuracy grid + alpha floor checks (rows=54): `runs/E0331_degradation_accuracy_psp_avel_evt_20260214-161014/degradation_accuracy.json`
  - Slide exports:
    - `docs/oral_assets/fig3_degradation_delta_acc_alpha0p5.png`
    - `docs/oral_assets/fig3_degradation_recall_d0_alpha0p5.png`
  - Note: for `psp_avel_evt`, degradations are applied in score space (external teacher is not recomputed from raw audio).

## 7) Cross-Dataset Proxy (Long Video)

We verify the “listen-then-look” indexing story in a long-video downstream task:

- EPIC-SOUNDS (expanded t256/v137, strict equal-budget; SEEDS=0,1,2):
  - Anchored: `runs/E0100_epic_video_cls_local_audio_anchored_full_ms120_t256_v137_s012_20260209-235834/metrics.json`
  - Uniform: `runs/E0100_epic_video_cls_local_uniform_full_ms120_t256_v137_s012_20260210-001346/metrics.json`
  - Random: `runs/E0100_epic_video_cls_local_random_full_ms120_t256_v137_s012_20260210-001929/metrics.json`

Optional diagnostic (tighter budget, `max_steps=60`): see `docs/experiment.md` E0100 notes.

Long-video QA add-on budget curves (slide exports):
- IntentQA: `docs/oral_assets/fig4_qa_budget_curve_intentqa.png`
- AVQA: `docs/oral_assets/fig4_qa_budget_curve_avqa.png`

## 8) Reproducibility (What We Provide)

We maintain:

- Runnable commands + required artifacts in `docs/experiment.md`
- A conclusion→evidence matrix in `docs/evidence_matrix.md`
- Per-experiment logs under `artifacts/experiments/*/run.log`

Dataset verification is captured by:
- `python scripts/datasets/verify_all.py` (outputs `runs/datasets_verify_*/datasets_verify.json`)

## 9) Latest Queue Update (E0980)

- C0003 hard gate is now proven by PSP/CPSP AVEL Stage-1 + keepadj+hconf Stage-2:
  - Full test402: `runs/E0980_full_test402_psp_evt_gini_keepadj_hconf_best_s0-9_20260214-031741/metrics.json` (Δ=+0.02169; p=0.00149; `SEEDS=0..9`)
  - Diagnose: `runs/E0980_full_test402_psp_evt_gini_keepadj_hconf_best_s0-9_20260214-031741/diagnose.json`
  - Evidence Alignment: `runs/E0981_evidence_alignment_psp_keepadj_hconf_best_test402_20260214-033440/evidence_alignment.json`

Long-video seed extension status (unchanged):
- IntentQA faithfulness seed=2: `runs/E0713_intentqa_faithfulness_val_s2_20260212-000949/faithfulness.json` (stable vs prior seeds).
- EgoSchema seed=2: `runs/E0714_egoschema_eval_subset256_s2_20260212-004316/metrics.json` (`uniform=0.5859`, `ql2l_clap=0.5352`, `ql2l_asr_bm25=0.5469`; matches prior seeds).
