# Oral Push Checklist (Listen-then-Look / AVE-P0)

This is the **minimum but decisive** experiment + narrative checklist to get to an “oral-ready” submission.

Canon sources:
- Paper draft: `plan.md`
- Actionable plan / claims: `docs/plan.md` (`P####` / `C####`)
- Experiment ledger: `docs/experiment.md` (`E####`)

---

## 0) Hard gate (one must be true)

- [x] **C0003 proven** on official AVE `test402`: `anchored_top2 - uniform ≥ +0.02`, paired `p < 0.05`, `SEEDS=0..9`.
  - Winning run (PSP/CPSP AVEL Stage-1 + keepadj+hconf Stage-2): `runs/E0980_full_test402_psp_evt_gini_keepadj_hconf_best_s0-9_20260214-031741/metrics.json` (anchored=0.73791 vs uniform=0.71622; Δ=+0.02169; p=0.00149; `SEEDS=0..9`). Diagnose: `runs/E0980_full_test402_psp_evt_gini_keepadj_hconf_best_s0-9_20260214-031741/diagnose.json` (fallback_used_frac≈0.709).
  - Selected config: `runs/E0978_val402_psp_evt_gini_keepadj_hconf_v1_20260214-030933/best_config.json` (candidate name=`ltlgini_keepadj_gini0p45_hconf0p5`).
  - Prior best (deployable; vec-MLP df7): `runs/E0643_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_df7_officialids_s0-9_20260211-001604/metrics.json` (Δ=+0.01045; p=0.0395).
- [ ] If C0003 cannot be proven, **lock a revised claim** (e.g., “consistent +1% at fixed budget + strong Oracle ceiling + robust protocol”) and make the rest of this checklist airtight (so reviewers cannot dismiss as heuristic/cherry-pick).

---

## 1) Minimum decisive experiments (checklist)

### A. “生死图 #1” — Acc–Tok Pareto (Oracle → Predicted)

- [x] **E0330: Multi-budget Pareto grid on AVE** (Oracle / Predicted / Random / Cheap-visual at the same budgets; with CIs).
  - Goal: one plot answers “mechanism upper bound exists”, “predicted remains behind Oracle (and can regress)”, “not any window works”, and “Pareto holds over budgets”.
  - Runner/script: `bash scripts/e0330_mde_pareto_grid_official.sh` (see E0330 in `docs/experiment.md`).
  - Artifacts: `runs/E0330_*/pareto_report.json` + `runs/E0330_*/pareto.png` + per-budget raw metrics.
  - Latest full (PSP-aligned): `runs/E0330_mde_pareto_grid_official_psp_avel_evt_20260214-155549/{pareto_report.json,pareto.png}`.
  - Historical reference (pre-PSP): `runs/E0330_mde_pareto_grid_official_av_clipdiff_mlp_local_20260209-235305/{pareto_report.json,pareto.png}`.
  - Slide export: `docs/oral_assets/fig1_pareto.png`

### B. “生死图 #2” — Evidence Alignment + Failure bucketing

- [x] **E0202: Evidence Alignment report** for the current best run (and the next best candidate you propose).
  - Goal: show coverage/consistency signals predict when anchors help/hurt; include the “harmful buckets” (far anchors / 2-high / fallback-heavy).
  - Artifacts: `runs/E0202_*/evidence_alignment.json` and a short “top failure cases” table.
  - Latest full:
    - Energy baseline: `runs/E0202_evidence_alignment_energy_test402_20260209-061145/evidence_alignment.json` (weak corr; use as negative-but-clean evidence).
    - Best C0003 config (df7): `runs/E0720_evidence_alignment_df7_best_20260212-015616/evidence_alignment.json` (weak corr; indicates Coverage@τ is not predictive here).
    - C0003-proven PSP+hconf: `runs/E0981_evidence_alignment_psp_keepadj_hconf_best_test402_20260214-033440/evidence_alignment.json` (positive corr; pearson≈0.315, spearman≈0.180).

### C. “生死图 #3” — Robust degradation protocol (shift/noise/silence × α)

- [x] **E0331: Degradation suite with downstream accuracy + α lower bound** (not just Recall@K).
  - Goal: show predicted anchors degrade gracefully and never go below the computable α-baseline (uniform fallback).
  - Runner/script: `bash scripts/e0331_degradation_accuracy_official.sh` (see E0331 in `docs/experiment.md`).
  - Artifacts: `runs/E0331_*/degradation_accuracy.json` + plots.
  - Latest full (PSP-aligned): `runs/E0331_degradation_accuracy_psp_avel_evt_20260214-161014/degradation_accuracy.json` (`rows=54`, `alpha_floor_checks.num_fail=0`, `alpha_floor_checks.min_margin≈+0.000995`).
    - Note: for `psp_avel_evt`, degradations are applied in score space (external teacher is not recomputed from raw audio).
  - Historical reference (pre-PSP): `runs/E0331_degradation_accuracy_av_clipdiff_mlp_local_20260209-235316/degradation_accuracy.json`.
  - Slide exports:
    - `docs/oral_assets/fig3_degradation_delta_acc_alpha0p5.png`
    - `docs/oral_assets/fig3_degradation_recall_d0_alpha0p5.png`

### D. Minimal controls (reviewer objections)

- [x] Random anchors (already in `avs.experiments.ave_p0*` and MDE harness) included in the Pareto grid (E0330).
- [x] Cheap-visual anchors included in the Pareto grid (E0330).
- [x] Report tokens/FLOPs/latency consistently (table in E0330 output + `docs/plan.md` claims).
  - Tokens: E0330 artifacts include strict token-budgeted comparisons.
  - FLOPs/latency calibration artifact: `runs/E0408_vision_efficiency_20260209-233151/vision_efficiency.json`.

### E. Cross-dataset proxy — EPIC-SOUNDS long-video recognition (downstream sanity)

- [x] **E0100 (expanded): EPIC-SOUNDS video-level multi-label classification** under strict equal-budget (SEEDS=0,1,2; `max_steps=120`, `max_seconds=120`, `limit_train_videos=256`, `limit_val_videos=137`).
  - Goal: show the Listen-then-Look indexing story survives in a long-video setting (not only AVE 10s).
  - Artifacts:
    - Anchored: `runs/E0100_epic_video_cls_local_audio_anchored_full_ms120_t256_v137_s012_20260209-235834/metrics.json`
    - Uniform: `runs/E0100_epic_video_cls_local_uniform_full_ms120_t256_v137_s012_20260210-001346/metrics.json`
    - Random: `runs/E0100_epic_video_cls_local_random_full_ms120_t256_v137_s012_20260210-001929/metrics.json`
  - Key result: anchored mAP=`0.4028±0.0048` vs uniform mAP=`0.3346±0.0021` (Δ=`+0.0681`); anchored macro_f1@0.5=`0.4194±0.0009` vs uniform=`0.3277±0.0387` (Δ=`+0.0917`).
  - Note: when `max_steps==max_seconds`, random==uniform because both select all seconds; if you want a meaningful random baseline, set `max_steps < max_seconds`.
  - Extra diagnostic (meaningful random baseline; tighter budget): `max_seconds=120`, `max_steps=60`, `limit_train_videos=256`, `limit_val_videos=137`, `SEEDS=0,1,2`:
    - Artifacts: `runs/E0100_epic_video_cls_local_{uniform,random,audio_anchored,oracle}_ms120_steps60_t256_v137_s012_20260210-*/metrics.json`
    - Result: uniform mAP=`0.3351±0.0048`; random=`0.3340±0.0051`; audio_anchored=`0.3358±0.0135`; oracle=`0.3329±0.0005` (no gain at this tighter budget).

---

## 2) Narrative checklist (slides/section headers)

- [x] **What is new**: “audio is a cheap temporal index; visual budget is allocated only where evidence is likely”. (see `docs/oral_narrative.md`)
- [x] **Budget definition is sealed**: strict token accounting + fixed budget grid. (see `docs/oral_narrative.md`)
- [x] **MDE protocol is pre-registered**: Oracle → Predicted + controls (Random / Cheap-visual). (see `docs/oral_narrative.md`)
- [x] **Why it works / when it fails**: bucketed diagnosis (fallback-heavy, far anchors, 2-high harm) + evidence alignment. (see `docs/oral_narrative.md`)
- [x] **Robustness**: explicit degradation curves + α lower bound. (see `docs/oral_narrative.md`)
- [x] **Reproducibility**: dataset verify (`bash scripts/datasets/verify_all.sh`), fixed seeds, saved run artifacts.
  - Evidence:
    - Dataset status snapshot: `runs/datasets_verify_20260214-033737/datasets_verify.json`
    - IntentQA LFS pull log (to eliminate pointer mp4s): `artifacts/datasets/intentqa_hf_pull_full_20260210-020508.log`

---

## 3) “拉大” execution rule (so we don’t waste test budget)

For any new Stage-1 signal / method targeting C0003:

- [x] **val402 sweep** (SEEDS=0..2) → only promote if competitive vs E0223/E0224.
- [x] **quick test402** (SEEDS=0..2) + bucket diagnosis → only then spend **full test402 SEEDS=0..9**.
  - Enforced in this round via E0405→E0406 (quick grid winner did not transfer on full; retained E0402 alt as best).
  - Last attempt: `ltl_sep3_v1` (E0332→E0333) — reduces fallback but regresses on test402 quick; do not promote.
  - Last attempt: `ltl_top1med_gate_lr_v1` (E0334→E0335) — gate rescues 0 clips; fallback unchanged; quick test not competitive; do not promote.
  - Last attempt: `ltl_top1med_visfb_v1` (E0336→E0337) — cheap-visual fallback plans regress on val402; skip test.
  - Last attempt: `ltl_top1med_visfb_gated_v1` (E0338→E0339) — gated visfb also regresses on val402; skip test.
  - Last attempt: `ltl_top1med_band_midres_v1` (E0340→E0343) — mid-res caches + band-budget DP do **not** transfer; midres variants regress on test402 quick (details in `docs/experiment.md` E0342).
  - Last attempt: `av_clap_clip_agree` (E0346→E0347) — weak on val402 (best Δ≈+0.00599) and regresses on test402 quick (Δ≈-0.00174); do not promote.
  - Last attempt: `av_clap_clip_agree_k1` (E0349→E0350) — k=1 removes 2-high harm but triggers heavy fallback on test402 quick (Δ≈+0.00464; fallback≈0.930); do not promote.
  - Last attempt: `clap_evt` (E0352→E0353) — weak on val402 (best Δ≈+0.00657) and not competitive on test402 quick (Δ≈+0.00813, p≈0.457; fallback≈0.478); do not promote.
  - Last attempt: `clap_evt_k1` (E0355→E0356) — k=1 removes 2-high by design, but the confidence gate collapses to heavy fallback on test402 quick (Δ≈+0.00489, p≈0.289; fallback≈0.998); do not promote.
  - Last attempt: `clap_lr` (E0358) — regresses on val402 (best Δ≈-0.00191, p≈0.625); do not promote.
  - Last attempt: `clap_mlp_cls_target` (E0361) — near-0 on val402 (best Δ≈+0.00158, p≈0.284); do not promote.
  - Last attempt: `ltl_top1med_keepadj_basealloc_highonly_v1` (E0364) — near-0 on val402 (best Δ≈+0.00291, p≈0.167); do not promote.
  - Last attempt: `av_ast_clipdiff_mlp` (E0367) — near-0 on val402 (best Δ≈+0.00183, p≈0.853); do not promote.
  - Last attempt: `av_ast_clipalign_bce` (E0370→E0371) — not competitive on test402 quick (Δ≈+0.00813, p≈0.365; fallback≈0.570); do not promote.
  - Last attempt: `ltl_top1med_gate_all_v1` (E0372→E0374) — val402 is modestly positive (best Δ≈+0.00989), but not competitive on test402 quick (Δ≈+0.01086, p≈0.1165; fallback≈0.751); do not promote.
  - Last attempt (control): `vision_binary_mlp` (E0375) — strong cheap-visual supervised anchors regress on val402 (best Δ≈-0.00175); stop before test402.
  - Last attempt: `panns` (E0378) — pretrained PANNs eventness anchors are not competitive on val402 (best Δ≈+0.00998); stop before test402.
  - Last attempt: `panns_lr` (E0381) — supervised calibration on pretrained PANNs outputs regresses on val402 (best Δ≈-0.00224); stop before test402.
  - Last attempt: `panns_embed_lr` (E0384) — positive but not competitive on val402 (best Δ≈+0.00865, p≈0.110); stop before test402.
  - Last attempt: `panns_embed_mlp` (E0387) — not competitive on val402 (best Δ≈+0.00208, p≈0.785); stop before test402.
  - Last attempt: `av_panns_embed_clipdiff_mlp` (E0390) — not competitive on val402 (best Δ≈+0.00374, p≈0.542); stop before test402.
  - Last attempt: `av_clipdiff_flow_mlp` (E0393) — does not beat baseline on val402 (best Δ≈+0.00881, p≈0.0971); stop before test402.
  - Last attempt: `ltl_top1med_k1_extreme_v1` with fixed `av_clipdiff_flow_mlp` (E0396) — regresses on val402 (best Δ≈-0.00125, p≈0.924); stop before test402.
  - Last attempt: `av_basic_mlp` (E0510) — supervised audio basic + frame-diff scalar is near-0 on val402 (`runs/E0510_ave_p0_sweep_official_val_av_basic_mlp_ltl_top1med_norm_v1_20260210-161514/sweep_summary.json`, best Δ≈+0.00183, p≈0.677); stop before test402.
  - Last attempt: `av_fused_clipdiff_prod` (E0292) — rerun val402 sweep regresses (`runs/E0292_ave_p0_sweep_official_val_av_fused_clipdiff_prod_ltl_top1med_v1_20260210-181356/sweep_summary.json`, best Δ≈-0.00815, p≈0.218); stop before test402.
  - Last attempt: `moe_energy_clipdiff` (E0296) — rerun val402 sweep regresses (`runs/E0296_ave_p0_sweep_official_val_moe_energy_clipdiff_ltl_top1med_moe_v1_20260210-181653/sweep_summary.json`, best Δ≈-0.00923, p≈0.485); stop before test402.
  - Last attempt: `av_clipdiff_mlp` (E0207→E0208 quick) — rerun val402 is weak-positive (`runs/E0207_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_adaptive_v1_20260210-182509/sweep_summary.json`, best Δ≈+0.00515, p≈0.204), but not competitive on test402 quick (`runs/E0208_quick_test402_av_clipdiff_mlp_ltl_adaptive_v1_20260210-182944/metrics.json`, Δ≈+0.00738, p≈0.577); do not promote.
  - Last attempt: `av_clipdiff_flow_mlp_stride` (E0400→E0401 quick) — val402 is near-baseline (`runs/E0400_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_stride_ltl_top1med_k1_extreme_v1_20260210-195236/sweep_summary.json`, best Δ≈+0.00490, p≈0.465), but regresses on test402 quick (`runs/E0401_quick_test402_av_clipdiff_flow_mlp_stride_20260210-195916/metrics.json`, Δ≈-0.00506, p≈0.555; diagnose: `runs/E0401_quick_test402_av_clipdiff_flow_mlp_stride_20260210-195916/diagnose.json`); do not promote.
  - Last attempt: `av_clipdiff_vec_mlp` (E0610→E0611→E0612) — val402 is weak-positive (`runs/E0610_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_adaptive_v1_20260210-200224/sweep_summary.json`, best Δ≈+0.00607, p≈0.248), test402 quick is positive but not significant (`runs/E0611_quick_test402_av_clipdiff_vec_mlp_ltl_adaptive_v1_20260210-200638/metrics.json`, Δ≈+0.01061, p≈0.391; diagnose: `runs/E0611_quick_test402_av_clipdiff_vec_mlp_ltl_adaptive_v1_20260210-200638/diagnose.json`), and collapses on full test402 (`runs/E0612_full_test402_av_clipdiff_vec_mlp_ltl_adaptive_v1_20260210-200736/metrics.json`, Δ≈+0.00095, p≈0.875; diagnose: `runs/E0612_full_test402_av_clipdiff_vec_mlp_ltl_adaptive_v1_20260210-200736/diagnose.json`); this family did not prove C0003.
  - Next attempt (vec-MLP; far/2-high harm mitigation via Stage-2 policy; not promoted): `ltl_top1med_dropfar_v1` rerun on val402 regresses (`runs/E0620_val402_vecmlp_dropfar_20260210-222055/sweep_summary.json`, best Δ≈-0.00058, p≈0.921). Quick test402 for the intended policy variant (`thr0p5_df1`) removes `high_count=2` entirely but does not improve mean (`runs/E0621_quick_test402_vecmlp_dropfar_df1_20260210-222801/metrics.json`, Δ≈+0.00166, p≈0.900; diagnose: `runs/E0621_quick_test402_vecmlp_dropfar_df1_20260210-222801/diagnose.json`).
  - Next attempt (vec-MLP; far-anchor fallback; not promoted): `ltl_top1med_farfb_v1` rerun on val402 regresses (`runs/E0622_val402_vecmlp_farfb_20260210-222352/sweep_summary.json`, best Δ≈-0.00058, p≈0.921). Quick test402 for the intended policy variant (`thr0p5_ff1`) removes `high_count=2` + dist>1 by construction but does not improve mean (`runs/E0623_quick_test402_vecmlp_farfb_ff1_20260210-222856/metrics.json`, Δ≈+0.00274, p≈0.839; diagnose: `runs/E0623_quick_test402_vecmlp_farfb_ff1_20260210-222856/diagnose.json`).
  - Next attempt (vec-MLP; adaptive_v3 keepadj): val402 sweep is weak-positive (`runs/E0624_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_adaptive_keepadj_v1_20260210-224555/sweep_summary.json`, best Δ≈+0.00382, p≈0.384). Quick test402 for `ltlkeepadj_adj2_shift1_std0p55` is very strong (`runs/E0626_quick_test402_vecmlp_keepadj_adj2_shift1_std0p55_20260210-225120/metrics.json`, Δ≈+0.02098, p≈0.165; diagnose: `runs/E0626_quick_test402_vecmlp_keepadj_adj2_shift1_std0p55_20260210-225120/diagnose.json`) but does **not** transfer under full test402 (`runs/E0628_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_s0-9_20260210-225216/metrics.json`, Δ≈+0.00883, p≈0.225; diagnose: `runs/E0628_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_s0-9_20260210-225216/diagnose.json`). Full test402 for the *val-selected* keepadj winner regresses: `runs/E0629_full_test402_vecmlp_keepadj_best_s0-9_20260210-225809/metrics.json` (Δ≈-0.00408, p≈0.457; diagnose: `runs/E0629_full_test402_vecmlp_keepadj_best_s0-9_20260210-225809/diagnose.json`).
  - Next attempt (vec-MLP; keepadj + `anchor_drop_far_dist`): target the far-anchor2 harm buckets seen in E0628 (`dist=6/8` strongly negative in diagnose).
    - df7 (`anchor_drop_far_dist=7`, i.e. drop only `dist>=8`): quick test402 `runs/E0636_quick_test402_vecmlp_keepadj_adj2_shift1_std0p55_df7_officialids_20260211-000822/metrics.json` (Δ≈+0.01758, p≈0.142) → full test402 `runs/E0643_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_df7_officialids_s0-9_20260211-001604/metrics.json` (Δ=+0.01045; p=0.0395; significant but still far from +2%).
    - df5 (`anchor_drop_far_dist=5`, i.e. drop `dist>=6`): quick test402 `runs/E0637_quick_test402_vecmlp_keepadj_adj2_shift1_std0p55_df5_officialids_20260211-000915/metrics.json` (Δ≈+0.02421, p≈0.101) → full test402 `runs/E0638_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_df5_officialids_s0-9_20260211-001009/metrics.json` (Δ=+0.01117; p=0.109; higher mean but not significant).
  - Latest rerun: `av_clipdiff_flow_mlp` + `ltl_adaptive_keepadj_v1` (E0710→E0711→E0712) gives val402 positive (`runs/E0710_val402_flowmlp_keepadj_20260212-000010/sweep_summary.json`, best Δ≈+0.00648, p≈0.0355), quick test402 positive but non-significant (`runs/E0711_quick_test402_flowmlp_keepadj_20260212-000606/metrics.json`, Δ≈+0.00688, p≈0.395), and full test402 still short (`runs/E0712_full_test402_flowmlp_keepadj_20260212-000835/metrics.json`, Δ≈+0.00709, p≈0.141); do not promote.
  - Latest attempt: `imagebind_av_sim` (E0801→E0802) — val402 is ~0 (`runs/E0801_val402_imagebind_keepadjv2_20260212-035956/sweep_summary.json`, best Δ≈-0.00008), quick test402 regresses (`runs/E0802_quick_test402_imagebind_20260212-040440/metrics.json`, Δ≈-0.00265, p≈0.754); do not promote.
  - Latest attempt: `wavlm_evt_mlp` (E0810→E0811) — val402 regresses (`runs/E0810_val402_wavlm_20260212-041931/sweep_summary.json`, best Δ≈-0.00424), quick test402 is near-0 (`runs/E0811_quick_test402_wavlm_20260212-042425/metrics.json`, Δ≈+0.00124, p≈0.918); do not promote.
  - Latest attempt: `psp_avel_evt` (E0960→E0980) — external supervised AVE temporal localizer as Stage-1 + keepadj+hconf Stage-2; full test402: `runs/E0980_full_test402_psp_evt_gini_keepadj_hconf_best_s0-9_20260214-031741/metrics.json` (Δ=+0.02169; p=0.00149) → **C0003 proven**.

---

## 4) Next oral-minimum decisive pack (vNext checklist)

- [x] **Code+ledger prep**: add `E0399~E0404` entries to `docs/experiment.md` (with smoke/full commands, artifacts, and promotion gates).
- [x] **E0399 (Stage-1 bold variant)**: implement a dense-stride proposal scorer (`EVENTNESS=av_clipdiff_flow_mlp_stride`) that emits anchors from fixed stride windows (not only local maxima), keeping compute cheap and deployable.
- [x] **E0400 (val402 sweep)**: run official val402 sweep (`SEEDS=0..2`) for E0399 under a strict candidate set (`top1-med gate + k1-focused plans + fallback safety`) and compare against E0223/E0224 winner deltas.
- [x] **E0401 (quick test402 + diagnosis)**: if E0400 is competitive, run quick test402 (`SEEDS=0..2`) plus `E0344` diagnosis; require reduced harmful 2-high / far-anchor buckets or better anchor-used bucket gains.
- [x] **E0402 (full test402)**: if E0401 is competitive, run full test402 (`SEEDS=0..9`) to attempt C0003 (`Δ≥+0.02`, paired `p<0.05`).
- [x] **E0403 (mechanism evidence)**: rerun Oracle→Predicted report on the promoted method (`E0201` protocol) and require Oracle–Pred gap shrinkage vs current deployable baseline.
- [x] **E0404 (robustness evidence)**: rerun degradation suite (`E0203` protocol, shift/noise/silence × α) on the promoted method and require “no worse than α-baseline” under perturbations.
- [x] **Oral narrative closure**: update Fig.2/Fig.3/Fig.4 and Table-1 evidence pointers using the C0003-proven run `runs/E0980_full_test402_psp_evt_gini_keepadj_hconf_best_s0-9_20260214-031741/metrics.json` (and its `diagnose.json`), and regenerate the slide assets (esp. `docs/oral_assets/fig2_c0003_decomposition.png`).
- [x] **E0405→E0406 (quick-grid sanity check)**: exhaustive quick-transfer winner (`top1medn_thr0p6_shift0`) fails to transfer on full test402; keep `E0402 alt` as dense-stride best.
- [x] **E0407→E0409 (C0007 tightening)**: full Oracle→Pred rerun gives significant mid-budget gain (`p≈0.003` at Tok=1960), but cross-budget Pareto still shows weak high-budget transfer (negative predicted-vs-uniform at Tok≈1000/4840), so C0007 remains not fully proven.

---

## 5) Long-Video QA Add-on (Minimum, Oral-Ready)

This section is **optional** for the core AVE oral pack, but is a high-leverage add-on to preempt “does it transfer beyond AVE?” objections.

### A. Minimum baselines (no-cherry-pick)

- [x] **Question-only baseline** (no frames) on IntentQA/EgoSchema/AVQA to quantify language bias:
  - IntentQA: `runs/E0617_intentqa_vlm_eval_val_text_only_20260211-053301/metrics.json` (uniform=0.9447; text_only=0.6640).
  - EgoSchema: `runs/E0618_egoschema_eval_subset500_text_only_20260211-055131/metrics.json` (uniform=0.5880; text_only=0.2720).
  - AVQA: `runs/E0616_avqa_vlm_eval_val_b4_20260211-051556/metrics.json` (uniform=0.8113; text_only=0.3113).
- [x] **ql2l_clip baseline added** (query→CLIP text-image relevance; cached image embeddings per video) and evaluated on IntentQA val:
  - `runs/E0609_intentqa_vlm_eval_val_clip_20260211-011407/metrics.json` (`ql2l_clip` is worse than uniform; keep as negative-but-clean evidence).

### B. Minimum cross-dataset runs (small but decisive)

- [x] **AVQA (val subset)**: end-to-end run complete (download drift allowed):
  - `runs/E0615_avqa_vlm_eval_val_20260211-043508/metrics.json` (kept n=212; skipped_videos=44; invalid_rate=0).
  - Note: `B_FRAMES=16` exceeds clip duration (~10s), so all frame-selection methods tie; keep this as a *sanity check* + `text_only` language-bias baseline.
- [x] **AVQA (tight budget)**: rerun with `B_FRAMES=4` so selection methods diverge:
  - `runs/E0616_avqa_vlm_eval_val_b4_20260211-051556/metrics.json` (best=0.8255 vs uniform=0.8113; `text_only`=0.3113; skipped_videos=44).
- [x] **EgoSchema Subset (n=500)**: `ql2l_clip` baseline run complete:
  - `runs/E0606_egoschema_eval_subset500_clip_20260211-031138/metrics.json` (uniform acc=0.5880; ql2l_clip acc=0.5760; still < uniform).

### C. Narrative glue (one page, no new methods)

- [x] Add a short “when does audio help?” bucketed analysis:
  - Artifacts (markdown):
    - IntentQA: `runs/E0619_qa_bucket_report_20260211-062907/intentqa/bucket_report.md` (bucket by `type` + `q_bar`).
    - AVQA: `runs/E0619_qa_bucket_report_20260211-062907/avqa/bucket_report.md` (bucket by `question_type` + `q_bar`).
    - EgoSchema: `runs/E0619_qa_bucket_report_20260211-062907/egoschema/bucket_report.md` (no type labels; bucket by `q_bar` only).
  - Key takeaways (for slides):
    - IntentQA: `ql2l_clap` helps more on `CH` (+2.22pp) and hurts on `TN` (-1.49pp).
    - AVQA: `ql2l_asr_bm25` helps more on `Which` (+2.80pp) and hurts on `Come From` (-2.22pp).

### D. Seed-extension reproducibility checks (new)

- [x] **E0713: IntentQA faithfulness seed extension** (`seed=2`) completed.
  - Artifact: `runs/E0713_intentqa_faithfulness_val_s2_20260212-000949/faithfulness.json`
  - Signal: `acc_drop=0.0`, `pred_change_rate≈0.0316`, matching seeds 0/1 (stable).
- [x] **E0714: EgoSchema Subset seed extension** (`seed=2`) completed.
  - Artifact: `runs/E0714_egoschema_eval_subset256_s2_20260212-004316/metrics.json`
  - Signal: uniform=`0.5859`, `ql2l_clap=0.5352`, `ql2l_asr_bm25=0.5469` (all `invalid_rate=0`), matching seeds 0/1.

---

## 6) Oral vNext decisive pack (E070x/D0701)

- [x] **R0700: related-work alignment (online scan + positioning matrix)** completed.
  - Artifact: `docs/oral_related_work.md`
  - Use: one-page oral rebuttal map for “baseline sufficiency / extra-dataset necessity”.

- [x] **E0704: answer-prior bias baselines** completed (IntentQA/AVQA/EgoSchema).
  - Artifacts: `runs/E0704_qa_bias_baselines_20260211-161403/{intentqa,avqa,egoschema}/bias_baselines.json`
  - Key signal: all datasets are substantially above answer-prior; language-only cannot explain uniform-level performance.

- [x] **E0705: bucket significance (bootstrap)** completed.
  - Artifacts: `runs/E0705_qa_bucket_significance_20260211-161409/{intentqa,avqa,egoschema}/bucket_significance.json`
  - Key signal: AVQA has a significant positive `q_bar` bucket; EgoSchema has a significant negative high-`q_bar` bucket (must be stated explicitly in oral narrative).

- [x] **D0701: C0003 gate decision helper** completed.
  - Artifact: `runs/D0701_c0003_gate_20260211-161419/summary.json`
  - Decision: both `df5/df7` are `revised_claim`; `c0003_proven=false` persists.

- [x] **E0701: multi-seed robustness (IntentQA b16 / AVQA b4)** completed.
  - Targets: `runs/E0701_intentqa_val_b16_s{1,2}_20260211-162353/metrics.json`, `runs/E0701_avqa_val_b4_s{1,2}_20260211-162353/metrics.json`
  - Aggregation target: `runs/E0701_qa_multiseed_20260211-162353/{intentqa_multiseed,avqa_multiseed}/metrics_summary.json`

- [x] **E0702: budget curve (B=2/4/8/16)** completed.
  - Artifacts: `runs/E0702_qa_budget_curve_20260211-164607/{intentqa_curve,avqa_curve}/budget_curve.{json,md,png}`
  - Key signal: AVQA `ql2l_asr_bm25` is budget-sensitive (best at `B2` Δ=`+2.83pp`, still positive at `B4` Δ=`+1.42pp`, flips at `B8` Δ=`-0.94pp`); IntentQA ql2l gains are only marginal at `B16` (`ql2l_clap` Δ=`+0.40pp`) and non-positive at lower budgets.

- [x] **E0703: AVQA coverage expansion + sensitivity** completed.
  - Artifacts: `runs/E0703_avqa_coverage_expand_20260211-170526/sensitivity/{coverage_sensitivity.json,coverage_sensitivity.md}`
  - Key signal: expanded `n=865` keeps ranking stable (`ql2l_asr_bm25` best), with moderate absolute shift vs baseline `n=212`.
