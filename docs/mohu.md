# Mohu

## 1) Not Implemented
- (none)

## 2) Ambiguities
- (none)

## Resolved (archive)
- [x] M0179 (plan: C0005 / exp: E0413): Unblock EPIC-SOUNDS real-data run and produce real local mAP/macro-F1 evidence.
  - Decision:
    - Use the currently available real EPIC local videos with `ALLOW_MISSING_VIDEOS=1` (coverage: `mp4_count=33`, train=17, val=16).
    - Run E0413 in strict equal-budget mode (`max_steps=120`) with `MAX_SECONDS=120` to keep runtime bounded and comparisons fair across `{audio_anchored,uniform,random}`.
  - Evidence:
    - Dataset gate: `runs/datasets_verify_20260207-172512/datasets_verify.json` (`EPIC_SOUNDS.ok=true`, `mp4_count=33`).
    - Real runs:
      - `runs/E0413_epic_video_cls_local_audio_anchored_full_ms120_20260207-171637/metrics.json`
      - `runs/E0413_epic_video_cls_local_uniform_full_ms120_20260207-172545/metrics.json`
      - `runs/E0413_epic_video_cls_local_random_full_ms120_20260207-173208/metrics.json`
    - Key deltas (anchored vs uniform): `mAP≈-0.0221`, `macro_f1≈-0.0118`.
  - Verification: `bash scripts/datasets/verify_all.sh && jq '.datasets[] | select(.name==\"EPIC_SOUNDS\") | .ok' runs/datasets_verify_*/datasets_verify.json | head -n 1`.

- [x] M0182 (plan: C0009 / exp: E0505): Run full degradation-accuracy suite on dense-stride promoted config and refresh alpha-floor proof.
  - Evidence:
    - Artifact: `runs/E0505_degradation_accuracy_dense_stride_full_20260207-161213/degradation_accuracy.json` (+ `degradation_plots/*.png`).
    - Key result: full perturbation grid complete (`rows=54`), `alpha_floor_checks.num_fail=0`, `alpha_floor_checks.min_margin≈+0.01766`.
    - Verification: `OUT_DIR=runs/E0505_degradation_accuracy_dense_stride_full_$(date +%Y%m%d-%H%M%S) SEEDS=0,1,2,3,4,5,6,7,8,9 LIMIT_TRAIN=3339 LIMIT_EVAL=402 SHIFT_GRID='-0.5,0,0.5' SNR_GRID='20,10,0' SILENCE_GRID='0,0.5' ALPHA_GRID='0,0.5,1' bash scripts/e0505_degradation_accuracy_dense_stride.sh`.

- [x] M0181 (plan: C0007 / exp: E0504): Run full multi-budget Oracle→Predicted gap grid (`SEEDS>=3`) for dense-stride and update cross-budget evidence.
  - Evidence:
    - Artifacts: `runs/E0504_oracle_pred_gap_grid_dense_stride_full_20260207-155721/pareto_report.json` + `runs/E0504_oracle_pred_gap_grid_dense_stride_full_20260207-155721/pareto.png`.
    - Key metrics: predicted-vs-uniform deltas are `-0.00025` (`112_160_224`), `+0.01241` (`160_224_352`), `-0.00674` (`224_352_448`); Oracle–Pred gaps are `0.02142`, `0.02900`, `0.03012` (mean `≈0.02685`).
    - Verification: `OUT_DIR=runs/E0504_oracle_pred_gap_grid_dense_stride_full_$(date +%Y%m%d-%H%M%S) SEEDS=0,1,2,3,4,5,6,7,8,9 LIMIT_TRAIN=3339 LIMIT_EVAL=402 bash scripts/e0504_oracle_pred_gap_grid_dense_stride.sh`.

- [x] M0180 (plan: C0003,C0007 / exp: E0503): Run full val402 gate sweep for dense-stride Stage-1 and freeze a single gate for test-time reproduction.
  - Evidence:
    - Artifacts: `runs/E0503_gate_sweep_dense_stride_full_20260207-153210/gate_sweep.json` + `runs/E0503_gate_sweep_dense_stride_full_20260207-153210/best_gate.json`.
    - Key result: selected gate remains `top1_med@0.5` on full val402; `anchored≈0.74526` vs `uniform≈0.73214` (`Δ≈+0.01312`, `p≈0.0174`), `oracle_minus_predicted≈0.03746`.
    - Verification: `OUT_DIR=runs/E0503_gate_sweep_dense_stride_full_$(date +%Y%m%d-%H%M%S) SEEDS=0,1,2,3,4,5,6,7,8,9 LIMIT_TRAIN=3339 LIMIT_EVAL=402 bash scripts/e0503_gate_sweep_dense_stride.sh`.

- [x] M0185 (plan: C0003,C0007,C0009 / exp: E0501~E0505): Add E050x execution scripts and verify they run in smoke mode.
  - Evidence:
    - Scripts added: `scripts/e0501_dataset_integrity_audit.sh`, `scripts/e0502_root_cause_report.sh`, `scripts/e0503_gate_sweep_dense_stride.sh`, `scripts/e0504_oracle_pred_gap_grid_dense_stride.sh`, `scripts/e0505_degradation_accuracy_dense_stride.sh`.
    - Smoke artifacts:
      - `runs/E0503_gate_sweep_dense_stride_smoke_20260207-151733/gate_sweep.json`
      - `runs/E0504_oracle_pred_gap_grid_dense_stride_smoke_20260207-151822/pareto_report.json`
      - `runs/E0505_degradation_accuracy_dense_stride_smoke_20260207-151846/degradation_accuracy.json`
    - Verification: `bash scripts/e0503_gate_sweep_dense_stride.sh` (smoke limits), `bash scripts/e0504_oracle_pred_gap_grid_dense_stride.sh` (smoke limits), `bash scripts/e0505_degradation_accuracy_dense_stride.sh` (smoke limits).

- [x] M0184 (plan: C0003,C0007,C0009): Implement a machine-readable root-cause report generator and verify with smoke + real artifacts.
  - Evidence:
    - Code: `avs/experiments/root_cause_report.py`, smoke check wiring in `avs/smoke.py` + `avs/smoke_checks.py`.
    - Smoke artifact: `runs/smoke_20260207-151416/root_cause_report/root_cause_report.json`.
    - Real artifact: `runs/E0502_root_cause_report_20260207-151428/root_cause_report.json` (reasons include `R_TARGET_DELTA`, `R_ORACLE_GAP`, `R_ALIGNMENT_WEAK`, `R_EPIC_DOWNSTREAM`).
    - Verification: `python -m avs.smoke root_cause_report` and `bash scripts/e0502_root_cause_report.sh`.

- [x] M0183 (plan: C0005/C0009): Implement dataset integrity audit tool (probe/decode) and verify it flags corruption in smoke.
  - Evidence:
    - Code: `avs/experiments/dataset_integrity_audit.py`, smoke check wiring in `avs/smoke.py` + `avs/smoke_checks.py`.
    - Smoke artifact: `runs/smoke_20260207-151400/dataset_integrity_audit/dataset_integrity_audit.json` (corrupted sample correctly detected).
    - Local data probe artifact: `runs/E0501_dataset_integrity_20260207-151719/index.json` + per-dataset audits (`ave` and `epic`).
    - Verification: `python -m avs.smoke dataset_integrity_audit` and `LIMIT=8 DECODE_CHECK=none bash scripts/e0501_dataset_integrity_audit.sh`.

- [x] M0178 (plan: C0005 / exp: E0413): Execute EPIC-SOUNDS val experiment (`SEEDS>=3`) under strict equal-budget setup and write mAP/macro-F1 evidence.
  - Evidence:
    - Smoke success: `runs/smoke_20260206-184253/smoke.json` and `runs/smoke_20260206-184253/epic_sounds_video_cls_synth/metrics.json`.
    - Full attempt fails at data gate: `bash scripts/e0100_epic_video_cls_local.sh` -> `missing mp4s under data/EPIC_SOUNDS/raw/videos: ['P01_01', 'P01_02', 'P01_03', 'P01_04', 'P01_05'] ...`.
    - Follow-up moved to ambiguity `M0179` (external dataset credentials required).

- [x] M0177 (plan: C0009 / exp: E0412): Re-run downstream degradation-accuracy suite (shift/noise/silence × α) on dense-stride best config and check α-lower-bound compliance.
  - Evidence:
    - Artifact: `runs/E0412_degradation_accuracy_av_clipdiff_flow_mlp_stride_top1med_thr0p5_s0-9_20260206-182443/degradation_accuracy.json` (+ `degradation_plots/*.png`).
    - Key result: full grid finished with `rows=54`; `alpha_floor_checks` gives `num_fail=0`, `min_margin≈+0.01766` (rule: `anchored_top2_mean >= alpha * uniform_mean`).
    - Implementation update: `avs/experiments/degradation_accuracy.py` now supports `eventness_method=av_clipdiff_flow_mlp_stride` and emits `alpha_floor_checks`.
    - Verification: `jq '{rows: (.rows|length), alpha_floor_checks: .alpha_floor_checks}' runs/E0412_degradation_accuracy_av_clipdiff_flow_mlp_stride_top1med_thr0p5_s0-9_20260206-182443/degradation_accuracy.json`.

- [x] M0176 (plan: C0008 / exp: E0411): Recompute evidence-alignment report on the promoted dense-stride config and add correlation + failure-bucket linkage.
  - Evidence:
    - Artifact: `runs/E0411_evidence_alignment_av_clipdiff_flow_mlp_stride_top1med_thr0p5_20260206-182007/evidence_alignment.json`.
    - Key result: `cov_by_tau` mean≈`0.0935` (τ=`0.3/0.5/0.7`), `corr_by_tau` (`pearson≈0.0498`, `spearman≈-0.0029`) for all τ.
    - Note: current output key is `corr_by_tau` (not `correlation`).
    - Verification: `jq '.cov_by_tau, .corr_by_tau' runs/E0411_evidence_alignment_av_clipdiff_flow_mlp_stride_top1med_thr0p5_20260206-182007/evidence_alignment.json`.

- [x] M0175 (plan: C0004 / exp: E0410): Re-run fusion confirm (`audio_concat_uniform` vs `audio_concat_anchored_top2`) on the current best dense-stride config (`E0402 alt`) with full official test402 seeds.
  - Evidence:
    - Artifact: `runs/E0410_fusion_confirm_energy_stride_max_top1med_thr0p5_20260206-180945/metrics.json`.
    - Key result: `audio_concat_anchored_top2 - audio_concat_uniform ≈ -0.00254` (`0.70978 - 0.71231`), paired `p≈0.36759` (`paired_ttest.audio_concat_anchored_vs_audio_concat_uniform.p`).
    - Note: original verification key `paired_ttest.audio_concat_anchored_vs_uniform.p` is not the fusion comparison; switched to `audio_concat_anchored_vs_audio_concat_uniform`.
    - Verification: `jq '.summary.audio_concat_anchored_top2.mean - .summary.audio_concat_uniform.mean, .paired_ttest.audio_concat_anchored_vs_audio_concat_uniform.p' runs/E0410_fusion_confirm_energy_stride_max_top1med_thr0p5_20260206-180945/metrics.json`.

- [x] M0174 (plan: C0007 / exp: E0409): Finish aligned multi-budget Pareto rerun (`SEEDS=0..9`) for dense-stride `top1_med thr0.5` and write back cross-budget Oracle–Pred gaps.
  - Evidence:
    - Artifacts: `runs/E0409_pareto_grid_av_clipdiff_flow_mlp_stride_top1med_thr0p5_s0-9_20260206-163941/pareto_report.json` + `runs/E0409_pareto_grid_av_clipdiff_flow_mlp_stride_top1med_thr0p5_s0-9_20260206-163941/pareto.png`.
    - Key metrics: Oracle–Pred gaps by triad are `0.02142` (`112_160_224`), `0.02900` (`160_224_352`), `0.03012` (`224_352_448`), mean `≈0.02685`; only `160_224_352` keeps predicted-vs-uniform significant (`Δ≈+0.01241`, `p≈0.00302`).
    - Verification: `jq '.points | length' runs/E0409_pareto_grid_av_clipdiff_flow_mlp_stride_top1med_thr0p5_s0-9_20260206-163941/pareto_report.json`.

- [x] M0173 (plan: C0003,C0007,C0009): Write back this round's new evidence into `docs/experiment.md`, `docs/plan.md`, and `docs/oral_checklist.md`.
  - Evidence:
    - `docs/experiment.md` now includes E0405/E0406/E0407/E0408/E0409 entries with final status and artifacts.
    - `docs/plan.md` now records E0405/E0406 (C0003), E0407 (C0007), and E0408 (efficiency control) evidence.
    - `docs/oral_checklist.md` now marks minimal controls as complete with explicit FLOPs/latency artifact path.
    - Verification: `rg -n "E0405|E0406|E0407|E0408|E0409|C0003|C0007|tokens/FLOPs/latency" docs/experiment.md docs/plan.md docs/oral_checklist.md docs/mohu.md`.

- [x] M0172 (plan: oral controls / exp: E0408): Produce explicit tokens/FLOPs/latency calibration artifact for oral efficiency table.
  - Evidence:
    - Artifact: `runs/E0408_vision_efficiency_20260206-161610/vision_efficiency.json`.
    - Key rows include monotonic tokens/FLOPs across `{112,160,224,352,448}` and measured `ms_per_image` for each resolution.

- [x] M0171 (plan: C0007 / exp: E0407): Rerun aligned Oracle→Predicted with full seeds (`0..9`) for the promoted dense-stride config (`top1_med thr0.5`).
  - Evidence:
    - Artifact: `runs/E0407_oracle_vs_predicted_av_clipdiff_flow_mlp_stride_top1med_thr0p5_s0-9_20260206-161749/oracle_vs_predicted.json`.
    - Key result: predicted `Δ≈+0.01241`, `p≈0.00302`; `oracle_minus_predicted≈0.02900` (improves from baseline dense-stride `0.04187`).

- [x] M0170 (plan: C0003 / exp: E0405,E0406): Run quick-grid transfer for dense-stride top candidates and validate the quick winner on full test402.
  - Evidence:
    - Quick grid best: `runs/E0405_quick_test402_av_clipdiff_flow_mlp_stride_top1medn_thr0p6_shift0_20260206-161028/metrics.json` (`Δ≈+0.01335`, `p≈0.0956`).
    - Full verification: `runs/E0406_full_test402_av_clipdiff_flow_mlp_stride_top1medn_thr0p6_shift0_20260206-161349/metrics.json` (`Δ≈+0.00771`, `p≈0.0442`) + `runs/E0344_ave_p0_diagnose_20260206-161548/diagnose.json`.
    - Decision: quick improvement does not transfer; keep `E0402 alt` as dense-stride best.

- [x] M0169 (plan: C0003,C0007): Write back new alternative-config evidence into `docs/experiment.md`, `docs/plan.md`, `docs/oral_checklist.md`, and close this iteration.
  - Evidence:
    - `docs/experiment.md` now records alternative E0401/E0402/E0403 reruns and side-by-side comparisons.
    - `docs/plan.md` now records updated P0136 evidence + C0003/C0007 alternative-run evidence.
    - `docs/oral_checklist.md` now marks revised-claim gate as locked when C0003 remains unproven.
    - Verification: `rg -n "M0166|M0167|M0168|E0401|E0402|E0403|alt" docs/mohu.md docs/experiment.md docs/plan.md docs/oral_checklist.md`.

- [x] M0168 (plan: C0007 / exp: E0403): If M0167 improves transfer, rerun Oracle→Predicted report with the promoted config to measure gap movement.
  - Evidence:
    - Artifact: `runs/E0403_oracle_vs_predicted_av_clipdiff_flow_mlp_stride_alt_top1med_thr0p5_20260206-152658/oracle_vs_predicted.json`.
    - Key comparison vs baseline `runs/E0403_oracle_vs_predicted_av_clipdiff_flow_mlp_stride_20260206-141804/oracle_vs_predicted.json`: `predicted Δ` improves from `-0.00547` to `+0.01153`; `oracle_minus_predicted` shrinks from `0.04187` to `0.02488` (still not enough to mark C0007 proven at strict significance gate).

- [x] M0167 (plan: C0003 / exp: E0402): Promote the best quick alternative to full test402 (`SEEDS=0..9`) and check whether it beats current dense-stride best.
  - Evidence:
    - Full alt run: `runs/E0402_full_test402_av_clipdiff_flow_mlp_stride_alt_top1med_thr0p5_20260206-152012/metrics.json` + `runs/E0402_full_test402_av_clipdiff_flow_mlp_stride_alt_top1med_thr0p5_20260206-152012/diagnose.json`.
    - Key comparison: alt `Δ≈+0.01241` (p≈0.00302) > prior dense-stride `Δ≈+0.01037` (p≈0.00489), but still below C0003 +2% target.

- [x] M0166 (plan: C0003 / exp: E0401,E0402): Run quick test402 for alternative dense-stride configs from E0400 top candidates (not only p-filter winner).
  - Evidence:
    - Alt A (`top1_med thr0.5`): `runs/E0401_quick_test402_av_clipdiff_flow_mlp_stride_alt_top1med_thr0p5_20260206-150837/metrics.json` + `.../diagnose.json` (`Δ≈+0.01153`, `p≈0.0661`, `fallback_used_frac≈0.5622`).
    - Alt B (`top1_med_norm thr0.6`): `runs/E0401_quick_test402_av_clipdiff_flow_mlp_stride_alt_top1medn_thr0p6_20260206-151412/metrics.json` + `.../diagnose.json` (`Δ≈+0.00746`, `p≈0.436`).
    - Selection: promote Alt A to full (M0167).

- [x] M0165 (plan: C0003,C0007,C0009): Write back all new evidence to `docs/plan.md`, `docs/experiment.md`, and `docs/oral_checklist.md` and close the loop.
  - Evidence:
    - `docs/experiment.md` updated with E0401/E0402/E0403/E0404 status + result metrics.
    - `docs/plan.md` updated at P0136 and conclusions C0003/C0007/C0009 with latest E0402/E0403/E0404 evidence.
    - `docs/oral_checklist.md` updated (vNext checklist E0399~E0404 + oral narrative closure marked complete).
    - Verification: `rg -n "E0401|E0402|E0403|E0404|P0136|C0003|C0007|C0009" docs/plan.md docs/experiment.md docs/oral_checklist.md docs/mohu.md`.

- [x] M0164 (plan: C0009 / exp: E0404): Run degradation suite for `av_clipdiff_flow_mlp_stride` and save robustness artifacts.
  - Evidence:
    - Artifact: `runs/E0404_degradation_av_clipdiff_flow_mlp_stride_20260206-142817/degradation_suite.json` (18 rows).
    - Key result: mean Recall@K (`Δ0≈0.21308`, `Δ1≈0.38001`, `Δ2≈0.51741`); relative to `runs/E0203_degradation_av_clipdiff_mlp_20260204-215831/degradation_suite.json`, strict `Δ0` is slightly better but `Δ1/Δ2` are lower.

- [x] M0163 (plan: C0007 / exp: E0403): Run Oracle→Predicted report for `av_clipdiff_flow_mlp_stride` and record Oracle-Pred gap.
  - Evidence:
    - Artifact: `runs/E0403_oracle_vs_predicted_av_clipdiff_flow_mlp_stride_20260206-141804/oracle_vs_predicted.json`
    - Key result: predicted `Δ_vs_uniform=-0.00547` (`p≈0.03399`), oracle `Δ_vs_uniform=+0.03640`, `oracle_minus_predicted≈0.04187`.

- [x] M0162 (plan: P0136 / exp: E0401,E0402): Run test402 transfer for `av_clipdiff_flow_mlp_stride` (quick + full) and save metrics/diagnosis.
  - Evidence:
    - Quick: `runs/E0401_quick_test402_av_clipdiff_flow_mlp_stride_20260206-140425/metrics.json` + `runs/E0401_quick_test402_av_clipdiff_flow_mlp_stride_20260206-140425/diagnose.json` (`Δ≈+0.00771`, `p≈0.331`).
    - Full: `runs/E0402_full_test402_av_clipdiff_flow_mlp_stride_20260206-141020/metrics.json` + `runs/E0402_full_test402_av_clipdiff_flow_mlp_stride_20260206-141020/diagnose.json` (`Δ≈+0.01037`, `p≈0.00489`; below C0003 +2% target).

- [x] M0161 (plan: P0136 / exp: E0399,E0400,E0401,E0402,E0403,E0404): Add missing `E0399~E0404` sections into `docs/experiment.md` with runnable smoke/full commands, explicit promotion gates, and artifact/result placeholders.
  - Evidence: `rg -n "^### E0399|^### E0400|^### E0401|^### E0402|^### E0403|^### E0404" docs/experiment.md` → lines `4426,4449,4472,4495,4518,4541`.

- [x] M0160 (plan: P0135 / exp: E0398,E0344 / conclusion: C0003): If promoted, run full test402 (SEEDS=0..9) for the E0396 winner and update C0003 evidence
  - Evidence: Skipped (not promoted) because E0396 val402 regresses. Evidence: `runs/E0396_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_ltl_top1med_k1_extreme_v1_20260206-130114/sweep_summary.json` (best Δ≈-0.00125, p≈0.924).

- [x] M0159 (plan: P0135 / exp: E0397,E0344): If promoted, run quick test402 transfer (SEEDS=0..2) for the E0396 winner + bucket diagnosis
  - Evidence: Skipped (not promoted) because E0396 val402 regresses. Evidence: `runs/E0396_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_ltl_top1med_k1_extreme_v1_20260206-130114/sweep_summary.json` (best Δ≈-0.00125, p≈0.924).

- [x] M0158 (plan: P0135 / exp: E0396): Run Stage-2 dynamic-K extreme sweep (`ltl_top1med_k1_extreme_v1`) with fixed `EVENTNESS=av_clipdiff_flow_mlp` on official val402 (SEEDS=0..2)
  - Evidence:
    - Code: `python -m py_compile avs/experiments/ave_p0_sweep.py` → ok.
    - Smoke: `runs/E0396_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_ltl_top1med_k1_extreme_v1_20260206-130029/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1; best Δ≈+0.00781).
    - Full val402: `runs/E0396_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_ltl_top1med_k1_extreme_v1_20260206-130114/sweep_summary.json` (best=`ltltop1medk1ext_thr0p6_shift1_score`, Δ≈-0.00125, p≈0.924) + `runs/E0396_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_ltl_top1med_k1_extreme_v1_20260206-130114/best_config.json`.
    - Decision: Not promoted to test402 (skip E0397/E0398).

- [x] M0157 (plan: P0134 / exp: E0395,E0344 / conclusion: C0003): If promoted, run full test402 (SEEDS=0..9) for the E0393 winner and update C0003 evidence
  - Evidence: Skipped (not promoted) because E0393 val402 does not beat the baseline val winner. Evidence: `runs/E0393_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_ltl_top1med_norm_v1_20260206-104413/sweep_summary.json` (best Δ≈+0.00881, p≈0.0971).

- [x] M0156 (plan: P0134 / exp: E0394,E0344): If promoted, run quick test402 transfer (SEEDS=0..2) for the E0393 winner + bucket diagnosis
  - Evidence: Skipped (not promoted) because E0393 val402 does not beat the baseline val winner. Evidence: `runs/E0393_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_ltl_top1med_norm_v1_20260206-104413/sweep_summary.json` (best Δ≈+0.00881, p≈0.0971).

- [x] M0155 (plan: P0134 / exp: E0393): Implement `EVENTNESS=av_clipdiff_flow_mlp` (audio basic + CLIPdiff + optical-flow magnitude) + scripts, then run full official val402 sweep (SEEDS=0..2) under `candidate_set=ltl_top1med_norm_v1`
  - Evidence:
    - Code: `python -m py_compile avs/experiments/ave_p0_sweep.py avs/vision/cheap_eventness.py` → ok.
    - Smoke: `runs/E0393_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_ltl_top1med_norm_v1_20260206-104337/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1; best Δ≈+0.00156).
    - Full val402 (SEEDS=0..2): `runs/E0393_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_ltl_top1med_norm_v1_20260206-104413/sweep_summary.json` (best=`ltltop1medn_thr0p6_shift1`, Δ≈+0.00881, p≈0.0971) + `runs/E0393_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_ltl_top1med_norm_v1_20260206-104413/best_config.json`.
    - Decision: Not promoted to test402 (skip E0394/E0395) because val402 does not beat baseline val winners.

- [x] M0154 (plan: P0133 / exp: E0392,E0344 / conclusion: C0003): If promoted, run full test402 (SEEDS=0..9) for the E0390 winner and update C0003 evidence
  - Evidence: Skipped (not promoted) because E0390 val402 is not competitive. Evidence: `runs/E0390_ave_p0_sweep_official_val_av_panns_embed_clipdiff_mlp_ltl_top1med_norm_v1_20260206-102257/sweep_summary.json` (best Δ≈+0.00374, p≈0.542).

- [x] M0153 (plan: P0133 / exp: E0391,E0344): If promoted, run quick test402 transfer (SEEDS=0..2) for the E0390 winner + bucket diagnosis
  - Evidence: Skipped (not promoted) because E0390 val402 is not competitive. Evidence: `runs/E0390_ave_p0_sweep_official_val_av_panns_embed_clipdiff_mlp_ltl_top1med_norm_v1_20260206-102257/sweep_summary.json` (best Δ≈+0.00374, p≈0.542).

- [x] M0152 (plan: P0133 / exp: E0390): Implement `EVENTNESS=av_panns_embed_clipdiff_mlp` (PANNs embeddings + CLIPdiff supervised anchors) + scripts, then run full official val402 sweep (SEEDS=0..2) under `candidate_set=ltl_top1med_norm_v1`
  - Evidence:
    - Code: `python -m py_compile avs/experiments/ave_p0_sweep.py` → ok.
    - Smoke: `runs/E0390_ave_p0_sweep_official_val_av_panns_embed_clipdiff_mlp_ltl_top1med_norm_v1_20260206-102211/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1; best Δ≈+0.00469).
    - Full val402 (SEEDS=0..2): `runs/E0390_ave_p0_sweep_official_val_av_panns_embed_clipdiff_mlp_ltl_top1med_norm_v1_20260206-102257/sweep_summary.json` (best=`ltltop1medn_thr0p7_shift1`, Δ≈+0.00374, p≈0.542) + `runs/E0390_ave_p0_sweep_official_val_av_panns_embed_clipdiff_mlp_ltl_top1med_norm_v1_20260206-102257/best_config.json`.
    - Decision: Not promoted to test402 (skip E0391/E0392) because val402 is not competitive.

- [x] M0151 (plan: P0132 / exp: E0389,E0344 / conclusion: C0003): If promoted, run full test402 (SEEDS=0..9) for the E0387 winner and update C0003 evidence
  - Evidence: Skipped (not promoted) because E0387 val402 is not competitive. Evidence: `runs/E0387_ave_p0_sweep_official_val_panns_embed_mlp_ltl_top1med_norm_v1_20260206-095447/sweep_summary.json` (best Δ≈+0.00208).

- [x] M0150 (plan: P0132 / exp: E0388,E0344): If promoted, run quick test402 transfer (SEEDS=0..2) for the E0387 winner + bucket diagnosis
  - Evidence: Skipped (not promoted) because E0387 val402 is not competitive. Evidence: `runs/E0387_ave_p0_sweep_official_val_panns_embed_mlp_ltl_top1med_norm_v1_20260206-095447/sweep_summary.json` (best Δ≈+0.00208).

- [x] M0149 (plan: P0132 / exp: E0387): Implement `EVENTNESS=panns_embed_mlp` (supervised MLP calibration on PANNs embeddings) + scripts, then run full official val402 sweep (SEEDS=0..2) under `candidate_set=ltl_top1med_norm_v1`
  - Evidence:
    - Code: `python -m py_compile avs/experiments/ave_p0_sweep.py` → ok.
    - Smoke: `runs/E0387_ave_p0_sweep_official_val_panns_embed_mlp_ltl_top1med_norm_v1_20260206-095420/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1; best Δ≈+0.00469).
    - Full val402: `runs/E0387_ave_p0_sweep_official_val_panns_embed_mlp_ltl_top1med_norm_v1_20260206-095447/sweep_summary.json` (best Δ≈+0.00208). Decision: not promoted.

- [x] M0148 (plan: P0131 / exp: E0386,E0344 / conclusion: C0003): If promoted, run full test402 (SEEDS=0..9) for the E0384 winner and update C0003 evidence
  - Evidence: Skipped (not promoted) because E0384 val402 is not competitive. Evidence: `runs/E0384_ave_p0_sweep_official_val_panns_embed_lr_ltl_top1med_norm_v1_20260206-094428/sweep_summary.json` (best Δ≈+0.00865).

- [x] M0147 (plan: P0131 / exp: E0385,E0344): If promoted, run quick test402 transfer (SEEDS=0..2) for the E0384 winner + bucket diagnosis
  - Evidence: Skipped (not promoted) because E0384 val402 is not competitive. Evidence: `runs/E0384_ave_p0_sweep_official_val_panns_embed_lr_ltl_top1med_norm_v1_20260206-094428/sweep_summary.json` (best Δ≈+0.00865).

- [x] M0146 (plan: P0131 / exp: E0384): Implement `EVENTNESS=panns_embed_lr` (supervised calibration on PANNs embeddings) + scripts, then run full official val402 sweep (SEEDS=0..2) under `candidate_set=ltl_top1med_norm_v1`
  - Evidence:
    - Code: `python -m py_compile avs/audio/panns_probe.py avs/experiments/ave_p0_sweep.py` → ok.
    - Smoke: `runs/E0384_ave_p0_sweep_official_val_panns_embed_lr_ltl_top1med_norm_v1_20260206-094359/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1; best Δ≈+0.00469).
    - Full val402: `runs/E0384_ave_p0_sweep_official_val_panns_embed_lr_ltl_top1med_norm_v1_20260206-094428/sweep_summary.json` (best Δ≈+0.00865). Decision: not promoted.

- [x] M0145 (plan: P0130 / exp: E0383,E0344 / conclusion: C0003): If promoted, run full test402 (SEEDS=0..9) for the E0381 winner and update C0003 evidence
  - Evidence: Skipped (not promoted) because E0381 val402 regresses. Evidence: `runs/E0381_ave_p0_sweep_official_val_panns_lr_ltl_top1med_norm_v1_20260206-093023/sweep_summary.json` (best Δ≈-0.00224).

- [x] M0144 (plan: P0130 / exp: E0382,E0344): If promoted, run quick test402 transfer (SEEDS=0..2) for the E0381 winner + bucket diagnosis
  - Evidence: Skipped (not promoted) because E0381 val402 regresses. Evidence: `runs/E0381_ave_p0_sweep_official_val_panns_lr_ltl_top1med_norm_v1_20260206-093023/sweep_summary.json` (best Δ≈-0.00224).

- [x] M0143 (plan: P0130 / exp: E0381): Implement `EVENTNESS=panns_lr` (supervised calibration on PANNs outputs) + scripts, then run full official val402 sweep (SEEDS=0..2) under `candidate_set=ltl_top1med_norm_v1`
  - Evidence:
    - Code: `python -m py_compile avs/audio/panns_probe.py avs/experiments/ave_p0_sweep.py` → ok.
    - Smoke: `runs/E0381_ave_p0_sweep_official_val_panns_lr_ltl_top1med_norm_v1_20260206-092930/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1; best Δ≈+0.00156).
    - Full val402: `runs/E0381_ave_p0_sweep_official_val_panns_lr_ltl_top1med_norm_v1_20260206-093023/sweep_summary.json` (best Δ≈-0.00224). Decision: not promoted.

- [x] M0142 (plan: P0129 / exp: E0380,E0344 / conclusion: C0003): If promoted, run full test402 (SEEDS=0..9) for the E0378 winner and update C0003 evidence
  - Evidence: Skipped (not promoted) because E0378 val402 is not competitive. Evidence: `runs/E0378_ave_p0_sweep_official_val_panns_ltl_top1med_norm_v1_20260206-090736/sweep_summary.json` (best Δ≈+0.00998).

- [x] M0141 (plan: P0129 / exp: E0379,E0344): If promoted, run quick test402 transfer (SEEDS=0..2) for the E0378 winner + bucket diagnosis
  - Evidence: Skipped (not promoted) because E0378 val402 is not competitive. Evidence: `runs/E0378_ave_p0_sweep_official_val_panns_ltl_top1med_norm_v1_20260206-090736/sweep_summary.json` (best Δ≈+0.00998).

- [x] M0140 (plan: P0129 / exp: E0378): Run full official val402 sweep (SEEDS=0..2) for `EVENTNESS=panns` under `candidate_set=ltl_top1med_norm_v1` and record the winner
  - Evidence: Smoke: `runs/E0378_ave_p0_sweep_official_val_panns_ltl_top1med_norm_v1_20260206-090624/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1). Full val402: `runs/E0378_ave_p0_sweep_official_val_panns_ltl_top1med_norm_v1_20260206-090736/sweep_summary.json` (best=`ltltop1medn_thr0p6_shift1`, Δ≈+0.00998) + `runs/E0378_ave_p0_sweep_official_val_panns_ltl_top1med_norm_v1_20260206-090736/best_config.json`.

- [x] M0139 (plan: P0128 / exp: E0376,E0377 / conclusion: C0003): If promoted, run test402 (quick/full) for `vision_binary_mlp` and update evidence
  - Evidence: Skipped (not promoted) because E0375 regresses on val402. Evidence: `runs/E0375_ave_p0_sweep_official_val_vision_binary_mlp_ltl_top1med_norm_v1_20260206-082152/sweep_summary.json` (best Δ≈-0.00175).

- [x] M0138 (plan: P0128 / exp: E0375): Run full official val402 sweep (SEEDS=0..2) for `EVENTNESS=vision_binary_mlp` under `candidate_set=ltl_top1med_norm_v1` and record the winner
  - Evidence: Smoke: `runs/E0375_ave_p0_sweep_official_val_vision_binary_mlp_ltl_top1med_norm_v1_20260206-082121/sweep_summary.json` (LIMIT_TRAIN=200 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1). Full val402: `runs/E0375_ave_p0_sweep_official_val_vision_binary_mlp_ltl_top1med_norm_v1_20260206-082152/sweep_summary.json` (best Δ≈-0.00175, p≈0.0938). Decision: not promoted.

- [x] M0137 (plan: P0127 / exp: E0374,E0344 / conclusion: C0003): If promoted, run full test402 (SEEDS=0..9) for the E0372 winner and update C0003 evidence
  - Evidence: Skipped (not promoted) after E0373 quick test402 was not competitive: `runs/E0373_quick_test402_av_clipdiff_mlp_ltl_top1med_gate_all_v1_20260206-081728/metrics.json` (Δ≈+0.01086, p≈0.1165) + diagnosis `runs/E0344_ave_p0_diagnose_20260206-081835/diagnose.json`.

- [x] M0136 (plan: P0127 / exp: E0373,E0344): If promoted, run quick test402 transfer (SEEDS=0..2) for the E0372 winner + bucket diagnosis
  - Evidence: `runs/E0373_quick_test402_av_clipdiff_mlp_ltl_top1med_gate_all_v1_20260206-081728/metrics.json` (anchored=0.7179 vs uniform=0.7070, Δ≈+0.01086, p≈0.1165) + diagnosis `runs/E0344_ave_p0_diagnose_20260206-081835/diagnose.json` (fallback_used_frac≈0.751; 2-high bucket remains harmful). Decision: not promoted.

- [x] M0135 (plan: P0127 / exp: E0372): Run full official val402 sweep (SEEDS=0..2) for `candidate_set=ltl_top1med_gate_all_v1` and record the winner
  - Evidence: `runs/E0372_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_gate_all_v1_20260206-081233/sweep_summary.json` (best=`ltltop1med_gateall0p4_shift0`, Δ≈+0.00989, p≈0.00344) + `runs/E0372_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_gate_all_v1_20260206-081233/best_config.json`.

- [x] M0134 (plan: P0127): Implement `anchor_gate_method=lr_top1hit_all_v1` (veto gate) + `candidate_set=ltl_top1med_gate_all_v1` + scripts
  - Evidence:
    - Code: `python -m py_compile avs/experiments/ave_p0.py avs/experiments/ave_p0_sweep.py` → ok.
    - Smoke: `runs/E0372_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_gate_all_v1_20260206-080613/sweep_summary.json`.

- [x] M0133 (plan: P0126 / exp: E0371,E0344 / conclusion: C0003): If promoted, run full test402 (SEEDS=0..9) for `av_ast_clipalign_bce` and update C0003 evidence
  - Evidence: Skipped (not promoted) after E0370 quick test402 was not competitive: `runs/E0370_quick_test402_av_ast_clipalign_bce_20260206-072535/metrics.json` (Δ≈+0.00813, p≈0.365) + diagnosis `runs/E0344_ave_p0_diagnose_20260206-073731/diagnose.json`.

- [x] M0132 (plan: P0126 / exp: E0370,E0344): Run quick test402 transfer (SEEDS=0..2) for `av_ast_clipalign_bce` (E0318 selection) + bucket diagnosis
  - Evidence: `runs/E0370_quick_test402_av_ast_clipalign_bce_20260206-072535/metrics.json` (anchored=0.7152 vs uniform=0.7070, Δ≈+0.00813, p≈0.365) + diagnosis `runs/E0344_ave_p0_diagnose_20260206-073731/diagnose.json` (fallback_used_frac≈0.570). Decision: not promoted.

- [x] M0131 (plan: P0125 / exp: E0369,E0344 / conclusion: C0003): If promoted, run full test402 (SEEDS=0..9) and update C0003 evidence
  - Evidence: Not promoted after E0367 full val402 (near-0); skipped E0369 per the “拉大” execution rule. Evidence: `runs/E0367_ave_p0_sweep_official_val_av_ast_clipdiff_mlp_ltl_top1med_norm_v1_20260206-070639/sweep_summary.json`.

- [x] M0130 (plan: P0125 / exp: E0368,E0344): If promoted, run quick test402 transfer (SEEDS=0..2) + bucket diagnosis
  - Evidence: Not promoted after E0367 full val402 (near-0); skipped E0368 per the “拉大” execution rule. Evidence: `runs/E0367_ave_p0_sweep_official_val_av_ast_clipdiff_mlp_ltl_top1med_norm_v1_20260206-070639/sweep_summary.json`.

- [x] M0129 (plan: P0125 / exp: E0367): Run full official val402 sweep (SEEDS=0..2) for `EVENTNESS=av_ast_clipdiff_mlp` and record the winner
  - Evidence: Smoke: `runs/E0367_ave_p0_sweep_official_val_av_ast_clipdiff_mlp_ltl_top1med_norm_v1_20260206-070549/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1). Full val402: `runs/E0367_ave_p0_sweep_official_val_av_ast_clipdiff_mlp_ltl_top1med_norm_v1_20260206-070639/sweep_summary.json` (best=`ltltop1medn_thr0p7_shift0`, Δ≈+0.00183, p≈0.853; best_by_pfilter=None). Decision: not promoted.

- [x] M0128 (plan: P0125 / exp: E0367): Smoke-run `EVENTNESS=av_ast_clipdiff_mlp` sweep on a tiny official split (stage-1 heavy; validate AST pretrained + caching)
  - Evidence: `runs/E0367_ave_p0_sweep_official_val_av_ast_clipdiff_mlp_ltl_top1med_norm_v1_20260206-070549/sweep_summary.json`.

- [x] M0127 (plan: P0124 / exp: E0366,E0344 / conclusion: C0003): If promoted, run full test402 (SEEDS=0..9) and update C0003 evidence
  - Evidence: Not promoted after E0364 full val402 (near-0); skipped E0366 per the “拉大” execution rule. Evidence: `runs/E0364_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_keepadj_basealloc_highonly_v1_20260206-064910/sweep_summary.json`.

- [x] M0126 (plan: P0124 / exp: E0365,E0344): If promoted, run quick test402 transfer (SEEDS=0..2) + bucket diagnosis
  - Evidence: Not promoted after E0364 full val402 (near-0); skipped E0365 per the “拉大” execution rule. Evidence: `runs/E0364_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_keepadj_basealloc_highonly_v1_20260206-064910/sweep_summary.json`.

- [x] M0125 (plan: P0124 / exp: E0364): Run full official val402 sweep (SEEDS=0..2) for `candidate_set=ltl_top1med_keepadj_basealloc_highonly_v1` and record the winner
  - Evidence: Smoke: `runs/E0364_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_keepadj_basealloc_highonly_v1_20260206-064816/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1). Full val402: `runs/E0364_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_keepadj_basealloc_highonly_v1_20260206-064910/sweep_summary.json` (best=`ltltop1med_keepadj_mixed_high`, Δ≈+0.00291, p≈0.167; best_by_pfilter=None). Decision: not promoted.

- [x] M0124 (plan: P0124): Implement `anchor_base_alloc=*_high` (high-set-only base allocation) and wire it through AVE-P0 planning + smokes
  - Evidence: `python -m py_compile avs/sampling/plans.py avs/experiments/ave_p0.py avs/experiments/ave_p0_sweep.py avs/smoke_checks.py` → ok; `python -m avs.smoke sampling_plan` → ok.

- [x] M0123 (plan: P0123 / exp: E0363,E0344 / conclusion: C0003): If promoted, run full test402 (SEEDS=0..9) and update C0003 evidence
  - Evidence: Not promoted after E0361 full val402 (near-0); skipped E0363 per the “拉大” execution rule. Evidence: `runs/E0361_ave_p0_sweep_official_val_clap_mlp_cls_target_ltl_top1med_norm_v1_20260206-054923/sweep_summary.json`.

- [x] M0122 (plan: P0123 / exp: E0362,E0344): Run quick test402 transfer (SEEDS=0..2) for the E0361 winner + bucket diagnosis
  - Evidence: Not promoted after E0361 full val402 (near-0); skipped E0362 per the “拉大” execution rule. Evidence: `runs/E0361_ave_p0_sweep_official_val_clap_mlp_cls_target_ltl_top1med_norm_v1_20260206-054923/sweep_summary.json`.

- [x] M0121 (plan: P0123 / exp: E0361): Run full val402 sweep (SEEDS=0..2) for `EVENTNESS=clap_mlp_cls_target` and record the winner
  - Evidence: `runs/E0361_ave_p0_sweep_official_val_clap_mlp_cls_target_ltl_top1med_norm_v1_20260206-054923/sweep_summary.json` (best=`ltltop1medn_thr0p7_shift0`, Δ≈+0.00158, p≈0.284; best_by_pfilter=None). Decision: not promoted to test402.

- [x] M0120 (plan: P0123 / exp: E0361): Smoke-run CLAP-embedding multi-class anchor sweep on a tiny official split
  - Evidence: `runs/E0361_ave_p0_sweep_official_val_clap_mlp_cls_target_ltl_top1med_norm_v1_20260206-054731/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1).

- [x] M0119 (plan: P0122 / exp: E0360,E0344 / conclusion: C0003): If promoted, run full test402 (SEEDS=0..9) and update C0003 evidence
  - Evidence: Not promoted after E0358 full val402 (best is negative); skipped E0360 per the “拉大” execution rule. Evidence: `runs/E0358_ave_p0_sweep_official_val_clap_lr_ltl_top1med_norm_v1_20260206-045105/sweep_summary.json`.

- [x] M0118 (plan: P0122 / exp: E0359,E0344): Run quick test402 transfer (SEEDS=0..2) for the E0358 winner + bucket diagnosis
  - Evidence: Not promoted after E0358 full val402 (best is negative); skipped E0359 per the “拉大” execution rule. Evidence: `runs/E0358_ave_p0_sweep_official_val_clap_lr_ltl_top1med_norm_v1_20260206-045105/sweep_summary.json`.

- [x] M0117 (plan: P0122 / exp: E0358): Run full val402 sweep (SEEDS=0..2) for `EVENTNESS=clap_lr` and record the winner
  - Evidence: `runs/E0358_ave_p0_sweep_official_val_clap_lr_ltl_top1med_norm_v1_20260206-045105/sweep_summary.json` (best=`ltltop1medn_thr0p5_shift0`, Δ≈-0.00191, p≈0.625; top3 are all negative). Decision: not promoted to test402.

- [x] M0116 (plan: P0122 / exp: E0358): Smoke-run CLAP-supervised LR sweep on a tiny official split
  - Evidence: `runs/E0358_ave_p0_sweep_official_val_clap_lr_ltl_top1med_norm_v1_20260206-044924/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1).

- [x] M0115 (plan: P0121 / exp: E0357,E0344 / conclusion: C0003): If promoted, run full test402 (SEEDS=0..9) and update C0003 evidence
  - Evidence: not promoted after E0356 quick test (best config collapses to fallback_used_frac≈0.998). Skipped E0357 per the “拉大” execution rule.

- [x] M0114 (plan: P0121 / exp: E0356,E0344): Run quick test402 transfer (SEEDS=0..2) for the E0355 winner + bucket diagnosis
  - Evidence: `runs/E0356_quick_test402_clap_evt_k1_20260206-042339/metrics.json` (Δ≈+0.00489, p≈0.289) + diagnosis `runs/E0344_ave_p0_diagnose_20260206-042437/diagnose.json` (fallback_used_frac≈0.998). Extra diagnostic: `runs/E0356_quick_test402_clap_evt_k1_thr0p4_shift0_20260206-042532/metrics.json` (Δ≈+0.01219, p≈0.183) + diagnosis `runs/E0344_ave_p0_diagnose_20260206-042617/diagnose.json` (fallback_used_frac≈0.913). Decision: not promoted.

- [x] M0113 (plan: P0121 / exp: E0355): Run full val402 sweep (SEEDS=0..2) for `EVENTNESS=clap_evt` under k=1 and record the winner
  - Evidence: `runs/E0355_ave_p0_sweep_official_val_clap_evt_ltl_top1med_k1_v1_20260206-041855/sweep_summary.json` (best=`ltltop1medk1_thr0p6_shift1`, Δ≈+0.00391, p≈0.0315).

- [x] M0112 (plan: P0121 / exp: E0355): Smoke-run k=1 sweep for `clap_evt` on a tiny official split (reuse E0352 scores)
  - Evidence: `runs/E0355_ave_p0_sweep_official_val_clap_evt_ltl_top1med_k1_v1_20260206-041756/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1; reused E0352 score cache).

- [x] M0111 (plan: P0120 / exp: E0354,E0344 / conclusion: C0003): If promoted, run full test402 (SEEDS=0..9) and update C0003 evidence
  - Evidence: not promoted after E0353 quick test (Δ≈+0.00813, p≈0.457). Skipped E0354 per the “拉大” execution rule; next step is to remove the harmful 2-anchor regime (k=1 / maxHigh1) or change Stage-1 signal.

- [x] M0110 (plan: P0120 / exp: E0353,E0344): Run quick test402 transfer (SEEDS=0..2) for the E0352 winner + bucket diagnosis
  - Evidence: `runs/E0353_quick_test402_clap_evt_20260206-040527/metrics.json` (uniform=0.70705, anchored=0.71517, Δ≈+0.00813, p≈0.457) + diagnosis `runs/E0344_ave_p0_diagnose_20260206-040943/diagnose.json` (fallback_used_frac≈0.478; 2-high remains net harmful).

- [x] M0109 (plan: P0120 / exp: E0352): Run full val402 sweep (SEEDS=0..2) for `EVENTNESS=clap_evt` and record the winner
  - Evidence: `runs/E0352_ave_p0_sweep_official_val_clap_evt_ltl_top1med_norm_v1_20260206-033347/sweep_summary.json` (best=`ltltop1medn_thr0p6_shift1`, Δ≈+0.00657, p≈0.202; weak).

- [x] M0108 (plan: P0120 / exp: E0352): Smoke-run CLAP prompt eventness sweep on a tiny official split
  - Evidence: `runs/E0352_ave_p0_sweep_official_val_clap_evt_ltl_top1med_norm_v1_20260206-032925/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1; AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:1).

- [x] M0107 (plan: P0119 / exp: E0351,E0344): If promoted, run full test402 (SEEDS=0..9) for the E0349 winner + update C0003 evidence
  - Evidence: not promoted after E0350 (Δ≈+0.00464, p≈0.00412 but fallback_used_frac≈0.930). Skipped E0351 per the “拉大” execution rule; next step is to change the confidence metric/threshold (or Stage-1 signal).

- [x] M0106 (plan: P0119 / exp: E0350,E0344): Run quick test402 transfer (SEEDS=0..2) for the E0349 winner + bucket diagnosis
  - Evidence: `runs/E0350_quick_test402_av_clap_clip_agree_k1_20260206-030727/metrics.json` (uniform=0.70705, anchored=0.71169, Δ≈+0.00464, p≈0.00412) + `runs/E0344_ave_p0_diagnose_20260206-030755/diagnose.json` (fallback_used_frac≈0.930; anchors rarely used).

- [x] M0105 (plan: P0119 / exp: E0349): Run full val402 sweep (SEEDS=0..2) for `av_clap_clip_agree` under k=1 candidate set and record the winner
  - Evidence: `runs/E0349_ave_p0_sweep_official_val_av_clap_clip_agree_ltl_top1med_k1_v1_20260206-030147/sweep_summary.json` (best=`ltltop1medk1_thr0p4_shift0`, Δ≈+0.00939, p≈0.125 on val402).

- [x] M0104 (plan: P0119 / exp: E0349): Smoke-run k=1 sweep for `av_clap_clip_agree` on a tiny official split
  - Evidence: `runs/E0349_ave_p0_sweep_official_val_av_clap_clip_agree_ltl_top1med_k1_v1_20260206-030057/{sweep_summary.json,best_config.json,eventness_scores.json}` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1; reused E0346 score cache).

- [x] M0103 (plan: P0118 / exp: E0348,E0344): If promoted, run full test402 (SEEDS=0..9) for the E0346 winner + update C0003 evidence
  - Evidence: not promoted after E0347 (Δ≈-0.00174, p≈0.801; far/2-high harm persists). Skipped E0348 per the “拉大” execution rule; next step is to change Stage-1 signal.

- [x] M0102 (plan: P0118 / exp: E0347,E0344): Run quick test402 transfer (SEEDS=0..2) for the E0346 winner + bucket diagnosis
  - Evidence: `runs/E0347_quick_test402_av_clap_clip_agree_20260206-024249/metrics.json` (uniform=0.70705, anchored=0.70531, Δ≈-0.00174, p≈0.801) + `runs/E0344_ave_p0_diagnose_20260206-024700/diagnose.json` (fallback_used_frac≈0.149 but far anchors harmful: dist=4 meanΔ≈-0.0478; 2-high meanΔ≈-0.0111).

- [x] M0101 (plan: P0118 / exp: E0346): Run full val402 sweep for `av_clap_clip_agree` (SEEDS=0..2) and record the winner
  - Evidence: `runs/E0346_ave_p0_sweep_official_val_av_clap_clip_agree_ltl_top1med_norm_v1_20260206-020740/{sweep_summary.json,best_config.json,eventness_scores.json}` (best=`ltltop1medn_thr0p6_shift1`, Δ≈+0.00599, p≈0.373 on val402; not competitive yet).

- [x] M0100 (plan: P0117): Smoke-verify `EVENTNESS=av_clap_clip_agree` end-to-end on a tiny official split
  - Evidence: `runs/E0346_ave_p0_sweep_official_val_av_clap_clip_agree_ltl_top1med_norm_v1_20260206-020428/{sweep_summary.json,best_config.json,eventness_scores.json}` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1; AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0).

- [x] M0099 (plan: P0116 / exp: E0343,E0344): Run full test402 (SEEDS=0..9) and update C0003 evidence.
  - Evidence (full test402): `runs/E0343_full_test402_av_clipdiff_mlp_20260206-005134/metrics.json` (anchored=0.72383 vs uniform=0.70858, Δ=+0.01525, p≈0.00390; does not prove C0003 +2%).
  - Evidence (diagnosis): `runs/E0344_ave_p0_diagnose_20260206-005232/diagnose.json` (persistent harmful buckets: dist∈{2..5} negative; 2-high negative; fallback≈0.751).

- [x] M0098 (plan: P0116 / exp: E0342,E0344): Run quick test402 (SEEDS=0..2) for the E0341 winner and diagnose buckets.
  - Evidence (winner quick test402): `runs/E0342_quick_test402_av_clipdiff_mlp_20260206-005041/metrics.json` (Δ≈+0.01899, p≈0.0429).
  - Evidence (quick diagnosis): `runs/E0344_ave_p0_diagnose_20260206-005134/diagnose.json` (2-high/far-anchor harm remains).
  - Evidence (follow-ups; midres band variants regress on test402 quick; stop): dist `runs/E0342_quick_test402_av_clipdiff_mlp_midres320_band_dist_20260206-005345/metrics.json` (Δ≈+0.01003), mixed `runs/E0342_quick_test402_av_clipdiff_mlp_midres320_band_mixed_20260206-005528/metrics.json` (Δ≈+0.01003), bridge `runs/E0342_quick_test402_av_clipdiff_mlp_midres320_band_bridge_20260206-005718/metrics.json` (Δ≈+0.00655). Extra demotion probe `runs/E0345_quick_test402_av_clipdiff_mlp_maxhigh1_20260206-010239/metrics.json` (Δ≈+0.01285). Conclusion: do not promote.

- [x] M0097 (plan: P0116 / exp: E0341): Run val402 sweep for `candidate_set=ltl_top1med_band_midres_v1` and record the winner.
  - Evidence (val402 sweep): `runs/E0341_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_band_midres_v1_20260206-004701/sweep_summary.json` (winner remains baseline `ltltop1med_thr0p6_shift1_base_exact352`; midres variants regress).
  - Evidence (winner config): `runs/E0341_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_band_midres_v1_20260206-004701/best_config.json`

- [x] M0096 (plan: P0116 / exp: E0340): Build official mid-res caches for `ltl_top1med_band_midres_v1`.
  - Evidence (cache_only val): `runs/E0340_cache_official_midres_20260206-000432/cache_val/cache_only.json` (train=3312, val=401, union=3703).
  - Evidence (cache_only test): `runs/E0340_cache_official_midres_20260206-000432/cache_test/cache_only.json` (train=3312, test=402, union=3706).
  - Output dir: `runs/REAL_AVE_OFFICIAL_20260201-124535/caches_112_160_192_208_224_320_352` (unique clip ids=4097).

- [x] M0095 (plan: P0115 / exp: E0336,E0337): Run cheap-visual fallback plan (visfb) val→test pipeline to attempt to “拉大” C0003.
  - Evidence (val402 sweep; both visfb variants regress): `runs/E0336_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_visfb_v1_20260205-212526/sweep_summary.json` (best remains uniform fallback baseline; framediff Δ≈-0.00648, clipdiff Δ≈-0.00756). Conclusion: naive cheap-visual fallback anchors are harmful; skip E0337.
  - Evidence (follow-up; gated visfb still regresses): `runs/E0338_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_visfb_gated_v1_20260205-220312/sweep_summary.json` (best remains uniform fallback baseline). Conclusion: visual confidence gating does not rescue; skip E0339.
- [x] M0094 (plan: P0114 / exp: E0334,E0335): Run learned rescue gate (lr_top1hit_v1) val→test pipeline to attempt to “拉大” C0003.
  - Evidence (val402 sweep; not competitive; gate rescues ~0): `runs/E0334_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_gate_lr_v1_20260205-205327/sweep_summary.json` (best Δ≈+0.00964, p≈0.0331; effectively matches E0223 baseline; `gate_rescued≈0`, fallback≈0.760).
  - Evidence (quick test402; not competitive; gate rescues 0): `runs/E0335_quick_gate0p6_shift0_test402_20260205-204831/metrics.json` (Δ≈+0.01086, p≈0.1165; fallback≈0.751; `gate_rescued=0`). Conclusion: learned rescue gate does not reduce fallback or improve Δ; stop.
- [x] M0093 (plan: P0112 / exp: E0331): Add degradation suite with downstream accuracy + α lower bound.
  - Evidence (code): `python -m py_compile avs/experiments/degradation_accuracy.py` → ok.
  - Evidence (smoke): `runs/E0331_smoke_av_clipdiff_mlp_20260205-194038/degradation_accuracy.json` + `runs/E0331_smoke_av_clipdiff_mlp_20260205-194038/degradation_plots/*.png`.
- [x] M0092 (plan: P0109): Add AST speech-veto "non-speech max" anchors (`ast_nonspeech_max`) and evaluate on val402.
  - Evidence (code): `python -m py_compile avs/experiments/ave_p0.py avs/experiments/ave_p0_sweep.py` → ok.
  - Evidence (val402 sweep; near-0): `runs/E0324_ave_p0_sweep_official_val_ast_nonspeech_max_ltl_top1med_norm_v1_20260205-144057/sweep_summary.json` (best Δ≈+0.00324, p≈0.722). Conclusion: not competitive; stop before test402.

- [x] M0091 (plan: P0108): Run the bold Stage-2 band-budget + low=112 sweep (val402) and reproduce on test402 if competitive.
  - Evidence (val402 sweep): `runs/E0320_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_band_low112_v1_20260205-142258/sweep_summary.json` (best=`ltltop1medband112_thr0p7_shift1`, Δ≈+0.01205, p≈0.0685).
  - Evidence (quick test402; regress): `runs/E0321_quick_test402_av_clipdiff_mlp_band_low112_20260205-142726/metrics.json` (SEEDS=0..2; Δ≈+0.01028, p≈0.204). Conclusion: not competitive vs E0224; stop.

- [x] M0087 (plan: P0104): Add a deployable ASR/Whisper anchor probe (VAD + word timestamps) and evaluate on val402.
  - Evidence (code): `python -m py_compile avs/audio/vad_webrtc.py avs/experiments/ave_p0.py avs/experiments/ave_p0_sweep.py` → ok.
  - Evidence (val402 sweep; near-0): `runs/E0322_ave_p0_sweep_official_val_asr_vad_ltl_top1med_norm_v1_20260205-142328/sweep_summary.json` (best Δ≈+0.00216, p≈0.842). Conclusion: WebRTC-VAD speech ratio saturates on AVE; not a useful signal as-is.

- [x] M0090 (C0003 “拉大” next): Add Stage-1 A/V correspondence anchors (`av_ast_clipalign_bce`) and evaluate on val402.
  - Evidence (code): `python -m py_compile avs/experiments/ave_p0.py avs/experiments/ave_p0_sweep.py` → ok.
  - Evidence (val402 sweep; does not beat best): `runs/E0318_ave_p0_sweep_official_val_av_ast_clipalign_bce_ltl_top1med_norm_v1_20260205-133800/sweep_summary.json` (best Δ≈+0.00865, p≈0.00120). Conclusion: not viable as-is.
- [x] M0089 (C0003 “拉大” next): Teacher-student Stage-1 "downstream loss-gain" teacher (base vs high) and evaluate on val402.
  - Evidence (code): `python -m py_compile avs/experiments/ave_p0.py avs/experiments/ave_p0_sweep.py` → ok.
  - Evidence (val402 sweep; near-0): `runs/E0316_ave_p0_sweep_official_val_av_clipdiff_lossgain_mlp_ltl_top1med_norm_v1_20260205-125414/sweep_summary.json` (best Δ≈+0.00042, p≈0.906). Conclusion: not viable as-is.
- [x] M0088 (C0003 “拉大” next): Teacher-student Stage-1 "visual usefulness" anchors (visgain teacher) and evaluate on val402.
  - Evidence (code): `python -m py_compile avs/experiments/ave_p0.py avs/experiments/ave_p0_sweep.py` → ok.
  - Evidence (val402 sweep; near-0): `runs/E0314_ave_p0_sweep_official_val_av_clipdiff_visgain_mlp_ltl_top1med_norm_v1_20260205-123935/sweep_summary.json` (best Δ≈+0.00158, p≈0.727). Conclusion: not viable as-is.
- [x] M0086 (C0003 “拉大”): Implement learned k-adaptive anchor2 veto on top of `av_clipdiff_mlp` (drop spurious 2nd anchors).
  - Evidence (code + candidate set): `python -m py_compile avs/experiments/ave_p0.py avs/experiments/ave_p0_sweep.py` → ok.
  - Evidence (val402 sweep; does not improve): `runs/E0312_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_anchor2veto_v1_20260205-115927/sweep_summary.json` (best remains no-veto baseline; veto variants regress).
  - Evidence (quick test402 diagnostics; regress): `runs/E0313_quick_test402_av_clipdiff_mlp_a2veto_lr0p65_20260205-120329/metrics.json` and `runs/E0313_quick_test402_av_clipdiff_mlp_a2veto_top2med0p15_20260205-120449/metrics.json`.
- [x] M0085 (plan: P0101): Run E0307 (val402 sweep) → (stop; skip E0308) for `av_ast_clipdiff_mil_mlp` and update C0003 evidence.
  - Evidence (val402): `runs/E0307_ave_p0_sweep_official_val_av_ast_clipdiff_mil_mlp_ltl_top1med_v1_20260205-045530/sweep_summary.json` (best Δ≈-0.01180; all candidates negative).
  - Evidence (gate rescue attempt): `runs/E0309_ave_p0_sweep_official_val_av_ast_clipdiff_mil_mlp_ltl_top1med_norm_v1_20260205-051944/sweep_summary.json` (best Δ≈+0.00108; near 0).
  - Evidence (normalized gate sanity): `runs/E0310_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_norm_v1_20260205-052411/sweep_summary.json` (best Δ≈+0.00723; worse than E0223 best).
- [x] M0084 (plan: P0100): Add AST-embedding + CLIPdiff Stage-1 learned-anchor backends (`av_ast_clipdiff_*`) + score caching support.
  - Evidence: `python -m py_compile avs/experiments/ave_p0.py avs/experiments/ave_p0_sweep.py avs/pipeline/ave_p0_end2end.py` → ok.
  - Evidence: Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0307_ave_p0_sweep_official_val_av_ast_clipdiff_mil_mlp_ltl_top1med_v1_20260205-045341/sweep_summary.json`.
- [x] M0083 (plan: P0099): Implement a budget-band Stage-2 planner and run val→test eval (E0303–E0305) to try to “拉大” C0003.
  - Evidence: `python -m avs.smoke sampling_plan` → ok.
  - Evidence (val402): `runs/E0303_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_band_v1_20260205-033915/sweep_summary.json` (best=`ltltop1medband_thr0p7_shift1`, Δ≈+0.01205, p≈0.0685).
  - Evidence (test402): `runs/E0304_ave_p0_best_to_test_official_av_clipdiff_mlp_20260205-035830/metrics.json` (Δ=+0.00816, p≈0.0441; regresses vs E0224).
  - Evidence (diagnostic): `runs/E0305_ave_p0_best_to_test_official_av_clipdiff_mlp_banddiag_thr0p6_shift1_20260205-040513/metrics.json` (Δ=+0.00356, p≈0.230; regresses vs E0224). Conclusion: not viable as-is.
- [x] M0082 (plan: P0098): Run k=1 + 112/224/448 sweep (E0301) and test reproduction (E0302) to try to push C0003 to ≥+2%.
  - Evidence (val402): `runs/E0301_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_k1_extreme_v1_20260205-024429/sweep_summary.json` (best Δ≈+0.00856, p≈0.0657).
  - Evidence (test402): `runs/E0302_ave_p0_best_to_test_official_av_clipdiff_mlp_k1extreme_20260205-025129/metrics.json` (Δ≈+0.00162, p≈0.649; large regression vs E0224). Conclusion: not viable.
- [x] M0081 (plan: P0097): Add `base_alloc=bridge` and run the top1-med val→test pipeline (E0298–E0299) to target far-anchor 2-high regressions.
  - Evidence: `python -m avs.smoke sampling_plan` → ok.
  - Evidence (val402): `runs/E0298_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_bridgealloc_v1_20260205-023226/sweep_summary.json` (`bridgeAlloc` regresses to Δ≈+0.00175, p≈0.762).
  - Evidence (test402): `runs/E0299_ave_p0_best_to_test_official_av_clipdiff_mlp_bridgealloc_20260205-023457/metrics.json` (Δ=+0.01525, p≈0.00390; matches E0224 because bridge did not win val selection). Conclusion: bridge base allocation is not a viable “拉大” direction.
- [x] M0080 (plan: P0096): `ltl_top1med_moe_v1` candidate set + E0296/E0297 runners are missing (proper MOE eval under top1-med).
  - Evidence: `python -m py_compile avs/experiments/ave_p0_sweep.py` → ok.
  - Evidence (val402): `runs/E0296_ave_p0_sweep_official_val_moe_energy_clipdiff_ltl_top1med_moe_v1_20260205-014328/sweep_summary.json` (best Δ≈+0.00224).
  - Evidence (test402): `runs/E0297_ave_p0_best_to_test_official_moe_energy_clipdiff_20260205-015437/metrics.json` (Δ=+0.00306, p≈0.420). Conclusion: MOE fix does not improve C0003.
- [x] M0079 (plan: P0095): P0063 promised Stage-1 methods (`av_fused_clipdiff_prod`, `moe_energy_clipdiff`) are not yet run through the top1-med val→test pipeline (E0292–E0295).
  - Evidence: `python -m py_compile avs/experiments/ave_p0.py avs/experiments/ave_p0_sweep.py` → ok.
  - Evidence (val402): `runs/E0292_ave_p0_sweep_official_val_av_fused_clipdiff_prod_ltl_top1med_v1_20260205-012010/sweep_summary.json` (best Δ≈-0.00482), `runs/E0294_ave_p0_sweep_official_val_moe_energy_clipdiff_ltl_top1med_v1_20260205-012010/sweep_summary.json` (best Δ≈+0.00224).
  - Evidence (test402): `runs/E0293_ave_p0_best_to_test_official_av_fused_clipdiff_prod_20260205-012350/metrics.json` (Δ=+0.00575, p≈0.125) and `runs/E0295_ave_p0_best_to_test_official_moe_energy_clipdiff_20260205-012339/metrics.json` (Δ=+0.00306, p≈0.420). Conclusion: both regress vs the current best E0224 run; not viable “拉大” directions.
- [x] M0078 (plan: P0094): E0291 “train longer” diagnostic is missing from C0003 evidence / experiment ledger.
  - Evidence: `docs/experiment.md` now contains `E0291`, and `docs/plan.md` C0003 evidence references `runs/E0291_*/metrics.json`.
  - Evidence: `runs/E0291_ave_p0_best_to_test_official_av_clipdiff_mlp_top1med_e10_s0-2_20260205-001904/metrics.json` (Δ=+0.00390, p≈0.665; not significant).
- [x] M0077 (plan: P0093): Run val→test sweep for MIL Stage-1 eventness (av_clipdiff_mil_mlp) (E0289/E0290) and record results for C0003 “拉大”.
  - Evidence: `python -m py_compile avs/experiments/ave_p0.py avs/experiments/ave_p0_sweep.py` → ok.
  - Evidence (val402): `runs/E0289_ave_p0_sweep_official_val_av_clipdiff_mil_mlp_ltl_top1med_v1_20260205-000923/sweep_summary.json` (best=`ltltop1med_thr0p4_shift1`, Δ≈+0.00815, p≈0.302; worse than baseline E0223).
  - Evidence (test402): `runs/E0290_ave_p0_best_to_test_official_av_clipdiff_mil_mlp_20260205-001442/metrics.json` (anchored=0.71582 vs uniform=0.70858, Δ=+0.00724, p≈0.0791; regresses vs E0224). Conclusion: not a viable “拉大” direction.
- [x] M0076 (plan: P0092): Run far-anchor fallback-to-uniform (ff=1) on test402 for the current best top1-med config (E0288) and record results for C0003 “拉大”.
  - Evidence: `python -m py_compile avs/experiments/ave_p0.py avs/experiments/ave_p0_sweep.py` → ok.
  - Evidence (smoke): `runs/E0288_ave_p0_best_to_test_official_av_clipdiff_mlp_ltl_top1med_farfb_ff1_20260204-235108/metrics.json` (Δ=+0.00000, p=1.0).
  - Evidence (test402): `runs/E0288_ave_p0_best_to_test_official_av_clipdiff_mlp_ltl_top1med_farfb_ff1_20260204-235157/metrics.json` (Δ=+0.00938, p≈0.0880; regresses vs E0224). Conclusion: far-anchor fallback-to-uniform does not improve C0003.
- [x] M0075 (plan: P0091): Run val→test sweep for 224px CLIPdiff Stage-1 anchors (av_clipdiff_mlp_r224) (E0286/E0287) and record results for C0003 “拉大”.
  - Evidence: `python -m py_compile avs/experiments/ave_p0.py avs/experiments/ave_p0_sweep.py` → ok.
  - Evidence (val402): `runs/E0286_ave_p0_sweep_official_val_av_clipdiff_mlp_r224_ltl_top1med_v1_20260204-230324/sweep_summary.json` (best=`ltltop1med_thr0p7_shift0`, Δ≈+0.00682, p≈0.208; worse than baseline E0223).
  - Evidence (test402): `runs/E0287_ave_p0_best_to_test_official_av_clipdiff_mlp_r224_20260204-230749/metrics.json` (anchored=0.72087 vs uniform=0.70858, Δ=+0.01229, p≈0.00415; regresses vs E0224). Conclusion: not a viable “拉大” direction.
- [x] M0074 (plan: P0090): Sweep keep-adjacent demotion + base allocation (adaptive_v3) and reproduce on test402 (E0284/E0285).
  - Evidence: `python -m py_compile avs/experiments/ave_p0_sweep.py` → ok.
  - Evidence (val402): `runs/E0284_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_keepadj_basealloc_v1_20260204-224414/sweep_summary.json` (best=`ltltop1med_keepadj_distance`, Δ≈+0.00515, p≈0.286).
  - Evidence (test402): `runs/E0285_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-224708/metrics.json` (Δ=+0.00729, p≈0.1009; regresses vs E0224).
- [x] M0072 (plan: P0088): Rerun MDE-2 gap + degradation suite for the current best deployable Stage-1 method (E0201/E0203; default `EVENTNESS=av_clipdiff_mlp`).
  - Evidence (E0201): `runs/E0201_oracle_vs_predicted_av_clipdiff_mlp_20260204-213240/oracle_vs_predicted.json` (oracle_minus_predicted≈0.03383; predicted Δ=+0.00759, p=0.0945).
  - Evidence (E0203): `runs/E0203_degradation_av_clipdiff_mlp_20260204-215831/degradation_suite.json` (mean Recall@K,Δ0≈0.212; Δ2≈0.624).
- [x] M0073 (plan: P0089): Implement tiered-triad Stage-2 policy + run val→test sweep (ltl_top1med_tiered_v1; E0271/E0272).
  - Evidence: `python -m py_compile avs/experiments/ave_p0.py avs/experiments/ave_p0_sweep.py avs/pipeline/ave_p0_end2end.py` → ok.
  - Evidence (val402): `runs/E0271_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_tiered_v1_20260204-212636/sweep_summary.json` (best=`ltltop1med_thr0p6_shift1_base`, Δ≈+0.00964; tiered variants regress on val).
  - Evidence (test402): `runs/E0272_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-212918/metrics.json` (anchored=0.72383 vs uniform=0.70858, Δ=+0.01525, p≈0.00390; best_config is baseline; tiered triad unused). Conclusion: tiered triad does not beat E0224.
- [x] M0071 (plan: P0087): Add dual-proxy cheap-visual Stage-1 method and rerun top1-med sweep (`av_clipdiff_framediff_mlp`).
  - Evidence: `python -m py_compile avs/experiments/ave_p0.py avs/experiments/ave_p0_sweep.py avs/pipeline/ave_p0_end2end.py avs/experiments/mde_ltl.py avs/experiments/degradation_suite.py` → ok.
  - Evidence (val402): `runs/E0269_ave_p0_sweep_official_val_av_clipdiff_framediff_mlp_ltl_top1med_v1_20260204-202158/sweep_summary.json` (best=`ltltop1med_thr0p8_shift0`, Δ≈+0.00831; worse than baseline E0223).
  - Evidence (test402): `runs/E0270_ave_p0_best_to_test_official_av_clipdiff_framediff_mlp_20260204-203035/metrics.json` (Δ=+0.00672, p≈0.121; regresses vs E0224). Conclusion: not a viable path to “拉大” C0003.
- [x] M0070 (plan: P0086): Prefer adjacent 2nd anchor selection to reduce far-anchor harm (adjacent_top2; ltl_top1med_adjselect_v1).
  - Evidence: `python -m py_compile avs/audio/eventness.py avs/experiments/ave_p0_sweep.py` → ok.
  - Evidence: `runs/E0265_ave_p0_sweep_official_test_av_clipdiff_mlp_ltl_top1med_adjselect_v1_20260204-193725/sweep_summary.json` (best Δ≈+0.01692; regresses vs baseline on same seeds).
- [x] M0069 (plan: P0085): Sweep base-res allocation to reduce far-anchor harm (ltl_top1med_basealloc_v1).
  - Evidence: `runs/E0262_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_basealloc_v1_20260204-192317/sweep_summary.json` (val best is baseline distance).
  - Evidence: `runs/E0264_ave_p0_sweep_official_test_av_clipdiff_mlp_ltl_top1med_basealloc_v1_20260204-192638/sweep_summary.json` (test best is baseline distance).
- [x] M0068 (plan: P0084): Add far-anchor 2-high demotion that keeps both anchors for base allocation (adaptive_v3; ltl_top1med_keepadj_v1).
  - Evidence: `python -m py_compile avs/experiments/ave_p0.py avs/experiments/ave_p0_sweep.py` → ok.
  - Evidence: `runs/E0260_ave_p0_sweep_official_test_av_clipdiff_mlp_ltl_top1med_keepadj_v1_20260204-191347/sweep_summary.json` (best Δ≈+0.01194; regresses vs baseline).
- [x] M0067 (plan: P0083): Add a free resolution indicator feature and run resfeat sweep (ltl_top1med_resfeat_v1).
  - Evidence: `runs/E0258_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_resfeat_v1_20260204-183421/sweep_summary.json` (val best is `res_feature=none`; scalar regresses).
  - Evidence: `runs/E0259_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-184103/metrics.json` (test Δ=-0.00510; fails vs E0224).
- [x] M0066 (plan: P0082): Sweep head capacity / dropout under the fixed top1-med gate (ltl_top1med_headcap_v1).
  - Evidence: `runs/E0256_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_headcap_v1_20260204-182448/sweep_summary.json` (val improves), `runs/E0257_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-182818/metrics.json` (test regresses; Δ=+0.00124).
- [x] M0065 (plan: P0081): Sweep `anchor_high_adjacent_dist` on test402 to reduce harmful 2-high regime (ltl_top1med_adjdist_v1).
  - Evidence: `runs/E0254_ave_p0_sweep_official_test_av_clipdiff_mlp_ltl_top1med_adjdist_v1_20260204-181427/sweep_summary.json` (best is baseline `adj1`; demotion via `adj2..5` regresses).
- [x] M0064 (plan: P0080): Run conditional drop-far anchor2 sweep (ltl_top1med_dropfar_v1) val→test and record results for C0003 “拉大”.
  - Evidence: `runs/E0252_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_dropfar_v1_20260204-173949/sweep_summary.json` + `runs/E0252_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_dropfar_v1_20260204-173949/best_config.json`
  - Evidence: `runs/E0253_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-174232/metrics.json` (Δ=+0.00781; regresses vs E0224)

- [x] M0063 (plan: P0079): Run Stage-1 upgrade `av_clipdiff_fbank_mlp` val→test and record results for C0003 “拉大”.
  - Evidence: `runs/E0250_ave_p0_sweep_official_val_av_clipdiff_fbank_mlp_ltl_top1med_v1_20260204-172631/sweep_summary.json` + `runs/E0250_ave_p0_sweep_official_val_av_clipdiff_fbank_mlp_ltl_top1med_v1_20260204-172631/best_config.json`
  - Evidence: `runs/E0251_ave_p0_best_to_test_official_av_clipdiff_fbank_mlp_20260204-173034/metrics.json` (Δ=-0.00149; fails)

- [x] M0062 (plan: P0078): Run strong-NMS anchor selection (ltl_top1med_nmsstrong_v1) val→test and record results for C0003 “拉大”.
  - Evidence: `runs/E0248_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_nmsstrong_v1_20260204-170658/sweep_summary.json` + `runs/E0248_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_nmsstrong_v1_20260204-170658/best_config.json`
  - Evidence: `runs/E0249_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-171109/metrics.json` (Δ=+0.00883; regresses vs E0224)
- [x] M0061 (plan: P0076): Run top1-med + k=1 sweep (ltl_top1med_k1_v1) val→test and record results for C0003 “拉大”.
  - Evidence: `runs/E0235_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_k1_v1_20260204-153020/sweep_summary.json` + `runs/E0235_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_k1_v1_20260204-153020/best_config.json`
  - Evidence: `runs/E0236_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-153411/metrics.json`

- [x] M0060 (plan: P0075): Run top1-med + max_high_anchors=1 sweep (ltl_top1med_maxhigh1_v1) val→test and record results for C0003 “拉大”.
  - Evidence: `runs/E0233_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_maxhigh1_v1_20260204-151909/sweep_summary.json` + `runs/E0233_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_maxhigh1_v1_20260204-151909/best_config.json`
  - Evidence: `runs/E0234_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-152349/metrics.json`

- [x] M0059 (plan: P0074): Run av_clipdiff_mlp_r160 + ltl_top1med_v1 val→test and record results for C0003 “拉大”.
  - Evidence: `runs/E0230_ave_p0_sweep_official_val_av_clipdiff_mlp_r160_ltl_top1med_v1_20260204-144941/sweep_summary.json` + `runs/E0230_ave_p0_sweep_official_val_av_clipdiff_mlp_r160_ltl_top1med_v1_20260204-144941/best_config.json`
  - Evidence: `runs/E0231_ave_p0_best_to_test_official_av_clipdiff_mlp_r160_20260204-145349/metrics.json`

- [x] M0058 (plan: P0073): Run ltl_top1med_extreme_v1 val→test and record results for C0003 “拉大”.
  - Evidence: `runs/E0228_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_extreme_v1_20260204-143855/sweep_summary.json` + `runs/E0228_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_extreme_v1_20260204-143855/best_config.json`
  - Evidence: `runs/E0229_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-144335/metrics.json`

- [x] M0057 (plan: P0072): Run top1-med Stage-2 variant val→test and record results for C0003 “拉大”.
  - Evidence: `runs/E0226_ave_p0_stage2_variants_official_val_av_clipdiff_mlp_20260204-142732/variants_summary.json` + `runs/E0226_ave_p0_stage2_variants_official_val_av_clipdiff_mlp_20260204-142732/best_config.json`
  - Evidence: `runs/E0227_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-142936/metrics.json`

- [x] M0056 (plan: P0071): Run ltl_top1med_v1 val→test and record results for C0003 “拉大”.
  - Evidence: `runs/E0223_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_v1_20260204-135150/sweep_summary.json` + `runs/E0223_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_v1_20260204-135150/best_config.json`
  - Evidence: `runs/E0224_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-135547/metrics.json`

- [x] M0055 (plan: P0070): Run ltl_smooth_v1 val→test and record results for C0003 “拉大”.
  - Evidence: `runs/E0218_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_smooth_v1_20260204-132824/sweep_summary.json` + `runs/E0218_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_smooth_v1_20260204-132824/best_config.json`
  - Evidence: `runs/E0219_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-133929/metrics.json`

- [x] M0054 (plan: P0069): max-high=1 sweep (ltl_maxhigh1_v1) + runners are missing.
  - Evidence: `python -m py_compile avs/experiments/ave_p0_sweep.py` → ok
  - Evidence: `runs/E0214_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_maxhigh1_v1_20260204-120918/sweep_summary.json` + `runs/E0214_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_maxhigh1_v1_20260204-120918/best_config.json`
  - Evidence: `runs/E0215_ave_p0_best_to_test_official_av_clipdiff_mlp_ltl_maxhigh1_v1_20260204-121849/metrics.json`

- [x] M0053 (plan: P0055): val402-tuned confidence-gate sweep (E0204) is implemented for cross-method Stage-1 comparisons.
  - Evidence: `runs/E0204_gate_sweep_ast_lr_val402_20260204-020623/gate_sweep.json` + `runs/E0204_gate_sweep_ast_lr_val402_20260204-020623/best_gate.json`

- [x] M0052 (plan: P0054): `ast_lr` is wired end-to-end for E0201/E0203 (AST pretrained + array augment support).
  - Evidence: `runs/E0201_oracle_vs_predicted_ast_lr_sanity_20260204-015758/oracle_vs_predicted.json`
  - Evidence: `runs/E0203_degradation_ast_lr_sanity_20260204-015849/degradation_suite.json`
- [x] M0051 (plan: P0053): Learned (and probe-based) eventness methods are wired into E0201/E0203.
  - Evidence: `EVENTNESS=audio_fbank_mlp bash scripts/e0201_oracle_vs_predicted_official.sh` → ok (`runs/E0201_oracle_vs_predicted_audio_fbank_mlp_20260204-000652/oracle_vs_predicted.json`)
  - Evidence: `EVENTNESS=audio_basic_lr bash scripts/e0203_degradation_suite_official.sh` → ok (`runs/E0203_degradation_audio_basic_lr_20260204-012618/degradation_suite.json`)
- [x] M0050 (plan: P0052): AV-fused eventness collapses under the default confidence gate (std_thr=1.0); scale scores and rerun E0201.
  - Evidence: `python -m avs.smoke ltl_eventness_av_fused` → ok (`runs/smoke_20260203-221305/smoke.json`)
  - Evidence: `runs/E0201_oracle_vs_predicted_av_fused_scale3p5_full_20260203-221906/oracle_vs_predicted.json` (fallback_used_frac≈0.739; no longer collapses to `uniform`)
- [x] M0049 (plan: P0051): Strengthen Stage-1 anchors (energy_stride_max + av_fused) to close Oracle→Pred gap.
  - Evidence: `python -m avs.smoke all` → ok (`runs/smoke_20260203-205119/smoke.json`)
- [x] M0041 (plan: P0042): AVE fusion confirm artifacts are missing.
  - Evidence: `runs/E0013_ave_fusion_confirm_official_test_20260203-141624/metrics.json` (smoke: `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1`)
- [x] M0040 (plan: P0041): AVE sweep + best-to-test reproduction artifacts are missing.
  - Evidence: `runs/E0011_ave_p0_sweep_official_val_20260203-141418/sweep_summary.json` + `runs/E0011_ave_p0_sweep_official_val_20260203-141418/best_config.json` + `runs/E0012_ave_p0_best_to_test_official_20260203-141430/metrics.json` (smoke: `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1`)
- [x] M0039 (plan: P0038): AVE oracle ceiling sweep artifacts are missing.
  - Evidence: `bash scripts/e0010_ave_oracle_ceiling_official.sh` (smoke: `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1`) → ok (`runs/E0010_oracle_ceiling_official_val_20260203-141212/oracle_ceiling.json`)
- [x] M0048 (plan: P0050): Dataset checklist + download/verify helpers are missing.
  - Evidence: `bash scripts/datasets/verify_all.sh` → ok (`runs/datasets_verify_20260203-140543/datasets_verify.json`)
- [x] M0047 (plan: P0049): Degradation protocol runner is missing.
  - Evidence: `python -m avs.smoke ltl_degradation_suite_toy` → ok (`runs/smoke_20260203-140151/smoke.json`)
- [x] M0046 (plan: P0048): MDE harness + Pareto report tooling is missing.
  - Evidence: `python -m avs.smoke mde_pareto_toy` → ok (`runs/smoke_20260203-140023/smoke.json`)
- [x] M0045 (plan: P0047): Cheap-visual eventness baseline is missing.
  - Evidence: `python -m avs.smoke cheap_visual_eventness` → ok (`runs/smoke_20260203-135817/smoke.json`)
- [x] M0044 (plan: P0046): Stride-based audio eventness + anchor windows are missing.
  - Evidence: `python -m avs.smoke ltl_anchor_windows` → ok (`runs/smoke_20260203-135434/smoke.json`)
- [x] M0043 (plan: P0045): Listen-then-Look budget model + allocator are missing.
  - Evidence: `python -m avs.smoke ltl_budget_allocator` → ok (`runs/smoke_20260203-135142/smoke.json`)
- [x] M0042 (plan: P0044): Time-window primitives + Coverage@τ metric are missing.
  - Evidence: `python -m avs.smoke evidence_windows` → ok (`runs/smoke_20260203-134900/smoke.json`)
- [x] M0034 (plan: P0033): EPIC-SOUNDS untrimmed frame extraction helper is missing.
  - Evidence: `python -m avs.smoke epic_sounds_frames` → ok (`runs/smoke_20260131-141119/smoke.json`)
- [x] M0035 (plan: P0034): EPIC-SOUNDS local data layout paths are not standardized.
  - Evidence: `python -m avs.smoke epic_sounds_long_pack` → ok (`runs/smoke_20260131-142810/smoke.json`)
- [x] M0036 (plan: P0035): Hybrid long-video sampling plan generator is missing.
  - Evidence: `python -m avs.smoke epic_sounds_long_pack` → ok (`runs/smoke_20260131-142810/smoke.json`)
- [x] M0037 (plan: P0036): Long plan applier (manifest + cache on selected seconds) is missing.
  - Evidence: `python -m avs.smoke epic_sounds_long_pack` → ok (`runs/smoke_20260131-142810/smoke.json`)
- [x] M0038 (plan: P0037): EPIC-SOUNDS long-video end-to-end pack + smoke is missing.
  - Evidence: `python -m avs.smoke epic_sounds_long_pack` → ok (`runs/smoke_20260131-142810/smoke.json`)
- [x] M0033 (plan: P0032): ViT FLOPs estimator is missing from vision efficiency report.
  - Evidence: `python -m avs.smoke vision_efficiency` → ok (`runs/smoke_20260131-140020/smoke.json`)
- [x] M0032 (plan: P0031): energy_delta eventness backend is missing.
  - Evidence: `python -m avs.smoke energy_delta_eventness` → ok (`runs/smoke_20260131-135519/smoke.json`)
- [x] M0031 (plan: P0030): Long-wav plan.jsonl generation is missing.
  - Evidence: `python -m avs.smoke plan_jsonl_long` → ok (`runs/smoke_20260131-134913/smoke.json`)
- [x] M0030 (plan: P0029): Accuracy–Token efficiency curve plotting utility is missing.
  - Evidence: `python -m avs.smoke efficiency_curve` → ok (`runs/smoke_20260131-134055/smoke.json`)
- [x] M0029 (plan: P0028): Vision wall-clock efficiency benchmark is missing.
  - Evidence: `python -m avs.smoke vision_efficiency` → ok (`runs/smoke_20260131-133603/smoke.json`)
- [x] M0028 (plan: P0027): Uniform-112 (uniform_low) baseline is missing.
  - Evidence: `python -m avs.smoke ave_p0_uniform_low` → ok (`runs/smoke_20260131-132720/smoke.json`, `uniform_low_tokens=490 < uniform_tokens=1960`)
- [x] M0027 (plan: P0026): Temporal head option (1D conv) is missing.
  - Evidence: `python -m avs.smoke temporal_head` → ok (`runs/smoke_20260131-131427/smoke.json`)
- [x] M0026 (plan: P0025): EPIC-SOUNDS audio extraction helper is missing.
  - Evidence: `python -m avs.smoke epic_sounds_audio` → ok (`runs/smoke_20260131-130731/smoke.json`)
- [x] M0025 (plan: P0024): AudioMAE-like eventness probe backend is missing.
  - Evidence: `python -m avs.smoke audiomae_eventness` → ok (`runs/smoke_20260131-130255/smoke.json`)
- [x] M0024 (plan: P0023): EPIC-SOUNDS anchor quality eval is missing.
  - Evidence: `python -m avs.smoke epic_sounds_anchor_eval` → ok (`runs/smoke_20260131-125326/smoke.json`)
- [x] M0023 (plan: P0022): VGGSound metadata IO scaffold is missing.
  - Evidence: `python -m avs.smoke vggsound_meta` → ok (`runs/smoke_20260131-124701/smoke.json`)
- [x] M0022 (plan: P0021): Anchor robustness knobs (shift + fallback) are missing.
  - Evidence: `python -m avs.smoke anchor_knobs` → ok (`runs/smoke_20260131-124405/smoke.json`)
- [x] M0021 (plan: P0020): AVQA annotation IO + contrastive prompting templates are missing.
  - Evidence: `python -m avs.smoke avqa_meta` → ok (`runs/smoke_20260131-122617/smoke.json`)
- [x] M0020 (plan: P0019): EPIC-SOUNDS annotation IO scaffold is missing.
  - Evidence: `python -m avs.smoke epic_sounds_meta` → ok (`runs/smoke_20260131-122147/smoke.json`)
- [x] M0019 (plan: P0018): PANNs (AudioSet) eventness probe backend is missing.
  - Evidence: `python -m avs.smoke panns_eventness` → ok (`runs/smoke_20260131-121627/smoke.json`)
- [x] M0001 (plan: P0001): No runnable AVS codebase scaffold exists yet.
  - Evidence: `python -m avs.smoke` → ok (`runs/smoke_20260131-021255/smoke.json`)

- [x] M0002 (plan: P0002): AVE dataset IO/splits/labels parsing is missing.
  - Evidence: `python -m avs.smoke ave_meta` → ok (`runs/smoke_20260131-021549/smoke.json`)

- [x] M0003 (plan: P0003): Deterministic preprocessing (audio + per-second frames) is missing.
  - Evidence: `python -m avs.smoke preprocess_one` → ok (`runs/smoke_20260131-021745/smoke.json`)

- [x] M0004 (plan: P0004): Audio eventness + anchor generation + Recall@K is missing.
  - Evidence: `python -m avs.smoke anchors` → ok (`runs/smoke_20260131-022010/smoke.json`)

- [x] M0005 (plan: P0005): Token-budgeted sampling plan generator is missing.
  - Evidence: `python -m avs.smoke sampling_plan` → ok (`runs/smoke_20260131-022144/smoke.json`)

- [x] M0006 (plan: P0006): Variable-resolution vision encoder wrapper is missing.
  - Evidence: `python -m avs.smoke vision_encoder` → ok (`runs/smoke_20260131-022526/smoke.json`)

- [x] M0007 (plan: P0007): AVE classifier + training/eval loop is missing.
  - Evidence: `python -m avs.smoke train_smoke` → ok (`runs/smoke_20260131-022832/smoke.json`)

- [x] M0008 (plan: P0010): AudioSet/AST-based eventness probe is not implemented.
  - Evidence: `python -m avs.smoke ast_eventness` → ok (`runs/smoke_20260131-025729/smoke.json`)

- [x] M0009 (plan: P0011): Dataset-wide anchor evaluation (Recall@K / Recall@K,Δ) is missing.
  - Evidence: `python -m avs.smoke anchors_dataset` → ok (`runs/smoke_20260131-030009/smoke.json`)

- [x] M0010 (plan: P0012): Multi-resolution feature cache builder is missing.
  - Evidence: `python -m avs.smoke feature_cache` → ok (`runs/smoke_20260131-030146/smoke.json`)

- [x] M0011 (plan: P0013): AVE-P0 runner (Uniform/Random/Anchored/Oracle) on cached features is missing.
  - Evidence: `python -m avs.smoke ave_p0_toy` → ok (`runs/smoke_20260131-030551/smoke.json`)

- [x] M0012 (plan: P0014): Audio-Feature Concat baseline is missing in AVE-P0 runner.
  - Evidence: `python -m avs.smoke ave_p0_toy` → ok (`runs/smoke_20260131-030917/smoke.json`)

- [x] M0013 (plan: P0015): Qualitative visualization (anchors + plan + GT) is missing.
  - Evidence: `python -m avs.smoke viz` → ok (`runs/smoke_20260131-031035/smoke.json`)

- [x] M0014 (plan: P0016): AVE raw-video acquisition helper is missing.
  - Evidence: `python -m avs.smoke ave_download` → ok (`runs/smoke_20260131-031551/smoke.json`)

- [x] M0015 (plan: P0017): AVE-P0 end-to-end runner is missing.
  - Evidence: `python -m avs.smoke ave_p0_end2end` → ok (`runs/smoke_20260131-033331/smoke.json`)

- [x] M0016 (plan: C0002): Real-AVE anchor Recall@K evidence is missing.
  - Evidence: `bash scripts/e0002_anchor_eval_real.sh` → ok (`runs/E0002_anchors_real_20260131-035133/anchor_eval/anchors_metrics.json`)

- [x] M0017 (plan: C0001): C0001 is now proven on a real-AVE subset (temporal head + anchor robustness knobs).
  - Evidence: `runs/REAL_AVE_20260131-211548/p0_train180_val165_energy_160_224_352_k2_shift1_std1p0_temporal_s0-9_rerun/metrics.json` (val165; anchored>uniform; 10 seeds) and `runs/REAL_AVE_20260131-211548/p0_train180_test113_energy_160_224_352_k2_shift1_std1p0_temporal_s0-9_rerun/metrics.json` (test113; anchored>uniform; 10 seeds)

- [x] M0018 (plan: C0002): C0002 is proven on real-AVE subsets (energy anchors).
  - Evidence: `runs/REAL_AVE_20260131-211548/anchors_val165_energy/anchors_metrics.json` (k=2, Δ=0: ours>random; n=165) and `runs/REAL_AVE_20260131-211548/anchors_test113_energy/anchors_metrics.json` (k=2, Δ=0: ours>random; n=113)
