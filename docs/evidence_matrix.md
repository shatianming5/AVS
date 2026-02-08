# Evidence Matrix

- Generated: `2026-02-09 02:31:46`
- Plan: `docs/plan.md`

| Conclusion | Checked in plan | Local artifacts present? | Notes |
| --- | --- | --- | --- |
| `C0001` | yes | no | missing 15/15 artifacts |
| `C0002` | yes | no | missing 6/6 artifacts |
| `C0003` | no | no | missing 107/107 artifacts |
| `C0004` | no | no | missing 3/3 artifacts |
| `C0005` | no | no | missing 5/5 artifacts |
| `C0006` | yes | no | missing 7/7 artifacts |
| `C0007` | no | no | missing 13/13 artifacts |
| `C0008` | no | no | missing 3/3 artifacts |
| `C0009` | no | no | missing 14/14 artifacts |

## C0001: On AVE, under a strictly equal ViT token budget, Audio-anchored sampling (with a lightweight temporal head) improves segment-level accuracy vs Uniform sampling and Random anchors.

### Missing
- `runs/AVE_P0/*/metrics.json` (missing)
- `runs/E0001_ave_p0_real_20260131-040050/p0_energy_seedfix/metrics.json` (missing)
- `runs/REAL_AVE_20260131-211548/p0_train180_val165_energy/metrics.json` (missing)
- `runs/REAL_AVE_20260131-211548/p0_train180_test113_energy/metrics.json` (missing)
- `runs/REAL_AVE_20260131-211548/p0_train180_val165_energy_160_224_352_k2/metrics.json` (missing)
- `runs/REAL_AVE_20260131-211548/p0_train180_val165_k1_energy_rerun/metrics.json` (missing)
- `runs/REAL_AVE_20260131-211548/p0_train180_test113_k1_energy_rerun/metrics.json` (missing)
- `runs/REAL_AVE_20260131-211548/p0_train180_val165_energy_160_224_352_k2_shift1_std1p0_temporal_s0-9_rerun/metrics.json` (missing)
- ... (+7 more)

### Present


## C0002: Audio-based anchors achieve higher Recall@K (and Recall@K,Δ) than random anchors on AVE.

### Missing
- `runs/anchors/*/anchors_metrics.json` (missing)
- `runs/E0002_anchors_real_20260131-045200/anchor_eval/anchors_metrics.json` (missing)
- `runs/REAL_AVE_20260131-211548/anchors_val165_energy/anchors_metrics.json` (missing)
- `runs/REAL_AVE_20260131-211548/anchors_test113_energy/anchors_metrics.json` (missing)
- `runs/REAL_AVE_OFFICIAL_20260201-124535/E0002_anchors_official_val/anchor_eval/anchors_metrics.json` (missing)
- `runs/REAL_AVE_OFFICIAL_20260201-124535/E0002_anchors_official_test/anchor_eval/anchors_metrics.json` (missing)

### Present


## C0003: On official AVE test402, sampling-only anchored_top2 improves >= +2.0% with p<0.05 (SEEDS=0..9).

### Missing
- `runs/E0402_full_test402_av_clipdiff_flow_mlp_stride_20260206-141020/metrics.json` (missing)
- `runs/E0402_full_test402_av_clipdiff_flow_mlp_stride_20260206-141020/diagnose.json` (missing)
- `runs/E0402_full_test402_av_clipdiff_flow_mlp_stride_alt_top1med_thr0p5_20260206-152012/metrics.json` (missing)
- `runs/E0402_full_test402_av_clipdiff_flow_mlp_stride_alt_top1med_thr0p5_20260206-152012/diagnose.json` (missing)
- `runs/E0503_gate_sweep_dense_stride_full_20260207-153210/gate_sweep.json` (missing)
- `runs/E0503_gate_sweep_dense_stride_full_20260207-153210/best_gate.json` (missing)
- `runs/E0405_quick_test402_av_clipdiff_flow_mlp_stride_top1medn_thr0p6_shift0_20260206-161028/metrics.json` (missing)
- `runs/E0406_full_test402_av_clipdiff_flow_mlp_stride_top1medn_thr0p6_shift0_20260206-161349/metrics.json` (missing)
- ... (+99 more)

### Present


## C0004: On official AVE test402, audio_concat_anchored_top2 improves over audio_concat_uniform (>= +1.0%, p<0.05; SEEDS=0..9).

### Missing
- `runs/E0013_ave_fusion_confirm_official_test_20260203-150226/metrics.json` (missing)
- `runs/E0020_ave_fusion_confirm_official_test_energy_v2_20260203-190804/metrics.json` (missing)
- `runs/E0410_fusion_confirm_energy_stride_max_top1med_thr0p5_20260206-180945/metrics.json` (missing)

### Present


## C0005: On EPIC-SOUNDS video-level multi-label recognition, audio-anchored selection improves mAP on val (SEEDS>=3).

### Missing
- `runs/E0100_*/metrics.json` (missing)
- `runs/smoke_20260206-184253/epic_sounds_video_cls_synth/metrics.json` (missing)
- `runs/E0413_epic_video_cls_local_audio_anchored_full_ms120_20260207-171637/metrics.json` (missing)
- `runs/E0413_epic_video_cls_local_uniform_full_ms120_20260207-172545/metrics.json` (missing)
- `runs/E0413_epic_video_cls_local_random_full_ms120_20260207-173208/metrics.json` (missing)

### Present


## C0006: Oracle anchors provide an upper bound that shows stable Acc–Tok Pareto improvements across a pre-registered budget grid on AVE.

### Missing
- `runs/E0010_*/oracle_ceiling.json` (missing)
- `runs/E0330_*/pareto_report.json` (missing)
- `runs/E0330_*/pareto.png` (missing)
- `runs/E0010_oracle_ceiling_official_val_20260203-144421/oracle_ceiling.json` (missing)
- `runs/E0010_oracle_ceiling_official_test_20260203-143455/oracle_ceiling.json` (missing)
- `runs/E0330_full_av_clipdiff_mlp_auto_20260205-184559/pareto_report.json` (missing)
- `runs/E0330_full_av_clipdiff_mlp_auto_20260205-184559/pareto.png` (missing)

### Present


## C0007: Predicted anchors stay within a small gap of Oracle anchors on the same budget grid (deployable stage-1).

### Missing
- `runs/E0201_*/oracle_vs_predicted.json` (missing)
- `runs/E0330_*/pareto_report.json` (missing)
- `runs/E0403_oracle_vs_predicted_av_clipdiff_flow_mlp_stride_20260206-141804/oracle_vs_predicted.json` (missing)
- `runs/E0403_oracle_vs_predicted_av_clipdiff_flow_mlp_stride_alt_top1med_thr0p5_20260206-152658/oracle_vs_predicted.json` (missing)
- `runs/E0407_oracle_vs_predicted_av_clipdiff_flow_mlp_stride_top1med_thr0p5_s0-9_20260206-161749/oracle_vs_predicted.json` (missing)
- `runs/E0409_pareto_grid_av_clipdiff_flow_mlp_stride_top1med_thr0p5_s0-9_20260206-163941/pareto_report.json` (missing)
- `runs/E0504_oracle_pred_gap_grid_dense_stride_full_20260207-155721/pareto_report.json` (missing)
- `runs/E0504_oracle_pred_gap_grid_dense_stride_full_20260207-155721/pareto.png` (missing)
- ... (+5 more)

### Present


## C0008: Evidence Alignment (Cov@τ) correlates with downstream accuracy and diagnoses failure cases.

### Missing
- `runs/E0202_*/evidence_alignment.json` (missing)
- `runs/E0202_evidence_alignment_energy_v2_test_20260203-194355/evidence_alignment.json` (missing)
- `runs/E0411_evidence_alignment_av_clipdiff_flow_mlp_stride_top1med_thr0p5_20260206-182007/evidence_alignment.json` (missing)

### Present


## C0009: Listen-then-Look degrades gracefully under shift/noise/silence, and α provides a computable accuracy lower bound.

### Missing
- `runs/E0203_*/degradation_suite.json` (missing)
- `runs/E0203_*/degradation_plots/*` (missing)
- `runs/E0404_*/degradation_suite.json` (missing)
- `runs/E0404_degradation_av_clipdiff_flow_mlp_stride_20260206-142817/degradation_suite.json` (missing)
- `runs/E0203_degradation_av_clipdiff_mlp_20260204-215831/degradation_suite.json` (missing)
- `runs/E0203_full_energy_20260203-210331/degradation_suite.json` (missing)
- `runs/E0203_full_energy_stride_max_20260203-210414/degradation_suite.json` (missing)
- `runs/E0203_full_av_fused_20260203-210458/degradation_suite.json` (missing)
- ... (+6 more)

### Present


