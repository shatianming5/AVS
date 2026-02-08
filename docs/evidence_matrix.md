# Evidence Matrix

- Generated: `2026-02-09 05:02:33`
- Plan: `docs/plan.md`

| Conclusion | Checked in plan | Local artifacts present? | Notes |
| --- | --- | --- | --- |
| `C0001` | yes | no | missing 15/15 artifacts |
| `C0002` | yes | no | missing 6/6 artifacts |
| `C0003` | no | no | missing 107/107 artifacts |
| `C0004` | no | no | missing 3/3 artifacts |
| `C0005` | yes | yes |  |
| `C0006` | yes | no | missing 7/7 artifacts |
| `C0007` | no | no | missing 13/13 artifacts |
| `C0008` | no | no | missing 3/3 artifacts |
| `C0009` | no | no | missing 14/14 artifacts |

## C0001: On AVE, under a strictly equal ViT token budget, Audio-anchored sampling (with a lightweight temporal head) improves segment-level accuracy vs Uniform sampling and Random anchors.

### Missing
- `runs/AVE_P0/*/metrics.json` (missing; glob, 0 matches)
- `runs/E0001_ave_p0_real_20260131-040050/p0_energy_seedfix/metrics.json` (missing)
- `runs/REAL_AVE_20260131-211548/p0_train180_val165_energy/metrics.json` (missing)
- `runs/REAL_AVE_20260131-211548/p0_train180_test113_energy/metrics.json` (missing)
- `runs/REAL_AVE_20260131-211548/p0_train180_val165_energy_160_224_352_k2/metrics.json` (missing)
- `runs/REAL_AVE_20260131-211548/p0_train180_val165_k1_energy_rerun/metrics.json` (missing)
- `runs/REAL_AVE_20260131-211548/p0_train180_test113_k1_energy_rerun/metrics.json` (missing)
- `runs/REAL_AVE_20260131-211548/p0_train180_val165_energy_160_224_352_k2_shift1_std1p0_temporal_s0-9_rerun/metrics.json` (missing)
- `runs/REAL_AVE_20260131-211548/p0_train180_test113_energy_160_224_352_k2_shift1_std1p0_temporal_s0-9_rerun/metrics.json` (missing)
- `runs/REAL_AVE_20260131-211548/p0_train195_val165_energy_160_224_352_k2_shift1_std1p0_temporal_s0-4/metrics.json` (missing)
- `runs/REAL_AVE_20260131-211548/p0_train195_test113_energy_160_224_352_k2_shift1_std1p0_temporal_s0-4/metrics.json` (missing)
- `runs/REAL_AVE_OFFICIAL_20260201-124535/p0_train3339_val402_energy_160_224_352_k2_shift1_std1.0_temporal_conv/metrics.json` (missing)
- `runs/REAL_AVE_OFFICIAL_20260201-124535/p0_train3339_test402_energy_160_224_352_k2_shift1_std1.0_temporal_conv/metrics.json` (missing)
- `runs/REAL_AVE_OFFICIAL_RERUN_20260201-152134/p0_train3339_val402_energy_160_224_352_k2_shift1_std1.0_temporal_conv_v2/metrics.json` (missing)
- `runs/REAL_AVE_OFFICIAL_RERUN_20260201-152134/p0_train3339_test402_energy_160_224_352_k2_shift1_std1.0_temporal_conv_v2/metrics.json` (missing)

### Present

## C0002: Audio-based anchors achieve higher Recall@K (and Recall@K,Δ) than random anchors on AVE.

### Missing
- `runs/anchors/*/anchors_metrics.json` (missing; glob, 0 matches)
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
- `runs/E0018_ave_p0_sweep_official_val_energy_v2_20260203-185629/sweep_summary.json` (missing)
- `runs/E0018_ave_p0_sweep_official_val_energy_v2_20260203-185629/best_config.json` (missing)
- `runs/E0019_ave_p0_best_to_test_official_energy_v2_20260203-190500/metrics.json` (missing)
- `runs/E0201_full_energy_gini0p35_test402_20260204-041950/oracle_vs_predicted.json` (missing)
- `runs/E0205_full_audio_basic_tcn_val402_20260204-063935/sweep_summary.json` (missing)
- `runs/E0206_ave_p0_best_to_test_official_audio_basic_tcn_20260204-070803/metrics.json` (missing)
- `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_mlp_20260204-075914/sweep_summary.json` (missing)
- `runs/E0208_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-080419/metrics.json` (missing)
- `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_std_v1_20260204-083739/sweep_summary.json` (missing)
- `runs/E0208_ave_p0_best_to_test_official_av_clipdiff_mlp_ltl_std_v1_20260204-084147/metrics.json` (missing)
- `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_std_v2_20260204-085103/sweep_summary.json` (missing)
- `runs/E0208_ave_p0_best_to_test_official_av_clipdiff_mlp_ltl_std_v2_20260204-085632/metrics.json` (missing)
- `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_gini_v2_20260204-090323/sweep_summary.json` (missing)
- `runs/E0208_ave_p0_best_to_test_official_av_clipdiff_mlp_ltl_gini_v2_20260204-090710/metrics.json` (missing)
- `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_gap_v1_20260204-090353/sweep_summary.json` (missing)
- `runs/E0208_ave_p0_best_to_test_official_av_clipdiff_mlp_ltl_gap_v1_20260204-090741/metrics.json` (missing)
- `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_extreme_v1_20260204-091309/sweep_summary.json` (missing)
- `runs/E0208_ave_p0_best_to_test_official_av_clipdiff_mlp_ltl_extreme_v1_20260204-091529/metrics.json` (missing)
- `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_mlp_cls_ltl_std_v1_20260204-092157/sweep_summary.json` (missing)
- `runs/E0208_ave_p0_best_to_test_official_av_clipdiff_mlp_cls_ltl_std_v1_20260204-092530/metrics.json` (missing)
- `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_mlp_20260204-102403/sweep_summary.json` (missing)
- `runs/E0208_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-103001/metrics.json` (missing)
- `runs/E0218_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_smooth_v1_20260204-132824/sweep_summary.json` (missing)
- `runs/E0219_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-133929/metrics.json` (missing)
- `runs/E0223_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_v1_20260204-135150/sweep_summary.json` (missing)
- `runs/E0224_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-135547/metrics.json` (missing)
- `runs/E0267_ave_p0_best_to_test_official_av_clipdiff_mlp_top1med_epochs10_wd0p01_20260204-194212/metrics.json` (missing)
- `runs/E0268_ave_p0_best_to_test_official_av_clipdiff_mlp_top1med_epochs10_wd0p0_20260204-194359/metrics.json` (missing)
- `runs/E0291_ave_p0_best_to_test_official_av_clipdiff_mlp_top1med_e10_s0-2_20260205-001904/metrics.json` (missing)
- `runs/E0292_ave_p0_sweep_official_val_av_fused_clipdiff_prod_ltl_top1med_v1_20260205-012010/sweep_summary.json` (missing)
- `runs/E0293_ave_p0_best_to_test_official_av_fused_clipdiff_prod_20260205-012350/metrics.json` (missing)
- `runs/E0294_ave_p0_sweep_official_val_moe_energy_clipdiff_ltl_top1med_v1_20260205-012010/sweep_summary.json` (missing)
- `runs/E0295_ave_p0_best_to_test_official_moe_energy_clipdiff_20260205-012339/metrics.json` (missing)
- `runs/E0226_ave_p0_stage2_variants_official_val_av_clipdiff_mlp_20260204-142732/variants_summary.json` (missing)
- `runs/E0227_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-142936/metrics.json` (missing)
- `runs/E0228_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_extreme_v1_20260204-143855/sweep_summary.json` (missing)
- `runs/E0229_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-144335/metrics.json` (missing)
- `runs/E0230_ave_p0_sweep_official_val_av_clipdiff_mlp_r160_ltl_top1med_v1_20260204-144941/sweep_summary.json` (missing)
- `runs/E0231_ave_p0_best_to_test_official_av_clipdiff_mlp_r160_20260204-145349/metrics.json` (missing)
- `runs/E0233_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_maxhigh1_v1_20260204-151909/sweep_summary.json` (missing)
- `runs/E0234_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-152349/metrics.json` (missing)
- `runs/E0235_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_k1_v1_20260204-153020/sweep_summary.json` (missing)
- `runs/E0236_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-153411/metrics.json` (missing)
- `runs/E0237_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_adaptivegap_v1_20260204-160956/sweep_summary.json` (missing)
- `runs/E0238_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-161232/metrics.json` (missing)
- `runs/E0239_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_highconf_v1_20260204-161417/sweep_summary.json` (missing)
- `runs/E0240_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-161835/metrics.json` (missing)
- `runs/E0241_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_scorealloc_v1_20260204-162247/sweep_summary.json` (missing)
- `runs/E0242_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-162356/metrics.json` (missing)
- `runs/E0245_ave_p0_sweep_official_val_av_clipdiff_mlp_autoshift_ltl_top1med_autoshift_v1_20260204-163436/sweep_summary.json` (missing)
- `runs/E0246_ave_p0_best_to_test_official_av_clipdiff_mlp_autoshift_20260204-163703/metrics.json` (missing)
- `runs/E0243_ave_p0_sweep_official_val_av_clip_mlp_cls_target_ltl_top1med_v1_20260204-162702/sweep_summary.json` (missing)
- `runs/E0247_ave_p0_sweep_official_val_av_clipdiff_mlp_cls_target_ltl_top1med_v1_20260204-164046/sweep_summary.json` (missing)
- `runs/E0248_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_nmsstrong_v1_20260204-170658/sweep_summary.json` (missing)
- `runs/E0249_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-171109/metrics.json` (missing)
- `runs/E0250_ave_p0_sweep_official_val_av_clipdiff_fbank_mlp_ltl_top1med_v1_20260204-172631/sweep_summary.json` (missing)
- `runs/E0251_ave_p0_best_to_test_official_av_clipdiff_fbank_mlp_20260204-173034/metrics.json` (missing)
- `runs/E0252_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_dropfar_v1_20260204-173949/sweep_summary.json` (missing)
- `runs/E0253_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-174232/metrics.json` (missing)
- `runs/E0021_ave_p0_sweep_official_val_energy_v3_20260203-194306/sweep_summary.json` (missing)
- `runs/E0022_ave_p0_best_to_test_official_energy_v3_20260203-195707/metrics.json` (missing)
- `runs/E0022b_energyv3_shift1_std0p6_test402_20260203-195743/metrics.json` (missing)
- `runs/E0022c_energyv3_shift0_std1p0_test402_20260203-195754/metrics.json` (missing)
- `runs/E0024_energy_ref_test402_epochs20_20260203-194640/metrics.json` (missing)
- `runs/E0025_energy_k5_maxHigh1_scoreAlloc_test402_20260203-200327/metrics.json` (missing)
- `runs/E0015_ave_p0_best_to_test_official_ast_20260203-172848/metrics.json` (missing)
- `runs/E0016_ave_p0_diagnose_20260203-173704/diagnose.json` (missing)
- `runs/E0201_full_energy_autoshift_clipdiff_test402_20260204-032327/oracle_vs_predicted.json` (missing)
- `runs/E0201_full_energy_autoshift_clipdiff_pos_test402_20260204-040122/oracle_vs_predicted.json` (missing)
- `runs/E0207_ave_p0_sweep_official_val_av_clip_mlp_cls_20260204-095132/sweep_summary.json` (missing)
- `runs/E0208_ave_p0_best_to_test_official_av_clip_mlp_cls_20260204-095509/metrics.json` (missing)
- `runs/E0207_ave_p0_sweep_official_val_av_clip_mlp_cls_target_20260204-095814/sweep_summary.json` (missing)
- `runs/E0208_ave_p0_best_to_test_official_av_clip_mlp_cls_target_20260204-100202/metrics.json` (missing)
- `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_tcn_20260204-100517/sweep_summary.json` (missing)
- `runs/E0208_ave_p0_best_to_test_official_av_clipdiff_tcn_20260204-100855/metrics.json` (missing)
- `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_20260204-101416/sweep_summary.json` (missing)
- `runs/E0284_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_keepadj_basealloc_v1_20260204-224414/sweep_summary.json` (missing)
- `runs/E0285_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-224708/metrics.json` (missing)
- `runs/E0303_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_band_v1_20260205-033915/sweep_summary.json` (missing)
- `runs/E0304_ave_p0_best_to_test_official_av_clipdiff_mlp_20260205-035830/metrics.json` (missing)
- `runs/E0305_ave_p0_best_to_test_official_av_clipdiff_mlp_banddiag_thr0p6_shift1_20260205-040513/metrics.json` (missing)
- `runs/E0320_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_band_low112_v1_20260205-142258/sweep_summary.json` (missing)
- `runs/E0321_quick_test402_av_clipdiff_mlp_band_low112_20260205-142726/metrics.json` (missing)
- `runs/E0322_ave_p0_sweep_official_val_asr_vad_ltl_top1med_norm_v1_20260205-142328/sweep_summary.json` (missing)
- `runs/E0324_ave_p0_sweep_official_val_ast_nonspeech_max_ltl_top1med_norm_v1_20260205-144057/sweep_summary.json` (missing)
- `runs/E0307_ave_p0_sweep_official_val_av_ast_clipdiff_mil_mlp_ltl_top1med_v1_20260205-045530/sweep_summary.json` (missing)
- `runs/E0309_ave_p0_sweep_official_val_av_ast_clipdiff_mil_mlp_ltl_top1med_norm_v1_20260205-051944/sweep_summary.json` (missing)
- `runs/E0310_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_norm_v1_20260205-052411/sweep_summary.json` (missing)
- `runs/E0311_ave_p0_sweep_official_val_av_ast_clipalign_nce_ltl_top1med_norm_v1_20260205-055421/sweep_summary.json` (missing)
- `runs/E0318_ave_p0_sweep_official_val_av_ast_clipalign_bce_ltl_top1med_norm_v1_20260205-133800/sweep_summary.json` (missing)
- `runs/E0312_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_anchor2veto_v1_20260205-115927/sweep_summary.json` (missing)
- `runs/E0313_quick_test402_av_clipdiff_mlp_a2veto_lr0p65_20260205-120329/metrics.json` (missing)
- `runs/E0313_quick_test402_av_clipdiff_mlp_a2veto_top2med0p15_20260205-120449/metrics.json` (missing)
- `runs/E0314_ave_p0_sweep_official_val_av_clipdiff_visgain_mlp_ltl_top1med_norm_v1_20260205-123935/sweep_summary.json` (missing)
- `runs/E0316_ave_p0_sweep_official_val_av_clipdiff_lossgain_mlp_ltl_top1med_norm_v1_20260205-125414/sweep_summary.json` (missing)
- `runs/E0326_ave_p0_sweep_official_val_av_clipdiff_speech_mlp_ltl_top1med_norm_v1_20260205-154338/sweep_summary.json` (missing)
- `runs/E0328_ave_p0_sweep_official_val_energy_nonspeech_ast_ltl_top1med_norm_v1_20260205-155950/sweep_summary.json` (missing)
- `runs/E0329_ave_p0_sweep_official_val_av_clipdiff_accflip_mlp_ltl_top1med_norm_v1_20260205-165330/sweep_summary.json` (missing)
- `runs/DIAG_E0224_test402_20260205-122658/diagnose.json` (missing)

### Present

## C0004: On official AVE test402, audio_concat_anchored_top2 improves over audio_concat_uniform (>= +1.0%, p<0.05; SEEDS=0..9).

### Missing
- `runs/E0013_ave_fusion_confirm_official_test_20260203-150226/metrics.json` (missing)
- `runs/E0020_ave_fusion_confirm_official_test_energy_v2_20260203-190804/metrics.json` (missing)
- `runs/E0410_fusion_confirm_energy_stride_max_top1med_thr0p5_20260206-180945/metrics.json` (missing)

### Present

## C0005: On EPIC-SOUNDS video-level multi-label recognition, audio-anchored selection improves mAP on val (SEEDS>=3).

- All referenced artifacts exist locally.

## C0006: Oracle anchors provide an upper bound that shows stable Acc–Tok Pareto improvements across a pre-registered budget grid on AVE.

### Missing
- `runs/E0010_*/oracle_ceiling.json` (missing; glob, 0 matches)
- `runs/E0330_*/pareto_report.json` (missing; glob, 0 matches)
- `runs/E0330_*/pareto.png` (missing; glob, 0 matches)
- `runs/E0010_oracle_ceiling_official_val_20260203-144421/oracle_ceiling.json` (missing)
- `runs/E0010_oracle_ceiling_official_test_20260203-143455/oracle_ceiling.json` (missing)
- `runs/E0330_full_av_clipdiff_mlp_auto_20260205-184559/pareto_report.json` (missing)
- `runs/E0330_full_av_clipdiff_mlp_auto_20260205-184559/pareto.png` (missing)

### Present

## C0007: Predicted anchors stay within a small gap of Oracle anchors on the same budget grid (deployable stage-1).

### Missing
- `runs/E0201_*/oracle_vs_predicted.json` (missing; glob, 0 matches)
- `runs/E0330_*/pareto_report.json` (missing; glob, 0 matches)
- `runs/E0403_oracle_vs_predicted_av_clipdiff_flow_mlp_stride_20260206-141804/oracle_vs_predicted.json` (missing)
- `runs/E0403_oracle_vs_predicted_av_clipdiff_flow_mlp_stride_alt_top1med_thr0p5_20260206-152658/oracle_vs_predicted.json` (missing)
- `runs/E0407_oracle_vs_predicted_av_clipdiff_flow_mlp_stride_top1med_thr0p5_s0-9_20260206-161749/oracle_vs_predicted.json` (missing)
- `runs/E0409_pareto_grid_av_clipdiff_flow_mlp_stride_top1med_thr0p5_s0-9_20260206-163941/pareto_report.json` (missing)
- `runs/E0504_oracle_pred_gap_grid_dense_stride_full_20260207-155721/pareto_report.json` (missing)
- `runs/E0504_oracle_pred_gap_grid_dense_stride_full_20260207-155721/pareto.png` (missing)
- `runs/E0201_full_energy_20260203-210017/oracle_vs_predicted.json` (missing)
- `runs/E0201_full_energy_stride_max_20260203-210017/oracle_vs_predicted.json` (missing)
- `runs/E0201_oracle_vs_predicted_av_fused_scale3p5_full_20260203-221906/oracle_vs_predicted.json` (missing)
- `runs/E0201_oracle_vs_predicted_av_clipdiff_mlp_20260204-213240/oracle_vs_predicted.json` (missing)
- `runs/E0330_full_av_clipdiff_mlp_auto_20260205-184559/pareto_report.json` (missing)

### Present

## C0008: Evidence Alignment (Cov@τ) correlates with downstream accuracy and diagnoses failure cases.

### Missing
- `runs/E0202_*/evidence_alignment.json` (missing; glob, 0 matches)
- `runs/E0202_evidence_alignment_energy_v2_test_20260203-194355/evidence_alignment.json` (missing)
- `runs/E0411_evidence_alignment_av_clipdiff_flow_mlp_stride_top1med_thr0p5_20260206-182007/evidence_alignment.json` (missing)

### Present

## C0009: Listen-then-Look degrades gracefully under shift/noise/silence, and α provides a computable accuracy lower bound.

### Missing
- `runs/E0203_*/degradation_suite.json` (missing; glob, 0 matches)
- `runs/E0203_*/degradation_plots/*` (missing; glob, 0 matches)
- `runs/E0404_*/degradation_suite.json` (missing; glob, 0 matches)
- `runs/E0404_degradation_av_clipdiff_flow_mlp_stride_20260206-142817/degradation_suite.json` (missing)
- `runs/E0203_degradation_av_clipdiff_mlp_20260204-215831/degradation_suite.json` (missing)
- `runs/E0203_full_energy_20260203-210331/degradation_suite.json` (missing)
- `runs/E0203_full_energy_stride_max_20260203-210414/degradation_suite.json` (missing)
- `runs/E0203_full_av_fused_20260203-210458/degradation_suite.json` (missing)
- `runs/E0331_smoke_av_clipdiff_mlp_20260205-194038/degradation_accuracy.json` (missing)
- `runs/E0331_full_av_clipdiff_mlp_20260205-194925/degradation_accuracy.json` (missing)
- `runs/E0331_full_av_clipdiff_mlp_20260205-194925/degradation_plots/*.png` (missing; glob, 0 matches)
- `runs/E0412_degradation_accuracy_av_clipdiff_flow_mlp_stride_top1med_thr0p5_s0-9_20260206-182443/degradation_accuracy.json` (missing)
- `runs/E0505_degradation_accuracy_dense_stride_full_20260207-161213/degradation_accuracy.json` (missing)
- `runs/E0505_degradation_accuracy_dense_stride_full_20260207-161213/degradation_plots/*.png` (missing; glob, 0 matches)

### Present

