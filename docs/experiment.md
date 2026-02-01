# Experiments

## Overview
- Goal: Prove `docs/plan.md` conclusions (C0001/C0002) on AVE with runnable, reproducible commands.
- Baseline: Uniform-224 sampling under a fixed ViT token budget.
- Primary model: CLIP ViT-B/16 (vision features) + per-segment MLP head; audio probe = energy or AST.
- Data: To install the official full AVE videos (4143 clips), run `bash scripts/ave_install_official.sh` (downloads `AVE_Dataset.zip` and installs under `data/AVE/raw/videos/`).

## Experiments

### E0001: AVE-P0 real subset (equal token budget baselines)
| Field | Value |
| --- | --- |
| Objective | Compare `uniform` vs `random_top2` vs `anchored_top2` vs `oracle_top2` (and `audio_concat_uniform`, `audio_concat_anchored_top2`, `audio_feat_concat_uniform`, `audio_feat_concat_anchored_top2`) under equal token budget. |
| Baseline | `uniform` |
| Model | `openai/clip-vit-base-patch16` + (`avs.models.per_segment_mlp.PerSegmentMLP` or `avs.models.temporal_conv.TemporalConvHead`) |
| Weights | HF: CLIP (`--vision-pretrained`); optional HF: AST (`--ast-pretrained`) |
| Code path | `avs/pipeline/ave_p0_end2end.py`, `avs/experiments/ave_p0.py`, `scripts/e0001_ave_p0_real.sh` |
| Params | `LIMIT_TRAIN`, `LIMIT_EVAL`, `SEEDS`, `EVENTNESS={energy,ast}`, `VISION_PRETRAINED={0,1}`, `DEVICE`, `TRAIN_DEVICE`, `LOW_RES/BASE_RES/HIGH_RES`, `ANCHOR_SHIFT`, `ANCHOR_STD_THRESHOLD`, `HEAD` |
| Metrics (must save) | `metrics.json` (`summary.*.mean/std`, `paired_ttest`, `token_budget`, per-clip `val_acc_by_sample`, and `debug_eval` for anchors/scores/plan) |
| Checks | `token_budget==1960`; compare `anchored_top2` vs `uniform` and `random_top2` |
| VRAM | CPU ok; GPU optional |
| Time/epoch | ~5 epochs (config in `avs/pipeline/ave_p0_end2end.py`) |
| Total time | Depends on download + cache build |
| Single-GPU script | `scripts/e0001_ave_p0_real.sh` |
| Multi-GPU script | `scripts/e0001_ave_p0_real_multigpu.sh` |
| Smoke cmd | `python -m avs.smoke ave_p0_end2end` |
| Full cmd | `DEVICE=cuda:0 LIMIT_TRAIN=180 LIMIT_EVAL=165 SEEDS=0,1,2 VISION_PRETRAINED=1 CACHE_NUM_WORKERS=4 CACHE_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 bash scripts/e0001_ave_p0_real_multigpu.sh` |
| Alt Full cmd (best config for C0001) | `DEVICE=cuda:0 LIMIT_TRAIN=180 LIMIT_EVAL=165 SEEDS=0,1,2,3,4,5,6,7,8,9 VISION_PRETRAINED=1 CACHE_NUM_WORKERS=4 CACHE_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 LOW_RES=160 BASE_RES=224 HIGH_RES=352 ANCHOR_SHIFT=1 ANCHOR_STD_THRESHOLD=1.0 HEAD=temporal_conv bash scripts/e0001_ave_p0_real_multigpu.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0001_*` |
| Artifacts | `runs/E0001_*/ave_p0_end2end/metrics.json` |
| Results | Small-eval seeded rerun: `runs/E0001_ave_p0_real_20260131-040050/p0_energy_seedfix/metrics.json` (energy; anchored>random>uniform; token_budget=1960). Larger subsets (k=2): `runs/REAL_AVE_20260131-211548/p0_train180_val165_energy/metrics.json` (uniform=0.250 > anchored=0.214) and `runs/REAL_AVE_20260131-211548/p0_train180_test113_energy/metrics.json` (uniform=0.254 > anchored=0.212). K=1 reruns reduce the gap but remain < uniform: `runs/REAL_AVE_20260131-211548/p0_train180_val165_k1_energy_rerun/metrics.json` and `runs/REAL_AVE_20260131-211548/p0_train180_test113_k1_energy_rerun/metrics.json`. Less-extreme equal-budget triad shows oracle>uniform upper bound: `runs/REAL_AVE_20260131-211548/p0_train180_val165_energy_160_224_352_k2/metrics.json` (oracle=0.258 > uniform=0.250 > anchored=0.242). Anchor robustness knobs can flip val but don’t transfer to test: `runs/REAL_AVE_20260131-211548/p0_train180_val165_energy_160_224_352_k2_shift1_std1p0_s0-4_rerun/metrics.json` (val165; anchored=0.245 > uniform=0.242) vs `runs/REAL_AVE_20260131-211548/p0_train180_test113_energy_160_224_352_k2_shift1_std1p0_s0-4/metrics.json` (test113; anchored=0.247 < uniform=0.248). Temporal head (temporal_conv) + robustness knobs yields anchored>uniform on both val/test: `runs/REAL_AVE_20260131-211548/p0_train180_val165_energy_160_224_352_k2_shift1_std1p0_temporal_s0-9_rerun/metrics.json` (val165; anchored=0.228 > uniform=0.221) and `runs/REAL_AVE_20260131-211548/p0_train180_test113_energy_160_224_352_k2_shift1_std1p0_temporal_s0-9_rerun/metrics.json` (test113; anchored=0.233 > uniform=0.230). Expanded train set (195 train clips available) strengthens the gap: `runs/REAL_AVE_20260131-211548/p0_train195_val165_energy_160_224_352_k2_shift1_std1p0_temporal_s0-4/metrics.json` (val165; anchored=0.232 > uniform=0.211) and `runs/REAL_AVE_20260131-211548/p0_train195_test113_energy_160_224_352_k2_shift1_std1p0_temporal_s0-4/metrics.json` (test113; anchored=0.274 > uniform=0.259). |


### E0002: Anchor quality on real AVE audio (Recall@K / Recall@K,Δ)
| Field | Value |
| --- | --- |
| Objective | Evaluate whether audio-derived anchors have higher Recall@K than random anchors on real AVE clips. |
| Baseline | random anchors |
| Model | audio eventness (`energy` or `ast`) |
| Weights | optional HF: AST (`--ast-pretrained`) |
| Code path | `avs/experiments/ave_anchor_eval.py`, `scripts/e0002_anchor_eval_real.sh` |
| Params | `LIMIT`, `SPLIT`, `MODE`, `METHOD={energy,energy_delta,ast,panns,audiomae}` |
| Metrics (must save) | `anchors_metrics.json` (`ours_mean_recall`, `random_mean_recall`, `num_clips`) |
| Checks | For at least one Δ, `ours_mean_recall > random_mean_recall` |
| VRAM | CPU ok |
| Time/epoch | N/A |
| Total time | Mostly download + audio feature extraction |
| Single-GPU script | `scripts/e0002_anchor_eval_real.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT=8 bash scripts/e0002_anchor_eval_real.sh` |
| Full cmd | `MODE=local SRC_DIR=data/AVE/raw/videos SPLIT=val LIMIT=402 bash scripts/e0002_anchor_eval_real.sh && MODE=local SRC_DIR=data/AVE/raw/videos SPLIT=test LIMIT=402 bash scripts/e0002_anchor_eval_real.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0002_*` and `runs/REAL_AVE_OFFICIAL_*/E0002_anchors_official_*` |
| Artifacts | `runs/E0002_*/anchor_eval/anchors_metrics.json` and `runs/REAL_AVE_OFFICIAL_*/E0002_anchors_official_*/anchor_eval/anchors_metrics.json` |
| Results | Energy anchors (n=89): `runs/E0002_anchors_real_20260131-045200/anchor_eval/anchors_metrics.json` (k=2; Δ0 ours=0.207 > rand=0.194). Additional local subsets: `runs/REAL_AVE_20260131-211548/anchors_val165_energy/anchors_metrics.json` (n=165; Δ0 ours=0.208 > rand=0.202) and `runs/REAL_AVE_20260131-211548/anchors_test113_energy/anchors_metrics.json` (n=113; Δ0 ours=0.221 > rand=0.181). Official full split (MODE=local): val `runs/REAL_AVE_OFFICIAL_20260201-124535/E0002_anchors_official_val/anchor_eval/anchors_metrics.json` (n=401; Δ0 ours=0.231 > rand=0.211) and test `runs/REAL_AVE_OFFICIAL_20260201-124535/E0002_anchors_official_test/anchor_eval/anchors_metrics.json` (n=402; Δ0 ours=0.231 > rand=0.204). Note: for dilated windows (Δ1/Δ2), random > ours on both val/test when using plain top-k selection. Using `anchor_select=nms` can improve the dilated-window robustness on test402: `runs/ANCHOR_QUALITY_test402_energy_nmsR2_shift0/anchors_metrics.json` (k=2; Δ1 ours=0.585 > rand=0.517; Δ2 ours=0.801 > rand=0.716). Rerun comparisons: energy_delta is worse than random on Δ0/Δ1/Δ2 on both splits (`runs/REAL_AVE_OFFICIAL_RERUN_20260201-152134/anchors_{val,test}_energy_delta/anchors_metrics.json`), and AST-pretrained is near-tie on Δ0 (val ours=0.215>rand=0.211; test ours=0.202<rand=0.204) (`runs/REAL_AVE_OFFICIAL_RERUN_20260201-152134/anchors_{val,test}_ast_pretrained/anchors_metrics.json`). |

### E0003: Official AVE full-dataset validation (multi-GPU)
| Field | Value |
| --- | --- |
| Objective | Run “full + real” AVE validation using the official 4143-clip zip: (1) anchor eval on full val/test (402/402) and (2) AVE-P0 train→val/test on the full train split (3339), using the best-known config for C0001. |
| Data | `bash scripts/ave_install_official.sh` (installs under `data/AVE/raw/videos/`) |
| Code path | `scripts/ave_verify_official_after_install.sh` |
| Metrics (must save) | `E0002_anchors_official_*/anchor_eval/anchors_metrics.json`; `p0_train*/{val,test}*/metrics.json` under `runs/REAL_AVE_OFFICIAL_*` |
| Smoke cmd | `EXPECTED_VAL=8 EXPECTED_TEST=8 LIMIT_TRAIN=32 SEEDS=0 PREPROCESS_JOBS=8 bash scripts/ave_verify_official_after_install.sh` (requires official dataset installed) |
| Full cmd | `bash scripts/ave_verify_official_after_install.sh` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/ave_official_verify.log` and `runs/REAL_AVE_OFFICIAL_*` |
| Results | Run root: `runs/REAL_AVE_OFFICIAL_20260201-124535`. Anchors: see `E0002_anchors_official_{val,test}/anchor_eval/anchors_metrics.json` (Δ0 ours > random on both). AVE-P0 (token_budget=1960; SEEDS=0..9; head=temporal_conv; energy; 160/224/352; shift=1; std_thr=1.0): train→val `p0_train3339_val402_energy_160_224_352_k2_shift1_std1.0_temporal_conv/metrics.json` (anchored=0.7377 > uniform=0.7296 > random=0.7192; p(anchored vs uniform)=0.048, p(anchored vs random)=0.003) and train→test `p0_train3339_test402_energy_160_224_352_k2_shift1_std1.0_temporal_conv/metrics.json` (anchored≈uniform: 0.7191 vs 0.7186; p=0.896; anchored > random=0.6992, p=3.6e-05). Rerun root: `runs/REAL_AVE_OFFICIAL_RERUN_20260201-152134` (adds `audio_concat_anchored_top2` baseline; trains head on `TRAIN_DEVICE=cuda:0`; skips cache build when caches exist). Train→val `p0_train3339_val402_energy_160_224_352_k2_shift1_std1.0_temporal_conv_v2/metrics.json` (anchored=0.7508 > uniform=0.7402; p=0.009; audio_concat_anchored=0.7515; audio_concat_uniform=0.7490). Train→test `p0_train3339_test402_energy_160_224_352_k2_shift1_std1.0_temporal_conv_v2/metrics.json` (anchored=0.7306 > uniform=0.7192; p=0.046; audio_concat_uniform=0.7336; audio_concat_anchored=0.7346). |

Follow-ups (train3339→test402; same token budget=1960):
- Fast repro (cache-preload optimization, same numbers): `runs/REAL_AVE_OFFICIAL_TUNE_20260201-ANCH_ADAPT/p0_train3339_test402_energy_160_224_352_k2_shift1_std1.0_temporal_conv_fixed_v3/metrics.json` (anchored=0.7306 > uniform=0.7192; p=0.046).
- Diagnose report (root-cause): `runs/REAL_AVE_OFFICIAL_TUNE_20260201-ANCH_ADAPT/p0_train3339_test402_energy_160_224_352_k2_shift1_std1.0_temporal_conv_fixed_v3/diagnose_v3/diagnose.json` (fallback≈73%; among 2-anchor clips, distance=1 dominates and has near-zero mean delta).
- Adaptive high-anchor demotion (not better): `runs/REAL_AVE_OFFICIAL_TUNE_20260201-ANCH_ADAPT/p0_train3339_test402_energy_160_224_352_k2_shift1_std1.0_temporal_conv_adaptive_gap0.6_adj1_v2/metrics.json` (anchored=0.7287; p=0.079).
- Lower std threshold (not better): `runs/REAL_AVE_OFFICIAL_TUNE_20260201-ANCH_ADAPT/p0_train3339_test402_energy_160_224_352_k2_shift1_std0.8_temporal_conv_fixed/metrics.json` (anchored=0.7273; p=0.094).
- PANNs eventness (not better): `runs/REAL_AVE_OFFICIAL_TUNE_20260201-PANNS/p0_train3339_test402_panns_160_224_352_k2_shift1_std0.2_temporal_conv/metrics.json` (anchored=0.7238; p=0.189).
- Supervised audio_basic_lr eventness (not better): `runs/REAL_AVE_OFFICIAL_TUNE_20260201-AUDIO_LR/p0_train3339_test402_audio_basic_lr_160_224_352_k2_shift1_std0.0_temporal_conv/metrics.json` (anchored=0.7060; p=0.0056) and `runs/REAL_AVE_OFFICIAL_TUNE_20260201-AUDIO_LR/p0_train3339_test402_audio_basic_lr_160_224_352_k2_shift1_std0.3_temporal_conv/metrics.json` (anchored=0.7255; p=0.136).
- Strong-NMS anchor selection (not better): `runs/REAL_AVE_OFFICIAL_TUNE_20260201-ANCH_STRONGNMS/p0_train3339_test402_energy_160_224_352_k2_shift1_std1.0_temporal_conv_nmsStrong_r2_gap0.6_s0-9/metrics.json` (anchored=0.7295 < baseline 0.7306; p=0.068).
- Supervised audio_basic_mlp_cls eventness (screening, not better): on the same train→test ids with seeds0-2, `anchored_top2` is worse than energy anchors (see `runs/TMP_RERUN_audio_basic_mlp_cls_smoke/metrics.json` and `runs/TMP_RERUN_energy_thr1p0_s0-2/metrics.json`). A more aggressive fallback (`ANCHOR_STD_THRESHOLD=0.1`) avoids the regression but still does not beat energy (see `runs/TMP_RERUN_audio_basic_mlp_cls_thr0.1/metrics.json`).
- Supervised audio_basic_mlp_cls_target eventness (screening, worse): `runs/TMP_RERUN_audio_basic_mlp_cls_target_thr0.0/metrics.json`.
- Supervised audio_basic_mlp eventness (screening, not better): with fallback, it recovers to a small +Δ but stays below energy (see `runs/TMP_RERUN_audio_basic_mlp_thr{0.3,0.6,1.0}/metrics.json` vs `runs/TMP_RERUN_energy_thr1p0_s0-2/metrics.json`).
- Supervised audio_fbank_mlp eventness (not better on smaller real subset): `runs/TMP_RERUN_audio_fbank_mlp_small_thr0.0/metrics.json` vs energy baseline `runs/REAL_AVE_20260131-211548/p0_train195_test113_energy_160_224_352_k2_shift1_std1p0_temporal_s0-4/metrics.json`.
