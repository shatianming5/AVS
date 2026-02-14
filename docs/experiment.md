# Experiments

## Overview
- Goal: Prove `docs/plan.md` conclusions (C0001-C0009) with runnable, reproducible commands and locally present artifacts.
- Baseline: Uniform-224 sampling under a fixed ViT token budget.
- Primary model: CLIP ViT-B/16 (vision features) + per-segment MLP head; audio probe = energy or AST.
- Data: To install the official full AVE videos (4143 clips), run `bash scripts/ave_install_official.sh` (downloads `AVE_Dataset.zip` and installs under `data/AVE/raw/videos/`).

## Reproducibility
- Date: 2026-02-09
- Environment: Python 3.10.12; torch 2.6.0+cu124 (CUDA 12.4); GPUs: 5x NVIDIA GeForce RTX 4090 D
- Datasets:
  - AVE (official zip): `data/AVE/raw/videos/` (install via `bash scripts/ave_install_official.sh`)
  - EPIC-SOUNDS (local subset): `data/EPIC_SOUNDS/raw/videos/` + annotations under `data/EPIC_SOUNDS/meta/`

## Checklist
### Run Queue (Oral-Competitive C0003 push; sequential)
- [x] E0801: ImageBind Stage-1 (`imagebind_av_sim`) — val402 sweep (SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=imagebind_av_sim CANDIDATE_SET=ltl_adaptive_keepadj_v2 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0801_val402_imagebind_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0801_*/sweep_summary.json`
    - `runs/E0801_*/best_config.json`
    - `runs/E0801_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`

- [x] E0802: ImageBind Stage-1 (`imagebind_av_sim`) — quick test402 (SEEDS=0..2) + diagnose
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0801_*/best_config.json EVENTNESS=imagebind_av_sim SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0802_quick_test402_imagebind_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0802_*/metrics.json OUT_DIR=runs/E0802_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0802_*/metrics.json`
    - `runs/E0802_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)

- [x] E0803: ImageBind Stage-1 (`imagebind_av_sim`) — full test402 (SEEDS=0..9) (skipped: E0802 not promoted)
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0801_*/best_config.json EVENTNESS=imagebind_av_sim SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0803_full_test402_imagebind_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0803_*/metrics.json OUT_DIR=runs/E0803_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts: []
  - required_metrics: []

- [x] E0810: WavLM Stage-1 (`wavlm_evt_mlp`) — val402 sweep (SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=wavlm_evt_mlp CANDIDATE_SET=ltl_adaptive_keepadj_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0810_val402_wavlm_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0810_*/sweep_summary.json`
    - `runs/E0810_*/best_config.json`
    - `runs/E0810_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`

- [x] E0811: WavLM Stage-1 (`wavlm_evt_mlp`) — quick test402 (SEEDS=0..2) + diagnose
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0810_*/best_config.json EVENTNESS=wavlm_evt_mlp SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0811_quick_test402_wavlm_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0811_*/metrics.json OUT_DIR=runs/E0811_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0811_*/metrics.json`
    - `runs/E0811_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)

- [x] E0812: WavLM Stage-1 (`wavlm_evt_mlp`) — full test402 (SEEDS=0..9) (skipped: E0811 not promoted)
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0810_*/best_config.json EVENTNESS=wavlm_evt_mlp SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0812_full_test402_wavlm_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0812_*/metrics.json OUT_DIR=runs/E0812_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts: []
  - required_metrics: []

- [x] E0820: Oracle-distilled Stage-1 (`av_wavlm_clip_lossgain_mlp`) — val402 sweep (SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_wavlm_clip_lossgain_mlp CANDIDATE_SET=ltl_adaptive_keepadj_v2 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0820_val402_wavlm_cliplossgain_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0820_*/sweep_summary.json`
    - `runs/E0820_*/best_config.json`
    - `runs/E0820_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`

- [x] E0821: Oracle-distilled Stage-1 (`av_wavlm_clip_lossgain_mlp`) — quick test402 (SEEDS=0..2) + diagnose
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0820_*/best_config.json EVENTNESS=av_wavlm_clip_lossgain_mlp SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0821_quick_test402_wavlm_cliplossgain_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0821_*/metrics.json OUT_DIR=runs/E0821_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0821_*/metrics.json`
    - `runs/E0821_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)

- [x] E0822: Oracle-distilled Stage-1 (`av_wavlm_clip_lossgain_mlp`) — full test402 (SEEDS=0..9) (skipped: E0821 not promoted)
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0820_*/best_config.json EVENTNESS=av_wavlm_clip_lossgain_mlp SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0822_full_test402_wavlm_cliplossgain_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0822_*/metrics.json OUT_DIR=runs/E0822_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts:
    - `runs/E0822_*/metrics.json`
    - `runs/E0822_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)

- [x] E0830: MIL Stage-1 (`av_wavlm_clip_mil_mlp`) — val402 sweep (SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_wavlm_clip_mil_mlp CANDIDATE_SET=ltl_adaptive_keepadj_v2 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0830_val402_wavlm_clipmil_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0830_*/sweep_summary.json`
    - `runs/E0830_*/best_config.json`
    - `runs/E0830_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`

- [x] E0831: MIL Stage-1 (`av_wavlm_clip_mil_mlp`) — quick test402 (SEEDS=0..2) + diagnose
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0830_*/best_config.json EVENTNESS=av_wavlm_clip_mil_mlp SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0831_quick_test402_wavlm_clipmil_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0831_*/metrics.json OUT_DIR=runs/E0831_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0831_*/metrics.json`
    - `runs/E0831_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)

- [x] E0832: MIL Stage-1 (`av_wavlm_clip_mil_mlp`) — full test402 (SEEDS=0..9) (skipped: E0831 not promoted)
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0830_*/best_config.json EVENTNESS=av_wavlm_clip_mil_mlp SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0832_full_test402_wavlm_clipmil_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0832_*/metrics.json OUT_DIR=runs/E0832_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts:
    - `runs/E0832_*/metrics.json`
    - `runs/E0832_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)

- [x] E0840: WavLM+CLIP supervised eventness (`av_wavlm_clip_evt_mlp`) — val402 sweep (SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_wavlm_clip_evt_mlp CANDIDATE_SET=ltl_top1med_norm_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0840_val402_wavlm_clipevt_mlp_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0840_*/sweep_summary.json`
    - `runs/E0840_*/best_config.json`
    - `runs/E0840_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`

- [x] E0841: WavLM+CLIP supervised eventness (`av_wavlm_clip_evt_mlp`) — quick test402 (SEEDS=0..2) + diagnose
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0840_*/best_config.json EVENTNESS=av_wavlm_clip_evt_mlp SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0841_quick_test402_wavlm_clipevt_mlp_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0841_*/metrics.json OUT_DIR=runs/E0841_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0841_*/metrics.json`
    - `runs/E0841_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)

- [x] E0842: WavLM+CLIP supervised eventness (`av_wavlm_clip_evt_mlp`) — full test402 (SEEDS=0..9) (skipped: E0841 not promoted)
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0840_*/best_config.json EVENTNESS=av_wavlm_clip_evt_mlp SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0842_full_test402_wavlm_clipevt_mlp_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0842_*/metrics.json OUT_DIR=runs/E0842_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts: []
  - required_metrics: []

- [x] E0850: WavLM+CLIP supervised eventness (`av_wavlm_clip_evt_tcn`) — val402 sweep (SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_wavlm_clip_evt_tcn CANDIDATE_SET=ltl_top1med_norm_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0850_val402_wavlm_clipevt_tcn_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0850_*/sweep_summary.json`
    - `runs/E0850_*/best_config.json`
    - `runs/E0850_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`

- [x] E0851: WavLM+CLIP supervised eventness (`av_wavlm_clip_evt_tcn`) — quick test402 (SEEDS=0..2) + diagnose
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0850_*/best_config.json EVENTNESS=av_wavlm_clip_evt_tcn SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0851_quick_test402_wavlm_clipevt_tcn_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0851_*/metrics.json OUT_DIR=runs/E0851_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0851_*/metrics.json`
    - `runs/E0851_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)

- [x] E0852: WavLM+CLIP supervised eventness (`av_wavlm_clip_evt_tcn`) — full test402 (SEEDS=0..9) (skipped: E0851 not promoted)
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0850_*/best_config.json EVENTNESS=av_wavlm_clip_evt_tcn SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0852_full_test402_wavlm_clipevt_tcn_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0852_*/metrics.json OUT_DIR=runs/E0852_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts: []
  - required_metrics: []

- [x] E0860: Vec-MLP Stage-1 (`av_clipdiff_vec_mlp`) — val402 sweep (`ltl_sep3_v1`; SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_clipdiff_vec_mlp CANDIDATE_SET=ltl_sep3_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 SCORES_JSON=runs/E0610_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_adaptive_v1_20260210-200224/eventness_scores.json OUT_DIR=runs/E0860_val402_vecmlp_sep3_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0860_*/sweep_summary.json`
    - `runs/E0860_*/best_config.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`

- [x] E0861: Vec-MLP Stage-1 (`av_clipdiff_vec_mlp`) — quick test402 (`ltl_sep3_v1`; SEEDS=0..2) + diagnose
  - command:
    - `BEST_CONFIG_JSON=runs/E0860_*/best_config.json EVENTNESS=av_clipdiff_vec_mlp SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0861_quick_test402_vecmlp_sep3_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0861_*/metrics.json OUT_DIR=runs/E0861_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0861_*/metrics.json`
    - `runs/E0861_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)

- [x] E0862: Vec-MLP Stage-1 (`av_clipdiff_vec_mlp`) — full test402 (`ltl_sep3_v1`; SEEDS=0..9) (skipped: E0861 not promoted)
  - command:
    - `BEST_CONFIG_JSON=runs/E0860_*/best_config.json EVENTNESS=av_clipdiff_vec_mlp SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0862_full_test402_vecmlp_sep3_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0862_*/metrics.json OUT_DIR=runs/E0862_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts: []
  - required_metrics: []

- [x] E0870: WavLM+CLIPdiff vec-MLP Stage-1 (`av_wavlm_clipdiff_vec_mlp`) — val402 sweep (SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_wavlm_clipdiff_vec_mlp CANDIDATE_SET=ltl_top1med_norm_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0870_val402_wavlm_clipdiff_vecmlp_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0870_*/sweep_summary.json`
    - `runs/E0870_*/best_config.json`
    - `runs/E0870_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`

- [x] E0871: WavLM+CLIPdiff vec-MLP Stage-1 (`av_wavlm_clipdiff_vec_mlp`) — quick test402 (SEEDS=0..2) + diagnose
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0870_*/best_config.json EVENTNESS=av_wavlm_clipdiff_vec_mlp SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0871_quick_test402_wavlm_clipdiff_vecmlp_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0871_*/metrics.json OUT_DIR=runs/E0871_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0871_*/metrics.json`
    - `runs/E0871_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)

- [x] E0872: WavLM+CLIPdiff vec-MLP Stage-1 (`av_wavlm_clipdiff_vec_mlp`) — full test402 (SEEDS=0..9) (skipped: E0871 not promoted)
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0870_*/best_config.json EVENTNESS=av_wavlm_clipdiff_vec_mlp SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0872_full_test402_wavlm_clipdiff_vecmlp_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0872_*/metrics.json OUT_DIR=runs/E0872_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts: []
  - required_metrics: []

- [x] E0880: WavLM+CLIP multi-class cls-target Stage-1 (`av_wavlm_clip_mlp_cls_target`) — val402 sweep (SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_wavlm_clip_mlp_cls_target CANDIDATE_SET=ltl_top1med_norm_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0880_val402_wavlm_clip_cls_target_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0880_*/sweep_summary.json`
    - `runs/E0880_*/best_config.json`
    - `runs/E0880_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`

- [x] E0881: WavLM+CLIP multi-class cls-target Stage-1 (`av_wavlm_clip_mlp_cls_target`) — quick test402 (SEEDS=0..2) + diagnose
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0880_*/best_config.json EVENTNESS=av_wavlm_clip_mlp_cls_target SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0881_quick_test402_wavlm_clip_cls_target_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0881_*/metrics.json OUT_DIR=runs/E0881_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0881_*/metrics.json`
    - `runs/E0881_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)

- [x] E0882: WavLM+CLIP multi-class cls-target Stage-1 (`av_wavlm_clip_mlp_cls_target`) — full test402 (SEEDS=0..9) (skipped: E0881 not promoted)
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0880_*/best_config.json EVENTNESS=av_wavlm_clip_mlp_cls_target SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0882_full_test402_wavlm_clip_cls_target_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0882_*/metrics.json OUT_DIR=runs/E0882_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts: []
  - required_metrics: []

- [x] E0883: Vec-MLP Stage-1 (`av_clipdiff_vec_mlp`) — val402 Stage-2 sweep (`ltl_maxhigh1_v1`; SEEDS=0..2)
  - command: `OUT_DIR=runs/E0883_val402_vecmlp_maxhigh1_$(date +%Y%m%d-%H%M%S) && mkdir -p "$OUT_DIR" && cp runs/E0610_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_adaptive_v1_20260210-200224/eventness_scores.json "$OUT_DIR/eventness_scores.json" && PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_clipdiff_vec_mlp CANDIDATE_SET=ltl_maxhigh1_v1 SEEDS=0,1,2 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:0 SCORES_JSON="$OUT_DIR/eventness_scores.json" OUT_DIR="$OUT_DIR" bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0883_*/sweep_summary.json`
    - `runs/E0883_*/best_config.json`
    - `runs/E0883_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`

- [x] E0884: Vec-MLP Stage-1 (`av_clipdiff_vec_mlp`) — quick test402 (`ltl_maxhigh1_v1`; SEEDS=0..2) + diagnose
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0883_*/best_config.json EVENTNESS=av_clipdiff_vec_mlp SEEDS=0,1,2 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0884_quick_test402_vecmlp_maxhigh1_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0884_*/metrics.json OUT_DIR=runs/E0884_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0884_*/metrics.json`
    - `runs/E0884_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)

- [x] E0885: Vec-MLP Stage-1 (`av_clipdiff_vec_mlp`) — full test402 (`ltl_maxhigh1_v1`; SEEDS=0..9) (skipped: E0884 not promoted)
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0883_*/best_config.json EVENTNESS=av_clipdiff_vec_mlp SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0885_full_test402_vecmlp_maxhigh1_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0885_*/metrics.json OUT_DIR=runs/E0885_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts: []
  - required_metrics: []

- [x] E0886: WavLM+CLIP multi-class margin Stage-1 (`av_wavlm_clip_mlp_cls`) — val402 sweep (SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_wavlm_clip_mlp_cls CANDIDATE_SET=ltl_top1med_norm_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0886_val402_wavlm_clip_cls_margin_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0886_*/sweep_summary.json`
    - `runs/E0886_*/best_config.json`
    - `runs/E0886_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`

- [x] E0887: WavLM+CLIP multi-class margin Stage-1 (`av_wavlm_clip_mlp_cls`) — quick test402 (SEEDS=0..2) + diagnose
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0886_*/best_config.json EVENTNESS=av_wavlm_clip_mlp_cls SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0887_quick_test402_wavlm_clip_cls_margin_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0887_*/metrics.json OUT_DIR=runs/E0887_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0887_*/metrics.json`
    - `runs/E0887_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)

- [x] E0888: WavLM+CLIP multi-class margin Stage-1 (`av_wavlm_clip_mlp_cls`) — full test402 (SEEDS=0..9) (skipped: E0887 not promoted)
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0886_*/best_config.json EVENTNESS=av_wavlm_clip_mlp_cls SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0888_full_test402_wavlm_clip_cls_margin_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0888_*/metrics.json OUT_DIR=runs/E0888_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts: []
  - required_metrics: []

- [x] E0890: WavLM+CLIP MIL Stage-1 (`av_wavlm_clip_mil_mlp`) — val402 sweep (`ltl_top1medn_maxhigh1_v1`; SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_wavlm_clip_mil_mlp CANDIDATE_SET=ltl_top1medn_maxhigh1_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0890_val402_wavlm_clip_mil_mlp_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0890_*/sweep_summary.json`
    - `runs/E0890_*/best_config.json`
    - `runs/E0890_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`

- [x] E0891: WavLM+CLIP MIL Stage-1 (`av_wavlm_clip_mil_mlp`) — quick test402 (SEEDS=0..2) + diagnose
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0890_*/best_config.json EVENTNESS=av_wavlm_clip_mil_mlp SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0891_quick_test402_wavlm_clip_mil_mlp_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0891_*/metrics.json OUT_DIR=runs/E0891_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0891_*/metrics.json`
    - `runs/E0891_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)

- [x] E0892: WavLM+CLIP MIL Stage-1 (`av_wavlm_clip_mil_mlp`) — full test402 (SEEDS=0..9) (skipped: E0891 not promoted)
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0890_*/best_config.json EVENTNESS=av_wavlm_clip_mil_mlp SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0892_full_test402_wavlm_clip_mil_mlp_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0892_*/metrics.json OUT_DIR=runs/E0892_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts: []
  - required_metrics: []

- [x] E0893: Vec-MLP Stage-2 ablation (df7 keepadj; force `max_high_anchors=1`) — quick test402 (SEEDS=0..2) + diagnose
  - command: `OUT_DIR=runs/E0893_quick_test402_vecmlp_df7_maxhigh1_$(date +%Y%m%d-%H%M%S) && mkdir -p "$OUT_DIR" && jq '.max_high_anchors=1 | .name="ltlkeepadj_adj2_shift1_std0p55_df7_maxhigh1_s0-2"' runs/E0643_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_df7_officialids_s0-9_20260211-001604/config.json > "$OUT_DIR/config.json" && PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON="$OUT_DIR/config.json" EVENTNESS=av_clipdiff_vec_mlp SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 SCORES_JSON=runs/E0610_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_adaptive_v1_20260210-200224/eventness_scores.json OUT_DIR="$OUT_DIR" bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh && IN_METRICS="$OUT_DIR/metrics.json" OUT_DIR="$OUT_DIR" bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0893_*/metrics.json`
    - `runs/E0893_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)

- [x] E0895: Vec-MLP Stage-2 ablation (df7 keepadj; `budget_mode=band`, extra_res=[112]) — quick test402 (SEEDS=0..2) + diagnose
  - command: `OUT_DIR=runs/E0895_quick_test402_vecmlp_df7_band112_$(date +%Y%m%d-%H%M%S) && mkdir -p "$OUT_DIR" && jq '.budget_mode="band" | .budget_extra_resolutions=[112] | .name="ltlkeepadj_adj2_shift1_std0p55_df7_band112_s0-2"' runs/E0643_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_df7_officialids_s0-9_20260211-001604/config.json > "$OUT_DIR/config.json" && PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON="$OUT_DIR/config.json" EVENTNESS=av_clipdiff_vec_mlp SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 SCORES_JSON=runs/E0610_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_adaptive_v1_20260210-200224/eventness_scores.json OUT_DIR="$OUT_DIR" bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh && IN_METRICS="$OUT_DIR/metrics.json" OUT_DIR="$OUT_DIR" bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0895_*/metrics.json`
    - `runs/E0895_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)

- [x] E0896: Vec-MLP Stage-2 ablation (df7 keepadj; `budget_mode=band`, extra_res=[112]) — full test402 (SEEDS=0..9) + diagnose
  - command: `OUT_DIR=runs/E0896_full_test402_vecmlp_df7_band112_$(date +%Y%m%d-%H%M%S) && mkdir -p "$OUT_DIR" && jq '.budget_mode="band" | .budget_extra_resolutions=[112] | .name="ltlkeepadj_adj2_shift1_std0p55_df7_band112_s0-9"' runs/E0643_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_df7_officialids_s0-9_20260211-001604/config.json > "$OUT_DIR/config.json" && PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON="$OUT_DIR/config.json" EVENTNESS=av_clipdiff_vec_mlp SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 SCORES_JSON=runs/E0610_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_adaptive_v1_20260210-200224/eventness_scores.json OUT_DIR="$OUT_DIR" bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh && IN_METRICS="$OUT_DIR/metrics.json" OUT_DIR="$OUT_DIR" bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts:
    - `runs/E0896_*/metrics.json`
    - `runs/E0896_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)

- [x] E0898: Vec-MLP Stage-2 ablation (df7 keepadj; maxhigh1; std0.35) — quick test402 (SEEDS=0..2) + diagnose
  - command: `OUT_DIR=runs/E0898_quick_test402_vecmlp_df7_maxhigh1_std0p35_$(date +%Y%m%d-%H%M%S) && mkdir -p "$OUT_DIR" && jq '.max_high_anchors=1 | .anchor_std_threshold=0.35 | .name="ltlkeepadj_adj2_shift1_std0p35_df7_maxhigh1_s0-2"' runs/E0643_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_df7_officialids_s0-9_20260211-001604/config.json > "$OUT_DIR/config.json" && PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON="$OUT_DIR/config.json" EVENTNESS=av_clipdiff_vec_mlp SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 SCORES_JSON=runs/E0610_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_adaptive_v1_20260210-200224/eventness_scores.json OUT_DIR="$OUT_DIR" bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh && IN_METRICS="$OUT_DIR/metrics.json" OUT_DIR="$OUT_DIR" bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0898_*/metrics.json`
    - `runs/E0898_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)

- [x] E0899: Vec-MLP Stage-2 ablation (df7 keepadj; maxhigh1; std0.45) — quick test402 (SEEDS=0..2) + diagnose
  - command: `OUT_DIR=runs/E0899_quick_test402_vecmlp_df7_maxhigh1_std0p45_$(date +%Y%m%d-%H%M%S) && mkdir -p "$OUT_DIR" && jq '.max_high_anchors=1 | .anchor_std_threshold=0.45 | .name="ltlkeepadj_adj2_shift1_std0p45_df7_maxhigh1_s0-2"' runs/E0643_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_df7_officialids_s0-9_20260211-001604/config.json > "$OUT_DIR/config.json" && PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON="$OUT_DIR/config.json" EVENTNESS=av_clipdiff_vec_mlp SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 SCORES_JSON=runs/E0610_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_adaptive_v1_20260210-200224/eventness_scores.json OUT_DIR="$OUT_DIR" bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh && IN_METRICS="$OUT_DIR/metrics.json" OUT_DIR="$OUT_DIR" bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0899_*/metrics.json`
    - `runs/E0899_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)

- [x] E0900: Vec-MLP Stage-2 ablation (df7 keepadj; maxhigh1; std0.65) — quick test402 (SEEDS=0..2) + diagnose
  - command: `OUT_DIR=runs/E0900_quick_test402_vecmlp_df7_maxhigh1_std0p65_$(date +%Y%m%d-%H%M%S) && mkdir -p "$OUT_DIR" && jq '.max_high_anchors=1 | .anchor_std_threshold=0.65 | .name="ltlkeepadj_adj2_shift1_std0p65_df7_maxhigh1_s0-2"' runs/E0643_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_df7_officialids_s0-9_20260211-001604/config.json > "$OUT_DIR/config.json" && PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON="$OUT_DIR/config.json" EVENTNESS=av_clipdiff_vec_mlp SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 SCORES_JSON=runs/E0610_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_adaptive_v1_20260210-200224/eventness_scores.json OUT_DIR="$OUT_DIR" bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh && IN_METRICS="$OUT_DIR/metrics.json" OUT_DIR="$OUT_DIR" bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0900_*/metrics.json`
    - `runs/E0900_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)

- [x] E0901: Vec-MLP Stage-2 ablation (df7 keepadj; `k=1`) — quick test402 (SEEDS=0..2) + diagnose
  - command: `OUT_DIR=runs/E0901_quick_test402_vecmlp_df7_k1_$(date +%Y%m%d-%H%M%S) && mkdir -p "$OUT_DIR" && jq '.k=1 | .max_high_anchors=1 | .name="ltlkeepadj_adj2_shift1_std0p55_df7_k1_s0-2"' runs/E0643_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_df7_officialids_s0-9_20260211-001604/config.json > "$OUT_DIR/config.json" && PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON="$OUT_DIR/config.json" EVENTNESS=av_clipdiff_vec_mlp SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 SCORES_JSON=runs/E0610_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_adaptive_v1_20260210-200224/eventness_scores.json OUT_DIR="$OUT_DIR" bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh && IN_METRICS="$OUT_DIR/metrics.json" OUT_DIR="$OUT_DIR" bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0901_*/metrics.json`
    - `runs/E0901_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)

- [x] E0902: WavLM+CLIP XAttn MIL Stage-1 (`av_wavlm_clip_xattn_mil`) — val402 sweep (`ltl_top1medn_maxhigh1_v1`; SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_wavlm_clip_xattn_mil CANDIDATE_SET=ltl_top1medn_maxhigh1_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0902_val402_wavlm_clip_xattn_mil_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0902_*/sweep_summary.json`
    - `runs/E0902_*/best_config.json`
    - `runs/E0902_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`

- [x] E0903: WavLM+CLIP XAttn MIL Stage-1 (`av_wavlm_clip_xattn_mil`) — val402 sweep (r224; clip+clipdiff; SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_wavlm_clip_xattn_mil CANDIDATE_SET=ltl_top1medn_maxhigh1_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 XATTN_TRAIN_DEVICE=cuda:0 XATTN_VIS_RES=224 XATTN_VIS_FEATS=clip+clipdiff OUT_DIR=runs/E0903_val402_wavlm_clip_xattn_mil_r224_clipdiff_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0903_*/sweep_summary.json`
    - `runs/E0903_*/best_config.json`
    - `runs/E0903_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - logs: `artifacts/experiments/E0903/run.log`

- [x] E0904: WavLM+CLIP XAttn MIL Stage-1 (`av_wavlm_clip_xattn_mil`) — val402 sweep (keepadjv2; cached scores from E0903)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_wavlm_clip_xattn_mil CANDIDATE_SET=ltl_adaptive_keepadj_v2 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 SCORES_JSON=runs/E0903_*/eventness_scores.json OUT_DIR=runs/E0904_val402_xattn_mil_r224_clipdiff_keepadjv2_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0904_*/sweep_summary.json`
    - `runs/E0904_*/best_config.json`
    - `runs/E0903_*/eventness_scores.json` (reused)
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - logs: `artifacts/experiments/E0904/run.log`

- [x] E0905: WavLM+CLIP XAttn MIL Stage-1 (`av_wavlm_clip_xattn_mil`) — val402 sweep (r352; clip; SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_wavlm_clip_xattn_mil CANDIDATE_SET=ltl_top1medn_maxhigh1_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 XATTN_TRAIN_DEVICE=cuda:0 XATTN_VIS_RES=352 XATTN_VIS_FEATS=clip OUT_DIR=runs/E0905_val402_xattn_mil_r352_clip_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0905_*/sweep_summary.json`
    - `runs/E0905_*/best_config.json`
    - `runs/E0905_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - logs: `artifacts/experiments/E0905/run.log`

- [x] E0906: High-res vision Stage-1 (`vision_binary_mlp_r352`) — val402 sweep (`ltl_top1medn_maxhigh1_v1`; SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=vision_binary_mlp_r352 CANDIDATE_SET=ltl_top1medn_maxhigh1_v1 SEEDS=0,1,2 TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0906_val402_vision_binary_mlp_r352_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0906_*/sweep_summary.json`
    - `runs/E0906_*/best_config.json`
    - `runs/E0906_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - logs: `artifacts/experiments/E0906/run.log`

- [x] E0907: High-res vision Stage-1 (`vision_binary_mlp_r352`) — val402 sweep (`ltl_gini_v2`; cached scores from E0906; SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=vision_binary_mlp_r352 CANDIDATE_SET=ltl_gini_v2 SEEDS=0,1,2 TRAIN_DEVICE=cuda:0 SCORES_JSON=runs/E0906_*/eventness_scores.json OUT_DIR=runs/E0907_val402_vision_binary_mlp_r352_gini_v2_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0907_*/sweep_summary.json`
    - `runs/E0907_*/best_config.json`
    - `runs/E0906_*/eventness_scores.json` (reused)
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - logs: `artifacts/experiments/E0907/run.log`

- [x] E0908: AVE-localizer-style Stage-1 (`av_wavlm_clip_avel_bilstm_cls_target`) — val402 sweep (r352; clip+clipdiff; `ltl_top1medn_maxhigh1_v1`; SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_wavlm_clip_avel_bilstm_cls_target CANDIDATE_SET=ltl_top1medn_maxhigh1_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 AVEL_TRAIN_DEVICE=cuda:0 AVEL_VIS_RES=352 AVEL_VIS_FEATS=clip+clipdiff AVEL_EPOCHS=60 AVEL_BS=256 AVEL_LR=1e-3 OUT_DIR=runs/E0908_val402_avel_bilstm_cls_r352_clipdiff_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0908_*/sweep_summary.json`
    - `runs/E0908_*/best_config.json`
    - `runs/E0908_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - logs: `artifacts/experiments/E0908/run.log`

- [x] E0909: AVE-localizer-style Stage-1 (`av_wavlm_clip_avel_bilstm_cls_target`) — val402 sweep (minmax-normalized score cache; sanity check; cached from E0908)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_wavlm_clip_avel_bilstm_cls_target CANDIDATE_SET=ltl_top1medn_maxhigh1_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 SCORES_JSON=runs/E0908_*/eventness_scores_minmax.json OUT_DIR=runs/E0909_val402_avel_bilstm_cls_r352_clipdiff_minmax_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0909_*/sweep_summary.json`
    - `runs/E0909_*/best_config.json`
    - `runs/E0908_*/eventness_scores_minmax.json` (reused)
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - logs: `artifacts/experiments/E0909/run.log`

- [x] E0910: AVE-localizer-style Stage-1 (`av_wavlm_clip_avel_bilstm_cls_target`) — val402 sweep (onset-deriv positive score cache; cached from E0908)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_wavlm_clip_avel_bilstm_cls_target CANDIDATE_SET=ltl_top1medn_maxhigh1_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 SCORES_JSON=runs/E0908_*/eventness_scores_onset_deriv_pos.json OUT_DIR=runs/E0910_val402_avel_bilstm_cls_onset_deriv_pos_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0910_*/sweep_summary.json`
    - `runs/E0910_*/best_config.json`
    - `runs/E0908_*/eventness_scores_onset_deriv_pos.json` (reused)
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - logs: `artifacts/experiments/E0910/run.log`

- [x] E0911: AVE-localizer-style Stage-1 (`av_wavlm_clip_avel_bilstm_cls_target`) — val402 sweep (`ltl_adaptive_keepadj_v2`; cached scores from E0908; SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_wavlm_clip_avel_bilstm_cls_target CANDIDATE_SET=ltl_adaptive_keepadj_v2 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 SCORES_JSON=runs/E0908_*/eventness_scores.json OUT_DIR=runs/E0911_val402_avel_bilstm_cls_keepadjv2_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0911_*/sweep_summary.json`
    - `runs/E0911_*/best_config.json`
    - `runs/E0908_*/eventness_scores.json` (reused)
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - logs: `artifacts/experiments/E0911/run.log`

- [x] E0912: AVE-localizer-style Stage-1 (`av_wavlm_clip_avel_bilstm_cls_target`) — val402 sweep (`ltl_gini_v2`; cached scores from E0908; SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_wavlm_clip_avel_bilstm_cls_target CANDIDATE_SET=ltl_gini_v2 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 SCORES_JSON=runs/E0908_*/eventness_scores.json OUT_DIR=runs/E0912_val402_avel_bilstm_cls_gini_v2_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0912_*/sweep_summary.json`
    - `runs/E0912_*/best_config.json`
    - `runs/E0908_*/eventness_scores.json` (reused)
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - logs: `artifacts/experiments/E0912/run.log`

- [x] E0913: AVE-localizer-style Stage-1 (`av_wavlm_clip_avel_bilstm_cls_target`) — val402 sweep (`ltl_adaptive_v2`; cached scores from E0908; SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_wavlm_clip_avel_bilstm_cls_target CANDIDATE_SET=ltl_adaptive_v2 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 SCORES_JSON=runs/E0908_*/eventness_scores.json OUT_DIR=runs/E0913_val402_avel_bilstm_cls_adaptive_v2_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0913_*/sweep_summary.json`
    - `runs/E0913_*/best_config.json`
    - `runs/E0908_*/eventness_scores.json` (reused)
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - logs: `artifacts/experiments/E0913/run.log`

- [x] E0914: XAttn supervised Stage-1 (`av_wavlm_clip_xattn_cls_target`) — val402 sweep (r352; clip+clipdiff; `ltl_top1medn_maxhigh1_v1`; SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_wavlm_clip_xattn_cls_target CANDIDATE_SET=ltl_top1medn_maxhigh1_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 XATTN_TRAIN_DEVICE=cuda:0 XATTN_VIS_RES=352 XATTN_VIS_FEATS=clip+clipdiff XATTN_EPOCHS=60 XATTN_BS=256 XATTN_EVAL_BS=256 XATTN_LR=2e-3 OUT_DIR=runs/E0914_val402_xattn_cls_target_r352_clipdiff_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0914_*/sweep_summary.json`
    - `runs/E0914_*/best_config.json`
    - `runs/E0914_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - logs: `artifacts/experiments/E0914/run.log`

- [x] E0915: Build EVA02 vision caches (timm backbone) for official AVE (train3339/val402/test402)
  - command:
    - Train+val cache build:
      - `python -m avs.pipeline.ave_p0_end2end --mode none --allow-missing --meta-dir data/AVE/meta --processed-dir runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed --caches-dir runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_eva02_base_patch16_clip_224_112_160_224_352 --split-train train --split-eval val --train-ids-file data/AVE/meta/download_ok_train_official.txt --eval-ids-file data/AVE/meta/download_ok_val_official.txt --limit-train 3339 --limit-eval 402 --seeds 0,1 --cache-only --cache-skip-existing --cache-num-workers 5 --cache-devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4 --cache-resolutions 112,160,224,352 --vision-model-name timm:eva02_base_patch16_clip_224 --vision-pretrained --out-dir runs/E0915_build_cache_eva02_clip_p16_112_160_224_352_20260212-225043`
    - Test cache build (incremental):
      - `python -m avs.pipeline.ave_p0_end2end --mode none --allow-missing --meta-dir data/AVE/meta --processed-dir runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed --caches-dir runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_eva02_base_patch16_clip_224_112_160_224_352 --split-train train --split-eval test --train-ids-file data/AVE/meta/download_ok_train_official.txt --eval-ids-file data/AVE/meta/download_ok_test_official.txt --limit-train 3339 --limit-eval 402 --seeds 0,1 --cache-only --cache-skip-existing --cache-num-workers 5 --cache-devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4 --cache-resolutions 112,160,224,352 --vision-model-name timm:eva02_base_patch16_clip_224 --vision-pretrained --out-dir runs/E0915_build_cache_eva02_clip_p16_112_160_224_352_test_20260212-230913`
  - configs: []
  - seeds: []
  - required_artifacts:
    - `runs/E0915_build_cache_eva02_clip_p16_112_160_224_352_20260212-225043/cache_build.json`
    - `runs/E0915_build_cache_eva02_clip_p16_112_160_224_352_test_20260212-230913/cache_build.json`
    - `runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_eva02_base_patch16_clip_224_112_160_224_352/*.npz`
  - required_metrics:
    - `cache_build.json`: `ok=true`, `missing_caches=[]`, `cache_resolutions=[112,160,224,352]`

- [x] E0916: EVA02 Stage-2 backbone swap (train+eval on EVA02 caches) — val402 sweep (SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_eva02_base_patch16_clip_224_112_160_224_352 EVENTNESS=av_clipdiff_vec_mlp CANDIDATE_SET=ltl_adaptive_keepadj_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0916_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_adaptive_keepadj_v1_eva02_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0916_*/sweep_summary.json`
    - `runs/E0916_*/best_config.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - decision: not promoted (val402 not competitive)

- [x] E0917: EVA02 Stage-1-only caches (Stage-2 stays baseline) — val402 sweep (keepadj; SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 STAGE1_CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_eva02_base_patch16_clip_224_112_160_224_352 EVENTNESS=av_clipdiff_vec_mlp CANDIDATE_SET=ltl_adaptive_keepadj_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0917_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_adaptive_keepadj_v1_stage1eva02_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0917_*/sweep_summary.json`
    - `runs/E0917_*/best_config.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - decision: not promoted (val402 not competitive)

- [x] E0918: EVA02 Stage-1-only caches (Stage-2 stays baseline) — val402 sweep (top1med_norm; SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 STAGE1_CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_eva02_base_patch16_clip_224_112_160_224_352 EVENTNESS=av_clipdiff_vec_mlp CANDIDATE_SET=ltl_top1med_norm_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0918_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_top1med_norm_v1_stage1eva02_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0918_*/sweep_summary.json`
    - `runs/E0918_*/best_config.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - decision: not promoted (val402 not competitive)

- [x] E0919: Build DINOv2 vision caches (timm backbone) for official AVE (train/val/test)
  - command:
    - Train+val cache build:
      - `HF_HUB_OFFLINE=1 python -m avs.pipeline.ave_p0_end2end --mode none --meta-dir data/AVE/meta --processed-dir runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed --preprocess-skip-existing --caches-dir runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch14_dinov2_112_160_224_352_448 --split-train train --split-eval val --train-ids-file data/AVE/meta/download_ok_train_official.txt --eval-ids-file data/AVE/meta/download_ok_val_official.txt --limit-train 3339 --limit-eval 402 --seeds 0,1 --cache-only --cache-skip-existing --cache-num-workers 4 --cache-devices cuda:0,cuda:1,cuda:2,cuda:4 --cache-resolutions 112,160,224,352,448 --vision-model-name timm:vit_base_patch14_dinov2 --vision-pretrained --out-dir runs/E0919_build_cache_dinov2_fill_trainval_20260213-001226`
    - Test cache build (incremental):
      - `HF_HUB_OFFLINE=1 python -m avs.pipeline.ave_p0_end2end --mode none --meta-dir data/AVE/meta --processed-dir runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed --preprocess-skip-existing --caches-dir runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch14_dinov2_112_160_224_352_448 --split-train train --split-eval test --train-ids-file data/AVE/meta/download_ok_train_official.txt --eval-ids-file data/AVE/meta/download_ok_test_official.txt --limit-train 3339 --limit-eval 402 --seeds 0,1 --cache-only --cache-skip-existing --cache-num-workers 4 --cache-devices cuda:0,cuda:1,cuda:2,cuda:4 --cache-resolutions 112,160,224,352,448 --vision-model-name timm:vit_base_patch14_dinov2 --vision-pretrained --out-dir runs/E0919_build_cache_dinov2_fill_test_20260213-002118`
  - configs: []
  - seeds: []
  - required_artifacts:
    - `runs/E0919_build_cache_dinov2_fill_trainval_20260213-001226/cache_build.json`
    - `runs/E0919_build_cache_dinov2_fill_trainval_20260213-001226/cache_only.json`
    - `runs/E0919_build_cache_dinov2_fill_test_20260213-002118/cache_build.json`
    - `runs/E0919_build_cache_dinov2_fill_test_20260213-002118/cache_only.json`
    - `runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch14_dinov2_112_160_224_352_448/*.npz`
  - required_metrics:
    - `cache_build.json`: `ok=true`, `missing_caches=[]`, `cache_resolutions=[112,160,224,352,448]`

- [x] E0920: DINOv2 Stage-1-only caches (`STAGE1_CACHES_DIR`) — val402 sweep (keepadj; SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 STAGE1_CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch14_dinov2_112_160_224_352_448 EVENTNESS=av_clipdiff_vec_mlp CANDIDATE_SET=ltl_adaptive_keepadj_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0920_val402_vecmlp_keepadj_stage1dinov2_20260213-001634 bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0920_val402_vecmlp_keepadj_stage1dinov2_20260213-001634/sweep_summary.json`
    - `runs/E0920_val402_vecmlp_keepadj_stage1dinov2_20260213-001634/best_config.json`
    - `runs/E0920_val402_vecmlp_keepadj_stage1dinov2_20260213-001634/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - results: best=`ltlkeepadj_adj2_shift0_std0p55`, anchored=0.75520 vs uniform=0.74680 (Δ≈+0.00840; p≈0.4031) → promoted to quick test402 (E0921).

- [x] E0921: DINOv2 Stage-1-only caches (`STAGE1_CACHES_DIR`) — quick test402 (SEEDS=0..2) + diagnose
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 STAGE1_CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch14_dinov2_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0920_val402_vecmlp_keepadj_stage1dinov2_20260213-001634/best_config.json EVENTNESS=av_clipdiff_vec_mlp SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0921_quick_test402_vecmlp_keepadj_stage1dinov2_20260213-002346 bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0921_quick_test402_vecmlp_keepadj_stage1dinov2_20260213-002346/metrics.json OUT_DIR=runs/E0921_quick_test402_vecmlp_keepadj_stage1dinov2_20260213-002346 bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0921_quick_test402_vecmlp_keepadj_stage1dinov2_20260213-002346/metrics.json`
    - `runs/E0921_quick_test402_vecmlp_keepadj_stage1dinov2_20260213-002346/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)
  - results: anchored=0.72189 vs uniform=0.71294 (Δ≈+0.00896; p≈0.5825) → not promoted.

- [x] E0922: DINOv2 Stage-1-only caches (`STAGE1_CACHES_DIR`) — full test402 (SEEDS=0..9) + diagnose (skipped: E0921 not promoted)
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 STAGE1_CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch14_dinov2_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0920_val402_vecmlp_keepadj_stage1dinov2_20260213-001634/best_config.json EVENTNESS=av_clipdiff_vec_mlp SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0922_full_test402_vecmlp_keepadj_stage1dinov2_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0922_*/metrics.json OUT_DIR=runs/E0922_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts:
    - `runs/E0922_*/metrics.json`
    - `runs/E0922_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)

- [x] E0923: DINOv2 Stage-1-only caches (`STAGE1_CACHES_DIR`) — val402 sweep (`av_clipdiff_flow_mlp_stride`; `ltl_top1med_k1_extreme_v1`; SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 STAGE1_CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch14_dinov2_112_160_224_352_448 EVENTNESS=av_clipdiff_flow_mlp_stride CANDIDATE_SET=ltl_top1med_k1_extreme_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0923_val402_flow_stride_stage1dinov2_$(date +%Y%m%d-%H%M%S) bash scripts/e0400_ave_p0_sweep_official_val_ltl_top1med_k1_extreme_v1_av_clipdiff_flow_mlp_stride.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0923_*/sweep_summary.json`
    - `runs/E0923_*/best_config.json`
    - `runs/E0923_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - results: best=`ltltop1medk1ext_thr0p6_shift0_score`, Δ≈+0.00333 (p≈0.5507) → promoted to quick test402 (E0924).

- [x] E0924: DINOv2 Stage-1-only caches (`STAGE1_CACHES_DIR`) — quick test402 (`av_clipdiff_flow_mlp_stride`; SEEDS=0..2) + diagnose
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 STAGE1_CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch14_dinov2_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0923_*/best_config.json EVENTNESS=av_clipdiff_flow_mlp_stride SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0924_quick_test402_flow_stride_stage1dinov2_$(date +%Y%m%d-%H%M%S) bash scripts/e0401_ave_p0_best_to_test_quick_official_av_clipdiff_flow_mlp_stride.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0924_*/metrics.json`
    - `runs/E0924_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)
  - results: anchored=0.71401 vs uniform=0.71294 (Δ≈+0.00108; p≈0.9222) → not promoted.

- [x] E0925: DINOv2 Stage-1-only caches (`STAGE1_CACHES_DIR`) — full test402 (`av_clipdiff_flow_mlp_stride`; SEEDS=0..9) + diagnose (skipped: E0924 not promoted)
  - command:
    - `STAGE1_CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch14_dinov2_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0923_*/best_config.json EVENTNESS=av_clipdiff_flow_mlp_stride SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0925_full_test402_flow_stride_stage1dinov2_$(date +%Y%m%d-%H%M%S) bash scripts/e0402_ave_p0_best_to_test_full_official_av_clipdiff_flow_mlp_stride.sh`
    - `IN_METRICS=runs/E0925_*/metrics.json OUT_DIR=runs/E0925_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts: []
  - required_metrics: []

- [x] E0926: DINOv2 Stage-1-only caches (`STAGE1_CACHES_DIR`) — val402 sweep (AVE-localizer BiLSTM; SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 STAGE1_CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch14_dinov2_112_160_224_352_448 EVENTNESS=av_wavlm_clip_avel_bilstm_cls_target CANDIDATE_SET=ltl_top1medn_maxhigh1_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 AVEL_TRAIN_DEVICE=cuda:0 AVEL_VIS_RES=352 AVEL_VIS_FEATS=clip+clipdiff AVEL_EPOCHS=60 AVEL_BS=256 AVEL_LR=1e-3 OUT_DIR=runs/E0926_val402_avel_bilstm_cls_stage1dinov2_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0926_*/sweep_summary.json`
    - `runs/E0926_*/best_config.json`
    - `runs/E0926_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - results: best=`ltltop1mednmax1_thr0p7_shift1`, Δ≈-0.00898 (p≈0.2298) → not promoted.

- [x] E0927: DINOv2 Stage-1-only caches (`STAGE1_CACHES_DIR`) — val402 sweep (XAttn cls-target; SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 STAGE1_CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch14_dinov2_112_160_224_352_448 EVENTNESS=av_wavlm_clip_xattn_cls_target CANDIDATE_SET=ltl_top1medn_maxhigh1_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 XATTN_TRAIN_DEVICE=cuda:0 XATTN_VIS_RES=352 XATTN_VIS_FEATS=clip+clipdiff XATTN_EPOCHS=60 XATTN_BS=256 XATTN_EVAL_BS=256 XATTN_LR=2e-3 OUT_DIR=runs/E0927_val402_xattn_cls_target_stage1dinov2_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0927_*/sweep_summary.json`
    - `runs/E0927_*/best_config.json`
    - `runs/E0927_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - results: best=`ltltop1mednmax1_thr0p5_shift1`, Δ≈+0.00324 (p≈0.6881) → not promoted.

- [x] E0928: DINOv2 Stage-1 scores reused (`SCORES_JSON`) — val402 sweep (`ltl_top1med_norm_v1`; SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_clipdiff_vec_mlp CANDIDATE_SET=ltl_top1med_norm_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 SCORES_JSON=runs/E0920_val402_vecmlp_keepadj_stage1dinov2_20260213-001634/eventness_scores.json OUT_DIR=runs/E0928_val402_vecmlp_top1medn_scores_dinov2_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0928_*/sweep_summary.json`
    - `runs/E0928_*/best_config.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - results: best=`ltltop1medn_thr0p6_shift1`, Δ≈+0.00017 (p≈0.9836) → not promoted.

- [x] E0930: Build SigLIP vision caches (timm backbone; stage-2 candidate) for official AVE (train/val/test)
  - command:
    - Train+val cache build:
      - `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m avs.pipeline.ave_p0_end2end --mode none --meta-dir data/AVE/meta --processed-dir runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed --preprocess-skip-existing --caches-dir runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch16_siglip_224_webli_112_160_224_352_448 --split-train train --split-eval val --train-ids-file data/AVE/meta/download_ok_train_official.txt --eval-ids-file data/AVE/meta/download_ok_val_official.txt --limit-train 3339 --limit-eval 402 --seeds 0,1 --cache-only --cache-skip-existing --cache-num-workers 4 --cache-devices cuda:0,cuda:1,cuda:2,cuda:4 --cache-resolutions 112,160,224,352,448 --vision-model-name timm:vit_base_patch16_siglip_224.webli --vision-pretrained --out-dir runs/E0930_build_cache_siglip_trainval_$(date +%Y%m%d-%H%M%S)`
    - Test cache build (incremental):
      - `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m avs.pipeline.ave_p0_end2end --mode none --meta-dir data/AVE/meta --processed-dir runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed --preprocess-skip-existing --caches-dir runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch16_siglip_224_webli_112_160_224_352_448 --split-train train --split-eval test --train-ids-file data/AVE/meta/download_ok_train_official.txt --eval-ids-file data/AVE/meta/download_ok_test_official.txt --limit-train 3339 --limit-eval 402 --seeds 0,1 --cache-only --cache-skip-existing --cache-num-workers 4 --cache-devices cuda:0,cuda:1,cuda:2,cuda:4 --cache-resolutions 112,160,224,352,448 --vision-model-name timm:vit_base_patch16_siglip_224.webli --vision-pretrained --out-dir runs/E0930_build_cache_siglip_test_$(date +%Y%m%d-%H%M%S)`
  - configs: []
  - seeds: []
  - required_artifacts:
    - `runs/E0930_build_cache_siglip_trainval_*/cache_build.json`
    - `runs/E0930_build_cache_siglip_test_*/cache_build.json`
    - `runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch16_siglip_224_webli_112_160_224_352_448/*.npz`
  - required_metrics:
    - `cache_build.json`: `ok=true`, `missing_caches=[]`, `cache_resolutions=[112,160,224,352,448]`
  - results:
    - train+val: `runs/E0930_build_cache_siglip_trainval_20260213-011101/cache_build.json` (ok=true, missing=0)
    - test: `runs/E0930_build_cache_siglip_test_20260213-013806/cache_build.json` (ok=true, missing=0)
    - caches: `runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch16_siglip_224_webli_112_160_224_352_448/` (4097 `.npz`, each has `res_{112,160,224,352,448}` arrays)

- [x] E0931: SigLIP Stage-2 backbone swap (train+eval on SigLIP caches) — val402 sweep (SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch16_siglip_224_webli_112_160_224_352_448 EVENTNESS=av_clipdiff_vec_mlp CANDIDATE_SET=ltl_adaptive_keepadj_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0931_val402_siglip_stage2_vecmlp_keepadj_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0931_*/sweep_summary.json`
    - `runs/E0931_*/best_config.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - results: `runs/E0931_val402_siglip_stage2_vecmlp_keepadj_20260213-014809/sweep_summary.json` best=`ltlkeepadj_adj2_shift1_std0p6`, anchored=0.42186 vs uniform=0.40889, Δ=+0.01297 (p=0.0982)

- [x] E0932: SigLIP Stage-2 backbone swap — quick test402 (SEEDS=0..2) + diagnose
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch16_siglip_224_webli_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0931_*/best_config.json EVENTNESS=av_clipdiff_vec_mlp SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0932_quick_test402_siglip_stage2_vecmlp_keepadj_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0932_*/metrics.json OUT_DIR=runs/E0932_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0932_*/metrics.json`
    - `runs/E0932_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)
  - results:
    - `runs/E0932_quick_test402_siglip_stage2_vecmlp_keepadj_20260213-015409/metrics.json`: anchored=0.33350 vs uniform=0.33085, Δ=+0.00265 (p=0.5427)
    - `runs/E0932_quick_test402_siglip_stage2_vecmlp_keepadj_20260213-015409/diagnose.json`: fallback_used_frac≈0.853 → not promoted; skip full test402.

- [x] E0933: SigLIP Stage-2 backbone swap — full test402 (SEEDS=0..9) + diagnose (only if E0932 is promoted)
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch16_siglip_224_webli_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0931_*/best_config.json EVENTNESS=av_clipdiff_vec_mlp SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0933_full_test402_siglip_stage2_vecmlp_keepadj_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0933_*/metrics.json OUT_DIR=runs/E0933_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts: []
  - required_metrics: []
  - results: skipped (E0932 quick test402 not promoted).

- [x] E0934: XAttn binary eventness Stage-1 (`av_wavlm_clip_xattn_evt`) — val402 sweep (SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_wavlm_clip_xattn_evt CANDIDATE_SET=ltl_adaptive_keepadj_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 XATTN_TRAIN_DEVICE=cuda:0 XATTN_VIS_RES=112 XATTN_VIS_FEATS=clip+clipdiff XATTN_EPOCHS=60 XATTN_BS=256 XATTN_EVAL_BS=256 XATTN_LR=2e-3 XATTN_PROJ_DIM=128 XATTN_DROPOUT=0.1 XATTN_EVT_CLIP_LOSS_WEIGHT=0.5 OUT_DIR=runs/E0934_val402_xattn_evt_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0934_*/sweep_summary.json`
    - `runs/E0934_*/best_config.json`
    - `runs/E0934_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - results: `runs/E0934_val402_xattn_evt_20260213-021028/sweep_summary.json` best=`ltlkeepadj_adj2_shift1_std0p45`, Δ=+0.00208 (p=0.8075)

- [x] E0935: XAttn binary eventness Stage-1 (`av_wavlm_clip_xattn_evt`) — quick test402 (SEEDS=0..2) + diagnose
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0934_*/best_config.json EVENTNESS=av_wavlm_clip_xattn_evt SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 XATTN_TRAIN_DEVICE=cuda:0 XATTN_VIS_RES=112 XATTN_VIS_FEATS=clip+clipdiff XATTN_EPOCHS=60 XATTN_BS=256 XATTN_EVAL_BS=256 XATTN_LR=2e-3 XATTN_PROJ_DIM=128 XATTN_DROPOUT=0.1 XATTN_EVT_CLIP_LOSS_WEIGHT=0.5 OUT_DIR=runs/E0935_quick_test402_xattn_evt_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0935_*/metrics.json OUT_DIR=runs/E0935_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0935_*/metrics.json`
    - `runs/E0935_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)
  - results:
    - `runs/E0935_quick_test402_xattn_evt_20260213-021503/metrics.json`: anchored=0.71915 vs uniform=0.71294, Δ=+0.00622 (p=0.5305)
    - `runs/E0935_quick_test402_xattn_evt_20260213-021503/diagnose.json`: fallback_used_frac≈0.077 → not promoted; skip full test402.

- [x] E0936: XAttn binary eventness Stage-1 (`av_wavlm_clip_xattn_evt`) — full test402 (SEEDS=0..9) + diagnose (only if E0935 is promoted)
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0934_*/best_config.json EVENTNESS=av_wavlm_clip_xattn_evt SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 XATTN_TRAIN_DEVICE=cuda:0 XATTN_VIS_RES=112 XATTN_VIS_FEATS=clip+clipdiff XATTN_EPOCHS=60 XATTN_BS=256 XATTN_EVAL_BS=256 XATTN_LR=2e-3 XATTN_PROJ_DIM=128 XATTN_DROPOUT=0.1 XATTN_EVT_CLIP_LOSS_WEIGHT=0.5 OUT_DIR=runs/E0936_full_test402_xattn_evt_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0936_*/metrics.json OUT_DIR=runs/E0936_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts: []
  - required_metrics: []
  - results: skipped (E0935 quick test402 not promoted).

- [x] E0937: XAttn binary eventness Stage-1 (`av_wavlm_clip_xattn_evt`) — val402 sweep (Stage-1 SigLIP cache; SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 STAGE1_CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch16_siglip_224_webli_112_160_224_352_448 EVENTNESS=av_wavlm_clip_xattn_evt CANDIDATE_SET=ltl_adaptive_keepadj_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 XATTN_TRAIN_DEVICE=cuda:0 XATTN_VIS_RES=112 XATTN_VIS_FEATS=clip+clipdiff XATTN_EPOCHS=60 XATTN_BS=256 XATTN_EVAL_BS=256 XATTN_LR=2e-3 XATTN_PROJ_DIM=128 XATTN_DROPOUT=0.1 XATTN_EVT_CLIP_LOSS_WEIGHT=0.5 OUT_DIR=runs/E0937_val402_xattn_evt_stage1siglip_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0937_*/sweep_summary.json`
    - `runs/E0937_*/best_config.json`
    - `runs/E0937_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - results: `runs/E0937_val402_xattn_evt_stage1siglip_20260213-021919/sweep_summary.json` best=`ltlkeepadj_adj1_shift0_std0p6`, Δ=+0.00682 (p=0.4311)

- [x] E0938: XAttn binary eventness Stage-1 (`av_wavlm_clip_xattn_evt`) — quick test402 (Stage-1 SigLIP cache; SEEDS=0..2) + diagnose
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 STAGE1_CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch16_siglip_224_webli_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0937_*/best_config.json EVENTNESS=av_wavlm_clip_xattn_evt SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 XATTN_TRAIN_DEVICE=cuda:0 XATTN_VIS_RES=112 XATTN_VIS_FEATS=clip+clipdiff XATTN_EPOCHS=60 XATTN_BS=256 XATTN_EVAL_BS=256 XATTN_LR=2e-3 XATTN_PROJ_DIM=128 XATTN_DROPOUT=0.1 XATTN_EVT_CLIP_LOSS_WEIGHT=0.5 OUT_DIR=runs/E0938_quick_test402_xattn_evt_stage1siglip_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0938_*/metrics.json OUT_DIR=runs/E0938_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0938_*/metrics.json`
    - `runs/E0938_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)
  - results:
    - `runs/E0938_quick_test402_xattn_evt_stage1siglip_20260213-022855/metrics.json`: anchored=0.71857 vs uniform=0.71294, Δ=+0.00564 (p=0.6689)
    - `runs/E0938_quick_test402_xattn_evt_stage1siglip_20260213-022855/diagnose.json`: fallback_used_frac≈0.152 → not promoted; skip full test402.

- [x] E0939: XAttn binary eventness Stage-1 (`av_wavlm_clip_xattn_evt`) — full test402 (Stage-1 SigLIP cache; SEEDS=0..9) + diagnose (only if E0938 is promoted)
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 STAGE1_CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch16_siglip_224_webli_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0937_*/best_config.json EVENTNESS=av_wavlm_clip_xattn_evt SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 XATTN_TRAIN_DEVICE=cuda:0 XATTN_VIS_RES=112 XATTN_VIS_FEATS=clip+clipdiff XATTN_EPOCHS=60 XATTN_BS=256 XATTN_EVAL_BS=256 XATTN_LR=2e-3 XATTN_PROJ_DIM=128 XATTN_DROPOUT=0.1 XATTN_EVT_CLIP_LOSS_WEIGHT=0.5 OUT_DIR=runs/E0939_full_test402_xattn_evt_stage1siglip_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0939_*/metrics.json OUT_DIR=runs/E0939_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts: []
  - required_metrics: []
  - results: skipped (E0938 quick test402 not promoted).

- [x] E0940: XAttn binary eventness Stage-1 (`av_wavlm_clip_xattn_evt`) — val402 sweep (Stage-1 DINOv2 cache; SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 STAGE1_CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch14_dinov2_112_160_224_352_448 EVENTNESS=av_wavlm_clip_xattn_evt CANDIDATE_SET=ltl_adaptive_keepadj_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 XATTN_TRAIN_DEVICE=cuda:0 XATTN_VIS_RES=112 XATTN_VIS_FEATS=clip+clipdiff XATTN_EPOCHS=60 XATTN_BS=256 XATTN_EVAL_BS=256 XATTN_LR=2e-3 XATTN_PROJ_DIM=128 XATTN_DROPOUT=0.1 XATTN_EVT_CLIP_LOSS_WEIGHT=0.5 OUT_DIR=runs/E0940_val402_xattn_evt_stage1dinov2_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0940_*/sweep_summary.json`
    - `runs/E0940_*/best_config.json`
    - `runs/E0940_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - results: `runs/E0940_val402_xattn_evt_stage1dinov2_20260213-023423/sweep_summary.json` best=`ltlkeepadj_adj2_shift1_std0p6`, Δ=-0.00191 (p=0.6639) → not promoted.

- [x] E0941: XAttn binary eventness Stage-1 (`av_wavlm_clip_xattn_evt`) — quick test402 (Stage-1 DINOv2 cache; SEEDS=0..2) + diagnose
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 STAGE1_CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch14_dinov2_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0940_*/best_config.json EVENTNESS=av_wavlm_clip_xattn_evt SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 XATTN_TRAIN_DEVICE=cuda:0 XATTN_VIS_RES=112 XATTN_VIS_FEATS=clip+clipdiff XATTN_EPOCHS=60 XATTN_BS=256 XATTN_EVAL_BS=256 XATTN_LR=2e-3 XATTN_PROJ_DIM=128 XATTN_DROPOUT=0.1 XATTN_EVT_CLIP_LOSS_WEIGHT=0.5 OUT_DIR=runs/E0941_quick_test402_xattn_evt_stage1dinov2_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0941_*/metrics.json OUT_DIR=runs/E0941_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts: []
  - required_metrics: []
  - results: skipped (E0940 val402 sweep is negative; do not promote to quick/full).

- [x] E0942: XAttn binary eventness Stage-1 (`av_wavlm_clip_xattn_evt`) — full test402 (Stage-1 DINOv2 cache; SEEDS=0..9) + diagnose
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 STAGE1_CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch14_dinov2_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0940_*/best_config.json EVENTNESS=av_wavlm_clip_xattn_evt SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 XATTN_TRAIN_DEVICE=cuda:0 XATTN_VIS_RES=112 XATTN_VIS_FEATS=clip+clipdiff XATTN_EPOCHS=60 XATTN_BS=256 XATTN_EVAL_BS=256 XATTN_LR=2e-3 XATTN_PROJ_DIM=128 XATTN_DROPOUT=0.1 XATTN_EVT_CLIP_LOSS_WEIGHT=0.5 OUT_DIR=runs/E0942_full_test402_xattn_evt_stage1dinov2_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0942_*/metrics.json OUT_DIR=runs/E0942_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts: []
  - required_metrics: []
  - results: skipped (E0940 val402 sweep not promotable).

- [x] E0943: XAttn binary eventness Stage-1 (`av_wavlm_clip_xattn_evt`) — val402 sweep (Stage-1 vis_res=352; SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_wavlm_clip_xattn_evt CANDIDATE_SET=ltl_adaptive_keepadj_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:2 XATTN_TRAIN_DEVICE=cuda:0 XATTN_VIS_RES=352 XATTN_VIS_FEATS=clip+clipdiff XATTN_EPOCHS=60 XATTN_BS=256 XATTN_EVAL_BS=256 XATTN_LR=2e-3 XATTN_PROJ_DIM=128 XATTN_DROPOUT=0.1 XATTN_EVT_CLIP_LOSS_WEIGHT=0.5 OUT_DIR=runs/E0943_val402_xattn_evt_vis352_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0943_*/sweep_summary.json`
    - `runs/E0943_*/best_config.json`
    - `runs/E0943_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - results: `runs/E0943_val402_xattn_evt_vis352_20260213-024540/sweep_summary.json` best=`ltlkeepadj_adj1_shift1_std0p5`, Δ=-0.00224 (p=0.5038) → not promoted.

- [x] E0944: XAttn binary eventness Stage-1 (`av_wavlm_clip_xattn_evt`) — quick test402 (Stage-1 vis_res=352; SEEDS=0..2) + diagnose
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0943_*/best_config.json EVENTNESS=av_wavlm_clip_xattn_evt SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:2 XATTN_TRAIN_DEVICE=cuda:0 XATTN_VIS_RES=352 XATTN_VIS_FEATS=clip+clipdiff XATTN_EPOCHS=60 XATTN_BS=256 XATTN_EVAL_BS=256 XATTN_LR=2e-3 XATTN_PROJ_DIM=128 XATTN_DROPOUT=0.1 XATTN_EVT_CLIP_LOSS_WEIGHT=0.5 OUT_DIR=runs/E0944_quick_test402_xattn_evt_vis352_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0944_*/metrics.json OUT_DIR=runs/E0944_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts: []
  - required_metrics: []
  - results: skipped (E0943 val402 sweep negative; do not promote to quick/full).

- [x] E0945: XAttn binary eventness Stage-1 (`av_wavlm_clip_xattn_evt`) — full test402 (Stage-1 vis_res=352; SEEDS=0..9) + diagnose
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0943_*/best_config.json EVENTNESS=av_wavlm_clip_xattn_evt SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:2 XATTN_TRAIN_DEVICE=cuda:0 XATTN_VIS_RES=352 XATTN_VIS_FEATS=clip+clipdiff XATTN_EPOCHS=60 XATTN_BS=256 XATTN_EVAL_BS=256 XATTN_LR=2e-3 XATTN_PROJ_DIM=128 XATTN_DROPOUT=0.1 XATTN_EVT_CLIP_LOSS_WEIGHT=0.5 OUT_DIR=runs/E0945_full_test402_xattn_evt_vis352_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0945_*/metrics.json OUT_DIR=runs/E0945_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts: []
  - required_metrics: []
  - results: skipped (E0943 not promotable).

- [x] E0946: WavLM+CLIPdiff vec-MLP Stage-1 (`av_wavlm_clipdiff_vec_mlp`) — val402 sweep (keepadj; SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_wavlm_clipdiff_vec_mlp CANDIDATE_SET=ltl_adaptive_keepadj_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:2 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0946_val402_wavlm_clipdiff_vecmlp_keepadj_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0946_*/sweep_summary.json`
    - `runs/E0946_*/best_config.json`
    - `runs/E0946_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - results: `runs/E0946_val402_wavlm_clipdiff_vecmlp_keepadj_20260213-030005/sweep_summary.json` best=`ltlkeepadj_adj1_shift0_std0p6`, Δ=+0.00283 (p=0.7692) → not promoted.

- [x] E0947: WavLM+CLIPdiff vec-MLP Stage-1 (`av_wavlm_clipdiff_vec_mlp`) — quick test402 (keepadj; SEEDS=0..2) + diagnose
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0946_*/best_config.json EVENTNESS=av_wavlm_clipdiff_vec_mlp SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:2 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0947_quick_test402_wavlm_clipdiff_vecmlp_keepadj_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0947_*/metrics.json OUT_DIR=runs/E0947_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts: []
  - required_metrics: []
  - results: skipped (E0946 not promotable on val402; do not promote to quick/full).

- [x] E0948: WavLM+CLIPdiff vec-MLP Stage-1 (`av_wavlm_clipdiff_vec_mlp`) — full test402 (keepadj; SEEDS=0..9) + diagnose
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0946_*/best_config.json EVENTNESS=av_wavlm_clipdiff_vec_mlp SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:2 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0948_full_test402_wavlm_clipdiff_vecmlp_keepadj_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0948_*/metrics.json OUT_DIR=runs/E0948_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts: []
  - required_metrics: []
  - results: skipped (E0946 not promotable).

- [x] E0949: CLIPdiff vec-MLP Stage-1 (`av_clipdiff_vec_mlp`) — val402 sweep (keepadj; Stage-1 SigLIP cache; SEEDS=0..2)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 STAGE1_CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch16_siglip_224_webli_112_160_224_352_448 EVENTNESS=av_clipdiff_vec_mlp CANDIDATE_SET=ltl_adaptive_keepadj_v1 SEEDS=0,1,2 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:2 OUT_DIR=runs/E0949_val402_vecmlp_keepadj_stage1siglip_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0949_*/sweep_summary.json`
    - `runs/E0949_*/best_config.json`
    - `runs/E0949_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - results: `runs/E0949_val402_vecmlp_keepadj_stage1siglip_20260213-030437/sweep_summary.json` best=`ltlkeepadj_adj2_shift0_std0p6`, Δ=+0.00490 (p=0.6126) → not promoted.

- [x] E0950: CLIPdiff vec-MLP Stage-1 (`av_clipdiff_vec_mlp`) — quick test402 (keepadj; Stage-1 SigLIP cache; SEEDS=0..2) + diagnose
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 STAGE1_CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch16_siglip_224_webli_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0949_*/best_config.json EVENTNESS=av_clipdiff_vec_mlp SEEDS=0,1,2 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:2 OUT_DIR=runs/E0950_quick_test402_vecmlp_keepadj_stage1siglip_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0950_*/metrics.json OUT_DIR=runs/E0950_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts: []
  - required_metrics: []
  - results: skipped (E0949 not promotable on val402; do not promote to quick/full).

- [x] E0951: CLIPdiff vec-MLP Stage-1 (`av_clipdiff_vec_mlp`) — full test402 (keepadj; Stage-1 SigLIP cache; SEEDS=0..9) + diagnose
  - command:
    - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 STAGE1_CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch16_siglip_224_webli_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0949_*/best_config.json EVENTNESS=av_clipdiff_vec_mlp SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:2 OUT_DIR=runs/E0951_full_test402_vecmlp_keepadj_stage1siglip_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0951_*/metrics.json OUT_DIR=runs/E0951_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts: []
  - required_metrics: []
  - results: skipped (E0949 not promotable).

- [x] E0952: Export CACE-Net Stage-1 eventness scores (external; processed frames)
  - command: `python scripts/e0952_export_cace_net_eventness.py --visual-source processed_frames --processed-dir runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed --weights /home/zechuan/.cache/huggingface/hub/models--xianghe--cace-net/snapshots/576350280d49c6fa02d971a671e4992c9ad2f3f7/expLoss_Seed3917_guide_Co-Guide_psai_0.3_Contrastive_True_contras-coeff_1.0__lambda_0.6/model_epoch_46_top1_80.796_task_Supervised_best_model_psai_0.3_lambda_0.6.pth.tar --out-json runs/E0952_export_cace_evt_$(date +%Y%m%d-%H%M%S)/cace_evt_scores.json --device cuda:0 --batch-size 16 --vgg-batch-size 128`
  - configs: []
  - seeds: []
  - required_artifacts:
    - `runs/E0952_*/cace_evt_scores.json`
  - required_metrics: []
  - results: `runs/E0952_export_cace_evt_20260213-040137/cace_evt_scores.json` (unique_vids=4097).

- [x] E0953: CACE-Net Stage-1 (`cace_net_evt`) — val402 sweep (`ltl_adaptive_keepadj_v1`; SEEDS=0..2)
  - command: `CACE_NET_SCORES_JSON=runs/E0952_*/cace_evt_scores.json PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=cace_net_evt CANDIDATE_SET=ltl_adaptive_keepadj_v1 SEEDS=0,1,2 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0953_val402_cace_evt_keepadj_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0953_*/sweep_summary.json`
    - `runs/E0953_*/best_config.json`
    - `runs/E0953_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - results: `runs/E0953_val402_cace_evt_keepadj_20260213-040611/sweep_summary.json` best=`ltlkeepadj_adj1_shift0_std0p6`, Δ=-0.00091 (p=0.8948) → not promoted.

- [x] E0954: CACE-Net Stage-1 (`cace_net_evt`) — val402 sweep (`ltl_gini_v2`; SEEDS=0..2)
  - command: `CACE_NET_SCORES_JSON=runs/E0952_*/cace_evt_scores.json PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=cace_net_evt CANDIDATE_SET=ltl_gini_v2 SEEDS=0,1,2 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0954_val402_cace_evt_gini_v2_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0954_*/sweep_summary.json`
    - `runs/E0954_*/best_config.json`
    - `runs/E0954_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - results: `runs/E0954_val402_cace_evt_gini_v2_20260213-041040/sweep_summary.json` best=`ltlgini2_gini0p5_shift1`, Δ=+0.00224 (p=0.7671) → promote to quick test402 only.

- [x] E0955: CACE-Net Stage-1 (`cace_net_evt`) — quick test402 (from E0954 best; SEEDS=0..2) + diagnose
  - command:
    - `CACE_NET_SCORES_JSON=runs/E0952_*/cace_evt_scores.json PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0954_*/best_config.json EVENTNESS=cace_net_evt SEEDS=0,1,2 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0955_quick_test402_cace_evt_gini_v2_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0955_*/metrics.json OUT_DIR=runs/E0955_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0955_*/metrics.json`
    - `runs/E0955_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)
  - results: `runs/E0955_quick_test402_cace_evt_gini_v2_20260213-041313/metrics.json` anchored=0.72114 vs uniform=0.71294 (Δ=+0.00821; p=0.5097) → not promoted (skip full).

- [x] E0956: CACE-Net Stage-1 (`cace_net_evt`) — val402 sweep (`ltl_top1med_norm_v1`; SEEDS=0..2)
  - command: `CACE_NET_SCORES_JSON=runs/E0952_*/cace_evt_scores.json PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=cace_net_evt CANDIDATE_SET=ltl_top1med_norm_v1 SEEDS=0,1,2 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0956_val402_cace_evt_top1med_norm_v1_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0956_*/sweep_summary.json`
    - `runs/E0956_*/best_config.json`
    - `runs/E0956_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - results: `runs/E0956_val402_cace_evt_top1med_norm_v1_20260213-041404/sweep_summary.json` best=`ltltop1medn_thr0p5_shift0`, Δ=-0.00474 (p=0.5405) → not promoted.

- [x] E0957: Export CACE-Net Stage-1 eventness scores (external; raw video sampling, 4 frames/sec)
  - command: `python scripts/e0952_export_cace_net_eventness.py --visual-source raw_video_sample16 --raw-videos-dir data/AVE/raw/videos --seconds 10 --frames-per-second 4 --weights /home/zechuan/.cache/huggingface/hub/models--xianghe--cace-net/snapshots/576350280d49c6fa02d971a671e4992c9ad2f3f7/expLoss_Seed3917_guide_Co-Guide_psai_0.3_Contrastive_True_contras-coeff_1.0__lambda_0.6/model_epoch_46_top1_80.796_task_Supervised_best_model_psai_0.3_lambda_0.6.pth.tar --out-json runs/E0957_export_cace_evt_rawfps4_$(date +%Y%m%d-%H%M%S)/cace_evt_scores.json --device cuda:0 --batch-size 4 --vgg-batch-size 128`
  - configs: []
  - seeds: []
  - required_artifacts:
    - `runs/E0957_*/cace_evt_scores.json`
  - required_metrics: []
  - results: `runs/E0957_export_cace_evt_rawfps4_20260213-042012/cace_evt_scores.json` (unique_vids=4097).

- [x] E0958: CACE-Net Stage-1 (`cace_net_evt`) — val402 sweep (rawfps4 scores; `ltl_gini_v2`; SEEDS=0..2)
  - command: `CACE_NET_SCORES_JSON=runs/E0957_*/cace_evt_scores.json PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=cace_net_evt CANDIDATE_SET=ltl_gini_v2 SEEDS=0,1,2 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0958_val402_cace_evt_rawfps4_gini_v2_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0958_*/sweep_summary.json`
    - `runs/E0958_*/best_config.json`
    - `runs/E0958_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - results: `runs/E0958_val402_cace_evt_rawfps4_gini_v2_20260213-043133/sweep_summary.json` best=`ltlgini2_gini0p45_shift0`, Δ=-0.00640 (p=0.5632) → not promoted.

- [x] E0960: Export PSP/CPSP AVEL Stage-1 eventness scores (external; processed frames)
  - command: `python scripts/e0960_export_psp_eventness.py --visual-source processed_frames --processed-dir runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed --out-json runs/E0960_export_psp_evt_$(date +%Y%m%d-%H%M%S)/psp_evt_scores.json --device cuda:0 --batch-size 32`
  - configs: []
  - seeds: []
  - required_artifacts:
    - `runs/E0960_*/psp_evt_scores.json`
  - required_metrics: []
  - results: `runs/E0960_export_psp_evt_20260213-050441/psp_evt_scores.json` (unique_vids=4097).

- [x] E0961: PSP/CPSP AVEL Stage-1 (`psp_avel_evt`) — val402 sweep (`ltl_gini_v2`; SEEDS=0..2)
  - command: `PSP_SCORES_JSON=runs/E0960_*/psp_evt_scores.json PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=psp_avel_evt CANDIDATE_SET=ltl_gini_v2 SEEDS=0,1,2 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0961_val402_psp_evt_gini_v2_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0961_*/sweep_summary.json`
    - `runs/E0961_*/best_config.json`
    - `runs/E0961_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - results: `runs/E0961_val402_psp_evt_gini_v2_20260213-050917/sweep_summary.json` best=`ltlgini2_gini0p5_shift0`, Δ=+0.00582 (p=0.5060) → promote to quick test402.

- [x] E0962: PSP/CPSP AVEL Stage-1 (`psp_avel_evt`) — quick test402 (from E0961 best; SEEDS=0..2) + diagnose
  - command:
    - `PSP_SCORES_JSON=runs/E0960_*/psp_evt_scores.json PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0961_*/best_config.json EVENTNESS=psp_avel_evt SEEDS=0,1,2 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0962_quick_test402_psp_evt_gini0p5_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0962_*/metrics.json OUT_DIR=runs/E0962_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0962_*/metrics.json`
    - `runs/E0962_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)
  - results: `runs/E0962_quick_test402_psp_evt_gini0p5_20260213-051204/metrics.json` anchored=0.73060 vs uniform=0.71294 (Δ=+0.01766; p=0.2408) + diagnose=`runs/E0962_quick_test402_psp_evt_gini0p5_20260213-051204/diagnose.json` (fallback_used_frac≈0.811) → promoted to full test402 to check significance.

- [x] E0963: PSP/CPSP AVEL Stage-1 (`psp_avel_evt`) — full test402 (SEEDS=0..9) + diagnose
  - command:
    - `PSP_SCORES_JSON=runs/E0960_*/psp_evt_scores.json PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0961_*/best_config.json EVENTNESS=psp_avel_evt SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0963_full_test402_psp_evt_gini0p5_s0-9_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0963_*/metrics.json OUT_DIR=runs/E0963_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts:
    - `runs/E0963_*/metrics.json`
    - `runs/E0963_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)
  - results: `runs/E0963_full_test402_psp_evt_gini0p5_s0-9_20260213-051328/metrics.json` anchored=0.72983 vs uniform=0.71622 (Δ=+0.01361; p=0.0319) + diagnose=`runs/E0963_full_test402_psp_evt_gini0p5_s0-9_20260213-051328/diagnose.json` (fallback_used_frac≈0.811) → new best full-test Δ but still < +2%.

- [x] E0964: PSP/CPSP AVEL Stage-1 (`psp_avel_evt`) — val402 sweep (`ltl_gini_v1`; SEEDS=0..2)
  - command: `PSP_SCORES_JSON=runs/E0960_*/psp_evt_scores.json PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=psp_avel_evt CANDIDATE_SET=ltl_gini_v1 SEEDS=0,1,2 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0964_val402_psp_evt_gini_v1_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0964_*/sweep_summary.json`
    - `runs/E0964_*/best_config.json`
    - `runs/E0964_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - results: `runs/E0964_val402_psp_evt_gini_v1_20260213-051607/sweep_summary.json` best=`ltl_gini0p20_scoreAlloc`, Δ=-0.00549 (p=0.5628) → not promoted.

- [x] E0965: PSP/CPSP AVEL Stage-1 (`psp_avel_evt`) — val402 sweep (`ltl_top1med_dropfar_v1`; SEEDS=0..2)
  - command: `PSP_SCORES_JSON=runs/E0960_*/psp_evt_scores.json PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=psp_avel_evt CANDIDATE_SET=ltl_top1med_dropfar_v1 SEEDS=0,1,2 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0965_val402_psp_evt_top1med_dropfar_v1_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0965_*/sweep_summary.json`
    - `runs/E0965_*/best_config.json`
    - `runs/E0965_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - results: `runs/E0965_val402_psp_evt_top1med_dropfar_v1_20260213-051848/sweep_summary.json` best=`ltltop1med_thr0p5_shift1_df1`, Δ=+0.00025 (p=0.9804) → not promoted.

- [x] E0966: PSP/CPSP AVEL Stage-1 (`psp_avel_evt`) — val402 sweep (`ltl_gini_dropfar_v1`; SEEDS=0..2)
  - command: `PSP_SCORES_JSON=runs/E0960_*/psp_evt_scores.json PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=psp_avel_evt CANDIDATE_SET=ltl_gini_dropfar_v1 SEEDS=0,1,2 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0966_val402_psp_evt_gini_dropfar_v1_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0966_*/sweep_summary.json`
    - `runs/E0966_*/best_config.json`
    - `runs/E0966_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - results: `runs/E0966_val402_psp_evt_gini_dropfar_v1_20260213-052227/sweep_summary.json` best=`ltlgini_df1_gini0p35_shift1`, Δ=+0.00732 (p=0.4417) → promote to quick test402 only.

- [x] E0967: PSP/CPSP AVEL Stage-1 (`psp_avel_evt`) — quick test402 (from E0966 best; SEEDS=0..2) + diagnose
  - command:
    - `PSP_SCORES_JSON=runs/E0960_*/psp_evt_scores.json PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0966_*/best_config.json EVENTNESS=psp_avel_evt SEEDS=0,1,2 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0967_quick_test402_psp_evt_gini_dropfar_best_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0967_*/metrics.json OUT_DIR=runs/E0967_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0967_*/metrics.json`
    - `runs/E0967_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)
  - results: `runs/E0967_quick_test402_psp_evt_gini_dropfar_best_20260213-052700/metrics.json` anchored=0.72231 vs uniform=0.71294 (Δ=+0.00937; p=0.4846) → not promoted (skip full).

- [x] E0970: PSP/CPSP AVEL Stage-1 (`psp_avel_evt`) — val402 sweep (`ltl_top1med_visfb_v1`; SEEDS=0..2)
  - command: `PSP_SCORES_JSON=runs/E0960_*/psp_evt_scores.json PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=psp_avel_evt CANDIDATE_SET=ltl_top1med_visfb_v1 SEEDS=0,1,2 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0970_val402_psp_evt_top1med_visfb_v1_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0970_*/sweep_summary.json`
    - `runs/E0970_*/best_config.json`
    - `runs/E0970_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - results: `runs/E0970_val402_psp_evt_top1med_visfb_v1_20260213-054752/sweep_summary.json` best=`ltltop1med_uniformfb_shift1`, Δ=-0.00665 (p=0.4460) → not promoted.

- [x] E0971: PSP/CPSP AVEL Stage-1 (`psp_avel_evt`) — val402 sweep (`ltl_gini_visfb_v1`; SEEDS=0..2)
  - command: `PSP_SCORES_JSON=runs/E0960_*/psp_evt_scores.json PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=psp_avel_evt CANDIDATE_SET=ltl_gini_visfb_v1 SEEDS=0,1,2 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0971_val402_psp_evt_gini_visfb_v1_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0971_*/sweep_summary.json`
    - `runs/E0971_*/best_config.json`
    - `runs/E0971_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - results: `runs/E0971_val402_psp_evt_gini_visfb_v1_20260213-055449/sweep_summary.json` best=`ltlgini_visfb_uniform_shift0`, Δ=+0.00582 (p=0.5060) → not promoted (visual fallback is harmful here).

- [x] E0972: Export PSP/CPSP AVEL Stage-1 eventness scores (raw videos; `raw_video_avg16`; sharded; attempted)
  - command (5 shards; merge would be done after):
    - `OUT_DIR=runs/E0972_export_psp_evt_rawfps4_$(date +%Y%m%d-%H%M%S); mkdir -p ${OUT_DIR}; for i in 0 1 2 3 4; do python scripts/e0960_export_psp_eventness.py --processed-dir runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed --visual-source raw_video_avg16 --raw-videos-dir data/AVE/raw/videos --sample-num 4 --shard-idx ${i} --num-shards 5 --device cuda:0 --batch-size 32 --out-json ${OUT_DIR}/psp_evt_scores_shard${i}.json; done`
  - configs: []
  - seeds: []
  - required_artifacts:
    - `runs/E0972_*/psp_evt_scores.json`
  - required_metrics: []
  - results: `runs/E0972_export_psp_evt_rawfps4_20260213-060205/` produced no JSON outputs (OUT_DIR empty); shard logs: `artifacts/experiments/E0972/shard*.log`.

- [x] E0973: PSP/CPSP AVEL Stage-1 (`psp_avel_evt`) — val402 sweep (`ltl_gap_v1`; SEEDS=0..2)
  - command: `PSP_SCORES_JSON=runs/E0960_*/psp_evt_scores.json PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=psp_avel_evt CANDIDATE_SET=ltl_gap_v1 SEEDS=0,1,2 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0973_val402_psp_evt_gap_v1_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0973_*/sweep_summary.json`
    - `runs/E0973_*/best_config.json`
    - `runs/E0973_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - results: `runs/E0973_val402_psp_evt_gap_v1_20260214-025354/sweep_summary.json` best=`ltlgap1_gap0p5_shift0`, Δ=-0.00249 (p=0.7516) → not promoted.

- [x] E0974: PSP/CPSP AVEL Stage-1 (`psp_avel_evt`) — val402 sweep (`ltl_gini_keepadj_v1`; SEEDS=0..2)
  - command: `PSP_SCORES_JSON=runs/E0960_*/psp_evt_scores.json PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=psp_avel_evt CANDIDATE_SET=ltl_gini_keepadj_v1 SEEDS=0,1,2 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0974_val402_psp_evt_gini_keepadj_v1_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0974_*/sweep_summary.json`
    - `runs/E0974_*/best_config.json`
    - `runs/E0974_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - results: `runs/E0974_val402_psp_evt_gini_keepadj_v1_20260214-025940/sweep_summary.json` best=`ltlgini_keepadj_df1_gini0p45_shift0`, Δ=+0.00623 (p=0.3742) → promote to quick test402.

- [x] E0975: PSP/CPSP AVEL Stage-1 (`psp_avel_evt`) — quick test402 (from E0974 best; SEEDS=0..2) + diagnose
  - command:
    - `PSP_SCORES_JSON=runs/E0960_*/psp_evt_scores.json PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0974_*/best_config.json EVENTNESS=psp_avel_evt SEEDS=0,1,2 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0975_quick_test402_psp_evt_gini_keepadj_best_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0975_*/metrics.json OUT_DIR=runs/E0975_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0975_*/metrics.json`
    - `runs/E0975_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)
  - results: `runs/E0975_quick_test402_psp_evt_gini_keepadj_best_20260214-030312/metrics.json` anchored=0.73441 vs uniform=0.71294 (Δ=+0.02148; p=0.1307) + diagnose=`runs/E0975_quick_test402_psp_evt_gini_keepadj_best_20260214-030312/diagnose.json` (fallback_used_frac≈0.709) → promoted to full test402.

- [x] E0976: PSP/CPSP AVEL Stage-1 (`psp_avel_evt`) — full test402 (from E0974 best; SEEDS=0..9) + diagnose
  - command:
    - `PSP_SCORES_JSON=runs/E0960_*/psp_evt_scores.json PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0974_*/best_config.json EVENTNESS=psp_avel_evt SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0976_full_test402_psp_evt_gini_keepadj_best_s0-9_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0976_*/metrics.json OUT_DIR=runs/E0976_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts:
    - `runs/E0976_*/metrics.json`
    - `runs/E0976_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)
  - results: `runs/E0976_full_test402_psp_evt_gini_keepadj_best_s0-9_20260214-030359/metrics.json` anchored=0.73348 vs uniform=0.71622 (Δ=+0.01726; p=0.00167) + diagnose=`runs/E0976_full_test402_psp_evt_gini_keepadj_best_s0-9_20260214-030359/diagnose.json` (fallback_used_frac≈0.709) → still < +2%.

- [x] E0977: PSP/CPSP AVEL Stage-1 (`psp_avel_evt`) — val402 sweep (`ltl_gini_keepadj_basealloc_v1`; SEEDS=0..2)
  - command: `PSP_SCORES_JSON=runs/E0960_*/psp_evt_scores.json PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=psp_avel_evt CANDIDATE_SET=ltl_gini_keepadj_basealloc_v1 SEEDS=0,1,2 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0977_val402_psp_evt_gini_keepadj_basealloc_v1_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0977_*/sweep_summary.json`
    - `runs/E0977_*/best_config.json`
    - `runs/E0977_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - results: `runs/E0977_val402_psp_evt_gini_keepadj_basealloc_v1_20260214-030703/sweep_summary.json` best=`ltlgini_keepadj_thr0p45_df1_shift0_distance`, Δ=+0.00623 (p=0.3742) → no improvement; not promoted.

- [x] E0978: PSP/CPSP AVEL Stage-1 (`psp_avel_evt`) — val402 sweep (`ltl_gini_keepadj_hconf_v1`; SEEDS=0..2)
  - command: `PSP_SCORES_JSON=runs/E0960_*/psp_evt_scores.json PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=psp_avel_evt CANDIDATE_SET=ltl_gini_keepadj_hconf_v1 SEEDS=0,1,2 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0978_val402_psp_evt_gini_keepadj_hconf_v1_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0978_*/sweep_summary.json`
    - `runs/E0978_*/best_config.json`
    - `runs/E0978_*/eventness_scores.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`
  - results: `runs/E0978_val402_psp_evt_gini_keepadj_hconf_v1_20260214-030933/sweep_summary.json` best=`ltlgini_keepadj_gini0p45_hconf0p5`, Δ=+0.00765 (p=0.6190) → promote to quick test402.

- [x] E0979: PSP/CPSP AVEL Stage-1 (`psp_avel_evt`) — quick test402 (from E0978 best; SEEDS=0..2) + diagnose
  - command:
    - `PSP_SCORES_JSON=runs/E0960_*/psp_evt_scores.json PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0978_*/best_config.json EVENTNESS=psp_avel_evt SEEDS=0,1,2 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0979_quick_test402_psp_evt_gini_keepadj_hconf_best_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0979_*/metrics.json OUT_DIR=runs/E0979_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0979_*/metrics.json`
    - `runs/E0979_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ)
  - results: `runs/E0979_quick_test402_psp_evt_gini_keepadj_hconf_best_20260214-031126/metrics.json` anchored=0.73806 vs uniform=0.71294 (Δ=+0.02512; p=0.1567) + diagnose=`runs/E0979_quick_test402_psp_evt_gini_keepadj_hconf_best_20260214-031126/diagnose.json` (fallback_used_frac≈0.709) → promoted to full test402.

- [x] E0980: PSP/CPSP AVEL Stage-1 (`psp_avel_evt`) — full test402 (from E0978 best; SEEDS=0..9) + diagnose (**C0003 proven**)
  - command:
    - `PSP_SCORES_JSON=runs/E0960_*/psp_evt_scores.json PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0978_*/best_config.json EVENTNESS=psp_avel_evt SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:0 OUT_DIR=runs/E0980_full_test402_psp_evt_gini_keepadj_hconf_best_s0-9_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
    - `IN_METRICS=runs/E0980_*/metrics.json OUT_DIR=runs/E0980_* bash scripts/e0344_ave_p0_diagnose.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts:
    - `runs/E0980_*/metrics.json`
    - `runs/E0980_*/diagnose.json`
  - required_metrics:
    - `metrics.json`: `paired_ttest.anchored_vs_uniform.p`, `summary.anchored_top2.mean`, `summary.uniform.mean` (report Δ; require Δ≥+0.02 and p<0.05)
  - results: `runs/E0980_full_test402_psp_evt_gini_keepadj_hconf_best_s0-9_20260214-031741/metrics.json` anchored=0.73791 vs uniform=0.71622 (Δ=+0.02169; p=0.00149) + diagnose=`runs/E0980_full_test402_psp_evt_gini_keepadj_hconf_best_s0-9_20260214-031741/diagnose.json` (fallback_used_frac≈0.709) → **C0003 hard gate met**.

### Run Queue (Long-Video QA; sequential)
- [x] E0600 (real; ppl): IntentQA VLM evaluation under budgeted frame selection (val n=253; seed=0)
  - command: `OUT_DIR=runs/E0600_intentqa_vlm_eval_full_20260210-041911 SPLIT=val LIMIT=256 METHODS=uniform,random,audio,cheap_visual,fused,ql2l_clap,ql2l_asr_bm25 B_FRAMES=16 MAX_SECONDS=120 SEED=0 STRATEGY=ppl DEVICE=cuda:1 DTYPE=bfloat16 QL2L_CLAP_DEVICE=cuda:2 QL2L_ASR_DEVICE=cpu ALLOW_MISSING_VIDEOS=1 MIN_ITEMS=250 bash scripts/e0600_intentqa_vlm_eval.sh`
  - configs: []
  - seeds: [0]
  - required_artifacts:
    - `runs/E0600_intentqa_vlm_eval_full_20260210-041911/metrics.json`
    - `runs/E0600_intentqa_vlm_eval_full_20260210-041911/predictions.jsonl`
    - `runs/E0600_intentqa_vlm_eval_full_20260210-041911/preprocess_meta.json`
  - required_metrics:
    - `metrics.json`: `summary[*].{acc,invalid_rate}`, `delta_vs_uniform`, `skipped_videos`
  - logs:
    - `artifacts/experiments/E0600_full_ppl/run.log` (aborted at 80/253; no artifacts written)
    - `artifacts/experiments/E0600_full_ppl_rerun2/run.log` (full)
  - backfill: done (see `### E0600`).

- [x] E0601 (real; ppl; ql2l_clap): IntentQA faithfulness proxy (delete-and-predict; budget-matched) (val n=253; seed=0)
  - command: `OUT_DIR=runs/E0601_intentqa_faithfulness_full_20260210-061137 SPLIT=val LIMIT=256 METHOD=ql2l_clap B_FRAMES=16 MAX_SECONDS=120 SEED=0 STRATEGY=ppl DEVICE=cuda:1 DTYPE=bfloat16 QL2L_CLAP_DEVICE=cuda:2 QL2L_ASR_DEVICE=cpu ALLOW_MISSING_VIDEOS=1 MIN_ITEMS=250 bash scripts/e0601_intentqa_faithfulness.sh`
  - configs: []
  - seeds: [0]
  - required_artifacts:
    - `runs/E0601_intentqa_faithfulness_full_20260210-061137/faithfulness.json`
    - `runs/E0601_intentqa_faithfulness_full_20260210-061137/rows.jsonl`
    - `runs/E0601_intentqa_faithfulness_full_20260210-061137/preprocess_meta.json`
  - required_metrics:
    - `faithfulness.json`: `accuracy`, `accuracy_deleted`, `acc_drop`, `pred_change_rate`, `invalid_rate`
  - logs: `artifacts/experiments/E0601_full_ql2l_clap_ppl/run.log`
  - backfill: done (see `### E0601`).

- [x] E0602 (real; ppl; CONFIG=Subset): EgoSchema VLM eval/pred generation (labeled subset) (test n=256; seed=0)
  - command: `OUT_DIR=runs/E0602_egoschema_eval_subset_full_20260210-064250 CONFIG=Subset SPLIT=test LIMIT=256 METHODS=uniform,ql2l_clap,ql2l_asr_bm25 B_FRAMES=16 MAX_SECONDS=120 SEED=0 STRATEGY=ppl DEVICE=cuda:1 DTYPE=bfloat16 QL2L_CLAP_DEVICE=cuda:2 QL2L_ASR_DEVICE=cpu bash scripts/e0602_egoschema_predict.sh`
  - configs: []
  - seeds: [0]
  - required_artifacts:
    - `runs/E0602_egoschema_eval_subset_full_20260210-064250/metrics.json`
    - `runs/E0602_egoschema_eval_subset_full_20260210-064250/predictions.jsonl`
    - `runs/E0602_egoschema_eval_subset_full_20260210-064250/preprocess_meta.json`
  - required_metrics:
    - `metrics.json`: `summary[*].acc` (not null for all methods), `summary[*].invalid_rate`
  - logs: `artifacts/experiments/E0602_full_subset_ppl/run.log`
  - backfill: done (see `### E0602`).

### Run Queue (Long-Video QA; extended; sequential)
- [x] E0604 (real; ppl; seeds=0,1): IntentQA VLM evaluation under budgeted frame selection (val; multi-seed)
  - command:
    - `OUT_DIR=runs/E0604_intentqa_vlm_eval_val_s0_20260210-125048 SPLIT=val LIMIT=256 METHODS=uniform,random,audio,cheap_visual,fused,ql2l_clap,ql2l_asr_bm25 B_FRAMES=16 MAX_SECONDS=120 SEED=0 STRATEGY=ppl DEVICE=cuda:1 DTYPE=bfloat16 QL2L_CLAP_DEVICE=cuda:2 QL2L_ASR_DEVICE=cpu ALLOW_MISSING_VIDEOS=1 MIN_ITEMS=250 bash scripts/e0600_intentqa_vlm_eval.sh`
    - `OUT_DIR=runs/E0604_intentqa_vlm_eval_val_s1_20260210-125048 SPLIT=val LIMIT=256 METHODS=uniform,random,cheap_visual,ql2l_clap B_FRAMES=16 MAX_SECONDS=120 SEED=1 STRATEGY=ppl DEVICE=cuda:1 DTYPE=bfloat16 QL2L_CLAP_DEVICE=cuda:2 QL2L_ASR_DEVICE=cpu ALLOW_MISSING_VIDEOS=1 MIN_ITEMS=250 bash scripts/e0600_intentqa_vlm_eval.sh`
  - configs: []
  - seeds: [0, 1]
  - required_artifacts:
    - `runs/E0604_intentqa_vlm_eval_val_s0_20260210-125048/metrics.json`
    - `runs/E0604_intentqa_vlm_eval_val_s0_20260210-125048/predictions.jsonl`
    - `runs/E0604_intentqa_vlm_eval_val_s0_20260210-125048/preprocess_meta.json`
    - `runs/E0604_intentqa_vlm_eval_val_s1_20260210-125048/metrics.json`
    - `runs/E0604_intentqa_vlm_eval_val_s1_20260210-125048/predictions.jsonl`
    - `runs/E0604_intentqa_vlm_eval_val_s1_20260210-125048/preprocess_meta.json`
  - required_metrics:
    - `metrics.json`: `summary[*].{acc,invalid_rate}`, `delta_vs_uniform`, `skipped_videos`
  - logs:
    - `artifacts/experiments/E0604_val_s0/run.log`
    - `artifacts/experiments/E0604_val_s1/run.log`
  - backfill: done (see `### E0604`).

- [x] E0605 (real; ppl; seeds=0,1; CONFIG=Subset full): EgoSchema VLM eval (labeled Subset; full n=500)
  - command:
    - `OUT_DIR=runs/E0605_egoschema_eval_subset500_s0_20260210-125048 CONFIG=Subset SPLIT=test LIMIT=0 METHODS=uniform,ql2l_clap,ql2l_asr_bm25 B_FRAMES=16 MAX_SECONDS=120 SEED=0 STRATEGY=ppl DEVICE=cuda:1 DTYPE=bfloat16 QL2L_CLAP_DEVICE=cuda:2 QL2L_ASR_DEVICE=cpu bash scripts/e0602_egoschema_predict.sh`
    - `OUT_DIR=runs/E0605_egoschema_eval_subset500_s1_20260210-183504 CONFIG=Subset SPLIT=test LIMIT=0 METHODS=uniform,ql2l_clap,ql2l_asr_bm25 B_FRAMES=16 MAX_SECONDS=120 SEED=1 STRATEGY=ppl DEVICE=cuda:1 DTYPE=bfloat16 QL2L_CLAP_DEVICE=cuda:2 QL2L_ASR_DEVICE=cpu bash scripts/e0602_egoschema_predict.sh`
  - configs: []
  - seeds: [0, 1]
  - required_artifacts:
    - `runs/E0605_egoschema_eval_subset500_s0_20260210-125048/metrics.json`
    - `runs/E0605_egoschema_eval_subset500_s0_20260210-125048/predictions.jsonl`
    - `runs/E0605_egoschema_eval_subset500_s0_20260210-125048/preprocess_meta.json`
    - `runs/E0605_egoschema_eval_subset500_s1_20260210-183504/metrics.json`
    - `runs/E0605_egoschema_eval_subset500_s1_20260210-183504/predictions.jsonl`
    - `runs/E0605_egoschema_eval_subset500_s1_20260210-183504/preprocess_meta.json`
  - required_metrics:
    - `metrics.json`: `summary[*].acc` (not null for all methods), `summary[*].invalid_rate`
  - logs:
    - `artifacts/experiments/E0605_subset500_s0/run.log`
    - `artifacts/experiments/E0605_subset500_s0_resume1/run.log` (resume)
    - `artifacts/experiments/E0605_subset500_s1/run.log`
  - backfill: done (see `### E0605`).

- [x] E0607 (real; ppl; seed=1; ql2l_clap): IntentQA faithfulness proxy (delete-and-predict; budget-matched) (val n=253)
  - command: `OUT_DIR=runs/E0607_intentqa_faithfulness_val_s1_20260210-194732 SPLIT=val LIMIT=256 METHOD=ql2l_clap B_FRAMES=16 MAX_SECONDS=120 SEED=1 STRATEGY=ppl DEVICE=cuda:1 DTYPE=bfloat16 QL2L_CLAP_DEVICE=cuda:2 QL2L_ASR_DEVICE=cpu ALLOW_MISSING_VIDEOS=1 MIN_ITEMS=250 bash scripts/e0601_intentqa_faithfulness.sh`
  - configs: []
  - seeds: [1]
  - required_artifacts:
    - `runs/E0607_intentqa_faithfulness_val_s1_20260210-194732/faithfulness.json`
    - `runs/E0607_intentqa_faithfulness_val_s1_20260210-194732/rows.jsonl`
    - `runs/E0607_intentqa_faithfulness_val_s1_20260210-194732/preprocess_meta.json`
  - required_metrics:
    - `faithfulness.json`: `accuracy`, `accuracy_deleted`, `acc_drop`, `pred_change_rate`, `invalid_rate`
  - logs: `artifacts/experiments/E0607_intentqa_faithfulness_val_s1/run.log`
  - backfill: done (see `### E0607`).

- [x] E0608 (real; ppl; seed=1; CONFIG=Subset): EgoSchema VLM eval/pred generation (labeled subset) (test n=256)
  - command: `OUT_DIR=runs/E0608_egoschema_eval_subset256_s1_20260210-201700 CONFIG=Subset SPLIT=test LIMIT=256 METHODS=uniform,ql2l_clap,ql2l_asr_bm25 B_FRAMES=16 MAX_SECONDS=120 SEED=1 STRATEGY=ppl DEVICE=cuda:1 DTYPE=bfloat16 QL2L_CLAP_DEVICE=cuda:2 QL2L_ASR_DEVICE=cpu bash scripts/e0602_egoschema_predict.sh`
  - configs: []
  - seeds: [1]
  - required_artifacts:
    - `runs/E0608_egoschema_eval_subset256_s1_20260210-201700/metrics.json`
    - `runs/E0608_egoschema_eval_subset256_s1_20260210-201700/predictions.jsonl`
    - `runs/E0608_egoschema_eval_subset256_s1_20260210-201700/preprocess_meta.json`
  - required_metrics:
    - `metrics.json`: `summary[*].acc` (not null for all methods), `summary[*].invalid_rate`
  - logs: `artifacts/experiments/E0608_egoschema_eval_subset256_s1/run.log`
  - backfill: done (see `### E0608`).

- [x] E0606 (real; ppl; seed=0; CONFIG=Subset full; +ql2l_clip): EgoSchema VLM eval (labeled Subset; full n=500)
  - command: `OUT_DIR=runs/E0606_egoschema_eval_subset500_clip_$(date +%Y%m%d-%H%M%S) CONFIG=Subset SPLIT=test LIMIT=0 METHODS=uniform,ql2l_clap,ql2l_asr_bm25,ql2l_clip B_FRAMES=16 MAX_SECONDS=120 SEED=0 STRATEGY=ppl DEVICE=cuda:1 DTYPE=bfloat16 QL2L_CLAP_DEVICE=cuda:2 QL2L_ASR_DEVICE=cpu QL2L_CLIP_DEVICE=cuda:3 bash scripts/e0602_egoschema_predict.sh`
  - configs: []
  - seeds: [0]
  - required_artifacts:
    - `runs/E0606_egoschema_eval_subset500_clip_20260211-031138/metrics.json`
    - `runs/E0606_egoschema_eval_subset500_clip_20260211-031138/predictions.jsonl`
    - `runs/E0606_egoschema_eval_subset500_clip_20260211-031138/preprocess_meta.json`
  - required_metrics:
    - `metrics.json`: `summary[*].acc` (not null for all methods), `summary[*].invalid_rate`
  - logs: `artifacts/experiments/E0606/run.log`
  - backfill: done (see `### E0606`).

- [x] E0618 (real; ppl; seed=0; text_only): EgoSchema language-bias baseline (uniform vs text_only; Subset n=500)
  - command: `OUT_DIR=runs/E0618_egoschema_eval_subset500_text_only_$(date +%Y%m%d-%H%M%S) CONFIG=Subset SPLIT=test LIMIT=0 METHODS=uniform,text_only B_FRAMES=16 MAX_SECONDS=120 SEED=0 STRATEGY=ppl DEVICE=cuda:1 DTYPE=bfloat16 QL2L_CLAP_DEVICE=cpu QL2L_ASR_DEVICE=cpu QL2L_CLIP_DEVICE=cpu bash scripts/e0602_egoschema_predict.sh`
  - configs: []
  - seeds: [0]
  - required_artifacts:
    - `runs/E0618_egoschema_eval_subset500_text_only_20260211-055131/metrics.json`
    - `runs/E0618_egoschema_eval_subset500_text_only_20260211-055131/predictions.jsonl`
    - `runs/E0618_egoschema_eval_subset500_text_only_20260211-055131/preprocess_meta.json`
  - required_metrics:
    - `metrics.json`: `summary[*].acc` (not null), `summary[*].invalid_rate`
  - logs: `artifacts/experiments/E0618/run.log`
  - backfill: done (see `### E0618`).

- [x] E0619 (analysis): QA bucket report (“when does audio help?”) on Long-Video QA add-on outputs
  - command: `OUT_DIR=runs/E0619_qa_bucket_report_20260211-062907 bash scripts/e0619_qa_bucket_report.sh`
  - configs: []
  - seeds: []
  - required_artifacts:
    - `runs/E0619_qa_bucket_report_20260211-062907/intentqa/bucket_report.json`
    - `runs/E0619_qa_bucket_report_20260211-062907/intentqa/bucket_report.md`
    - `runs/E0619_qa_bucket_report_20260211-062907/avqa/bucket_report.json`
    - `runs/E0619_qa_bucket_report_20260211-062907/avqa/bucket_report.md`
    - `runs/E0619_qa_bucket_report_20260211-062907/egoschema/bucket_report.json`
    - `runs/E0619_qa_bucket_report_20260211-062907/egoschema/bucket_report.md`
  - required_metrics:
    - `bucket_report.json`: `overall[*].acc` + per-bucket `delta_vs_uniform` for the primary method.
  - logs: `artifacts/experiments/E0619/run.log`
  - backfill: done (see `### E0619`).

- [x] E0609 (real; ppl; seed=0; +ql2l_clip): IntentQA VLM evaluation under budgeted frame selection (val n=253)
  - command: `OUT_DIR=runs/E0609_intentqa_vlm_eval_val_clip_20260211-011407 SPLIT=val LIMIT=256 METHODS=uniform,random,audio,cheap_visual,fused,ql2l_clap,ql2l_asr_bm25,ql2l_clip B_FRAMES=16 MAX_SECONDS=120 SEED=0 STRATEGY=ppl DEVICE=cuda:1 DTYPE=bfloat16 QL2L_CLAP_DEVICE=cuda:2 QL2L_ASR_DEVICE=cpu QL2L_CLIP_DEVICE=cuda:3 ALLOW_MISSING_VIDEOS=1 MIN_ITEMS=250 bash scripts/e0600_intentqa_vlm_eval.sh`
  - configs: []
  - seeds: [0]
  - required_artifacts:
    - `runs/E0609_intentqa_vlm_eval_val_clip_20260211-011407/metrics.json`
    - `runs/E0609_intentqa_vlm_eval_val_clip_20260211-011407/predictions.jsonl`
    - `runs/E0609_intentqa_vlm_eval_val_clip_20260211-011407/preprocess_meta.json`
  - required_metrics:
    - `metrics.json`: `summary[*].{acc,invalid_rate}`, `delta_vs_uniform`, `skipped_videos`
  - logs: `artifacts/experiments/E0609/run.log`
  - backfill: done (see `### E0609`).

- [x] E0617 (real; ppl; seed=0; text_only): IntentQA language-bias baseline (uniform vs text_only; val n=253)
  - command: `OUT_DIR=runs/E0617_intentqa_vlm_eval_val_text_only_20260211-053301 SPLIT=val LIMIT=256 METHODS=uniform,text_only B_FRAMES=16 MAX_SECONDS=120 SEED=0 STRATEGY=ppl DEVICE=cuda:1 DTYPE=bfloat16 QL2L_CLAP_DEVICE=cpu QL2L_ASR_DEVICE=cpu QL2L_CLIP_DEVICE=cpu ALLOW_MISSING_VIDEOS=1 MIN_ITEMS=250 bash scripts/e0600_intentqa_vlm_eval.sh`
  - configs: []
  - seeds: [0]
  - required_artifacts:
    - `runs/E0617_intentqa_vlm_eval_val_text_only_20260211-053301/metrics.json`
    - `runs/E0617_intentqa_vlm_eval_val_text_only_20260211-053301/predictions.jsonl`
    - `runs/E0617_intentqa_vlm_eval_val_text_only_20260211-053301/preprocess_meta.json`
  - required_metrics:
    - `metrics.json`: `summary[*].{acc,invalid_rate}`, `delta_vs_uniform`, `skipped_videos`
  - logs: `artifacts/experiments/E0617/run.log`
  - backfill: done (see `### E0617`).

- [x] E0615 (real; ppl; seed=0): AVQA VLM evaluation under budgeted frame selection (val subset; download drift allowed)
  - command: `OUT_DIR=runs/E0615_avqa_vlm_eval_val_$(date +%Y%m%d-%H%M%S) SPLIT=val LIMIT=256 METHODS=uniform,random,audio,cheap_visual,fused,ql2l_clap,ql2l_asr_bm25,ql2l_clip,text_only B_FRAMES=16 MAX_SECONDS=120 SEED=0 STRATEGY=ppl DEVICE=cuda:1 DTYPE=bfloat16 QL2L_CLAP_DEVICE=cuda:2 QL2L_ASR_DEVICE=cpu QL2L_CLIP_DEVICE=cuda:3 ALLOW_MISSING_VIDEOS=1 MIN_ITEMS=200 bash scripts/e0615_avqa_vlm_eval.sh`
  - configs: []
  - seeds: [0]
  - required_artifacts:
    - `runs/E0615_avqa_vlm_eval_val_20260211-043508/metrics.json`
    - `runs/E0615_avqa_vlm_eval_val_20260211-043508/predictions.jsonl`
    - `runs/E0615_avqa_vlm_eval_val_20260211-043508/preprocess_meta.json`
  - required_metrics:
    - `metrics.json`: `summary[*].{acc,invalid_rate}`, `delta_vs_uniform`, `skipped_videos`
  - logs: `artifacts/experiments/E0615/run.log`
  - backfill: done (see `### E0615`).

- [x] E0616 (real; ppl; seed=0; tight budget): AVQA VLM evaluation under budgeted frame selection (`B_FRAMES=4`; selection methods must diverge)
  - command: `OUT_DIR=runs/E0616_avqa_vlm_eval_val_b4_$(date +%Y%m%d-%H%M%S) SPLIT=val LIMIT=256 METHODS=uniform,random,audio,cheap_visual,fused,ql2l_clap,ql2l_asr_bm25,ql2l_clip,text_only B_FRAMES=4 MAX_SECONDS=120 SEED=0 STRATEGY=ppl DEVICE=cuda:1 DTYPE=bfloat16 QL2L_CLAP_DEVICE=cuda:2 QL2L_ASR_DEVICE=cpu QL2L_CLIP_DEVICE=cuda:3 ALLOW_MISSING_VIDEOS=1 MIN_ITEMS=200 bash scripts/e0615_avqa_vlm_eval.sh`
  - configs: []
  - seeds: [0]
  - required_artifacts:
    - `runs/E0616_avqa_vlm_eval_val_b4_20260211-051556/metrics.json`
    - `runs/E0616_avqa_vlm_eval_val_b4_20260211-051556/predictions.jsonl`
    - `runs/E0616_avqa_vlm_eval_val_b4_20260211-051556/preprocess_meta.json`
  - required_metrics:
    - `metrics.json`: `summary[*].{acc,invalid_rate}`, `delta_vs_uniform`, `skipped_videos`
  - logs: `artifacts/experiments/E0616/run.log`
  - backfill: done (see `### E0616`).

- [x] E0710 (real; C0003 queue; val402 sweep): dense-stride flow-MLP rerun with `ltl_adaptive_keepadj_v1` on official val402 (`SEEDS=0..2`)
  - command: `EVENTNESS=av_clipdiff_flow_mlp CANDIDATE_SET=ltl_adaptive_keepadj_v1 SEEDS=0,1,2 PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 IDS_TRAIN=data/AVE/meta/download_ok_train_official.txt IDS_EVAL=data/AVE/meta/download_ok_val_official.txt OUT_DIR=runs/E0710_val402_flowmlp_keepadj_$(date +%Y%m%d-%H%M%S) bash scripts/e0393_ave_p0_sweep_official_val_ltl_top1med_norm_v1_av_clipdiff_flow_mlp.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0710_val402_flowmlp_keepadj_20260212-000010/sweep_summary.json`
    - `runs/E0710_val402_flowmlp_keepadj_20260212-000010/best_config.json`
    - `runs/E0710_val402_flowmlp_keepadj_20260212-000010/ltlkeepadj_adj2_shift0_std0p5/metrics.json`
  - required_metrics:
    - `sweep_summary.json`: `best.anchored_minus_uniform_mean`, `best.anchored_vs_uniform_p`, `best.metrics_path`
  - logs: `artifacts/experiments/E0710/run.log`
  - backfill: done (see Results table `E0710` row).

- [x] E0711 (real; C0003 queue; quick test402): quick-transfer gate for E0710 val winner (`SEEDS=0..2`)
  - command: `BEST_CONFIG_JSON=runs/E0710_val402_flowmlp_keepadj_20260212-000010/best_config.json IDS_TRAIN=data/AVE/meta/download_ok_train_official.txt IDS_EVAL=data/AVE/meta/download_ok_test_official.txt PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 SEEDS=0,1,2 OUT_DIR=runs/E0711_quick_test402_flowmlp_keepadj_$(date +%Y%m%d-%H%M%S) bash scripts/e0394_ave_p0_best_to_test_quick_official_av_clipdiff_flow_mlp.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0711_quick_test402_flowmlp_keepadj_20260212-000606/metrics.json`
    - `runs/E0711_quick_test402_flowmlp_keepadj_20260212-000606/diagnose.json`
  - required_metrics:
    - `metrics.json`: `summary.{anchored_top2,uniform}.mean`, `paired_ttest.anchored_vs_uniform.p`
  - logs: `artifacts/experiments/E0711/run.log`
  - backfill: done (see Results table `E0711` row).

- [x] E0712 (real; C0003 queue; full test402): full test402 rerun for E0710 val winner (`SEEDS=0..9`)
  - command: `BEST_CONFIG_JSON=runs/E0710_val402_flowmlp_keepadj_20260212-000010/best_config.json IDS_TRAIN=data/AVE/meta/download_ok_train_official.txt IDS_EVAL=data/AVE/meta/download_ok_test_official.txt PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 SEEDS=0,1,2,3,4,5,6,7,8,9 OUT_DIR=runs/E0712_full_test402_flowmlp_keepadj_$(date +%Y%m%d-%H%M%S) bash scripts/e0395_ave_p0_best_to_test_full_official_av_clipdiff_flow_mlp.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts:
    - `runs/E0712_full_test402_flowmlp_keepadj_20260212-000835/metrics.json`
    - `runs/E0712_full_test402_flowmlp_keepadj_20260212-000835/diagnose.json`
  - required_metrics:
    - `metrics.json`: `summary.{anchored_top2,uniform}.mean`, `paired_ttest.anchored_vs_uniform.p`
  - logs: `artifacts/experiments/E0712/run.log`
  - backfill: done (see Results table `E0712` row).

- [x] E0713 (real; Long-Video QA extension; seed=2): IntentQA faithfulness rerun (`ql2l_clap`, val n=253)
  - command: `OUT_DIR=runs/E0713_intentqa_faithfulness_val_s2_$(date +%Y%m%d-%H%M%S) SPLIT=val LIMIT=256 METHOD=ql2l_clap B_FRAMES=16 MAX_SECONDS=120 SEED=2 STRATEGY=ppl DEVICE=cuda:1 DTYPE=bfloat16 QL2L_CLAP_DEVICE=cuda:2 QL2L_ASR_DEVICE=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ALLOW_MISSING_VIDEOS=1 MIN_ITEMS=250 bash scripts/e0601_intentqa_faithfulness.sh`
  - configs: []
  - seeds: [2]
  - required_artifacts:
    - `runs/E0713_intentqa_faithfulness_val_s2_20260212-000949/faithfulness.json`
    - `runs/E0713_intentqa_faithfulness_val_s2_20260212-000949/rows.jsonl`
    - `runs/E0713_intentqa_faithfulness_val_s2_20260212-000949/preprocess_meta.json`
  - required_metrics:
    - `faithfulness.json`: `accuracy`, `accuracy_deleted`, `acc_drop`, `pred_change_rate`, `invalid_rate`
  - logs: `artifacts/experiments/E0713/run.log`
  - backfill: done (see Results table `E0713` row).

- [x] E0714 (real; Long-Video QA extension; seed=2): EgoSchema Subset eval rerun (`methods=uniform,ql2l_clap,ql2l_asr_bm25`, test n=256)
  - command: `OUT_DIR=runs/E0714_egoschema_eval_subset256_s2_$(date +%Y%m%d-%H%M%S) CONFIG=Subset SPLIT=test LIMIT=256 METHODS=uniform,ql2l_clap,ql2l_asr_bm25 B_FRAMES=16 MAX_SECONDS=120 SEED=2 STRATEGY=ppl DEVICE=cuda:1 DTYPE=bfloat16 QL2L_CLAP_DEVICE=cuda:2 QL2L_ASR_DEVICE=cpu QL2L_CLIP_DEVICE=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 bash scripts/e0602_egoschema_predict.sh`
  - configs: []
  - seeds: [2]
  - required_artifacts:
    - `runs/E0714_egoschema_eval_subset256_s2_20260212-004316/metrics.json`
    - `runs/E0714_egoschema_eval_subset256_s2_20260212-004316/predictions.jsonl`
    - `runs/E0714_egoschema_eval_subset256_s2_20260212-004316/preprocess_meta.json`
  - required_metrics:
    - `metrics.json`: `summary[*].acc`, `summary[*].invalid_rate`
  - logs: `artifacts/experiments/E0714/run.log`
  - backfill: done (see Results table `E0714` row).

- [x] E0003: Official AVE full-dataset validation (multi-GPU)
  - command: `RUN_ROOT=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402 bash scripts/ave_verify_official_after_install.sh`
  - configs: []
  - seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - required_artifacts:
    - `runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/E0002_anchors_official_val/anchor_eval/anchors_metrics.json`
    - `runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/E0002_anchors_official_test/anchor_eval/anchors_metrics.json`
    - `runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/p0_train3339_val402_energy_160_224_352_k2_shift1_std1.0_temporal_conv/metrics.json`
    - `runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/p0_train3339_test402_energy_160_224_352_k2_shift1_std1.0_temporal_conv/metrics.json`
  - required_metrics:
    - `metrics.json`: `metrics.summary.{uniform,random_top2,anchored_top2,oracle_top2}.mean`, `metrics.token_budget`, `metrics.paired_ttest.anchored_vs_uniform.p`
    - `anchors_metrics.json`: `metrics.by_delta["0"].ours_mean_recall > metrics.by_delta["0"].random_mean_recall` on val/test
  - logs: `artifacts/experiments/E0003/run.log`

- [x] E0100: EPIC-SOUNDS video-level multi-label classification (downstream proxy)
  - command:
    - `SELECTION=audio_anchored MAX_SECONDS=120 MAX_STEPS=120 LIMIT_TRAIN_VIDEOS=64 LIMIT_VAL_VIDEOS=64 SEEDS=0,1,2 OUT_DIR=runs/E0100_epic_video_cls_local_audio_anchored_full_ms120_s64_20260209-045119 bash scripts/e0100_epic_video_cls_local.sh`
    - `SELECTION=uniform MAX_SECONDS=120 MAX_STEPS=120 LIMIT_TRAIN_VIDEOS=64 LIMIT_VAL_VIDEOS=64 SEEDS=0,1,2 OUT_DIR=runs/E0100_epic_video_cls_local_uniform_full_ms120_s64_20260209-045119 bash scripts/e0100_epic_video_cls_local.sh`
    - `SELECTION=random MAX_SECONDS=120 MAX_STEPS=120 LIMIT_TRAIN_VIDEOS=64 LIMIT_VAL_VIDEOS=64 SEEDS=0,1,2 OUT_DIR=runs/E0100_epic_video_cls_local_random_full_ms120_s64_20260209-045119 bash scripts/e0100_epic_video_cls_local.sh`
    - `SELECTION=audio_anchored MAX_SECONDS=120 MAX_STEPS=120 LIMIT_TRAIN_VIDEOS=256 LIMIT_VAL_VIDEOS=137 SEEDS=0,1,2 OUT_DIR=runs/E0100_epic_video_cls_local_audio_anchored_full_ms120_t256_v137_s012_20260209-235834 bash scripts/e0100_epic_video_cls_local.sh`
    - `SELECTION=uniform MAX_SECONDS=120 MAX_STEPS=120 LIMIT_TRAIN_VIDEOS=256 LIMIT_VAL_VIDEOS=137 SEEDS=0,1,2 OUT_DIR=runs/E0100_epic_video_cls_local_uniform_full_ms120_t256_v137_s012_20260210-001346 bash scripts/e0100_epic_video_cls_local.sh`
    - `SELECTION=random MAX_SECONDS=120 MAX_STEPS=120 LIMIT_TRAIN_VIDEOS=256 LIMIT_VAL_VIDEOS=137 SEEDS=0,1,2 OUT_DIR=runs/E0100_epic_video_cls_local_random_full_ms120_t256_v137_s012_20260210-001929 bash scripts/e0100_epic_video_cls_local.sh`
    - `SELECTION=uniform MAX_SECONDS=120 MAX_STEPS=60 LIMIT_TRAIN_VIDEOS=256 LIMIT_VAL_VIDEOS=137 SEEDS=0,1,2 OUT_DIR=runs/E0100_epic_video_cls_local_uniform_ms120_steps60_t256_v137_s012_20260210-012533 bash scripts/e0100_epic_video_cls_local.sh`
    - `SELECTION=random MAX_SECONDS=120 MAX_STEPS=60 LIMIT_TRAIN_VIDEOS=256 LIMIT_VAL_VIDEOS=137 SEEDS=0,1,2 OUT_DIR=runs/E0100_epic_video_cls_local_random_ms120_steps60_t256_v137_s012_20260210-013248 bash scripts/e0100_epic_video_cls_local.sh`
    - `SELECTION=audio_anchored MAX_SECONDS=120 MAX_STEPS=60 LIMIT_TRAIN_VIDEOS=256 LIMIT_VAL_VIDEOS=137 SEEDS=0,1,2 OUT_DIR=runs/E0100_epic_video_cls_local_audio_anchored_ms120_steps60_t256_v137_s012_20260210-013750 bash scripts/e0100_epic_video_cls_local.sh`
    - `SELECTION=oracle MAX_SECONDS=120 MAX_STEPS=60 LIMIT_TRAIN_VIDEOS=256 LIMIT_VAL_VIDEOS=137 SEEDS=0,1,2 OUT_DIR=runs/E0100_epic_video_cls_local_oracle_ms120_steps60_t256_v137_s012_20260210-014715 bash scripts/e0100_epic_video_cls_local.sh`
  - configs: []
  - seeds: [0, 1, 2]
  - required_artifacts:
    - `runs/E0100_epic_video_cls_local_audio_anchored_full_ms120_s64_20260209-045119/metrics.json`
    - `runs/E0100_epic_video_cls_local_uniform_full_ms120_s64_20260209-045119/metrics.json`
    - `runs/E0100_epic_video_cls_local_random_full_ms120_s64_20260209-045119/metrics.json`
    - `runs/E0100_epic_video_cls_local_audio_anchored_full_ms120_t256_v137_s012_20260209-235834/metrics.json`
    - `runs/E0100_epic_video_cls_local_uniform_full_ms120_t256_v137_s012_20260210-001346/metrics.json`
    - `runs/E0100_epic_video_cls_local_random_full_ms120_t256_v137_s012_20260210-001929/metrics.json`
    - `runs/E0100_epic_video_cls_local_uniform_ms120_steps60_t256_v137_s012_20260210-012533/metrics.json`
    - `runs/E0100_epic_video_cls_local_random_ms120_steps60_t256_v137_s012_20260210-013248/metrics.json`
    - `runs/E0100_epic_video_cls_local_audio_anchored_ms120_steps60_t256_v137_s012_20260210-013750/metrics.json`
    - `runs/E0100_epic_video_cls_local_oracle_ms120_steps60_t256_v137_s012_20260210-014715/metrics.json`
  - required_metrics:
    - `metrics.json`: `summary.mAP.mean`, `summary["macro_f1@0.5"].mean`, plus budget fields (`max_steps`, `max_seconds`, `base_res`)
  - logs:
    - `artifacts/experiments/E0100/run.log` (pilot s64)
    - `artifacts/experiments/E0100_full/run.log` (expanded t256/v137)
    - `artifacts/experiments/E0100_steps60/run.log` (uniform, steps60)
    - `artifacts/experiments/E0100_steps60_random/run.log`
    - `artifacts/experiments/E0100_steps60_anchored/run.log`
    - `artifacts/experiments/E0100_steps60_oracle/run.log`

- [x] E0202: Evidence Alignment (Cov@tau) vs accuracy correlation report (energy / test402)
  - command: `python -m avs.experiments.evidence_alignment_report --in-metrics runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/p0_train3339_test402_energy_160_224_352_k2_shift1_std1.0_temporal_conv/metrics.json --meta-dir data/AVE/meta --out-dir runs/E0202_evidence_alignment_energy_test402_20260209-061145`
  - configs: []
  - seeds: []
  - required_artifacts: [`runs/E0202_evidence_alignment_energy_test402_20260209-061145/evidence_alignment.json`]
  - required_metrics:
    - `evidence_alignment.json`: `cov_by_tau` + `corr_by_tau` with Pearson/Spearman (tau_grid includes `0.3,0.5,0.7`)
  - logs: `artifacts/experiments/E0202/run.log`

- [x] E0720: Evidence Alignment report on best C0003 config (df7 / test402)
  - command: `python -m avs.experiments.evidence_alignment_report --in-metrics runs/E0643_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_df7_officialids_s0-9_20260211-001604/metrics.json --meta-dir data/AVE/meta --out-dir runs/E0720_evidence_alignment_df7_best_20260212-015616`
  - configs: []
  - seeds: []
  - required_artifacts: [`runs/E0720_evidence_alignment_df7_best_20260212-015616/evidence_alignment.json`]
  - required_metrics:
    - `evidence_alignment.json`: `cov_by_tau` + `corr_by_tau` with Pearson/Spearman (tau_grid includes `0.3,0.5,0.7`)

- [x] E0203: Degradation suite (shift/noise/silence grid) on official AVE test402 (energy)
  - command: `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_160_224_352 EVENTNESS=energy AUDIO_DEVICE=cpu bash scripts/e0203_degradation_suite_official.sh`
  - configs: []
  - seeds: []
  - required_artifacts: [`runs/E0203_degradation_energy_20260209-061156/degradation_suite.json`]
  - required_metrics:
    - `degradation_suite.json`: 18-row grid complete; per-row `recall_by_delta` present; `fallback_used_frac` reported
  - logs: `artifacts/experiments/E0203/run.log`

## Results
| id | status | key_metrics | artifacts | notes |
|---|---|---|---|---|
| E0003 | success | test402 anchored_top2.mean=0.72025 vs uniform.mean=0.71622 (Δ=+0.00403; p=0.383); anchors Δ0 ours>random on val/test | `runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/` | allow-missing drops some clips (train=3312, val=401, test=402) |
| E0100 | success | mAP=`0.4028±0.0048` (anchored) vs `0.3346±0.0021` (uniform; Δ=+0.0681); macro_f1@0.5=`0.4194±0.0009` vs `0.3277±0.0387` (Δ=+0.0917) | `runs/E0100_epic_video_cls_local_*_full_ms120_t256_v137_s012_202602*/metrics.json` | expanded t256/v137 rerun (plus pilot s64); strict equal-budget (`max_steps × base_res`). Note: when `max_steps==max_seconds`, random==uniform because both select all seconds. Additional diagnostic (`max_steps=60`, `max_seconds=120`, t256/v137): uniform mAP=`0.3351±0.0048`; random=`0.3340±0.0051`; audio_anchored=`0.3358±0.0135`; oracle=`0.3329±0.0005` (no gain at this tighter budget). |
| E0202 | success | Cov@tau mean≈0.0593; corr pearson≈0.0625 spearman≈-0.0314 (weak) | `runs/E0202_evidence_alignment_energy_test402_20260209-061145/evidence_alignment.json` | diagnostic only (not a predictor) |
| E0203 | success | mean Recall@K across 18 conditions: Δ0≈0.2240 (min≈0.2174, max≈0.2307), Δ1≈0.4767, Δ2≈0.6420 | `runs/E0203_degradation_energy_20260209-061156/degradation_suite.json` | fallback_used_frac=0.0 |
| E0600 | success | IntentQA val (n=253): uniform acc=0.9447; cheap_visual acc=0.9526; ql2l_clap acc=0.9486 | `runs/E0600_intentqa_vlm_eval_full_20260210-041911/` | Qwen2-VL-2B; `budget_frames=16`, `max_seconds=120`, `strategy=ppl` |
| E0601 | success | IntentQA faithfulness (n=253; ql2l_clap): acc=0.9486; acc_drop=0.0000; pred_change_rate=0.0316 | `runs/E0601_intentqa_faithfulness_full_20260210-061137/` | delete-and-predict proxy; invalid_rate=0 |
| E0602 | success | EgoSchema Subset test (n=256): uniform acc=0.5859; ql2l_clap acc=0.5352; ql2l_asr_bm25 acc=0.5469 | `runs/E0602_egoschema_eval_subset_full_20260210-064250/` | Qwen2-VL-2B; `budget_frames=16`, `max_seconds=120`, `strategy=ppl` |
| E0604 | success | IntentQA val (n=253): seed0 uniform=0.9447, cheap_visual=0.9526, ql2l_clap=0.9486; seed1 uniform=0.9447, cheap_visual=0.9526, ql2l_clap=0.9486 | `runs/E0604_intentqa_vlm_eval_val_s*_20260210-125048/` | seed1 ran reduced METHODS (uniform,random,cheap_visual,ql2l_clap) due to runtime |
| E0605 | success | EgoSchema Subset test (n=500; seeds=0,1): uniform acc=0.5880; ql2l_clap acc=0.5480; ql2l_asr_bm25 acc=0.5560 (invalid_rate=0) | `runs/E0605_egoschema_eval_subset500_s0_20260210-125048/` `runs/E0605_egoschema_eval_subset500_s1_20260210-183504/` | identical metrics across seeds; seed0 resumed after partial run |
| E0607 | success | IntentQA faithfulness (n=253; ql2l_clap; seed1): acc=0.9486; acc_drop=0.0000; pred_change_rate=0.0316 | `runs/E0607_intentqa_faithfulness_val_s1_20260210-194732/` | matches seed0; invalid_rate=0 |
| E0608 | success | EgoSchema Subset test (n=256; seed1): uniform acc=0.5859; ql2l_clap acc=0.5352; ql2l_asr_bm25 acc=0.5469 | `runs/E0608_egoschema_eval_subset256_s1_20260210-201700/` | matches seed0; invalid_rate=0 |
| E0606 | success | EgoSchema Subset test (n=500; seed0): uniform acc=0.5880; ql2l_clip acc=0.5760; ql2l_clap=0.5480; ql2l_asr_bm25=0.5560 (invalid_rate=0) | `runs/E0606_egoschema_eval_subset500_clip_20260211-031138/` | ql2l_clip improves over other ql2l baselines but still < uniform |
| E0618 | success | EgoSchema Subset test (n=500): uniform acc=0.5880; text_only acc=0.2720 (Δ=-0.3160) | `runs/E0618_egoschema_eval_subset500_text_only_20260211-055131/` | invalid_rate=0 |
| E0609 | success | IntentQA val (n=253): uniform acc=0.9447; ql2l_clap acc=0.9486; cheap_visual acc=0.9526; ql2l_clip acc=0.9368 (Δ=-0.0079) | `runs/E0609_intentqa_vlm_eval_val_clip_20260211-011407/` | adds CLIP query-relevance baseline; skipped_videos=1 |
| E0617 | success | IntentQA val (n=253): uniform acc=0.9447; text_only acc=0.6640 (Δ=-0.2806) | `runs/E0617_intentqa_vlm_eval_val_text_only_20260211-053301/` | skipped_videos=1; invalid_rate=0 |
| E0615 | success | AVQA val (n=212; seed0): uniform acc=0.8160; text_only acc=0.3113 (others identical to uniform) | `runs/E0615_avqa_vlm_eval_val_20260211-043508/` | budget_frames=16 >= clip duration, so all frame-selection methods degenerate to full coverage; rerun with tighter budget for method separation |
| E0616 | success | AVQA val (n=212; B_FRAMES=4): uniform acc=0.8113; best=0.8255 (+0.0142; cheap_visual/fused/ql2l_asr_bm25); ql2l_clip=0.8208; audio/ql2l_clap=0.8160; text_only=0.3113 | `runs/E0616_avqa_vlm_eval_val_b4_20260211-051556/` | skipped_videos=44; invalid_rate=0 |
| E0710 | success | val402 sweep best=`ltlkeepadj_adj2_shift0_std0p5`: anchored-uniform Δ=+0.00648; p=0.0355 | `runs/E0710_val402_flowmlp_keepadj_20260212-000010/` | uses official ids + explicit processed/cache dirs from `REAL_AVE_OFFICIAL_RERUN_20260209-054402` |
| E0711 | success | quick test402: anchored=0.71982 vs uniform=0.71294 (Δ=+0.00688; p=0.395) | `runs/E0711_quick_test402_flowmlp_keepadj_20260212-000606/` | quick transfer positive but not significant |
| E0712 | success | full test402: anchored=0.72331 vs uniform=0.71622 (Δ=+0.00709; p=0.141) | `runs/E0712_full_test402_flowmlp_keepadj_20260212-000835/` | fails C0003 hard gate; below current best full-test Δ |
| E0713 | success | IntentQA faithfulness (seed=2, n=253): accuracy=0.9486; accuracy_deleted=0.9486; acc_drop=0.0000; pred_change_rate=0.0316 | `runs/E0713_intentqa_faithfulness_val_s2_20260212-000949/` | matches seeds 0/1 behavior; offline HF mode for stability |
| E0714 | success | EgoSchema Subset test (seed=2, n=256): uniform acc=0.5859; ql2l_clap acc=0.5352; ql2l_asr_bm25 acc=0.5469 (all invalid_rate=0) | `runs/E0714_egoschema_eval_subset256_s2_20260212-004316/` | matches seeds 0/1 behavior; seed-extension reproducibility check passed |
| E0720 | success | Cov@tau mean≈0.1032; corr pearson≈0.0747 spearman≈-0.0304 (weak) | `runs/E0720_evidence_alignment_df7_best_20260212-015616/evidence_alignment.json` | Evidence Alignment on best C0003 config (E0643 df7); diagnostic only (not predictive) |
| E0801 | success | val402 sweep best=`ltlkeepadjv2_adj1_shift0_std0p33`: anchored=0.74672 vs uniform=0.74680 (Δ=-0.00008; p=0.9929) | `runs/E0801_val402_imagebind_keepadjv2_20260212-035956/` | ImageBind AV-sim Stage-1; effectively neutral/negative on val |
| E0802 | success | quick test402: anchored=0.71028 vs uniform=0.71294 (Δ=-0.00265; p=0.7538) | `runs/E0802_quick_test402_imagebind_20260212-040440/` | fallback_used_frac≈0.739; fails promotion gate (no full test) |
| E0810 | success | val402 sweep best=`ltlkeepadj_adj1_shift0_std0p55`: anchored=0.74256 vs uniform=0.74680 (Δ=-0.00424; p=0.6662) | `runs/E0810_val402_wavlm_20260212-041931/` | WavLM supervised Stage-1 (`wavlm_evt_mlp`) fails val sweep |
| E0811 | success | quick test402: anchored=0.71418 vs uniform=0.71294 (Δ=+0.00124; p=0.9178) | `runs/E0811_quick_test402_wavlm_20260212-042425/` | fallback_used_frac≈0.231; fails promotion gate (no full test) |
| E0820 | success | val402 sweep best=`ltlkeepadjv2_adj2_shift0_std0p25`: anchored=0.74564 vs uniform=0.74680 (Δ=-0.00116; p=0.9163) | `runs/E0820_val402_wavlm_cliplossgain_20260212-112651/` | Oracle-distilled loss-gain Stage-1 (`av_wavlm_clip_lossgain_mlp`); fallback_used_frac≈0.983; fails val sweep |
| E0821 | success | quick test402: anchored=0.71443 vs uniform=0.71294 (Δ=+0.00149; p=0.9007) | `runs/E0821_quick_test402_wavlm_cliplossgain_20260212-113152/` | fallback_used_frac≈0.970; fails promotion gate (no full test) |
| E0830 | success | val402 sweep best=`ltlkeepadjv2_adj2_shift0_std0p25`: anchored=0.74530 vs uniform=0.74680 (Δ=-0.00150; p=0.5286) | `runs/E0830_val402_wavlm_clipmil_20260212-115213/` | WavLM+CLIP MIL Stage-1 (`av_wavlm_clip_mil_mlp`); fallback_used_frac=0.0; fails val sweep |
| E0831 | success | quick test402: anchored=0.71153 vs uniform=0.71294 (Δ=-0.00141; p=0.8594) | `runs/E0831_quick_test402_wavlm_clipmil_20260212-115656/` | fallback_used_frac=0.0; fails promotion gate (no full test) |
| E0840 | success | val402 sweep best=`ltltop1medn_thr0p5_shift1`: anchored=0.74023 vs uniform=0.74680 (Δ=-0.00657; p=0.5425) | `runs/E0840_val402_wavlm_clipevt_mlp_20260212-121939/` | WavLM+CLIP supervised eventness (BCE MLP); fails val sweep |
| E0841 | success | quick test402: anchored=0.71526 vs uniform=0.71294 (Δ=+0.00232; p=0.2430) | `runs/E0841_quick_test402_wavlm_clipevt_mlp_20260212-122228/` | fallback_used_frac≈0.483; fails promotion gate (no full test) |
| E0850 | success | val402 sweep best=`ltltop1medn_thr0p5_shift0`: anchored=0.74813 vs uniform=0.74680 (Δ=+0.00133; p=0.9001) | `runs/E0850_val402_wavlm_clipevt_tcn_20260212-122411/` | WavLM+CLIP supervised eventness (TCN); weak val; not promoted |
| E0851 | success | quick test402: anchored=0.70597 vs uniform=0.71294 (Δ=-0.00697; p=0.6731) | `runs/E0851_quick_test402_wavlm_clipevt_tcn_20260212-122630/` | fallback_used_frac≈0.679; harmful on quick; skip full |
| E0860 | success | val402 sweep best=`ltlsep3_thr0p6_shift1`: anchored=0.74140 vs uniform=0.74680 (Δ=-0.00540; p=0.2532) | `runs/E0860_val402_vecmlp_sep3_20260212-122905/` | Stage-2 sep3 gate attempt for vec-MLP; harmful on val |
| E0861 | success | quick test402: anchored=0.70614 vs uniform=0.71294 (Δ=-0.00680; p=0.5991) | `runs/E0861_quick_test402_vecmlp_sep3_20260212-123220/` | fallback_used_frac≈0.231; harmful on quick; skip full |
| E0870 | success | val402 sweep best=`ltltop1medn_thr0p5_shift0`: anchored=0.74638 vs uniform=0.74680 (Δ=-0.00042; p=0.9509) | `runs/E0870_val402_wavlm_clipdiff_vecmlp_20260212-123504/` | WavLM+CLIPdiff vec-MLP Stage-1; near-zero on val |
| E0871 | success | quick test402: anchored=0.71940 vs uniform=0.71294 (Δ=+0.00647; p=0.5134) | `runs/E0871_quick_test402_wavlm_clipdiff_vecmlp_20260212-123718/` | fallback_used_frac≈0.510; still far from +2%; not promoted |
| E0880 | success | val402 sweep best=`ltltop1medn_thr0p7_shift0`: anchored=0.74397 vs uniform=0.74680 (Δ=-0.00283; p=0.8283) | `runs/E0880_val402_wavlm_clip_cls_target_20260212-125727/` | WavLM+CLIP multi-class cls-target Stage-1 (`av_wavlm_clip_mlp_cls_target`); negative on val |
| E0881 | success | quick test402: anchored=0.71766 vs uniform=0.71294 (Δ=+0.00473; p=0.6175) | `runs/E0881_quick_test402_wavlm_clip_cls_target_20260212-130104/` | fallback_used_frac≈0.933; not promoted |
| E0883 | success | val402 sweep best=`ltlmax1_thr0p45_balanced_window3`: Δ=+0.00939; p=0.2750 | `runs/E0883_val402_vecmlp_maxhigh1_20260212-130225/` | Stage-2 max_high=1 sweep (`ltl_maxhigh1_v1`) for vec-MLP; preserves base-res context |
| E0884 | success | quick test402: anchored=0.72405 vs uniform=0.71294 (Δ=+0.01111; p=0.3234) | `runs/E0884_quick_test402_vecmlp_maxhigh1_20260212-130830/` | fallback_used_frac≈0.311; not promoted |
| E0886 | success | val402 sweep best=`ltltop1medn_thr0p7_shift1`: Δ=+0.00166; p=0.8877 | `runs/E0886_val402_wavlm_clip_cls_margin_20260212-131558/` | WavLM+CLIP multi-class margin Stage-1 (`av_wavlm_clip_mlp_cls`); near-zero on val |
| E0887 | success | quick test402: anchored=0.71526 vs uniform=0.71294 (Δ=+0.00232; p=0.6781) | `runs/E0887_quick_test402_wavlm_clip_cls_margin_20260212-131757/` | fallback_used_frac≈0.923; not promoted |
| E0890 | success | val402 sweep best=`ltltop1mednmax1_thr0p5_shift0`: anchored=0.74214 vs uniform=0.74680 (Δ=-0.00466; p=0.3436) | `runs/E0890_val402_wavlm_clip_mil_mlp_20260212-133043/` | WavLM+CLIP MIL Stage-1 (`av_wavlm_clip_mil_mlp`) rerun with `ltl_top1medn_maxhigh1_v1`; negative on val |
| E0891 | success | quick test402: anchored=0.70730 vs uniform=0.71294 (Δ=-0.00564; p=0.5272) | `runs/E0891_quick_test402_wavlm_clip_mil_mlp_20260212-133312/` | anchors_len_fallback_frac≈0.532; harmful on quick; skip full |
| E0893 | success | quick test402: anchored=0.72347 vs uniform=0.71294 (Δ=+0.01053; p=0.4780) | `runs/E0893_quick_test402_vecmlp_df7_maxhigh1_20260212-133857/` | df7 keepadj ablation (`max_high_anchors=1`): removes 2-high but does not improve Δ |
| E0895 | success | quick test402: anchored=0.72828 vs uniform=0.71294 (Δ=+0.01534; p=0.1997) | `runs/E0895_quick_test402_vecmlp_df7_band112_20260212-134114/` | df7 keepadj ablation (`budget_mode=band`, extra_res=[112]) looks promising on quick |
| E0896 | success | full test402: anchored=0.72410 vs uniform=0.71622 (Δ=+0.00789; p=0.2531) | `runs/E0896_full_test402_vecmlp_df7_band112_20260212-134215/` | df7 keepadj band112 regresses on full; 2-high bucket negative again |
| E0898 | success | quick test402: anchored=0.72164 vs uniform=0.71294 (Δ=+0.00871; p=0.5286) | `runs/E0898_quick_test402_vecmlp_df7_maxhigh1_std0p35_20260212-134530/` | maxhigh1 std sweep (0.35): fallback decreases (≈0.189) but Δ worsens |
| E0899 | success | quick test402: anchored=0.71791 vs uniform=0.71294 (Δ=+0.00498; p=0.7523) | `runs/E0899_quick_test402_vecmlp_df7_maxhigh1_std0p45_20260212-134637/` | maxhigh1 std sweep (0.45): fallback≈0.311; Δ worsens |
| E0900 | success | quick test402: anchored=0.71667 vs uniform=0.71294 (Δ=+0.00373; p=0.7991) | `runs/E0900_quick_test402_vecmlp_df7_maxhigh1_std0p65_20260212-134717/` | maxhigh1 std sweep (0.65): fallback≈0.624; Δ worsens |
| E0901 | success | quick test402: anchored=0.72504 vs uniform=0.71294 (Δ=+0.01211; p=0.3983) | `runs/E0901_quick_test402_vecmlp_df7_k1_20260212-135506/` | df7 keepadj ablation (`k=1`): fallback≈0.552; still below +2%; not promoted |
| E0902 | success | val402 sweep best=`ltltop1mednmax1_thr0p7_shift1`: anchored=0.74314 vs uniform=0.74680 (Δ=-0.00366; p=0.2573) | `runs/E0902_val402_wavlm_clip_xattn_mil_20260212-140250/` | WavLM+CLIP XAttn MIL Stage-1 (`av_wavlm_clip_xattn_mil`); default vis_res=112; negative on val |
| E0903 | success | val402 sweep best=`ltltop1mednmax1_thr0p6_shift0`: anchored=0.74746 vs uniform=0.74680 (Δ=+0.00066; p=0.9113) | `runs/E0903_val402_wavlm_clip_xattn_mil_r224_clipdiff_20260212-141657/` | XAttn MIL Stage-1 (r224; clip+clipdiff); near-zero on val |
| E0904 | success | val402 sweep best=`ltlkeepadjv2_adj2_shift1_std0p25`: anchored=0.74123 vs uniform=0.74680 (Δ=-0.00557; p=0.4279) | `runs/E0904_val402_xattn_mil_r224_clipdiff_keepadjv2_20260212-141941/` | keepadjv2 Stage-2 sweep with cached E0903 scores; harmful on val |
| E0905 | success | val402 sweep best=`ltltop1mednmax1_thr0p5_shift0`: anchored=0.74722 vs uniform=0.74680 (Δ=+0.00042; p=0.9496) | `runs/E0905_val402_xattn_mil_r352_clip_20260212-142552/` | XAttn MIL Stage-1 (r352; clip); near-zero on val |
| E0906 | success | val402 sweep best=`ltltop1mednmax1_thr0p6_shift0`: anchored=0.73716 vs uniform=0.74680 (Δ=-0.00964; p=0.1157) | `runs/E0906_val402_vision_binary_mlp_r352_20260212-143611/` | High-res vision Stage-1 (`vision_binary_mlp_r352`) is harmful under top1-med gate |
| E0907 | success | val402 sweep best=`ltlgini2_gini0p35_shift0`: anchored=0.74331 vs uniform=0.74680 (Δ=-0.00349; p=0.7066) | `runs/E0907_val402_vision_binary_mlp_r352_gini_v2_20260212-143837/` | gini gate reduces harm vs E0906 but remains negative on val |
| E0908 | success | val402 sweep best=`ltltop1mednmax1_thr0p5_shift1`: anchored=0.74796 vs uniform=0.74680 (Δ=+0.00116; p=0.9216) | `runs/E0908_val402_avel_bilstm_cls_r352_clipdiff_20260212-205411/` | AVE-localizer-style Stage-1 (`av_wavlm_clip_avel_bilstm_cls_target`; BiLSTM cls-target) is near-zero on val |
| E0909 | success | val402 sweep best=`ltltop1mednmax1_thr0p5_shift1`: anchored=0.74796 vs uniform=0.74680 (Δ=+0.00116; p=0.9216) | `runs/E0909_val402_avel_bilstm_cls_r352_clipdiff_minmax_20260212-210652/` | minmax-normalized scores make no difference (ranking-only) |
| E0910 | success | val402 sweep best=`ltltop1mednmax1_thr0p7_shift1`: anchored=0.74680 vs uniform=0.74680 (Δ=-0.00000; p≈1.0) | `runs/E0910_val402_avel_bilstm_cls_onset_deriv_pos_rerun_20260212-213925/` | onset-like score transform collapses to uniform |
| E0911 | success | val402 sweep best=`ltlkeepadjv2_adj1_shift0_std0p25`: anchored=0.73982 vs uniform=0.74680 (Δ=-0.00698; p=0.3580) | `runs/E0911_val402_avel_bilstm_cls_keepadjv2_20260212-214325/` | keepadjv2 Stage-2 configs are harmful under this Stage-1 |
| E0912 | success | val402 sweep best=`ltlgini2_gini0p4_shift1`: anchored=0.74888 vs uniform=0.74680 (Δ=+0.00208; p=0.7854) | `runs/E0912_val402_avel_bilstm_cls_gini_v2_20260212-214729/` | gini gate helps slightly but not significant |
| E0913 | success | val402 sweep best=`ltladjv2_adj1_shift0_std0p2_scoreAlloc`: anchored=0.74007 vs uniform=0.74680 (Δ=-0.00673; p=0.4637) | `runs/E0913_val402_avel_bilstm_cls_adaptive_v2_20260212-215337/` | adaptive Stage-2 configs are harmful under this Stage-1 |
| E0914 | success | val402 sweep best=`ltltop1mednmax1_thr0p5_shift1`: anchored=0.74422 vs uniform=0.74680 (Δ=-0.00258; p=0.6928) | `runs/E0914_val402_xattn_cls_target_r352_clipdiff_20260212-221441/` | XAttn supervised Stage-1 (`av_wavlm_clip_xattn_cls_target`) is harmful on val |
| E0915 | success | EVA02 caches built for official AVE: train+val=3703 clips, test=394 clips (total caches=4097 `.npz`; r∈{112,160,224,352}) | `runs/E0915_build_cache_eva02_clip_p16_112_160_224_352_20260212-225043/` `runs/E0915_build_cache_eva02_clip_p16_112_160_224_352_test_20260212-230913/` | caches: `runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_eva02_base_patch16_clip_224_112_160_224_352/` |
| E0916 | success | val402 sweep (EVA02 Stage-2): best=`ltlkeepadj_adj1_shift0_std0p6`: anchored=0.76110 vs uniform=0.75869 (Δ=+0.00241; p=0.6538) | `runs/E0916_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_adaptive_keepadj_v1_eva02_20260212-231218/` | uniform is high already; selection gain collapses; not promoted |
| E0917 | success | val402 sweep (EVA02 Stage-1 only; keepadj): best=`ltlkeepadj_adj1_shift0_std0p45`: anchored=0.75004 vs uniform=0.74680 (Δ=+0.00324; p=0.5702) | `runs/E0917_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_adaptive_keepadj_v1_stage1eva02_20260212-231759/` | uses `STAGE1_CACHES_DIR` override; not promoted |
| E0918 | success | val402 sweep (EVA02 Stage-1 only; top1med_norm): best=`ltltop1medn_thr0p5_shift0`: anchored=0.74863 vs uniform=0.74680 (Δ=+0.00183; p=0.8223) | `runs/E0918_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_top1med_norm_v1_stage1eva02_20260212-232240/` | not promoted |
| E0919 | success | DINOv2 caches built for official AVE: train+val=3703 clips, test=394 clips (total caches=4097 `.npz`; r∈{112,160,224,352,448}) | `runs/E0919_build_cache_dinov2_fill_trainval_20260213-001226/` `runs/E0919_build_cache_dinov2_fill_test_20260213-002118/` | caches: `runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch14_dinov2_112_160_224_352_448/` |
| E0920 | success | val402 sweep (DINOv2 Stage-1 only; keepadj): best=`ltlkeepadj_adj2_shift0_std0p55`: anchored=0.75520 vs uniform=0.74680 (Δ=+0.00840; p=0.4031) | `runs/E0920_val402_vecmlp_keepadj_stage1dinov2_20260213-001634/` | uses `STAGE1_CACHES_DIR` override; promoted to quick test402 |
| E0921 | success | quick test402: anchored=0.72189 vs uniform=0.71294 (Δ=+0.00896; p=0.5825) | `runs/E0921_quick_test402_vecmlp_keepadj_stage1dinov2_20260213-002346/` | diagnose: `runs/E0921_quick_test402_vecmlp_keepadj_stage1dinov2_20260213-002346/diagnose.json`; not promoted to full |
| E0923 | success | val402 sweep (`av_clipdiff_flow_mlp_stride`; DINOv2 Stage-1 only): best=`ltltop1medk1ext_thr0p6_shift0_score`, Δ=+0.00333 (p=0.5507) | `runs/E0923_val402_flow_stride_stage1dinov2_20260213-003820/` | uses `STAGE1_CACHES_DIR` override; promoted to quick test402 |
| E0924 | success | quick test402 (`av_clipdiff_flow_mlp_stride`; DINOv2 Stage-1 only): anchored=0.71401 vs uniform=0.71294 (Δ=+0.00108; p=0.9222) | `runs/E0924_quick_test402_flow_stride_stage1dinov2_20260213-004520/` | diagnose: `runs/E0924_quick_test402_flow_stride_stage1dinov2_20260213-004520/diagnose.json`; not promoted |
| E0926 | success | val402 sweep (AVE-localizer BiLSTM cls-target; DINOv2 Stage-1 only): best=`ltltop1mednmax1_thr0p7_shift1`, Δ=-0.00898 (p=0.2298) | `runs/E0926_val402_avel_bilstm_cls_stage1dinov2_20260213-005021/` | not promoted |
| E0927 | success | val402 sweep (XAttn cls-target; DINOv2 Stage-1 only): best=`ltltop1mednmax1_thr0p5_shift1`, Δ=+0.00324 (p=0.6881) | `runs/E0927_val402_xattn_cls_target_stage1dinov2_20260213-005232/` | not promoted |
| E0928 | success | val402 sweep (DINOv2 Stage-1 scores reused; top1med_norm): best=`ltltop1medn_thr0p6_shift1`, Δ=+0.00017 (p=0.9836) | `runs/E0928_val402_vecmlp_top1medn_scores_dinov2_20260213-005513/` | uses `SCORES_JSON` from E0920; not promoted |
| E0930 | success | SigLIP cache build: ok=true, missing=0; res=[112,160,224,352,448]; num_npz=4097 | `runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch16_siglip_224_webli_112_160_224_352_448/` | build logs: `runs/E0930_build_cache_siglip_trainval_20260213-011101/cache_build.json`, `runs/E0930_build_cache_siglip_test_20260213-013806/cache_build.json` |
| E0931 | success | val402 sweep (SigLIP stage-2): best=`ltlkeepadj_adj2_shift1_std0p6`, anchored=0.42186 vs uniform=0.40889 (Δ=+0.01297; p=0.0982) | `runs/E0931_val402_siglip_stage2_vecmlp_keepadj_20260213-014809/` | candidate_set=`ltl_adaptive_keepadj_v1`; EVENTNESS=`av_clipdiff_vec_mlp` |
| E0932 | success | quick test402 (SigLIP stage-2): anchored=0.33350 vs uniform=0.33085 (Δ=+0.00265; p=0.5427) | `runs/E0932_quick_test402_siglip_stage2_vecmlp_keepadj_20260213-015409/` | diagnose fallback_used_frac≈0.853; not promoted to full |
| E0934 | success | val402 sweep (XAttn binary eventness): best=`ltlkeepadj_adj2_shift1_std0p45`, anchored-uniform Δ=+0.00208; p=0.8075 | `runs/E0934_val402_xattn_evt_20260213-021028/` | EVENTNESS=`av_wavlm_clip_xattn_evt`; `XATTN_VIS_FEATS=clip+clipdiff` |
| E0935 | success | quick test402 (XAttn binary eventness): anchored=0.71915 vs uniform=0.71294 (Δ=+0.00622; p=0.5305) | `runs/E0935_quick_test402_xattn_evt_20260213-021503/` | diagnose fallback_used_frac≈0.077; fails promotion gate (no full test) |
| E0937 | success | val402 sweep (XAttn eventness; Stage-1 SigLIP cache): best=`ltlkeepadj_adj1_shift0_std0p6`, anchored-uniform Δ=+0.00682; p=0.4311 | `runs/E0937_val402_xattn_evt_stage1siglip_20260213-021919/` | `STAGE1_CACHES_DIR` SigLIP; Stage-2 remains CLIP cache |
| E0938 | success | quick test402 (XAttn eventness; Stage-1 SigLIP cache): anchored=0.71857 vs uniform=0.71294 (Δ=+0.00564; p=0.6689) | `runs/E0938_quick_test402_xattn_evt_stage1siglip_20260213-022855/` | diagnose fallback_used_frac≈0.152; fails promotion gate (no full test) |
| E0940 | success | val402 sweep (XAttn eventness; Stage-1 DINOv2 cache): best=`ltlkeepadj_adj2_shift1_std0p6`, anchored-uniform Δ=-0.00191; p=0.6639 | `runs/E0940_val402_xattn_evt_stage1dinov2_20260213-023423/` | not promoted (negative on val402) |
| E0943 | success | val402 sweep (XAttn eventness; Stage-1 vis_res=352): best=`ltlkeepadj_adj1_shift1_std0p5`, anchored-uniform Δ=-0.00224; p=0.5038 | `runs/E0943_val402_xattn_evt_vis352_20260213-024540/` | not promoted (negative on val402) |
| E0946 | success | val402 sweep (WavLM+CLIPdiff vec-MLP; keepadj): best=`ltlkeepadj_adj1_shift0_std0p6`, anchored-uniform Δ=+0.00283; p=0.7692 | `runs/E0946_val402_wavlm_clipdiff_vecmlp_keepadj_20260213-030005/` | not promoted |
| E0949 | success | val402 sweep (CLIPdiff vec-MLP; keepadj; Stage-1 SigLIP cache): best=`ltlkeepadj_adj2_shift0_std0p6`, anchored-uniform Δ=+0.00490; p=0.6126 | `runs/E0949_val402_vecmlp_keepadj_stage1siglip_20260213-030437/` | not promoted |
| E0952 | success | export CACE-Net eventness scores (processed frames): unique_vids=4097 | `runs/E0952_export_cace_evt_20260213-040137/` | `cace_evt_scores.json` |
| E0953 | success | val402 sweep (CACE evt; keepadj): best=`ltlkeepadj_adj1_shift0_std0p6`, Δ=-0.00091; p=0.8948 | `runs/E0953_val402_cace_evt_keepadj_20260213-040611/` | not promoted |
| E0954 | success | val402 sweep (CACE evt; gini_v2): best=`ltlgini2_gini0p5_shift1`, Δ=+0.00224; p=0.7671 | `runs/E0954_val402_cace_evt_gini_v2_20260213-041040/` | promoted to quick test402 only |
| E0955 | success | quick test402 (CACE evt; gini_v2 best): anchored=0.72114 vs uniform=0.71294 (Δ=+0.00821; p=0.5097) | `runs/E0955_quick_test402_cace_evt_gini_v2_20260213-041313/` | diagnose: `runs/E0955_quick_test402_cace_evt_gini_v2_20260213-041313/diagnose.json`; skip full |
| E0956 | success | val402 sweep (CACE evt; top1med_norm): best=`ltltop1medn_thr0p5_shift0`, Δ=-0.00474; p=0.5405 | `runs/E0956_val402_cace_evt_top1med_norm_v1_20260213-041404/` | not promoted |
| E0957 | success | export CACE-Net eventness scores (raw videos; 4 frames/sec): unique_vids=4097 | `runs/E0957_export_cace_evt_rawfps4_20260213-042012/` | `cace_evt_scores.json` |
| E0958 | success | val402 sweep (CACE evt; rawfps4; gini_v2): best=`ltlgini2_gini0p45_shift0`, Δ=-0.00640; p=0.5632 | `runs/E0958_val402_cace_evt_rawfps4_gini_v2_20260213-043133/` | not promoted |
| E0960 | success | export PSP/CPSP eventness scores (processed frames): unique_vids=4097 | `runs/E0960_export_psp_evt_20260213-050441/` | `psp_evt_scores.json` |
| E0961 | success | val402 sweep (PSP evt; gini_v2): best=`ltlgini2_gini0p5_shift0`, Δ=+0.00582; p=0.5060 | `runs/E0961_val402_psp_evt_gini_v2_20260213-050917/` | promoted to quick |
| E0962 | success | quick test402 (PSP evt; gini_v2 best): anchored=0.73060 vs uniform=0.71294 (Δ=+0.01766; p=0.2408) | `runs/E0962_quick_test402_psp_evt_gini0p5_20260213-051204/` | diagnose: `runs/E0962_quick_test402_psp_evt_gini0p5_20260213-051204/diagnose.json`; promoted to full |
| E0963 | success | full test402 (PSP evt; gini_v2 best): anchored=0.72983 vs uniform=0.71622 (Δ=+0.01361; p=0.0319) | `runs/E0963_full_test402_psp_evt_gini0p5_s0-9_20260213-051328/` | diagnose: `runs/E0963_full_test402_psp_evt_gini0p5_s0-9_20260213-051328/diagnose.json`; still < +2% |
| E0964 | success | val402 sweep (PSP evt; gini_v1): best=`ltl_gini0p20_scoreAlloc`, Δ=-0.00549; p=0.5628 | `runs/E0964_val402_psp_evt_gini_v1_20260213-051607/` | not promoted |
| E0965 | success | val402 sweep (PSP evt; top1med_dropfar): best=`ltltop1med_thr0p5_shift1_df1`, Δ=+0.00025; p=0.9804 | `runs/E0965_val402_psp_evt_top1med_dropfar_v1_20260213-051848/` | not promoted |
| E0966 | success | val402 sweep (PSP evt; gini_dropfar): best=`ltlgini_df1_gini0p35_shift1`, Δ=+0.00732; p=0.4417 | `runs/E0966_val402_psp_evt_gini_dropfar_v1_20260213-052227/` | promoted to quick only |
| E0967 | success | quick test402 (PSP evt; gini_dropfar best): anchored=0.72231 vs uniform=0.71294 (Δ=+0.00937; p=0.4846) | `runs/E0967_quick_test402_psp_evt_gini_dropfar_best_20260213-052700/` | not promoted |
| E0970 | success | val402 sweep (PSP evt; top1med_visfb): best=`ltltop1med_uniformfb_shift1`, Δ=-0.00665; p=0.4460 | `runs/E0970_val402_psp_evt_top1med_visfb_v1_20260213-054752/` | visual fallback variants regress; not promoted |
| E0971 | success | val402 sweep (PSP evt; gini_visfb): best=`ltlgini_visfb_uniform_shift0`, Δ=+0.00582; p=0.5060 | `runs/E0971_val402_psp_evt_gini_visfb_v1_20260213-055449/` | best stays uniform fallback; not promoted |
| E0972 | failed | export PSP evt (raw videos; rawfps4; sharded): no outputs written | `runs/E0972_export_psp_evt_rawfps4_20260213-060205/` | shard logs: `artifacts/experiments/E0972/shard*.log`; OUT_DIR empty |
| E0973 | success | val402 sweep (PSP evt; gap_v1): best=`ltlgap1_gap0p5_shift0`, Δ=-0.00249; p=0.7516 | `runs/E0973_val402_psp_evt_gap_v1_20260214-025354/` | not promoted |
| E0974 | success | val402 sweep (PSP evt; gini_keepadj_v1): best=`ltlgini_keepadj_df1_gini0p45_shift0`, Δ=+0.00623; p=0.3742 | `runs/E0974_val402_psp_evt_gini_keepadj_v1_20260214-025940/` | promoted to quick |
| E0975 | success | quick test402 (PSP evt; keepadj best): anchored=0.73441 vs uniform=0.71294 (Δ=+0.02148; p=0.1307) | `runs/E0975_quick_test402_psp_evt_gini_keepadj_best_20260214-030312/` | diagnose fallback_used_frac≈0.709; promoted to full |
| E0976 | success | full test402 (PSP evt; keepadj best): anchored=0.73348 vs uniform=0.71622 (Δ=+0.01726; p=0.00167) | `runs/E0976_full_test402_psp_evt_gini_keepadj_best_s0-9_20260214-030359/` | diagnose fallback_used_frac≈0.709; still < +2% |
| E0977 | success | val402 sweep (PSP evt; keepadj basealloc): best=`ltlgini_keepadj_thr0p45_df1_shift0_distance`, Δ=+0.00623; p=0.3742 | `runs/E0977_val402_psp_evt_gini_keepadj_basealloc_v1_20260214-030703/` | no improvement; not promoted |
| E0978 | success | val402 sweep (PSP evt; keepadj hconf): best=`ltlgini_keepadj_gini0p45_hconf0p5`, Δ=+0.00765; p=0.6190 | `runs/E0978_val402_psp_evt_gini_keepadj_hconf_v1_20260214-030933/` | promoted to quick |
| E0979 | success | quick test402 (PSP evt; keepadj hconf best): anchored=0.73806 vs uniform=0.71294 (Δ=+0.02512; p=0.1567) | `runs/E0979_quick_test402_psp_evt_gini_keepadj_hconf_best_20260214-031126/` | diagnose fallback_used_frac≈0.709; promoted to full |
| E0980 | success | full test402 (PSP evt; keepadj hconf best): anchored=0.73791 vs uniform=0.71622 (Δ=+0.02169; p=0.00149) | `runs/E0980_full_test402_psp_evt_gini_keepadj_hconf_best_s0-9_20260214-031741/` | diagnose fallback_used_frac≈0.709; **C0003 hard gate met** |

> Note: The authoritative runnable queue for the current `docs/plan.md` is the checklist above. The `## Experiments` catalog below is an archive; its internal `[ ]` fields are not a TODO list.

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
| Logs | `artifacts/experiments/E0003/run.log` and `runs/REAL_AVE_OFFICIAL_RERUN_*` |
| Results | Local rerun root: `runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402`. Anchors: `E0002_anchors_official_val/anchor_eval/anchors_metrics.json` (n=401; Δ0 ours=0.2315 > random=0.2111) and `E0002_anchors_official_test/anchor_eval/anchors_metrics.json` (n=402; Δ0 ours=0.2307 > random=0.2035). AVE-P0 (token_budget=1960; SEEDS=0..9; head=temporal_conv; energy; 160/224/352; shift=1; std_thr=1.0; allow-missing drops some clips: train=3312, val=401): train→val `p0_train3339_val402_energy_160_224_352_k2_shift1_std1.0_temporal_conv/metrics.json` (anchored=0.7457 > uniform=0.7402 > random=0.7268; p(anchored vs uniform)=0.239) and train→test `p0_train3339_test402_energy_160_224_352_k2_shift1_std1.0_temporal_conv/metrics.json` (anchored=0.7202 > uniform=0.7162 > random=0.7025; p(anchored vs uniform)=0.383). |

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


### E0010: AVE oracle ceiling sweep (MDE-1 Oracle anchors)
| Field | Value |
| --- | --- |
| Objective | Estimate the mechanism upper bound by comparing `uniform` vs `oracle_top2` under equal token budget on official AVE val402/test402 across a small, pre-registered config set. |
| Baseline | `uniform` |
| Model | Cached CLIP ViT-B/16 features + (`avs.models.per_segment_mlp.PerSegmentMLP` or `avs.models.temporal_conv.TemporalConvHead`) |
| Weights | HF: CLIP (`VISION_PRETRAINED=1`) |
| Code path | `avs/experiments/ave_p0_oracle_sweep.py`, `scripts/e0010_ave_oracle_ceiling_official.sh` |
| Params | `SPLIT_EVAL={val,test}`, `LIMIT_TRAIN`, `LIMIT_EVAL`, `SEEDS`, `EPOCHS`, `BATCH_SIZE`, `LR`, `CACHE_RESOLUTIONS`, `CACHES_DIR` |
| Metrics (must save) | `oracle_ceiling.json` (with per-config `metrics_path`, mean/std, paired p-values) |
| Checks | `oracle_minus_uniform_mean` is non-trivial for at least one config; report best oracle gain and its p-value. |
| VRAM | Train: ~1–2GB per seed on GPU; cache build heavy (feature extraction). |
| Time/epoch | ~seconds to minutes (head only) |
| Total time | ~tens of minutes (depends on caches) |
| Single-GPU script | `scripts/e0010_ave_oracle_ceiling_official.sh` |
| Multi-GPU script | `scripts/e0010_ave_oracle_ceiling_official.sh` (uses all visible GPUs for cache build) |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 BATCH_SIZE=16 bash scripts/e0010_ave_oracle_ceiling_official.sh` |
| Full cmd | `SPLIT_EVAL=val bash scripts/e0010_ave_oracle_ceiling_official.sh && SPLIT_EVAL=test bash scripts/e0010_ave_oracle_ceiling_official.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0010_*` |
| Artifacts | `runs/E0010_*/oracle_ceiling.json` and per-config `metrics.json` under the same dir |
| Results | Full val402: `runs/E0010_oracle_ceiling_official_val_20260203-144421/oracle_ceiling.json` (best oracle config: `160_224_352_temporal_k5`, oracle=0.7931 vs uniform=0.7311, Δ=+0.0620, p=8.9e-05). Full test402: `runs/E0010_oracle_ceiling_official_test_20260203-143455/oracle_ceiling.json` (best oracle config: `160_224_352_temporal_k5`, oracle=0.7663 vs uniform=0.7126, Δ=+0.0536, p=8.9e-06). |


### E0011: AVE fixed-space sweep on val402 (select best config)
| Field | Value |
| --- | --- |
| Objective | Run a fixed-space sweep to select the best sampling config on val402, producing a machine-readable `sweep_summary.json` + `best_config.json`. |
| Baseline | `uniform` |
| Model | Cached CLIP ViT-B/16 features + temporal head (`temporal_conv` default in candidates) |
| Weights | HF: CLIP (`VISION_PRETRAINED=1`) |
| Code path | `avs/experiments/ave_p0_sweep.py`, `scripts/e0011_ave_p0_sweep_official.sh` |
| Params | `EVENTNESS`, `ANCHOR_STD_THRESHOLD`, `SEEDS`, `P_FILTER`, candidate grid in `avs/experiments/ave_p0_sweep.py` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, and per-config `metrics.json` |
| Checks | Summary includes `anchored_top2 - uniform` mean deltas and paired p-values; best config is reproducible (stable ordering). |
| VRAM | Similar to E0010; cache build heavy if missing. |
| Time/epoch | ~seconds to minutes |
| Total time | ~tens of minutes to hours (depends on grid size) |
| Single-GPU script | `scripts/e0011_ave_p0_sweep_official.sh` |
| Multi-GPU script | `scripts/e0011_ave_p0_sweep_official.sh` (uses all visible GPUs for cache build) |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 BATCH_SIZE=16 bash scripts/e0011_ave_p0_sweep_official.sh` |
| Full cmd | `bash scripts/e0011_ave_p0_sweep_official.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0011_*` |
| Artifacts | `runs/E0011_*/sweep_summary.json`, `runs/E0011_*/best_config.json`, `runs/E0011_*/candidates/*/metrics.json` |
| Results | Full val402 sweep: `runs/E0011_ave_p0_sweep_official_val_20260203-143455/sweep_summary.json` (best=`runs/E0011_ave_p0_sweep_official_val_20260203-143455/best_config.json`=`base_160_224_352_k5`; candidate metrics: `runs/E0011_ave_p0_sweep_official_val_20260203-143455/base_160_224_352_k5/metrics.json`, anchored=0.7429 vs uniform=0.7294, Δ=+0.0134, p=0.017). |


### E0012: AVE best-config reproduction on test402 (val→test)
| Field | Value |
| --- | --- |
| Objective | Reproduce the val-selected `best_config.json` on official test402 with `SEEDS=0..9` (strict equal token budget). |
| Baseline | `uniform` |
| Model | Same as the selected config in `best_config.json` |
| Weights | HF: CLIP (`VISION_PRETRAINED=1`) |
| Code path | `scripts/e0012_ave_p0_best_to_test_official.sh`, `avs/experiments/ave_p0_rerun.py` |
| Params | `BEST_CONFIG_JSON`, `SEEDS`, `LIMIT_TRAIN`, `LIMIT_EVAL`, `SPLIT_EVAL=test` |
| Metrics (must save) | `metrics.json` + `diagnose.json` (optional) |
| Checks | Paired t-test `anchored_top2` vs `uniform` and report delta; store the exact `best_config.json` used. |
| VRAM | Similar to E0003 (head only) |
| Time/epoch | ~seconds to minutes |
| Total time | ~tens of minutes (depends on epochs) |
| Single-GPU script | `scripts/e0012_ave_p0_best_to_test_official.sh` |
| Multi-GPU script | `scripts/e0012_ave_p0_best_to_test_official.sh` |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 BATCH_SIZE=16 bash scripts/e0011_ave_p0_sweep_official.sh && BEST_CONFIG_JSON=$(pwd)/runs/E0011_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 BATCH_SIZE=16 bash scripts/e0012_ave_p0_best_to_test_official.sh` |
| Full cmd | `bash scripts/e0011_ave_p0_sweep_official.sh && BEST_CONFIG_JSON=$(pwd)/runs/E0011_*/best_config.json bash scripts/e0012_ave_p0_best_to_test_official.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0012_*` |
| Artifacts | `runs/E0012_*/metrics.json` |
| Results | Full test402: `runs/E0012_ave_p0_best_to_test_official_20260203-145743/metrics.json` (best config=`base_160_224_352_k5`; anchored=0.7144 vs uniform=0.7123, Δ=+0.0021, p=0.593; anchored vs random p=0.036). Follow-up (tuned `anchor_std_threshold=0.9`): `runs/E0012_energy_k5_std0p9_test_full_20260203-153623/metrics.json` (anchored=0.7196 vs uniform=0.7123, Δ=+0.0073, p=0.181). |


### E0013: AVE fusion confirm under best sampling config (audio_concat_* baselines)
| Field | Value |
| --- | --- |
| Objective | Confirm whether fusion improves on top of sampling by comparing `audio_concat_anchored_top2` vs `audio_concat_uniform` under the best sampling config. |
| Baseline | `audio_concat_uniform` |
| Model | Same as the selected config + audio concat head |
| Weights | HF: CLIP (`VISION_PRETRAINED=1`) |
| Code path | `scripts/e0013_ave_fusion_confirm_official.sh`, `avs/experiments/ave_p0_fusion_confirm.py` |
| Params | `BEST_CONFIG_JSON`, `SEEDS`, `LIMIT_TRAIN`, `LIMIT_EVAL` |
| Metrics (must save) | `metrics.json` (must include `audio_concat_uniform` and `audio_concat_anchored_top2`) |
| Checks | Paired t-test `audio_concat_anchored_top2` vs `audio_concat_uniform` and report delta. |
| VRAM | Similar to E0003 (head only) |
| Time/epoch | ~seconds to minutes |
| Total time | ~tens of minutes |
| Single-GPU script | `scripts/e0013_ave_fusion_confirm_official.sh` |
| Multi-GPU script | `scripts/e0013_ave_fusion_confirm_official.sh` |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 BATCH_SIZE=16 bash scripts/e0011_ave_p0_sweep_official.sh && BEST_CONFIG_JSON=$(pwd)/runs/E0011_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 BATCH_SIZE=16 bash scripts/e0013_ave_fusion_confirm_official.sh` |
| Full cmd | `bash scripts/e0011_ave_p0_sweep_official.sh && BEST_CONFIG_JSON=$(pwd)/runs/E0011_*/best_config.json bash scripts/e0013_ave_fusion_confirm_official.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0013_*` |
| Artifacts | `runs/E0013_*/metrics.json` |
| Results | Full test402: `runs/E0013_ave_fusion_confirm_official_test_20260203-150226/metrics.json` (audio_concat_anchored=0.7213 vs audio_concat_uniform=0.7162, Δ=+0.0051, p=0.367). Note: initial full job failed because `BEST_CONFIG_JSON` auto-detection matched a non-sweep `runs/E0011_*` dir; fixed in `scripts/e0013_ave_fusion_confirm_official.sh` by selecting the newest `runs/E0011_*/best_config.json`. |


### E0014: AVE AST sweep on val402 (select best config; candidate_set=ast_v1)
| Field | Value |
| --- | --- |
| Objective | Run an AST-based fixed-space sweep to select the best sampling config on val402, producing `sweep_summary.json` + `best_config.json` for later test402 reproduction. |
| Baseline | `uniform` |
| Model | Cached CLIP ViT-B/16 features + temporal head (`temporal_conv` default in candidates) |
| Weights | HF: AST (`--ast-pretrained`) |
| Code path | `avs/experiments/ave_p0_sweep.py` (candidate_set=`ast_v1`), `scripts/e0014_ave_p0_sweep_official_ast.sh` |
| Params | `SEEDS`, `EPOCHS`, `P_FILTER`, `AUDIO_DEVICE`, `TRAIN_DEVICE`, candidate grid in `avs/experiments/ave_p0_sweep.py::_candidates_ast_v1` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, and per-config `metrics.json` |
| Checks | Summary includes `anchored_top2 - uniform` mean deltas and paired p-values; best config is reproducible (stable ordering). |
| VRAM | AST inference + head training (light; caches must exist). |
| Time/epoch | ~seconds to minutes |
| Total time | ~tens of minutes (depends on epochs and devices) |
| Single-GPU script | `scripts/e0014_ave_p0_sweep_official_ast.sh` |
| Multi-GPU script | `scripts/e0014_ave_p0_sweep_official_ast.sh` (run multiple sweeps in parallel by splitting seeds or candidate subsets manually) |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 BATCH_SIZE=16 bash scripts/e0014_ave_p0_sweep_official_ast.sh` |
| Full cmd | `bash scripts/e0014_ave_p0_sweep_official_ast.sh` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0014_*` |
| Artifacts | `runs/E0014_*/sweep_summary.json`, `runs/E0014_*/best_config.json`, `runs/E0014_*/<candidate>/metrics.json` |
| Results | Val402: `runs/E0014_ave_p0_sweep_official_val_ast_20260203-163626/sweep_summary.json` (best=`ast_160_224_352_k5_std0p05`, Δ=+0.0037, p=0.458). Note: extreme triad `112/224/448` is strongly negative on val (Δ≈-0.061, p≈6e-7). |


### E0015: AVE AST best-config reproduction on test402 (val→test)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0014 on official AVE test402 with `SEEDS=0..9` and paired tests. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | HF: AST (`--ast-pretrained`) |
| Code path | `scripts/e0015_ave_p0_best_to_test_official_ast.sh` |
| Params | `BEST_CONFIG_JSON`, `SEEDS`, `LIMIT_TRAIN`, `LIMIT_EVAL`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (must include `summary.*.mean/std`, `paired_ttest`, and `debug_eval` for anchors/fallback stats) |
| Checks | `paired_ttest.anchored_vs_uniform.p` is recorded; report delta and fallback rate. |
| VRAM | Similar to E0003 (head only) |
| Time/epoch | ~seconds to minutes |
| Total time | ~tens of minutes |
| Single-GPU script | `scripts/e0015_ave_p0_best_to_test_official_ast.sh` |
| Multi-GPU script | `scripts/e0015_ave_p0_best_to_test_official_ast.sh` |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 BATCH_SIZE=16 bash scripts/e0014_ave_p0_sweep_official_ast.sh && BEST_CONFIG_JSON=$(ls -t runs/E0014_*/best_config.json | head -n 1) LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 BATCH_SIZE=16 bash scripts/e0015_ave_p0_best_to_test_official_ast.sh` |
| Full cmd | `bash scripts/e0014_ave_p0_sweep_official_ast.sh && BEST_CONFIG_JSON=$(ls -t runs/E0014_*/best_config.json | head -n 1) bash scripts/e0015_ave_p0_best_to_test_official_ast.sh` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0015_*` |
| Artifacts | `runs/E0015_*/metrics.json` |
| Results | Test402: `runs/E0015_ave_p0_best_to_test_official_ast_20260203-172848/metrics.json` (anchored=0.7003 vs uniform=0.7123, Δ=-0.0120, p=0.024; fallback_used≈0.169). |


### E0016: Diagnose why AST anchored gains do/do-not transfer (root-cause report)
| Field | Value |
| --- | --- |
| Objective | Produce an audit-friendly diagnosis: anchored gain distribution, fallback histograms, anchor distance buckets, and Recall@K↔delta correlation, to explain why test gains are (or are not) large. |
| Baseline | N/A (analysis-only) |
| Model | N/A |
| Weights | N/A |
| Code path | `avs/experiments/ave_p0_diagnose.py`, `scripts/e0016_ave_p0_diagnose_official.sh` |
| Params | `IN_METRICS` (full-run `metrics.json`), `--deltas` |
| Metrics (must save) | `diagnose.json` |
| Checks | Report includes fallback rate and per-bucket mean deltas; top improve/degrade lists exist. |
| VRAM | CPU |
| Time/epoch | N/A |
| Total time | minutes |
| Single-GPU script | `scripts/e0016_ave_p0_diagnose_official.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `IN_METRICS=$(ls -t runs/E0012_*/metrics.json | head -n 1) bash scripts/e0016_ave_p0_diagnose_official.sh` |
| Full cmd | `IN_METRICS=$(ls -t runs/E0015_*/metrics.json | head -n 1) bash scripts/e0016_ave_p0_diagnose_official.sh` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0016_*` |
| Artifacts | `runs/E0016_*/diagnose.json` |
| Results | `runs/E0016_ave_p0_diagnose_20260203-173704/diagnose.json` (mean Δ=-0.0120; fallback≈0.169; 2-anchor clips dominate and are harmful: mean Δ≈-0.017 for anchors_len=2; adjacent anchors are worse). |


### E0017: AVE AST fusion confirm under best sampling config (audio_concat_* baselines)
| Field | Value |
| --- | --- |
| Objective | Confirm whether fusion improves on top of AST-based sampling by comparing `audio_concat_anchored_top2` vs `audio_concat_uniform` under the best AST sampling config. |
| Baseline | `audio_concat_uniform` |
| Model | Same as the selected config + audio concat head |
| Weights | HF: AST (`--ast-pretrained`) |
| Code path | `avs/experiments/ave_p0_fusion_confirm.py`, `scripts/e0017_ave_fusion_confirm_official_ast.sh` |
| Params | `BEST_CONFIG_JSON`, `SEEDS`, `LIMIT_TRAIN`, `LIMIT_EVAL`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (must include `audio_concat_uniform` and `audio_concat_anchored_top2` plus paired tests) |
| Checks | Paired t-test `audio_concat_anchored_top2` vs `audio_concat_uniform` and report delta. |
| VRAM | Similar to E0003 (head only) |
| Time/epoch | ~seconds to minutes |
| Total time | ~tens of minutes |
| Single-GPU script | `scripts/e0017_ave_fusion_confirm_official_ast.sh` |
| Multi-GPU script | `scripts/e0017_ave_fusion_confirm_official_ast.sh` |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 BATCH_SIZE=16 bash scripts/e0014_ave_p0_sweep_official_ast.sh && BEST_CONFIG_JSON=$(ls -t runs/E0014_*/best_config.json | head -n 1) LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 BATCH_SIZE=16 bash scripts/e0017_ave_fusion_confirm_official_ast.sh` |
| Full cmd | `bash scripts/e0014_ave_p0_sweep_official_ast.sh && BEST_CONFIG_JSON=$(ls -t runs/E0014_*/best_config.json | head -n 1) bash scripts/e0017_ave_fusion_confirm_official_ast.sh` |
| Smoke | [x] |
| Full | [ ] |
| Logs | `runs/E0017_*` |
| Artifacts | `runs/E0017_*/metrics.json` |
| Results |  |


### E0018: AVE energy_v2 sweep on val402 (bigger gain search; transfer-focused)
| Field | Value |
| --- | --- |
| Objective | Run an energy-based sweep (`candidate_set=energy_v2`) to search for larger, more reliable anchored gains (diverse selection + adaptive high allocation) on val402. |
| Baseline | `uniform` |
| Model | Cached CLIP ViT-B/16 features + temporal head (`temporal_conv`) |
| Weights | HF: CLIP (`VISION_PRETRAINED=1`) |
| Code path | `avs/experiments/ave_p0_sweep.py` (candidate_set=`energy_v2`), `scripts/e0018_ave_p0_sweep_official_val_energy_v2.sh` |
| Params | `SEEDS`, `EPOCHS` (default 5), `P_FILTER`, `TRAIN_DEVICE`, candidate list in `avs/experiments/ave_p0_sweep.py::_candidates_energy_v2` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, and per-config `metrics.json` |
| Checks | Best config has positive Δ on val and is reproducible (stable ordering + guardrail). |
| VRAM | Low (head training); cache build dominates if missing. |
| Total time | tens of minutes |
| Single-GPU script | `scripts/e0018_ave_p0_sweep_official_val_energy_v2.sh` |
| Multi-GPU script | `scripts/e0018_ave_p0_sweep_official_val_energy_v2.sh` (split seeds across GPUs if desired) |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 BATCH_SIZE=16 bash scripts/e0018_ave_p0_sweep_official_val_energy_v2.sh` |
| Full cmd | `bash scripts/e0018_ave_p0_sweep_official_val_energy_v2.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0018_*` |
| Artifacts | `runs/E0018_*/sweep_summary.json`, `runs/E0018_*/best_config.json`, `runs/E0018_*/<candidate>/metrics.json` |
| Results | Val401: `runs/E0018_ave_p0_sweep_official_val_energy_v2_20260203-185629/sweep_summary.json` (best=`energy_ref_k2_topk_std1p0`, anchored=0.7378 vs uniform=0.7243, Δ=+0.01344, p=0.00175). Best config: `runs/E0018_ave_p0_sweep_official_val_energy_v2_20260203-185629/best_config.json`. |


### E0019: AVE energy_v2 best-config reproduction on test402 (val→test)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0018 on official AVE test402 with `SEEDS=0..9` and paired tests. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | HF: CLIP (`VISION_PRETRAINED=1`) |
| Code path | `scripts/e0019_ave_p0_best_to_test_official_energy_v2.sh` |
| Params | `BEST_CONFIG_JSON`, `SEEDS`, `LIMIT_TRAIN`, `LIMIT_EVAL`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (must include `summary.*.mean/std`, `paired_ttest`, and `debug_eval` for anchors/fallback stats) |
| Checks | Report delta and `paired_ttest.anchored_vs_uniform.p`. |
| VRAM | Similar to E0012 (head only) |
| Total time | tens of minutes |
| Single-GPU script | `scripts/e0019_ave_p0_best_to_test_official_energy_v2.sh` |
| Multi-GPU script | `scripts/e0019_ave_p0_best_to_test_official_energy_v2.sh` |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 BATCH_SIZE=16 bash scripts/e0018_ave_p0_sweep_official_val_energy_v2.sh && BEST_CONFIG_JSON=$(ls -t runs/E0018_*/best_config.json | head -n 1) LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 BATCH_SIZE=16 bash scripts/e0019_ave_p0_best_to_test_official_energy_v2.sh` |
| Full cmd | `bash scripts/e0018_ave_p0_sweep_official_val_energy_v2.sh && BEST_CONFIG_JSON=$(ls -t runs/E0018_*/best_config.json | head -n 1) bash scripts/e0019_ave_p0_best_to_test_official_energy_v2.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0019_*` |
| Artifacts | `runs/E0019_*/metrics.json` |
| Results | Test402: `runs/E0019_ave_p0_best_to_test_official_energy_v2_20260203-190500/metrics.json` (uniform=0.7086, anchored=0.7188, Δ=+0.01017, p=0.00466; fallback_used≈0.731). |


### E0020: AVE fusion confirm under energy_v2 best sampling config (audio_concat_* baselines)
| Field | Value |
| --- | --- |
| Objective | Confirm whether fusion improves on top of the energy_v2-selected sampling config by comparing `audio_concat_anchored_top2` vs `audio_concat_uniform`. |
| Baseline | `audio_concat_uniform` |
| Model | Same as the selected config + audio concat head |
| Weights | HF: CLIP (`VISION_PRETRAINED=1`) |
| Code path | `avs/experiments/ave_p0_fusion_confirm.py`, `scripts/e0020_ave_fusion_confirm_official_energy_v2.sh` |
| Params | `BEST_CONFIG_JSON`, `SEEDS`, `LIMIT_TRAIN`, `LIMIT_EVAL`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (must include `audio_concat_uniform` and `audio_concat_anchored_top2` plus paired tests) |
| Checks | Paired t-test `audio_concat_anchored_top2` vs `audio_concat_uniform` and report delta. |
| VRAM | Similar to E0013 (head only) |
| Total time | tens of minutes |
| Single-GPU script | `scripts/e0020_ave_fusion_confirm_official_energy_v2.sh` |
| Multi-GPU script | `scripts/e0020_ave_fusion_confirm_official_energy_v2.sh` |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 BATCH_SIZE=16 bash scripts/e0018_ave_p0_sweep_official_val_energy_v2.sh && BEST_CONFIG_JSON=$(ls -t runs/E0018_*/best_config.json | head -n 1) LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 BATCH_SIZE=16 bash scripts/e0020_ave_fusion_confirm_official_energy_v2.sh` |
| Full cmd | `bash scripts/e0018_ave_p0_sweep_official_val_energy_v2.sh && BEST_CONFIG_JSON=$(ls -t runs/E0018_*/best_config.json | head -n 1) bash scripts/e0020_ave_fusion_confirm_official_energy_v2.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0020_*` |
| Artifacts | `runs/E0020_*/metrics.json` |
| Results | Test402: `runs/E0020_ave_fusion_confirm_official_test_energy_v2_20260203-190804/metrics.json` (audio_concat_uniform=0.7214, audio_concat_anchored=0.7195, Δ=-0.0020, p=0.598). |


### E0021: AVE energy_v3 sweep on val402 (push for ≥+2% on test; extreme+gated candidates)
| Field | Value |
| --- | --- |
| Objective | Run an expanded energy sweep (`candidate_set=energy_v3`) targeting larger anchored gains via (1) less conservative confidence gating, (2) shift=0 variants to avoid 1-anchor drop, and (3) extreme triad variants with stable `max_high_anchors=1`. |
| Baseline | `uniform` |
| Model | Cached CLIP ViT-B/16 features + temporal head (`temporal_conv`) |
| Weights | HF: CLIP (`VISION_PRETRAINED=1`) |
| Code path | `avs/experiments/ave_p0_sweep.py` (candidate_set=`energy_v3`), `scripts/e0021_ave_p0_sweep_official_val_energy_v3.sh` |
| Params | `SEEDS`, `EPOCHS` (default 5), `P_FILTER`, `TRAIN_DEVICE`, candidate list in `avs/experiments/ave_p0_sweep.py::_candidates_energy_v3` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, and per-config `metrics.json` |
| Checks | Best config has positive Δ on val and is reproducible; prioritize configs with lower fallback and/or higher transfer. |
| VRAM | Low (head training); cache build dominates if missing. |
| Total time | tens of minutes |
| Single-GPU script | `scripts/e0021_ave_p0_sweep_official_val_energy_v3.sh` |
| Multi-GPU script | `scripts/e0021_ave_p0_sweep_official_val_energy_v3.sh` (split seeds across GPUs if desired) |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 BATCH_SIZE=16 bash scripts/e0021_ave_p0_sweep_official_val_energy_v3.sh` |
| Full cmd | `bash scripts/e0021_ave_p0_sweep_official_val_energy_v3.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0021_*` |
| Artifacts | `runs/E0021_*/sweep_summary.json`, `runs/E0021_*/best_config.json`, `runs/E0021_*/<candidate>/metrics.json` |
| Results | Val401: `runs/E0021_ave_p0_sweep_official_val_energy_v3_20260203-194306/sweep_summary.json` (best=`energyv3_ref_shift1_std1p0`, anchored=0.7456 vs uniform=0.7321, Δ=+0.01344, p=0.00175). Best config: `runs/E0021_ave_p0_sweep_official_val_energy_v3_20260203-194306/best_config.json`. |


### E0022: AVE energy_v3 best-config reproduction on test402 (val→test)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0021 on official AVE test402 with `SEEDS=0..9` and paired tests. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | HF: CLIP (`VISION_PRETRAINED=1`) |
| Code path | `scripts/e0022_ave_p0_best_to_test_official_energy_v3.sh` |
| Params | `BEST_CONFIG_JSON`, `SEEDS`, `LIMIT_TRAIN`, `LIMIT_EVAL`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (must include `summary.*.mean/std`, `paired_ttest`, and `debug_eval` for anchors/fallback stats) |
| Checks | Report delta and `paired_ttest.anchored_vs_uniform.p`; target Δ≥+0.02 and p<0.05. |
| VRAM | Similar to E0019 (head only) |
| Total time | tens of minutes |
| Single-GPU script | `scripts/e0022_ave_p0_best_to_test_official_energy_v3.sh` |
| Multi-GPU script | `scripts/e0022_ave_p0_best_to_test_official_energy_v3.sh` |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 BATCH_SIZE=16 bash scripts/e0021_ave_p0_sweep_official_val_energy_v3.sh && BEST_CONFIG_JSON=$(ls -t runs/E0021_*/best_config.json | head -n 1) LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 BATCH_SIZE=16 bash scripts/e0022_ave_p0_best_to_test_official_energy_v3.sh` |
| Full cmd | `bash scripts/e0021_ave_p0_sweep_official_val_energy_v3.sh && BEST_CONFIG_JSON=$(ls -t runs/E0021_*/best_config.json | head -n 1) bash scripts/e0022_ave_p0_best_to_test_official_energy_v3.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0022_*` |
| Artifacts | `runs/E0022_*/metrics.json` |
| Results | Test402: `runs/E0022_ave_p0_best_to_test_official_energy_v3_20260203-195707/metrics.json` (uniform=0.7086, anchored=0.7188, Δ=+0.01017, p=0.00466). |


### E0023: AVE fusion confirm under energy_v3 best sampling config (audio_concat_* baselines)
| Field | Value |
| --- | --- |
| Objective | Confirm whether fusion improves on top of the energy_v3-selected sampling config by comparing `audio_concat_anchored_top2` vs `audio_concat_uniform`. |
| Baseline | `audio_concat_uniform` |
| Model | Same as the selected config + audio concat head |
| Weights | HF: CLIP (`VISION_PRETRAINED=1`) |
| Code path | `avs/experiments/ave_p0_fusion_confirm.py`, `scripts/e0023_ave_fusion_confirm_official_energy_v3.sh` |
| Params | `BEST_CONFIG_JSON`, `SEEDS`, `LIMIT_TRAIN`, `LIMIT_EVAL`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (must include `audio_concat_uniform` and `audio_concat_anchored_top2` plus paired tests) |
| Checks | Paired t-test `audio_concat_anchored_top2` vs `audio_concat_uniform` and report delta. |
| VRAM | Similar to E0020 (head only) |
| Total time | tens of minutes |
| Single-GPU script | `scripts/e0023_ave_fusion_confirm_official_energy_v3.sh` |
| Multi-GPU script | `scripts/e0023_ave_fusion_confirm_official_energy_v3.sh` |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 BATCH_SIZE=16 bash scripts/e0021_ave_p0_sweep_official_val_energy_v3.sh && BEST_CONFIG_JSON=$(ls -t runs/E0021_*/best_config.json | head -n 1) LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 BATCH_SIZE=16 bash scripts/e0023_ave_fusion_confirm_official_energy_v3.sh` |
| Full cmd | `bash scripts/e0021_ave_p0_sweep_official_val_energy_v3.sh && BEST_CONFIG_JSON=$(ls -t runs/E0021_*/best_config.json | head -n 1) bash scripts/e0023_ave_fusion_confirm_official_energy_v3.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0023_*` |
| Artifacts | `runs/E0023_*/metrics.json` |
| Results |  |


### E0024: AVE longer training diagnostic (EPOCHS=20) on test402 (energy_ref)
| Field | Value |
| --- | --- |
| Objective | Check whether simply training the head longer closes the test402 gap for the current best sampling config. |
| Baseline | `uniform` |
| Model | Same as the current best config; only change is `EPOCHS=20` |
| Weights | HF: CLIP (`VISION_PRETRAINED=1`) |
| Code path | `scripts/e0019_ave_p0_best_to_test_official_energy_v2.sh` (override `EPOCHS=20`) |
| Params | `BEST_CONFIG_JSON`, `EPOCHS=20`, `SEEDS`, `LIMIT_TRAIN`, `LIMIT_EVAL`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` |
| Checks | If Δ does not increase vs E0019/E0022, rule out “train longer” as the main lever. |
| VRAM | Similar to E0019 |
| Total time | longer than E0019 (more epochs) |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=2 bash scripts/e0019_ave_p0_best_to_test_official_energy_v2.sh` |
| Full cmd | `EPOCHS=20 bash scripts/e0019_ave_p0_best_to_test_official_energy_v2.sh` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0024_*` |
| Artifacts | `runs/E0024_*/metrics.json` |
| Results | Test402: `runs/E0024_energy_ref_test402_epochs20_20260203-194640/metrics.json` (Δ=+0.00281, p=0.587). |


### E0025: AVE “increase K + score alloc” diagnostic on test402
| Field | Value |
| --- | --- |
| Objective | Check whether allocating more anchors (K>2) and using score-based base allocation improves transfer on test402. |
| Baseline | `uniform` |
| Model | Same backbone/head; change anchor/budget knobs only (`K=5`, `max_high_anchors=1`, `anchor_base_alloc=score`). |
| Weights | HF: CLIP (`VISION_PRETRAINED=1`) |
| Code path | `avs/experiments/ave_p0.py` (custom knobs) |
| Params | `K`, `MAX_HIGH_ANCHORS`, `ANCHOR_BASE_ALLOC`, `SEEDS`, split/ids files |
| Metrics (must save) | `metrics.json` |
| Checks | If Δ does not beat the best config (E0019/E0022), focus back on Stage-1 anchor quality. |
| VRAM | Similar to E0019 |
| Total time | tens of minutes |
| Smoke cmd | (ad-hoc) small subset run with `SEEDS=0,1` |
| Full cmd | (ad-hoc) full test402 run with `SEEDS=0..9` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0025_*` |
| Artifacts | `runs/E0025_*/metrics.json` |
| Results | Test402: `runs/E0025_energy_k5_maxHigh1_scoreAlloc_test402_20260203-200327/metrics.json` (Δ=+0.00876, p=0.0106). |


### E0100: EPIC-SOUNDS video-level multi-label classification (downstream proxy)
| Field | Value |
| --- | --- |
| Objective | Evaluate long-video benefit on a downstream task: compare `uniform` vs `audio_anchored` selection under fixed visual budget (`max_steps × base_res`) on EPIC-SOUNDS. |
| Baseline | `uniform` |
| Model | CLIP ViT-B/16 visual features + `avs.models.video_multilabel_head.VideoMultiLabelHead` |
| Weights | HF: CLIP (`--vision-pretrained`) |
| Code path | `avs/experiments/epic_sounds_video_cls.py`, `scripts/e0100_epic_video_cls_local.sh` |
| Params | `MAX_STEPS`, `BASE_RES`, `ANCHOR_RADIUS`, `BACKGROUND_STRIDE`, `SEEDS`, `VISION_PRETRAINED`, `DEVICE` |
| Metrics (must save) | `metrics.json` (mAP/macro-F1 + budget fields) |
| Checks | `audio_anchored` improves mAP over `uniform` on val for `SEEDS>=3`. |
| VRAM | GPU optional (feature extraction); head training is light. |
| Time/epoch | minutes |
| Total time | depends on EPIC data availability |
| Single-GPU script | `scripts/e0100_epic_video_cls_local.sh` |
| Multi-GPU script | `scripts/e0100_epic_video_cls_local.sh` (cache build can use multiple GPUs if wired) |
| Smoke cmd | `python -m avs.smoke epic_sounds_video_cls_synth` |
| Full cmd | `bash scripts/e0100_epic_video_cls_local.sh` (requires EPIC videos installed under `data/EPIC_SOUNDS/raw/videos/`) |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0100_*`, `artifacts/experiments/E0100/run.log` |
| Artifacts | `runs/E0100_epic_video_cls_local_audio_anchored_full_ms120_s64_20260209-045119/metrics.json`, `runs/E0100_epic_video_cls_local_uniform_full_ms120_s64_20260209-045119/metrics.json`, `runs/E0100_epic_video_cls_local_random_full_ms120_s64_20260209-045119/metrics.json` |
| Results | Real local EPIC run (subset; `limit_train_videos=64`, `limit_val_videos=64`, `max_seconds=120`, `max_steps=120`, `SEEDS=0,1,2`; budget=`max_steps × base_res`): audio_anchored `runs/E0100_epic_video_cls_local_audio_anchored_full_ms120_s64_20260209-045119/metrics.json` (mAP=`0.4356±0.0090`, macro_f1@0.5=`0.4041±0.0120`) vs uniform `runs/E0100_epic_video_cls_local_uniform_full_ms120_s64_20260209-045119/metrics.json` (mAP=`0.3826±0.0074`, macro_f1@0.5=`0.3512±0.0207`), ΔmAP=`+0.0530`, Δmacro_f1=`+0.0528`. random matches uniform on this setting: `runs/E0100_epic_video_cls_local_random_full_ms120_s64_20260209-045119/metrics.json`. |


### E0201: Oracle vs Predicted gap report (Listen-then-Look MDE-2)
| Field | Value |
| --- | --- |
| Objective | Quantify the Oracle→Predicted gap under a **fixed equal-token** protocol and confirm Predicted remains better than `uniform`/`random`. |
| Baseline | `oracle` (upper bound) |
| Model | Same backbone/head as the MDE harness |
| Weights | Per harness config |
| Code path | `avs/experiments/mde_ltl.py` |
| Params | `EVENTNESS_METHOD ∈ {energy, energy_delta, energy_stride_max, energy_autoshift_clipdiff, energy_autoshift_clipdiff_pos, av_fused, av_fused_prod, av_fused_clipdiff, av_fused_clipdiff_prod, moe_energy_clipdiff, vision_clipdiff, panns, psp_avel_evt, ast, ast_lr, ast_emb_lr, ast_evt_mlp, ast_mlp_cls, ast_mlp_cls_target, audio_basic_lr, audio_basic_mlp, audio_basic_tcn, audio_fbank_mlp, audio_fbank_tcn, audio_basic_mlp_cls, audio_basic_mlp_cls_target, av_basic_lr, av_basic_mlp, av_clipdiff_lr, av_clipdiff_mlp, av_clipdiff_framediff_mlp, av_clipdiff_fbank_mlp, av_clipdiff_vec_mlp, av_clipdiff_mlp_cls, av_clipdiff_mlp_cls_target, av_clip_mlp_cls, av_clip_mlp_cls_target, av_clipdiff_tcn, vision_mlp_cls, vision_mlp_cls_target, vision_binary_lr, vision_binary_mlp}`, `AST_PRETRAINED`, `AUDIO_DEVICE`, anchor source `{oracle,predicted,random,cheap_visual}`, `SEEDS`, split/ids files, optional confidence gate (`ANCHOR_CONF_METRIC/ANCHOR_CONF_THRESHOLD`). |
| Metrics (must save) | `oracle_vs_predicted.json` (summary + oracle_minus_predicted + p-values) |
| Checks | `oracle_minus_predicted` shrinks; Predicted beats `uniform` and `random` (p<0.05) under at least one eventness setting. |
| VRAM | TBD |
| Time/epoch | TBD |
| Total time | TBD |
| Single-GPU script | `EVENTNESS=energy bash scripts/e0201_oracle_vs_predicted_official.sh` |
| Multi-GPU script | Run different `EVENTNESS` on different GPUs: `{energy,energy_stride_max,av_fused}` |
| Smoke cmd | `python -m avs.experiments.mde_ltl oracle_vs_predicted --mode toy` |
| Full cmd | `EVENTNESS=energy bash scripts/e0201_oracle_vs_predicted_official.sh` (then rerun with `EVENTNESS=energy_stride_max` and `EVENTNESS=av_fused`) |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0201_*` |
| Artifacts | `runs/E0201_*/oracle_vs_predicted.json` |
| Results | Full (test402; SEEDS=0..9; token_budget=1960): `runs/E0201_full_energy_20260203-210017/oracle_vs_predicted.json` (energy: anchored=0.7188 vs uniform=0.7086, Δ=+0.01017, p=0.00466; oracle=0.7500; oracle_minus_predicted=0.03124). Energy + val-selected confidence gate (`gini@0.35`, from E0204): `runs/E0201_full_energy_gini0p35_test402_20260204-041950/oracle_vs_predicted.json` (anchored=0.7193 vs uniform=0.7086, Δ=+0.01075, p=0.0244; oracle_minus_predicted=0.03067). `runs/E0201_full_energy_stride_max_20260203-210017/oracle_vs_predicted.json` (energy_stride_max: anchored=0.7175 vs uniform=0.7086, Δ=+0.00888, p=0.00426; oracle_minus_predicted=0.03254). `runs/E0201_full_av_fused_20260203-210018/oracle_vs_predicted.json` collapsed due to confidence-gate scale mismatch (`std_thr=1.0` on `[0,1]` fused scores → fallback≈1.0). After fixing scale (P0052), a too-large scale made the gate overly permissive and regressed: `runs/E0201_oracle_vs_predicted_av_fused_20260203-214142/oracle_vs_predicted.json` (fallback_used_frac≈0.144; Δ=-0.00276). With the current calibrated scale (AV_FUSED_SCORE_SCALE=3.5): `runs/E0201_oracle_vs_predicted_av_fused_scale3p5_full_20260203-221906/oracle_vs_predicted.json` (av_fused: anchored=0.7190 vs uniform=0.7086, Δ=+0.01045, p=0.00995; oracle_minus_predicted=0.03097; fallback_used_frac≈0.739; `INCLUDE_CHEAP_VISUAL=0`). Autoshift diagnostics (does not beat energy): `runs/E0201_full_energy_autoshift_clipdiff_test402_20260204-032327/oracle_vs_predicted.json` (energy_autoshift_clipdiff: anchored=0.7154 vs uniform=0.7086, Δ=+0.00687, p=0.00574; oracle_minus_predicted=0.03455). `runs/E0201_full_energy_autoshift_clipdiff_pos_test402_20260204-040122/oracle_vs_predicted.json` (energy_autoshift_clipdiff_pos: anchored=0.7111 vs uniform=0.7086, Δ=+0.00254, p=0.495; oracle_minus_predicted=0.03888). Diagnostics: `runs/E0201_oracle_vs_predicted_audio_basic_lr_s012_20260203-223008/oracle_vs_predicted.json` (audio_basic_lr; SEEDS=0..2; fallback_used_frac≈0.978 under std_thr=1.0, so anchored≈uniform). `runs/E0201_oracle_vs_predicted_energy_gini0p4_s012_20260203-223901/oracle_vs_predicted.json` (energy + gini gate 0.4; SEEDS=0..2; Δ≈+0.0077). `runs/E0201_oracle_vs_predicted_energy_gini0p38_s012_20260203-224239/oracle_vs_predicted.json` (energy + gini gate 0.38; SEEDS=0..2; regresses). New diagnostics: `runs/E0201_oracle_vs_predicted_audio_fbank_mlp_20260204-000652/oracle_vs_predicted.json` (audio_fbank_mlp: anchored=0.7138 vs uniform=0.7086, Δ=+0.00522, p=0.0997). `runs/E0201_full_audio_basic_tcn_gini0p35_test402_20260204-062236/oracle_vs_predicted.json` (audio_basic_tcn + gini@0.35: anchored=0.7114 vs uniform=0.7086, Δ=+0.00279, p=0.528; oracle_minus_predicted=0.03863). `runs/E0201_full_audio_fbank_tcn_gini0p35_test402_20260204-062236/oracle_vs_predicted.json` (audio_fbank_tcn + gini@0.35: anchored=0.7081 vs uniform=0.7086, Δ=-0.00045, p=0.878; oracle_minus_predicted=0.04187). `runs/E0201_oracle_vs_predicted_panns_20260204-001813/oracle_vs_predicted.json` (panns; regresses). Gate-tuned ast_lr (val402 best gate=`gini@0.3`): `runs/E0201_oracle_vs_predicted_ast_lr_gini0p3_test402_20260204-021751/oracle_vs_predicted.json` (ast_lr: anchored=0.70886 vs uniform=0.70858, Δ=+0.00027, p=0.938; oracle_minus_predicted≈0.0411). Current best deployable Stage-1 (`av_clipdiff_mlp`; top1-med gate@0.6): `runs/E0201_oracle_vs_predicted_av_clipdiff_mlp_20260204-213240/oracle_vs_predicted.json` (predicted: anchored=0.7162 vs uniform=0.7086, Δ=+0.00759, p=0.0945; oracle_minus_predicted=0.03383; cheap_visual: anchored=0.7143 vs uniform=0.7086, Δ=+0.00575, p=0.0661). PSP Stage-1 (scores cache from E0978; base config from E0978; SEEDS=0..9): `runs/E0201_oracle_vs_predicted_psp_avel_evt_20260214-153740/oracle_vs_predicted.json` (anchored=0.73697 vs uniform=0.73007, Δ=+0.00689, p=0.1445; oracle=0.75438; cheap_visual included). |


### E0204: Confidence-gate sweep on val402 (select gate, then reuse on test)
| Field | Value |
| --- | --- |
| Objective | Avoid tuning `anchor_conf_metric/threshold` on test402 by selecting a gate on val402 for a given Stage-1 method, then re-running test402 full with the selected gate. |
| Baseline | Default gate (legacy `anchor_std_threshold`) |
| Model | Same backbone/head as E0201 (P0 head-only training) |
| Weights | Per harness config (AST optional via `--ast-pretrained`) |
| Code path | `avs/experiments/mde_ltl.py` (`gate_sweep`) |
| Params | `EVENTNESS_METHOD`, `GATE_METRIC`, `GATE_THRESHOLDS`, `SEEDS`, `AST_PRETRAINED`, `AUDIO_DEVICE`, plus fixed P0 cfg knobs (res triad, k, select, alloc). |
| Metrics (must save) | `gate_sweep.json`, `best_gate.json`, and per-threshold `metrics_gate_*.json`. |
| Checks | Best gate is selected on val; reusing it on test does not regress vs the default gate. |
| VRAM | TBD |
| Time/epoch | TBD |
| Total time | TBD |
| Single-GPU script | N/A |
| Multi-GPU script | Run different methods on different GPUs (AST inference can be GPU-bound). |
| Smoke cmd | `python -m avs.experiments.mde_ltl gate_sweep --mode toy` |
| Full cmd | `python -m avs.experiments.mde_ltl gate_sweep --mode ave_official --split-eval val --limit-eval 402 --eventness-method ast_lr --ast-pretrained --audio-device cuda:0 --gate-metric gini --gate-thresholds 0,0.1,0.2,0.3,0.4` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0204_*` |
| Artifacts | `runs/E0204_*/gate_sweep.json`, `runs/E0204_*/best_gate.json`, `runs/E0204_*/metrics_gate_*.json` |
| Results | Val402 (SEEDS=0..2): energy best gate=`gini@0.35`, Δ≈+0.01189 (p≈0.030): `runs/E0204_gate_sweep_energy_val402_20260204-041625/gate_sweep.json` + `runs/E0204_gate_sweep_energy_val402_20260204-041625/best_gate.json`. Other sweeps do not beat energy: moe_energy_clipdiff best gate=`gini@0.3`, Δ≈+0.01097 (p≈0.0819): `runs/E0204_gate_sweep_moe_energy_clipdiff_val402_20260204-050919/gate_sweep.json`. energy_stride_max best gate=`gini@0.4`, Δ≈+0.00648: `runs/E0204_gate_sweep_energy_stride_max_val402_20260204-050600/gate_sweep.json`. energy+window_topk best gate=`gini@0.4`, Δ≈+0.00623: `runs/E0204_gate_sweep_energy_window3_val402_20260204-050559/gate_sweep.json`. ast_evt_mlp best gate=`gini@0.3`, Δ≈+0.00349: `runs/E0204_gate_sweep_ast_evt_mlp_val402_20260204-054808/gate_sweep.json`. ast_lr best gate=`gini@0.3`, Δ≈+0.00058: `runs/E0204_gate_sweep_ast_lr_val402_20260204-020623/gate_sweep.json`. av_basic_lr best gate=`gini@0.4`, Δ≈+0.00075: `runs/E0204_gate_sweep_av_basic_lr_val402_20260204-052137/gate_sweep.json`. audio_basic_tcn best gate=`gini@0.45` but regresses (Δ≈-0.00291): `runs/E0204_gate_sweep_audio_basic_tcn_val402_20260204-062235/gate_sweep.json` + `runs/E0204_gate_sweep_audio_basic_tcn_val402_20260204-062235/best_gate.json`. audio_fbank_tcn best gate=`gini@0.45` but regresses (Δ≈-0.00249): `runs/E0204_gate_sweep_audio_fbank_tcn_val402_20260204-062235/gate_sweep.json` + `runs/E0204_gate_sweep_audio_fbank_tcn_val402_20260204-062235/best_gate.json`. vision-based gates regress (Δ<0): `runs/E0204_gate_sweep_vision_mlp_cls_val402_20260204-050601/gate_sweep.json`, `runs/E0204_gate_sweep_vision_clipdiff_val402_20260204-050918/gate_sweep.json`, `runs/E0204_gate_sweep_vision_binary_lr_val402_20260204-053417/gate_sweep.json`, `runs/E0204_gate_sweep_vision_binary_mlp_val402_20260204-053418/gate_sweep.json`. |


### E0205: Audio-TCN config sweep on val402 (select best anchored plan)
| Field | Value |
| --- | --- |
| Objective | Under a fixed head-only protocol, select the best anchored plan config for a learned audio-TCN Stage-1 method (`audio_basic_tcn` / `audio_fbank_tcn`) on val402 (no tuning on test). |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as energy sweeps |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` |
| Params | `EVENTNESS ∈ {audio_basic_tcn,audio_fbank_tcn}`, `SEEDS`, `EPOCHS/BATCH_SIZE/LR`, splits/ids files, `ALLOW_MISSING`, `candidate_set=energy_v3` (stage-2 plan knob sweep), and cached `eventness_scores.json` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Best config on val has positive Δ and passes `p_filter` (default 0.1) when possible. |
| VRAM | ~<2GB per run (head-only; caches on disk) |
| Time/epoch | ~seconds to minutes |
| Total time | Depends on number of candidates × seeds |
| Single-GPU script | `EVENTNESS=audio_basic_tcn bash scripts/e0205_ave_p0_sweep_official_val_audio_tcn.sh` |
| Multi-GPU script | Run `audio_basic_tcn` and `audio_fbank_tcn` on different GPUs (train_device). |
| Smoke cmd | `EVENTNESS=audio_basic_tcn LIMIT_TRAIN=256 LIMIT_EVAL=64 SEEDS=0,1 EPOCHS=1 bash scripts/e0205_ave_p0_sweep_official_val_audio_tcn.sh` |
| Full cmd | `EVENTNESS=audio_basic_tcn SEEDS=0,1,2,3,4,5,6,7,8,9 bash scripts/e0205_ave_p0_sweep_official_val_audio_tcn.sh` |
| Smoke | [x] |
| Full | [ ] |
| Logs | `runs/E0205_*` |
| Artifacts | `runs/E0205_*/sweep_summary.json`, `runs/E0205_*/best_config.json`, `runs/E0205_*/eventness_scores.json` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0205_ave_p0_sweep_official_val_audio_basic_tcn_20260204-063746/sweep_summary.json` (best config=`energyv3_shift0_std1p0`; indicative only). Val402 (SEEDS=0..2): `runs/E0205_full_audio_basic_tcn_val402_20260204-063935/sweep_summary.json` (best=`energyv3_extreme_112_224_448_maxHigh1_shift0_std0p6`, Δ≈+0.01837, p≈0.0426; note: still needs SEEDS=0..9 “full selection” to be definitive). |


### E0206: Audio-TCN best-to-test reproduction on test402
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0205 on official test402 (SEEDS=0..9), and compare to the energy baseline for C0003 “拉大”. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`) |
| Params | `BEST_CONFIG_JSON` (from E0205), `EVENTNESS`, `SEEDS`, `EPOCHS`, and (optional) `SCORES_JSON` reused from E0205 to avoid recomputing Stage-1 scores. |
| Metrics (must save) | `metrics.json` (includes `summary` + `paired_ttest.anchored_vs_uniform`) |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; compare to energy best (~+1%). |
| VRAM | ~<2GB per run (head-only; caches on disk) |
| Time/epoch | ~seconds to minutes |
| Total time | ~tens of minutes |
| Single-GPU script | `BEST_CONFIG_JSON=... EVENTNESS=audio_basic_tcn bash scripts/e0206_ave_p0_best_to_test_official_audio_tcn.sh` |
| Multi-GPU script | Run different `EVENTNESS` on different GPUs. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0205_*/best_config.json EVENTNESS=audio_basic_tcn LIMIT_TRAIN=256 LIMIT_EVAL=64 SEEDS=0,1 EPOCHS=1 bash scripts/e0206_ave_p0_best_to_test_official_audio_tcn.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0205_*/best_config.json EVENTNESS=audio_basic_tcn SEEDS=0,1,2,3,4,5,6,7,8,9 bash scripts/e0206_ave_p0_best_to_test_official_audio_tcn.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0206_*` |
| Artifacts | `runs/E0206_*/metrics.json` |
| Results | Smoke (train64/test32; SEEDS=0,1; EPOCHS=1): `runs/E0206_ave_p0_best_to_test_official_audio_basic_tcn_20260204-064200/metrics.json`. Full (test402; SEEDS=0..9): `runs/E0206_ave_p0_best_to_test_official_audio_basic_tcn_20260204-070803/metrics.json` (best_config from `runs/E0205_full_audio_basic_tcn_val402_20260204-063935/best_config.json`: `energyv3_extreme_112_224_448_maxHigh1_shift0_std0p6`; anchored=0.7196 vs uniform=0.7086, Δ=+0.01097, p=0.0142). |


### E0207: Stage-2 plan sweep on val402 for clipdiff-augmented anchors (LTL “拉大”)
| Field | Value |
| --- | --- |
| Objective | Run a fixed-space Stage-2 plan sweep on official AVE val402 for a given Stage-1 anchor method (e.g., `av_clipdiff_mlp`) and write `best_config.json` for later test402 reproduction. |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as energy sweeps |
| Weights | Uses existing CLIP caches; no extra pretrained weights (unless `EVENTNESS` requires it) |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` |
| Params | `EVENTNESS`, `CANDIDATE_SET`, `SEEDS`, `EPOCHS/BATCH_SIZE/LR`, splits/ids files, `ALLOW_MISSING`, and cached `eventness_scores.json` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Best config on val has positive Δ and passes `p_filter` (default 0.1) when possible. |
| VRAM | ~<2GB per run (head-only; caches on disk) |
| Time/epoch | ~seconds to minutes |
| Total time | Depends on number of candidates × seeds |
| Single-GPU script | `EVENTNESS=av_clipdiff_lr bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` |
| Multi-GPU script | Run different `EVENTNESS` on different GPUs (train_device). |
| Smoke cmd | `EVENTNESS=av_clipdiff_lr LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` |
| Full cmd | `EVENTNESS=av_clipdiff_lr SEEDS=0,1,2 bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0207_*` |
| Artifacts | `runs/E0207_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_lr_20260204-073551/sweep_summary.json`. Full val sweeps (SEEDS=0..2): `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_lr_20260204-073816/sweep_summary.json` (energy_v3; best=`energyv3_shift0_std0p4_adaptiveGap0p6`), `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_lr_20260204-075118/sweep_summary.json` (ltl_gini_v1; best=`ltl_gini0p20_scoreAlloc`), `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_mlp_20260204-075914/sweep_summary.json` (energy_v3; best=`energyv3_shift1_std0p6`), `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_std_v1_20260204-083739/sweep_summary.json` (ltl_std_v1; best=`ltlstd_shift0_std0p45`), `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_std_v2_20260204-085103/sweep_summary.json` (ltl_std_v2; best=`ltlstd2_shift0_std0p5_mixedAlloc`), `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_gini_v2_20260204-090323/sweep_summary.json` (ltl_gini_v2; best=`ltlgini2_gini0p45_shift0`), `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_gap_v1_20260204-090353/sweep_summary.json` (ltl_gap_v1; best=`ltlgap1_gap0p3_shift0`), `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_extreme_v1_20260204-091309/sweep_summary.json` (ltl_extreme_v1; best=`ltlextreme1_shift1_std0p6`), `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_mlp_cls_ltl_std_v1_20260204-092157/sweep_summary.json` (av_clipdiff_mlp_cls; ltl_std_v1; best=`ltlstd_shift0_std0p55`), `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_mlp_cls_target_ltl_std_v1_20260204-092729/sweep_summary.json` (av_clipdiff_mlp_cls_target; ltl_std_v1; best=`ltlstd_shift1_std0p55`), `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_tcn_20260204-081451/sweep_summary.json` (energy_v3), `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_tcn_20260204-081451/best_config.json`, `runs/E0207_ave_p0_sweep_official_val_av_clip_mlp_cls_20260204-095132/sweep_summary.json` (av_clip_mlp_cls; ltl_std_v1; best=`ltlstd_shift1_std0p5`), `runs/E0207_ave_p0_sweep_official_val_av_clip_mlp_cls_target_20260204-095814/sweep_summary.json` (av_clip_mlp_cls_target; ltl_std_v1; best=`ltlstd_shift1_std0p55`), `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_tcn_20260204-100517/sweep_summary.json` (av_clipdiff_tcn; ltl_std_v1; best=`ltlstd_shift1_std0p55`), `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_20260204-101416/sweep_summary.json` (av_clipdiff_vec_mlp; ltl_std_v1; best=`ltlstd_shift1_std0p55`), `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_mlp_20260204-102403/sweep_summary.json` (av_clipdiff_mlp; ltl_adaptive_v1; best=`ltladj1_shift0_std0p45`). |

Notes (2026-02-10 reruns; artifact paths locally present):
- `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_adaptive_v1`: `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_adaptive_v1_20260210-182509/sweep_summary.json` (best=`ltladj1_shift0_std0p5`, Δ≈+0.00515, p≈0.204).
- `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_std_v1`: `runs/E0207_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_std_v1_20260210-183049/sweep_summary.json` (best=`ltlstd_shift1_std0p45`, Δ≈+0.00457, p≈0.618).
- `EVENTNESS=av_clipdiff_vec_mlp`, `CANDIDATE_SET=ltl_adaptive_v1`: `runs/E0610_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_adaptive_v1_20260210-200224/sweep_summary.json` (best=`ltladj2_shift1_std0p55`, Δ≈+0.00607, p≈0.248).
- `EVENTNESS=av_clipdiff_vec_mlp`, `CANDIDATE_SET=ltl_top1med_dropfar_v1`: `runs/E0620_val402_vecmlp_dropfar_20260210-222055/sweep_summary.json` (best=`ltltop1med_thr0p5_shift1_df0`, Δ≈-0.00058, p≈0.921).
- `EVENTNESS=av_clipdiff_vec_mlp`, `CANDIDATE_SET=ltl_top1med_farfb_v1`: `runs/E0622_val402_vecmlp_farfb_20260210-222352/sweep_summary.json` (best=`ltltop1med_thr0p5_shift1_ff0`, Δ≈-0.00058, p≈0.921).
- `EVENTNESS=av_clipdiff_vec_mlp`, `CANDIDATE_SET=ltl_adaptive_keepadj_v1` (adaptive_v3): `runs/E0624_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_adaptive_keepadj_v1_20260210-224555/sweep_summary.json` (best=`ltlkeepadj_adj2_shift1_std0p45`, Δ≈+0.00382, p≈0.384).


### E0208: Best-to-test reproduction on test402 for clipdiff-augmented anchors (LTL “拉大”)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0207 on official AVE test402 (SEEDS=0..9) and compare to the best energy baseline for C0003 “拉大”. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights (unless `EVENTNESS` requires it) |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0207), `EVENTNESS`, `SEEDS`, `EPOCHS`, and (optional) `SCORES_JSON` reused from E0207 to avoid recomputing Stage-1 scores. |
| Metrics (must save) | `metrics.json` (includes `summary` + `paired_ttest.anchored_vs_uniform`) |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; compare to energy best (~+1%). |
| VRAM | ~<2GB per run (head-only; caches on disk) |
| Time/epoch | ~seconds to minutes |
| Total time | ~tens of minutes |
| Single-GPU script | `BEST_CONFIG_JSON=... EVENTNESS=av_clipdiff_lr bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh` |
| Multi-GPU script | Run different `EVENTNESS` on different GPUs. |
| Smoke cmd | `EVENTNESS=av_clipdiff_lr LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh && BEST_CONFIG_JSON=runs/E0207_*/best_config.json EVENTNESS=av_clipdiff_lr LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh` |
| Full cmd | `EVENTNESS=av_clipdiff_lr SEEDS=0,1,2 bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh && BEST_CONFIG_JSON=runs/E0207_*/best_config.json EVENTNESS=av_clipdiff_lr SEEDS=0,1,2,3,4,5,6,7,8,9 bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0208_*` |
| Artifacts | `runs/E0208_*/metrics.json` |
| Results | Smoke (train64/test32; SEEDS=0,1; EPOCHS=1): `runs/E0208_ave_p0_best_to_test_official_av_clipdiff_lr_20260204-073616/metrics.json`. Full test402 (SEEDS=0..9): `runs/E0208_ave_p0_best_to_test_official_av_clipdiff_lr_20260204-074318/metrics.json` (Δ=+0.00759, p=0.0313; fallback≈0.826 under std_thr=0.4), `runs/E0208_ave_p0_best_to_test_official_av_clipdiff_lr_20260204-075503/metrics.json` (ltl_gini_v1; Δ=+0.00122, p=0.704; fallback≈0.047), `runs/E0208_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-080419/metrics.json` (anchored=0.72045 vs uniform=0.70858, Δ=+0.01187, p=0.00142; fallback≈0.883), `runs/E0208_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-080617/metrics.json` (alt config; Δ=+0.00985, p=0.00532; fallback≈0.709), `runs/E0208_ave_p0_best_to_test_official_av_clipdiff_mlp_ltl_std_v1_20260204-084147/metrics.json` (ltl_std_v1 best_config=`ltlstd_shift0_std0p45`; anchored=0.72065 vs uniform=0.70858, Δ=+0.01206, p=0.00464; fallback≈0.754), `runs/E0208_ave_p0_best_to_test_official_av_clipdiff_mlp_ltl_std_v2_20260204-085632/metrics.json` (ltl_std_v2 best_config=`ltlstd2_shift0_std0p5_mixedAlloc`; anchored=0.71724 vs uniform=0.70858, Δ=+0.00866, p=0.0472; fallback≈0.816), `runs/E0208_ave_p0_best_to_test_official_av_clipdiff_mlp_ltl_gini_v2_20260204-090710/metrics.json` (ltl_gini_v2 best_config=`ltlgini2_gini0p45_shift0`; anchored=0.71948 vs uniform=0.70858, Δ=+0.01090, p=0.0274), `runs/E0208_ave_p0_best_to_test_official_av_clipdiff_mlp_ltl_gap_v1_20260204-090741/metrics.json` (ltl_gap_v1 best_config=`ltlgap1_gap0p3_shift0`; anchored=0.71600 vs uniform=0.70858, Δ=+0.00741, p=0.0604), `runs/E0208_ave_p0_best_to_test_official_av_clipdiff_mlp_ltl_extreme_v1_20260204-091529/metrics.json` (ltl_extreme_v1 best_config=`ltlextreme1_shift1_std0p6`; anchored=0.71413 vs uniform=0.70858, Δ=+0.00555, p=0.0849), `runs/E0208_ave_p0_best_to_test_official_av_clipdiff_mlp_cls_ltl_std_v1_20260204-092530/metrics.json` (av_clipdiff_mlp_cls; ltl_std_v1 best_config=`ltlstd_shift0_std0p55`; anchored=0.71510 vs uniform=0.70858, Δ=+0.00652, p=0.00801), `runs/E0208_ave_p0_best_to_test_official_av_clipdiff_tcn_20260204-082007/metrics.json` (Δ=+0.00468, p=0.142), `runs/E0208_ave_p0_best_to_test_official_av_clip_mlp_cls_20260204-095509/metrics.json` (av_clip_mlp_cls; ltl_std_v1; Δ=+0.00281, p=0.328), `runs/E0208_ave_p0_best_to_test_official_av_clip_mlp_cls_target_20260204-100202/metrics.json` (av_clip_mlp_cls_target; ltl_std_v1; Δ=-0.00597, p=0.131), `runs/E0208_ave_p0_best_to_test_official_av_clipdiff_tcn_20260204-100855/metrics.json` (av_clipdiff_tcn; ltl_std_v1; Δ=+0.00540, p=0.161), `runs/E0208_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-103001/metrics.json` (av_clipdiff_mlp; ltl_adaptive_v1 best_config=`ltladj1_shift0_std0p45`; anchored=0.72234 vs uniform=0.70858, Δ=+0.01376, p=1.4e-05). |

Notes (2026-02-10 reruns; artifact paths locally present):
- Quick test402 (SEEDS=0..2) for `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_adaptive_v1`: `runs/E0208_quick_test402_av_clipdiff_mlp_ltl_adaptive_v1_20260210-182944/metrics.json` (Δ≈+0.00738, p≈0.577).
- Quick test402 (SEEDS=0..2) for `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_std_v1`: `runs/E0208_quick_test402_av_clipdiff_mlp_ltl_std_v1_20260210-183321/metrics.json` (Δ≈+0.00124, p≈0.927).
- Quick test402 (SEEDS=0..2) for `EVENTNESS=av_clipdiff_vec_mlp`, `CANDIDATE_SET=ltl_adaptive_v1`: `runs/E0611_quick_test402_av_clipdiff_vec_mlp_ltl_adaptive_v1_20260210-200638/metrics.json` (Δ≈+0.01061, p≈0.391; diagnose: `runs/E0611_quick_test402_av_clipdiff_vec_mlp_ltl_adaptive_v1_20260210-200638/diagnose.json`).
- Full test402 (SEEDS=0..9) for the above: `runs/E0612_full_test402_av_clipdiff_vec_mlp_ltl_adaptive_v1_20260210-200736/metrics.json` (Δ≈+0.00095, p≈0.875; diagnose: `runs/E0612_full_test402_av_clipdiff_vec_mlp_ltl_adaptive_v1_20260210-200736/diagnose.json`).
- Quick test402 (SEEDS=0..2) for `EVENTNESS=av_clipdiff_vec_mlp` with dropfar (`thr0p5_df1`): `runs/E0621_quick_test402_vecmlp_dropfar_df1_20260210-222801/metrics.json` (Δ≈+0.00166, p≈0.900; diagnose: `runs/E0621_quick_test402_vecmlp_dropfar_df1_20260210-222801/diagnose.json`; note: removes `high_count=2` bucket entirely but does not improve mean).
- Quick test402 (SEEDS=0..2) for `EVENTNESS=av_clipdiff_vec_mlp` with farfb (`thr0p5_ff1`): `runs/E0623_quick_test402_vecmlp_farfb_ff1_20260210-222856/metrics.json` (Δ≈+0.00274, p≈0.839; diagnose: `runs/E0623_quick_test402_vecmlp_farfb_ff1_20260210-222856/diagnose.json`; note: removes `high_count=2` + dist>1 by construction but does not improve mean).
- Quick test402 (SEEDS=0..2) for `EVENTNESS=av_clipdiff_vec_mlp`, `CANDIDATE_SET=ltl_adaptive_keepadj_v1` (adaptive_v3 winner): `runs/E0625_quick_test402_vecmlp_keepadj_v1_20260210-225003/metrics.json` (Δ≈+0.00829, p≈0.432; diagnose: `runs/E0625_quick_test402_vecmlp_keepadj_v1_20260210-225003/diagnose.json`).
- Quick test402 (SEEDS=0..2) for `EVENTNESS=av_clipdiff_vec_mlp`, `CANDIDATE_SET=ltl_adaptive_keepadj_v1` (target config `ltlkeepadj_adj2_shift1_std0p55`): `runs/E0626_quick_test402_vecmlp_keepadj_adj2_shift1_std0p55_20260210-225120/metrics.json` (Δ≈+0.02098, p≈0.165; diagnose: `runs/E0626_quick_test402_vecmlp_keepadj_adj2_shift1_std0p55_20260210-225120/diagnose.json`).
- Full test402 (SEEDS=0..9) for the above target config: `runs/E0628_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_s0-9_20260210-225216/metrics.json` (Δ≈+0.00883, p≈0.225; diagnose: `runs/E0628_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_s0-9_20260210-225216/diagnose.json`; highly unstable per-seed deltas, so C0003 remains unproven).
- Full test402 (SEEDS=0..9) for the val-selected keepadj winner (E0624 best_config): `runs/E0629_full_test402_vecmlp_keepadj_best_s0-9_20260210-225809/metrics.json` (Δ≈-0.00408, p≈0.457; diagnose: `runs/E0629_full_test402_vecmlp_keepadj_best_s0-9_20260210-225809/diagnose.json`; does not transfer).
- Follow-up (2026-02-11; official ids via `data/AVE/meta/download_ok_{train,test}_official.txt`): re-run keepadj with `anchor_drop_far_dist` to target far-anchor2 harm buckets (E0628 diagnose: `dist=6/8` are strongly negative).
  - df7 (`anchor_drop_far_dist=7`, drops only `dist>=8`): quick test402 (SEEDS=0..2): `runs/E0636_quick_test402_vecmlp_keepadj_adj2_shift1_std0p55_df7_officialids_20260211-000822/metrics.json` (Δ≈+0.01758, p≈0.142). Full test402 (SEEDS=0..9): `runs/E0643_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_df7_officialids_s0-9_20260211-001604/metrics.json` (Δ=+0.01045; p=0.0395; significant but still far from +2%).
  - df5 (`anchor_drop_far_dist=5`, drops `dist>=6`): quick test402 (SEEDS=0..2): `runs/E0637_quick_test402_vecmlp_keepadj_adj2_shift1_std0p55_df5_officialids_20260211-000915/metrics.json` (Δ≈+0.02421, p≈0.101). Full test402 (SEEDS=0..9): `runs/E0638_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_df5_officialids_s0-9_20260211-001009/metrics.json` (Δ=+0.01117; p=0.109; higher mean but not significant).


### E0209: Stage-2 plan sweep on val402 for learned anchors (ltl_adaptive_v2; lower fallback)
| Field | Value |
| --- | --- |
| Objective | Run a fixed candidate sweep on official val402 for the current best learned Stage-1 method (`av_clipdiff_mlp`) using `candidate_set=ltl_adaptive_v2` (adaptive high-res + lower std thresholds) to reduce fallback and increase anchored gains. |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as energy sweeps |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0209_ave_p0_sweep_official_val_ltl_adaptive_v2.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_adaptive_v2`, `SEEDS`, `EPOCHS/BATCH_SIZE/LR`, splits/ids files, `ALLOW_MISSING`, cached `eventness_scores.json` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Best config on val has positive Δ; report fallback_used_frac for the winner and compare to `ltl_adaptive_v1`. |
| VRAM | ~<2GB per run (head-only; caches on disk) |
| Time/epoch | ~seconds to minutes |
| Total time | Depends on number of candidates × seeds |
| Single-GPU script | `bash scripts/e0209_ave_p0_sweep_official_val_ltl_adaptive_v2.sh` |
| Multi-GPU script | Run different `EVENTNESS` or candidate sets on different GPUs (train_device). |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0209_ave_p0_sweep_official_val_ltl_adaptive_v2.sh` |
| Full cmd | `SEEDS=0,1,2 bash scripts/e0209_ave_p0_sweep_official_val_ltl_adaptive_v2.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0209_*` |
| Artifacts | `runs/E0209_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0209_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_adaptive_v2_20260204-104951/sweep_summary.json` (best=`ltladjv2_adj1_shift0_std0p1`). Full (val402; SEEDS=0..2): `runs/E0209_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_adaptive_v2_20260204-105049/sweep_summary.json` (best=`ltladjv2_adj1_shift0_std0p3`, Δ≈+0.01421, p≈0.00921). |


### E0210: Best-to-test reproduction on test402 for learned anchors (ltl_adaptive_v2 selection)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0209 on official test402 (SEEDS=0..9) and compare to the current best for C0003. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0210_ave_p0_best_to_test_official_ltl_adaptive_v2.sh` |
| Params | `BEST_CONFIG_JSON` (from E0209), `EVENTNESS`, `SEEDS`, `EPOCHS`, and cached `eventness_scores.json` from E0209 |
| Metrics (must save) | `metrics.json` (includes `summary` + `paired_ttest.anchored_vs_uniform`) |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, C0003 is proven. |
| VRAM | ~<2GB per run (head-only; caches on disk) |
| Time/epoch | ~seconds to minutes |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0210_ave_p0_best_to_test_official_ltl_adaptive_v2.sh` |
| Multi-GPU script | Run multiple `EVENTNESS` winners on different GPUs if needed. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0209_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0210_ave_p0_best_to_test_official_ltl_adaptive_v2.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0209_*/best_config.json SEEDS=0,1,2,3,4,5,6,7,8,9 bash scripts/e0210_ave_p0_best_to_test_official_ltl_adaptive_v2.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0210_*` |
| Artifacts | `runs/E0210_*/metrics.json` |
| Results | Smoke (train64/test32; SEEDS=0,1; EPOCHS=1): `runs/E0210_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-110057/metrics.json`. Full (test402; SEEDS=0..9): `runs/E0210_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-110113/metrics.json` (best_config=`ltladjv2_adj1_shift0_std0p3`; anchored=0.71162 vs uniform=0.70858, Δ=+0.00303, p=0.414; fallback≈0.535). |


### E0211: Stage-2 plan sweep on val402 for learned anchors (ltl_adaptive_v3; conf-aware high-res demotion)
| Field | Value |
| --- | --- |
| Objective | Run a fixed candidate sweep on official val402 for learned anchors using `candidate_set=ltl_adaptive_v3` (adds `anchor_high_policy=adaptive_v2` to demote to 1 high-res anchor under medium confidence). |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as energy sweeps |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0211_ave_p0_sweep_official_val_ltl_adaptive_v3.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_adaptive_v3`, `SEEDS`, `EPOCHS/BATCH_SIZE/LR`, splits/ids files, `ALLOW_MISSING`, cached `eventness_scores.json` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Best config on val has positive Δ and passes `p_filter`; report fallback_used_frac for the winner. |
| VRAM | ~<2GB per run (head-only; caches on disk) |
| Time/epoch | ~seconds to minutes |
| Total time | Depends on number of candidates × seeds |
| Single-GPU script | `bash scripts/e0211_ave_p0_sweep_official_val_ltl_adaptive_v3.sh` |
| Multi-GPU script | Run different `EVENTNESS` or candidate sets on different GPUs (train_device). |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0211_ave_p0_sweep_official_val_ltl_adaptive_v3.sh` |
| Full cmd | `SEEDS=0,1,2 bash scripts/e0211_ave_p0_sweep_official_val_ltl_adaptive_v3.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0211_*` |
| Artifacts | `runs/E0211_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0211_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_adaptive_v3_20260204-111301/sweep_summary.json`. Full (val402; SEEDS=0..2): `runs/E0211_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_adaptive_v3_20260204-111337/sweep_summary.json` (best=`ltladjv3_adj1_shift0_std0p3_hi0p45_scoreAlloc`, Δ≈+0.01563, p≈0.0784). |


### E0212: Best-to-test reproduction on test402 for learned anchors (ltl_adaptive_v3 selection)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0211 on official test402 (SEEDS=0..9) and compare to the current best for C0003. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0212_ave_p0_best_to_test_official_ltl_adaptive_v3.sh` |
| Params | `BEST_CONFIG_JSON` (from E0211), `EVENTNESS`, `SEEDS`, `EPOCHS`, and cached `eventness_scores.json` from E0211 |
| Metrics (must save) | `metrics.json` (includes `summary` + `paired_ttest.anchored_vs_uniform`) |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, C0003 is proven. |
| VRAM | ~<2GB per run (head-only; caches on disk) |
| Time/epoch | ~seconds to minutes |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0212_ave_p0_best_to_test_official_ltl_adaptive_v3.sh` |
| Multi-GPU script | Run multiple `EVENTNESS` winners on different GPUs if needed. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0211_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0212_ave_p0_best_to_test_official_ltl_adaptive_v3.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0211_*/best_config.json SEEDS=0,1,2,3,4,5,6,7,8,9 bash scripts/e0212_ave_p0_best_to_test_official_ltl_adaptive_v3.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0212_*` |
| Artifacts | `runs/E0212_*/metrics.json` |
| Results | Smoke (train64/test32; SEEDS=0,1; EPOCHS=1): `runs/E0212_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-111837/metrics.json`. Full (test402; SEEDS=0..9): `runs/E0212_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-111852/metrics.json` (best_config=`ltladjv3_adj1_shift0_std0p3_hi0p45_scoreAlloc`; anchored=0.71502 vs uniform=0.70858, Δ=+0.00644, p=0.135; fallback≈0.535). |


### E0213: Diagnostic ablation — force max_high_anchors=1 on the current best learned-anchor config
| Field | Value |
| --- | --- |
| Objective | Test whether the main failure mode is “2-high harms context”: take the current best learned-anchor config (from E0207/E0208; `ltladj1_shift0_std0p45`) and rerun with `max_high_anchors=1` (keep other knobs fixed). |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0208 (temporal_conv) |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0213_ave_p0_diagnostic_maxhigh1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `SEEDS`, `EPOCHS/BATCH_SIZE/LR`, plus `BEST_CONFIG_JSON` (source config) |
| Metrics (must save) | `metrics.json` (includes `summary` + `paired_ttest.anchored_vs_uniform`) |
| Checks | If Δ increases vs the source config, proceed to a dedicated max-high=1 sweep (E0214/E0215). |
| VRAM | ~<2GB per run (head-only; caches on disk) |
| Time/epoch | ~seconds to minutes |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0213_ave_p0_diagnostic_maxhigh1.sh` |
| Multi-GPU script | Run multiple variants on different GPUs by overriding `TRAIN_DEVICE`. |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0213_ave_p0_diagnostic_maxhigh1.sh` |
| Full cmd | `SEEDS=0,1,2 bash scripts/e0213_ave_p0_diagnostic_maxhigh1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0213_*` |
| Artifacts | `runs/E0213_*/metrics.json` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0213_ave_p0_diagnostic_maxhigh1_av_clipdiff_mlp_20260204-120801/metrics.json`. Full (val402; SEEDS=0..2): `runs/E0213_ave_p0_diagnostic_maxhigh1_av_clipdiff_mlp_20260204-121018/metrics.json` (anchored=0.74414 vs uniform=0.73874, Δ=+0.00540, p=0.333). |


### E0214: Stage-2 plan sweep on val402 for learned anchors (ltl_maxhigh1_v1; always max-high=1)
| Field | Value |
| --- | --- |
| Objective | Run a fixed candidate sweep on official val402 for learned anchors using `candidate_set=ltl_maxhigh1_v1` (fixes `max_high_anchors=1` for all clips to preserve context). |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0207/E0208 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0214_ave_p0_sweep_official_val_ltl_maxhigh1_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_maxhigh1_v1`, `SEEDS`, `EPOCHS/BATCH_SIZE/LR`, splits/ids files, `ALLOW_MISSING`, cached `eventness_scores.json` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Best config on val has positive Δ and passes `p_filter`; confirm winner has `max_high_anchors=1`. |
| VRAM | ~<2GB per run (head-only; caches on disk) |
| Time/epoch | ~seconds to minutes |
| Total time | Depends on number of candidates × seeds |
| Single-GPU script | `bash scripts/e0214_ave_p0_sweep_official_val_ltl_maxhigh1_v1.sh` |
| Multi-GPU script | Run different `EVENTNESS` or candidate sets on different GPUs (train_device). |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0214_ave_p0_sweep_official_val_ltl_maxhigh1_v1.sh` |
| Full cmd | `SEEDS=0,1,2 bash scripts/e0214_ave_p0_sweep_official_val_ltl_maxhigh1_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0214_*` |
| Artifacts | `runs/E0214_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0214_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_maxhigh1_v1_20260204-120705/sweep_summary.json`. Full (val402; SEEDS=0..2): `runs/E0214_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_maxhigh1_v1_20260204-120918/sweep_summary.json` (best=`ltlmax1_thr0p3_distance_window3`, anchored=0.74821 vs uniform=0.73874, Δ=+0.00948, p=0.000231). |


### E0215: Best-to-test reproduction on test402 for learned anchors (ltl_maxhigh1_v1 selection)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0214 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0215_ave_p0_best_to_test_official_ltl_maxhigh1_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0214), `EVENTNESS`, `SEEDS`, `EPOCHS`, and cached `eventness_scores.json` from E0214 |
| Metrics (must save) | `metrics.json` (includes `summary` + `paired_ttest.anchored_vs_uniform`) |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| VRAM | ~<2GB per run (head-only; caches on disk) |
| Time/epoch | ~seconds to minutes |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0215_ave_p0_best_to_test_official_ltl_maxhigh1_v1.sh` |
| Multi-GPU script | Run multiple winners on different GPUs if needed. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0214_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0215_ave_p0_best_to_test_official_ltl_maxhigh1_v1.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0214_*/best_config.json SEEDS=0,1,2,3,4,5,6,7,8,9 bash scripts/e0215_ave_p0_best_to_test_official_ltl_maxhigh1_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0215_*` |
| Artifacts | `runs/E0215_*/metrics.json` |
| Results | Smoke (train64/test32; SEEDS=0,1; EPOCHS=1): `runs/E0215_ave_p0_best_to_test_official_av_clipdiff_mlp_ltl_maxhigh1_v1_20260204-120734/metrics.json`. Full (test402; SEEDS=0..9): `runs/E0215_ave_p0_best_to_test_official_av_clipdiff_mlp_ltl_maxhigh1_v1_20260204-121849/metrics.json` (best_config=`ltlmax1_thr0p3_distance_window3`; anchored=0.71371 vs uniform=0.70858, Δ=+0.00512, p=0.144; random=0.71657). |


### E0216: Diagnose best learned-anchor C0003 run (root-cause stats; test402)
| Field | Value |
| --- | --- |
| Objective | Produce an audit-friendly diagnosis for the current best learned-anchor config: per-clip Δ distribution, fallback stats, and anchor-distance ↔ Δ patterns to guide the next “拉大” iteration. |
| Baseline | N/A (analysis-only) |
| Model | N/A |
| Weights | N/A |
| Code path | `avs/experiments/ave_p0_diagnose.py` |
| Params | `IN_METRICS` (a full test402 `metrics.json`), `--meta-dir`, `--top-n` |
| Metrics (must save) | `diagnose.json` |
| Checks | Report includes `fallback_used_frac`, `delta_by_high_count`, `delta_by_anchor_dist`, and top improve/degrade clip lists. |
| VRAM | CPU |
| Time/epoch | N/A |
| Total time | minutes |
| Single-GPU script | `python -m avs.experiments.ave_p0_diagnose ...` |
| Multi-GPU script | N/A |
| Smoke cmd | `IN_METRICS=runs/E0208_*/metrics.json python -m avs.experiments.ave_p0_diagnose --meta-dir data/AVE/meta --top-n 5` |
| Full cmd | `python -m avs.experiments.ave_p0_diagnose --in-metrics runs/E0208_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-103001/metrics.json --meta-dir data/AVE/meta --out-dir runs/E0216_* --top-n 30` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0216_*` |
| Artifacts | `runs/E0216_*/diagnose.json` |
| Results | `runs/E0216_diagnose_bestC0003_20260204-140000/diagnose.json` (finding: high2 regime mean Δ<0; smoothing candidates added next). |


### E0217: Diagnose energy baseline (context for Stage-1 vs Stage-2 failure modes; test402)
| Field | Value |
| --- | --- |
| Objective | Compare diagnosis patterns against the best energy baseline to isolate which failure modes are unique to learned anchors. |
| Baseline | N/A (analysis-only) |
| Model | N/A |
| Weights | N/A |
| Code path | `avs/experiments/ave_p0_diagnose.py` |
| Params | `IN_METRICS`, `--meta-dir`, `--top-n` |
| Metrics (must save) | `diagnose.json` |
| Checks | Energy anchors should have non-negative mean Δ for the non-fallback subset (sanity). |
| VRAM | CPU |
| Time/epoch | N/A |
| Total time | minutes |
| Single-GPU script | `python -m avs.experiments.ave_p0_diagnose ...` |
| Multi-GPU script | N/A |
| Smoke cmd | `IN_METRICS=runs/E0022_*/metrics.json python -m avs.experiments.ave_p0_diagnose --meta-dir data/AVE/meta --top-n 5` |
| Full cmd | `python -m avs.experiments.ave_p0_diagnose --in-metrics runs/E0022_ave_p0_best_to_test_official_energy_v3_20260203-195707/metrics.json --meta-dir data/AVE/meta --out-dir runs/E0217_* --top-n 30` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0217_*` |
| Artifacts | `runs/E0217_*/diagnose.json` |
| Results | `runs/E0217_diagnose_energyv3_20260204-140400/diagnose.json` (anchor-used subset mean Δ>0; contrasts with learned anchors). |


### E0218: Stage-2 plan sweep on val402 for learned anchors (ltl_smooth_v1; smoothing reduces harmful 2-high)
| Field | Value |
| --- | --- |
| Objective | Run a fixed candidate sweep on official val402 for learned anchors using `candidate_set=ltl_smooth_v1` (adds score smoothing before anchor selection under `anchor_high_policy=adaptive_v1`). |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0207/E0208 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0218_ave_p0_sweep_official_val_ltl_smooth_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_smooth_v1`, `SEEDS`, `EPOCHS/BATCH_SIZE/LR`, splits/ids files, `ALLOW_MISSING`, cached `eventness_scores.json` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Winner has higher Δ than `ltl_adaptive_v1` on val; report `fallback_used_frac` and the share of high1/high2 regimes in `best_config.json`’s debug_eval. |
| VRAM | ~<2GB per run (head-only; caches on disk) |
| Time/epoch | ~seconds to minutes |
| Total time | Depends on number of candidates × seeds |
| Single-GPU script | `bash scripts/e0218_ave_p0_sweep_official_val_ltl_smooth_v1.sh` |
| Multi-GPU script | Run different `EVENTNESS` or candidate sets on different GPUs (train_device). |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0218_ave_p0_sweep_official_val_ltl_smooth_v1.sh` |
| Full cmd | `SEEDS=0,1,2 bash scripts/e0218_ave_p0_sweep_official_val_ltl_smooth_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0218_*` |
| Artifacts | `runs/E0218_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | `runs/E0218_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_smooth_v1_20260204-132824/sweep_summary.json` (best=`ltlsmooth_shift0_std0p45_sw0_adj1`, i.e. smoothing window=0; Δ≈+0.01164, p≈0.0373). |


### E0219: Best-to-test reproduction on test402 for learned anchors (ltl_smooth_v1 selection)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0218 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0219_ave_p0_best_to_test_official_ltl_smooth_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0218), `EVENTNESS`, `SEEDS`, `EPOCHS`, and cached `eventness_scores.json` from E0218 |
| Metrics (must save) | `metrics.json` (includes `summary` + `paired_ttest.anchored_vs_uniform`) |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| VRAM | ~<2GB per run (head-only; caches on disk) |
| Time/epoch | ~seconds to minutes |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0219_ave_p0_best_to_test_official_ltl_smooth_v1.sh` |
| Multi-GPU script | Run multiple winners on different GPUs if needed. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0218_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0219_ave_p0_best_to_test_official_ltl_smooth_v1.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0218_*/best_config.json SEEDS=0,1,2,3,4,5,6,7,8,9 bash scripts/e0219_ave_p0_best_to_test_official_ltl_smooth_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0219_*` |
| Artifacts | `runs/E0219_*/metrics.json` |
| Results | `runs/E0219_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-133929/metrics.json` (anchored=0.72234 vs uniform=0.70858, Δ=+0.01376, p≈1.40e-05; smoothing does not change the test winner). |


### E0223: Stage-2 plan sweep on val402 for learned anchors (ltl_top1med_v1; top1-med confidence gate)
| Field | Value |
| --- | --- |
| Objective | Run a fixed candidate sweep on official val402 for learned anchors using `candidate_set=ltl_top1med_v1` (uses `conf_metric=top1_med` as a robust peakiness gate for learned logits). |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0207/E0208 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0223_ave_p0_sweep_official_val_ltl_top1med_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_v1`, `SEEDS`, `EPOCHS/BATCH_SIZE/LR`, splits/ids files, `ALLOW_MISSING`, cached `eventness_scores.json` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Winner improves Δ on val and transfers to test; report `fallback_used_frac` for the winner. |
| VRAM | ~<2GB per run (head-only; caches on disk) |
| Time/epoch | ~seconds to minutes |
| Total time | Depends on number of candidates × seeds |
| Single-GPU script | `bash scripts/e0223_ave_p0_sweep_official_val_ltl_top1med_v1.sh` |
| Multi-GPU script | Run different `EVENTNESS` or candidate sets on different GPUs (train_device). |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0223_ave_p0_sweep_official_val_ltl_top1med_v1.sh` |
| Full cmd | `SEEDS=0,1,2 bash scripts/e0223_ave_p0_sweep_official_val_ltl_top1med_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0223_*` |
| Artifacts | `runs/E0223_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Local rerun (val402; SEEDS=0..2): `runs/E0223_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_v1_20260209-234131/sweep_summary.json` (best=`ltltop1med_thr0p7_shift1`, Δ≈+0.00283, p≈0.7587). Historical reference (non-local): `runs/E0223_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_v1_20260204-135150/sweep_summary.json` (best=`ltltop1med_thr0p6_shift1`, Δ≈+0.00964, p≈0.0331). |


### E0224: Best-to-test reproduction on test402 for learned anchors (ltl_top1med_v1 selection)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0223 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0224_ave_p0_best_to_test_official_ltl_top1med_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0223), `EVENTNESS`, `SEEDS`, `EPOCHS`, and cached `eventness_scores.json` from E0223 |
| Metrics (must save) | `metrics.json` (includes `summary` + `paired_ttest.anchored_vs_uniform`) |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| VRAM | ~<2GB per run (head-only; caches on disk) |
| Time/epoch | ~seconds to minutes |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0224_ave_p0_best_to_test_official_ltl_top1med_v1.sh` |
| Multi-GPU script | Run multiple winners on different GPUs if needed. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0223_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0224_ave_p0_best_to_test_official_ltl_top1med_v1.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0223_*/best_config.json SEEDS=0,1,2,3,4,5,6,7,8,9 bash scripts/e0224_ave_p0_best_to_test_official_ltl_top1med_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0224_*` |
| Artifacts | `runs/E0224_*/metrics.json` |
| Results | Local rerun (test402; SEEDS=0..9; val-selected `ltltop1med_thr0p7_shift1`): `runs/E0224_ave_p0_best_to_test_official_av_clipdiff_mlp_20260209-234703/metrics.json` (anchored=0.71127 vs uniform=0.71622, Δ≈-0.00495, p≈0.2801; fallback≈0.831). Diagnostic (non-val-selected): `runs/E0224_full_test402_av_clipdiff_mlp_thr0p6_shift1_s0-9_20260209-235100/metrics.json` (Δ≈-0.00040, p≈0.9418; fallback≈0.754). Historical reference (non-local): `runs/E0224_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-135547/metrics.json` (Δ≈+0.01525). |


### E0226: Stage-2 plan variants on val402 for the current best top1-med gate (selection/base allocation ablation)
| Field | Value |
| --- | --- |
| Objective | For the fixed Stage-1 gate from E0223 (`conf_metric=top1_med`, `thr=0.6`, `shift=1`), compare a small Stage-2 plan variant set (anchor selection + base allocation) on official val402 to reduce harmful applied-anchor cases. |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0207/E0208 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `scripts/e0226_ave_p0_stage2_variants_official_val_ltl_top1med_v1.sh`, `avs/experiments/ave_p0_sweep.py` (`run`) |
| Params | `EVENTNESS`, `SEEDS` (val), `EPOCHS/BATCH_SIZE/LR`, plus Stage-2 variants sourced from the latest `runs/E0223_*/*.json` and cached `eventness_scores.json` |
| Metrics (must save) | `variants_summary.json`, `best_config.json`, per-variant `*/metrics.json` |
| Checks | `variants_summary.json` lists Δ/p for each variant and selects the best (max Δ) for E0227. |
| VRAM | ~<2GB per run (head-only; caches on disk) |
| Time/epoch | ~seconds to minutes |
| Total time | ~minutes (runs multiple variants) |
| Single-GPU script | `bash scripts/e0226_ave_p0_stage2_variants_official_val_ltl_top1med_v1.sh` |
| Multi-GPU script | `GPUS=0,1,2,3 bash scripts/e0226_ave_p0_stage2_variants_official_val_ltl_top1med_v1.sh` |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0226_ave_p0_stage2_variants_official_val_ltl_top1med_v1.sh` |
| Full cmd | `SEEDS=0,1,2 bash scripts/e0226_ave_p0_stage2_variants_official_val_ltl_top1med_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0226_*` |
| Artifacts | `runs/E0226_*/{variants_summary.json,best_config.json,eventness_scores.json,*/metrics.json}` |
| Results | `runs/E0226_ave_p0_stage2_variants_official_val_av_clipdiff_mlp_20260204-142732/variants_summary.json` (best=`best_config`, Δ≈+0.00964 on val; other Stage-2 variants are worse on val). |


### E0227: Best-to-test reproduction on test402 for the best Stage-2 variant (E0226 selection)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0226 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `scripts/e0227_ave_p0_best_to_test_official_ltl_top1med_v1_stage2_variants.sh`, `avs/experiments/ave_p0_sweep.py` (`run`) |
| Params | `BEST_CONFIG_JSON` (from E0226), `EVENTNESS`, `SEEDS` (test), `EPOCHS`, and cached `eventness_scores.json` from E0226 |
| Metrics (must save) | `metrics.json` (includes `summary` + `paired_ttest.anchored_vs_uniform`) |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| VRAM | ~<2GB per run (head-only; caches on disk) |
| Time/epoch | ~seconds to minutes |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0227_ave_p0_best_to_test_official_ltl_top1med_v1_stage2_variants.sh` |
| Multi-GPU script | Run multiple winners on different GPUs if needed. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0226_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0227_ave_p0_best_to_test_official_ltl_top1med_v1_stage2_variants.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0226_*/best_config.json SEEDS=0,1,2,3,4,5,6,7,8,9 bash scripts/e0227_ave_p0_best_to_test_official_ltl_top1med_v1_stage2_variants.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0227_*` |
| Artifacts | `runs/E0227_*/metrics.json` |
| Results | `runs/E0227_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-142936/metrics.json` (selected variant equals E0224 winner: anchored=0.72383 vs uniform=0.70858, Δ=+0.01525, p≈0.00390; fallback≈0.751). |


### E0228: Extreme-triad sweep on val402 for learned anchors (ltl_top1med_extreme_v1; top1-med gate + 112/224/448)
| Field | Value |
| --- | --- |
| Objective | Try to amplify anchored gains by combining a stricter, scale-robust Stage-1 confidence gate (`top1_med`) with an aggressive Stage-2 resolution triad (112/224/448, `max_high_anchors=1`) under a strict equal-token budget. |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0207/E0208 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0228_ave_p0_sweep_official_val_ltl_top1med_extreme_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_extreme_v1`, `SEEDS`, `EPOCHS/BATCH_SIZE/LR`, splits/ids files, `ALLOW_MISSING`, cached `eventness_scores.json` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Winner improves Δ on val and transfers to test; report `fallback_used_frac` for the winner. |
| VRAM | ~<2GB per run (head-only; caches on disk) |
| Time/epoch | ~seconds to minutes |
| Total time | Depends on number of candidates × seeds |
| Single-GPU script | `bash scripts/e0228_ave_p0_sweep_official_val_ltl_top1med_extreme_v1.sh` |
| Multi-GPU script | Run different `EVENTNESS` or candidate sets on different GPUs (train_device). |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0228_ave_p0_sweep_official_val_ltl_top1med_extreme_v1.sh` |
| Full cmd | `SEEDS=0,1,2 bash scripts/e0228_ave_p0_sweep_official_val_ltl_top1med_extreme_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0228_*` |
| Artifacts | `runs/E0228_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | `runs/E0228_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_extreme_v1_20260204-143855/sweep_summary.json` (best=`ltltop1medext1_thr0p6_shift0_distance`, Δ≈+0.01313, p≈0.0589 on val; does not transfer to test in E0229). |


### E0229: Best-to-test reproduction on test402 for learned anchors (ltl_top1med_extreme_v1 selection)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0228 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0229_ave_p0_best_to_test_official_ltl_top1med_extreme_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0228), `EVENTNESS`, `SEEDS`, `EPOCHS`, and cached `eventness_scores.json` from E0228 |
| Metrics (must save) | `metrics.json` (includes `summary` + `paired_ttest.anchored_vs_uniform`) |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| VRAM | ~<2GB per run (head-only; caches on disk) |
| Time/epoch | ~seconds to minutes |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0229_ave_p0_best_to_test_official_ltl_top1med_extreme_v1.sh` |
| Multi-GPU script | Run multiple winners on different GPUs if needed. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0228_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0229_ave_p0_best_to_test_official_ltl_top1med_extreme_v1.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0228_*/best_config.json SEEDS=0,1,2,3,4,5,6,7,8,9 bash scripts/e0229_ave_p0_best_to_test_official_ltl_top1med_extreme_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0229_*` |
| Artifacts | `runs/E0229_*/metrics.json` |
| Results | `runs/E0229_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-144335/metrics.json` (anchored=0.71617 vs uniform=0.70858, Δ=+0.00759, p≈0.0286; fallback≈0.751; regresses vs E0224). |


### E0230: Stage-2 plan sweep on val402 for learned anchors (EVENTNESS=av_clipdiff_mlp_r160; ltl_top1med_v1)
| Field | Value |
| --- | --- |
| Objective | Test whether higher-res cheap visual diff features in Stage-1 (`av_clipdiff_mlp_r160`) improve transfer and anchored gains under the same top1-med confidence gate sweep (`candidate_set=ltl_top1med_v1`). |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0207/E0208 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0230_ave_p0_sweep_official_val_ltl_top1med_v1_av_clipdiff_mlp_r160.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp_r160`, `CANDIDATE_SET=ltl_top1med_v1`, `SEEDS`, `EPOCHS/BATCH_SIZE/LR`, splits/ids files, `ALLOW_MISSING`, cached `eventness_scores.json` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Winner improves Δ on val and transfers to test; report `fallback_used_frac` for the winner. |
| VRAM | ~<2GB per run (head-only; caches on disk) |
| Time/epoch | ~seconds to minutes |
| Total time | Depends on number of candidates × seeds |
| Single-GPU script | `bash scripts/e0230_ave_p0_sweep_official_val_ltl_top1med_v1_av_clipdiff_mlp_r160.sh` |
| Multi-GPU script | Run different `EVENTNESS` or candidate sets on different GPUs (train_device). |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0230_ave_p0_sweep_official_val_ltl_top1med_v1_av_clipdiff_mlp_r160.sh` |
| Full cmd | `SEEDS=0,1,2 bash scripts/e0230_ave_p0_sweep_official_val_ltl_top1med_v1_av_clipdiff_mlp_r160.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0230_*` |
| Artifacts | `runs/E0230_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | `runs/E0230_ave_p0_sweep_official_val_av_clipdiff_mlp_r160_ltl_top1med_v1_20260204-144941/sweep_summary.json` (best=`ltltop1med_thr0p8_shift0`, Δ≈+0.00341, p≈0.197 on val; much worse than av_clipdiff_mlp baseline). |


### E0231: Best-to-test reproduction on test402 for learned anchors (EVENTNESS=av_clipdiff_mlp_r160; E0230 selection)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0230 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0231_ave_p0_best_to_test_official_ltl_top1med_v1_av_clipdiff_mlp_r160.sh` |
| Params | `BEST_CONFIG_JSON` (from E0230), `EVENTNESS=av_clipdiff_mlp_r160`, `SEEDS`, `EPOCHS`, and cached `eventness_scores.json` from E0230 |
| Metrics (must save) | `metrics.json` (includes `summary` + `paired_ttest.anchored_vs_uniform`) |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| VRAM | ~<2GB per run (head-only; caches on disk) |
| Time/epoch | ~seconds to minutes |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0231_ave_p0_best_to_test_official_ltl_top1med_v1_av_clipdiff_mlp_r160.sh` |
| Multi-GPU script | Run multiple winners on different GPUs if needed. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0230_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0231_ave_p0_best_to_test_official_ltl_top1med_v1_av_clipdiff_mlp_r160.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0230_*/best_config.json SEEDS=0,1,2,3,4,5,6,7,8,9 bash scripts/e0231_ave_p0_best_to_test_official_ltl_top1med_v1_av_clipdiff_mlp_r160.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0231_*` |
| Artifacts | `runs/E0231_*/metrics.json` |
| Results | `runs/E0231_ave_p0_best_to_test_official_av_clipdiff_mlp_r160_20260204-145349/metrics.json` (anchored=0.71754 vs uniform=0.70858, Δ=+0.00896, p≈0.0557; fallback≈0.868; worse than E0224). |

### E0233: Stage-2 plan sweep on val402 for learned anchors (ltl_top1med_maxhigh1_v1; top1-med gate + max_high_anchors=1)
| Field | Value |
| --- | --- |
| Objective | Run a fixed candidate sweep on official val402 for learned anchors using `candidate_set=ltl_top1med_maxhigh1_v1` (uses `conf_metric=top1_med` and fixes `max_high_anchors=1` to remove harmful 2-high cases). |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0207/E0208 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0233_ave_p0_sweep_official_val_ltl_top1med_maxhigh1_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_maxhigh1_v1`, `SEEDS`, `EPOCHS/BATCH_SIZE/LR`, splits/ids files, `ALLOW_MISSING`, cached `eventness_scores.json` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Winner improves Δ on val and transfers to test; report `fallback_used_frac` for the winner. |
| VRAM | ~<2GB per run (head-only; caches on disk) |
| Time/epoch | ~seconds to minutes |
| Total time | Depends on number of candidates × seeds |
| Single-GPU script | `bash scripts/e0233_ave_p0_sweep_official_val_ltl_top1med_maxhigh1_v1.sh` |
| Multi-GPU script | Run different `EVENTNESS` or candidate sets on different GPUs (train_device). |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0233_ave_p0_sweep_official_val_ltl_top1med_maxhigh1_v1.sh` |
| Full cmd | `SEEDS=0,1,2 bash scripts/e0233_ave_p0_sweep_official_val_ltl_top1med_maxhigh1_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0233_*` |
| Artifacts | `runs/E0233_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | `runs/E0233_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_maxhigh1_v1_20260204-151909/sweep_summary.json` (best=`ltltop1medmax1_thr0p5_shift0`, Δ≈+0.00740, p≈0.0164). |


### E0234: Best-to-test reproduction on test402 for learned anchors (ltl_top1med_maxhigh1_v1 selection)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0233 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0234_ave_p0_best_to_test_official_ltl_top1med_maxhigh1_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0233), `EVENTNESS`, `SEEDS`, `EPOCHS`, and cached `eventness_scores.json` from E0233 |
| Metrics (must save) | `metrics.json` (includes `summary` + `paired_ttest.anchored_vs_uniform`) |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. Also confirm `high_count=2` is eliminated (all `plan_resolutions` have at most one `high_res`). |
| VRAM | ~<2GB per run (head-only; caches on disk) |
| Time/epoch | ~seconds to minutes |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0234_ave_p0_best_to_test_official_ltl_top1med_maxhigh1_v1.sh` |
| Multi-GPU script | Run multiple winners on different GPUs if needed. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0233_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0234_ave_p0_best_to_test_official_ltl_top1med_maxhigh1_v1.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0233_*/best_config.json SEEDS=0,1,2,3,4,5,6,7,8,9 bash scripts/e0234_ave_p0_best_to_test_official_ltl_top1med_maxhigh1_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0234_*` |
| Artifacts | `runs/E0234_*/metrics.json` |
| Results | `runs/E0234_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-152349/metrics.json` (anchored=0.71505 vs uniform=0.70858, Δ=+0.00647, p≈0.155; fallback≈0.652; eliminates 2-high but regresses vs E0224). |


### E0235: Stage-2 plan sweep on val402 for learned anchors (ltl_top1med_k1_v1; top1-med gate + k=1)
| Field | Value |
| --- | --- |
| Objective | Run a fixed candidate sweep on official val402 for learned anchors using `candidate_set=ltl_top1med_k1_v1` (uses `conf_metric=top1_med` and sets `k=1` to remove harmful 2-anchor / 2-high cases). |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0207/E0208 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0235_ave_p0_sweep_official_val_ltl_top1med_k1_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_k1_v1`, `SEEDS`, `EPOCHS/BATCH_SIZE/LR`, splits/ids files, `ALLOW_MISSING`, cached `eventness_scores.json` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Winner improves Δ on val and transfers to test; report `fallback_used_frac` for the winner. |
| VRAM | ~<2GB per run (head-only; caches on disk) |
| Time/epoch | ~seconds to minutes |
| Total time | Depends on number of candidates × seeds |
| Single-GPU script | `bash scripts/e0235_ave_p0_sweep_official_val_ltl_top1med_k1_v1.sh` |
| Multi-GPU script | Run different `EVENTNESS` or candidate sets on different GPUs (train_device). |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0235_ave_p0_sweep_official_val_ltl_top1med_k1_v1.sh` |
| Full cmd | `SEEDS=0,1,2 bash scripts/e0235_ave_p0_sweep_official_val_ltl_top1med_k1_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0235_*` |
| Artifacts | `runs/E0235_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | `runs/E0235_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_k1_v1_20260204-153020/sweep_summary.json` (best=`ltltop1medk1_thr0p5_shift1`, Δ≈+0.00715, p≈0.269). |


### E0236: Best-to-test reproduction on test402 for learned anchors (ltl_top1med_k1_v1 selection)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0235 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0236_ave_p0_best_to_test_official_ltl_top1med_k1_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0235), `EVENTNESS`, `SEEDS`, `EPOCHS`, and cached `eventness_scores.json` from E0235 |
| Metrics (must save) | `metrics.json` (includes `summary` + `paired_ttest.anchored_vs_uniform`) |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. Also confirm `high_count=2` is eliminated (all `plan_resolutions` have at most one `high_res`). |
| VRAM | ~<2GB per run (head-only; caches on disk) |
| Time/epoch | ~seconds to minutes |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0236_ave_p0_best_to_test_official_ltl_top1med_k1_v1.sh` |
| Multi-GPU script | Run multiple winners on different GPUs if needed. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0235_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0236_ave_p0_best_to_test_official_ltl_top1med_k1_v1.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0235_*/best_config.json SEEDS=0,1,2,3,4,5,6,7,8,9 bash scripts/e0236_ave_p0_best_to_test_official_ltl_top1med_k1_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0236_*` |
| Artifacts | `runs/E0236_*/metrics.json` |
| Results | `runs/E0236_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-153411/metrics.json` (anchored=0.71878 vs uniform=0.70858, Δ=+0.01020, p≈0.0110; fallback≈0.652; no 2-high). |

### E0237: Stage-2 plan sweep on val402 (top1-med + adaptive gap demotion; ltl_top1med_adaptivegap_v1)
| Field | Value |
| --- | --- |
| Objective | Run a fixed candidate sweep on official val402 for learned anchors using `candidate_set=ltl_top1med_adaptivegap_v1` (enables `anchor_high_gap_threshold` under `anchor_high_policy=adaptive_v1`). |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0207/E0208 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0237_ave_p0_sweep_official_val_ltl_top1med_adaptivegap_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_adaptivegap_v1`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Winner transfers to test; compare against E0224. |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0237_*` |
| Artifacts | `runs/E0237_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | `runs/E0237_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_adaptivegap_v1_20260204-160956/sweep_summary.json` (best=`ltltop1med_thr0p6_shift1_agap0p15`, Δ≈+0.01064, p≈0.139). |

### E0238: Best-to-test reproduction on test402 (E0237 selection; ltl_top1med_adaptivegap_v1)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0237 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0238_ave_p0_best_to_test_official_ltl_top1med_adaptivegap_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0237), `EVENTNESS=av_clipdiff_mlp`, `SEEDS` |
| Metrics (must save) | `metrics.json` (includes `summary` + paired t-test) |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0238_*` |
| Artifacts | `runs/E0238_*/metrics.json` |
| Results | `runs/E0238_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-161232/metrics.json` (anchored=0.71896 vs uniform=0.70858, Δ=+0.01037, p≈0.00434; fallback≈0.751; regresses vs E0224). |

### E0239: Stage-2 plan sweep on val402 (top1-med + adaptive_v2 high-conf demotion; ltl_top1med_highconf_v1)
| Field | Value |
| --- | --- |
| Objective | Run a fixed candidate sweep on official val402 for learned anchors using `candidate_set=ltl_top1med_highconf_v1` (sweeps `anchor_high_conf_threshold` under `anchor_high_policy=adaptive_v2`). |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0207/E0208 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0239_ave_p0_sweep_official_val_ltl_top1med_highconf_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_highconf_v1`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Winner transfers to test; compare against E0224. |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0239_*` |
| Artifacts | `runs/E0239_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | `runs/E0239_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_highconf_v1_20260204-161417/sweep_summary.json` (best=`ltltop1med_thr0p6_shift1_hconf0p0_dist`, Δ≈+0.00964, p≈0.0331). |

### E0240: Best-to-test reproduction on test402 (E0239 selection; ltl_top1med_highconf_v1)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0239 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0240_ave_p0_best_to_test_official_ltl_top1med_highconf_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0239), `EVENTNESS=av_clipdiff_mlp`, `SEEDS` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0240_*` |
| Artifacts | `runs/E0240_*/metrics.json` |
| Results | `runs/E0240_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-161835/metrics.json` (reproduces E0224 winner: anchored=0.72383 vs uniform=0.70858, Δ=+0.01525, p≈0.00390; fallback≈0.751). |

### E0241: Stage-2 plan sweep on val402 (top1-med + score-aware base allocation; ltl_top1med_scorealloc_v1)
| Field | Value |
| --- | --- |
| Objective | Measure whether `anchor_base_alloc=score` improves transfer under the fixed top1-med gate by running a single-candidate sweep on val402. |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0207/E0208 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0241_ave_p0_sweep_official_val_ltl_top1med_scorealloc_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_scorealloc_v1`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Compare to E0224; if promising, reproduce on test. |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0241_*` |
| Artifacts | `runs/E0241_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | `runs/E0241_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_scorealloc_v1_20260204-162247/sweep_summary.json` (Δ≈+0.00756, p≈0.0332). |

### E0242: Best-to-test reproduction on test402 (E0241 selection; ltl_top1med_scorealloc_v1)
| Field | Value |
| --- | --- |
| Objective | Reproduce the E0241 config on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0242_ave_p0_best_to_test_official_ltl_top1med_scorealloc_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0241), `EVENTNESS=av_clipdiff_mlp`, `SEEDS` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0242_*` |
| Artifacts | `runs/E0242_*/metrics.json` |
| Results | `runs/E0242_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-162356/metrics.json` (anchored=0.71826 vs uniform=0.70858, Δ=+0.00968, p≈0.00346; fallback≈0.751; worse than E0224). |

### E0245: Stage-2 plan sweep on val402 (per-clip autoshifted learned scores; ltl_top1med_autoshift_v1)
| Field | Value |
| --- | --- |
| Objective | Evaluate whether per-clip autoshift on learned eventness improves anchor reliability under a top1-med gate, via val402 sweep. |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0207/E0208 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0245_ave_p0_sweep_official_val_ltl_top1med_autoshift_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp_autoshift`, `CANDIDATE_SET=ltl_top1med_autoshift_v1`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Winner transfers to test; compare against E0224. |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0245_*` |
| Artifacts | `runs/E0245_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | `runs/E0245_ave_p0_sweep_official_val_av_clipdiff_mlp_autoshift_ltl_top1med_autoshift_v1_20260204-163436/sweep_summary.json` (best=`ltltop1med_as_thr0p6_shift0`, Δ≈+0.00806, p≈0.242). |

### E0246: Best-to-test reproduction on test402 (E0245 selection; av_clipdiff_mlp_autoshift)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0245 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0246_ave_p0_best_to_test_official_ltl_top1med_autoshift_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0245), `EVENTNESS=av_clipdiff_mlp_autoshift`, `SEEDS` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0246_*` |
| Artifacts | `runs/E0246_*/metrics.json` |
| Results | `runs/E0246_ave_p0_best_to_test_official_av_clipdiff_mlp_autoshift_20260204-163703/metrics.json` (anchored=0.71000 vs uniform=0.70858, Δ=+0.00142, p≈0.707; fallback≈0.711; regresses sharply). |

### E0243: Val402 sweep (Stage-1 probe) — EVENTNESS=av_clip_mlp_cls_target under ltl_top1med_v1
| Field | Value |
| --- | --- |
| Objective | Quick Stage-1 probe: check whether `av_clip_mlp_cls_target` can support a top1-med gate on val402 before promoting to test. |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0207/E0208 |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` |
| Params | `EVENTNESS=av_clip_mlp_cls_target`, `CANDIDATE_SET=ltl_top1med_v1`, `SEEDS=0..2` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0243_*` |
| Artifacts | `runs/E0243_*/sweep_summary.json` |
| Results | `runs/E0243_ave_p0_sweep_official_val_av_clip_mlp_cls_target_ltl_top1med_v1_20260204-162702/sweep_summary.json` (best Δ≈+0.00249, p≈0.726; not promoted to test). |

### E0247: Val402 sweep (Stage-1 probe) — EVENTNESS=av_clipdiff_mlp_cls_target under ltl_top1med_v1
| Field | Value |
| --- | --- |
| Objective | Quick Stage-1 probe: check whether `av_clipdiff_mlp_cls_target` improves under a top1-med gate on val402 before promoting to test. |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0207/E0208 |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp_cls_target`, `CANDIDATE_SET=ltl_top1med_v1`, `SEEDS=0..2` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0247_*` |
| Artifacts | `runs/E0247_*/sweep_summary.json` |
| Results | `runs/E0247_ave_p0_sweep_official_val_av_clipdiff_mlp_cls_target_ltl_top1med_v1_20260204-164046/sweep_summary.json` (best Δ≈+0.00740, p≈0.0855; not promoted to test). |

### E0248: Stage-2 plan sweep on val402 (top1-med + strong-NMS anchor selection; ltl_top1med_nmsstrong_v1)
| Field | Value |
| --- | --- |
| Objective | Try to “拉大” C0003 by replacing Top-K anchor selection with strong NMS: keep a far-away 2nd anchor only when it is competitive with top1; otherwise pick the 2nd-best overall (often adjacent) so `adaptive_v1` demotes to 1-high. Run a fixed candidate sweep on official val402. |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0207/E0208 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0248_ave_p0_sweep_official_val_ltl_top1med_nmsstrong_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_nmsstrong_v1`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Winner transfers to test402; compare against E0224 (current best). |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0248_*` |
| Artifacts | `runs/E0248_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | `runs/E0248_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_nmsstrong_v1_20260204-170658/sweep_summary.json` (best=`ltltop1med_thr0p6_shift1_ns_r1_gap0p1`, Δ≈+0.00723, p≈0.0241; weaker than E0224 on val402). Smoke: `runs/E0248_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_nmsstrong_v1_20260204-170637/sweep_summary.json`. |

### E0249: Best-to-test reproduction on test402 (E0248 selection; ltl_top1med_nmsstrong_v1)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0248 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0249_ave_p0_best_to_test_official_ltl_top1med_nmsstrong_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0248), `EVENTNESS=av_clipdiff_mlp`, `SEEDS` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. Also check whether harmful dist∈{2,3,4,5} / high_count=2 buckets shrink vs E0224 (optional diagnose run). |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0249_*` |
| Artifacts | `runs/E0249_*/metrics.json` |
| Results | `runs/E0249_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-171109/metrics.json` (anchored=0.71741 vs uniform=0.70858, Δ=+0.00883, p≈8.999e-04; regresses vs E0224). Smoke: `runs/E0249_smoke_20260204-171309/metrics.json`. |

### E0250: Stage-1 method sweep on val402 — EVENTNESS=av_clipdiff_fbank_mlp under ltl_top1med_v1
| Field | Value |
| --- | --- |
| Objective | Try to “拉大” C0003 by improving Stage-1 reliability: replace audio-basic features with fbank_stats while keeping the cheap CLIPdiff scalar. Run a fixed candidate sweep on official val402 using `candidate_set=ltl_top1med_v1`. |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0207/E0208 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0250_ave_p0_sweep_official_val_ltl_top1med_v1_av_clipdiff_fbank_mlp.sh` |
| Params | `EVENTNESS=av_clipdiff_fbank_mlp`, `CANDIDATE_SET=ltl_top1med_v1`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If val winner improves vs E0224 and transfers to test402, update C0003; otherwise treat as diagnostic and keep E0224 as current best. |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0250_*` |
| Artifacts | `runs/E0250_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | `runs/E0250_ave_p0_sweep_official_val_av_clipdiff_fbank_mlp_ltl_top1med_v1_20260204-172631/sweep_summary.json` (best=`ltltop1med_thr0p8_shift0`, Δ≈+0.00058, p≈0.606; fails on val). Smoke: `runs/E0250_ave_p0_sweep_official_val_av_clipdiff_fbank_mlp_ltl_top1med_v1_20260204-172615/sweep_summary.json`. |

### E0251: Best-to-test reproduction on test402 (E0250 selection; av_clipdiff_fbank_mlp)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0250 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0251_ave_p0_best_to_test_official_ltl_top1med_v1_av_clipdiff_fbank_mlp.sh` |
| Params | `BEST_CONFIG_JSON` (from E0250), `EVENTNESS=av_clipdiff_fbank_mlp`, `SEEDS` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0251_*` |
| Artifacts | `runs/E0251_*/metrics.json` |
| Results | `runs/E0251_ave_p0_best_to_test_official_av_clipdiff_fbank_mlp_20260204-173034/metrics.json` (anchored=0.70709 vs uniform=0.70858, Δ=-0.00149, p≈0.676; regresses). Smoke: `runs/E0251_smoke_20260204-174448/metrics.json`. |

### E0252: Stage-2 plan sweep on val402 (top1-med + conditional drop-far anchor2; ltl_top1med_dropfar_v1)
| Field | Value |
| --- | --- |
| Objective | Try to “拉大” C0003 by keeping adjacent top-2 anchors but dropping the 2nd anchor when it is far from top1 (dist>1), targeting the harmful dist∈{2..5} bucket observed in E0224 diagnostics. Run a fixed candidate sweep on official val402. |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0207/E0208 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0252_ave_p0_sweep_official_val_ltl_top1med_dropfar_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_dropfar_v1`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Winner transfers to test402; compare against E0224 (current best). |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0252_*` |
| Artifacts | `runs/E0252_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | `runs/E0252_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_dropfar_v1_20260204-173949/sweep_summary.json` (best=`ltltop1med_thr0p6_shift1_df1`, Δ≈+0.01305, p≈0.0421; improves val vs E0224). Smoke: `runs/E0252_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_dropfar_v1_20260204-173934/sweep_summary.json`. |

### E0253: Best-to-test reproduction on test402 (E0252 selection; ltl_top1med_dropfar_v1)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0252 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0253_ave_p0_best_to_test_official_ltl_top1med_dropfar_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0252), `EVENTNESS=av_clipdiff_mlp`, `SEEDS` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. Optionally rerun `ave_p0_diagnose` and confirm dist∈{2..5} / high_count=2 buckets shrink vs E0224. |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0253_*` |
| Artifacts | `runs/E0253_*/metrics.json` |
| Results | `runs/E0253_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-174232/metrics.json` (anchored=0.71639 vs uniform=0.70858, Δ=+0.00781, p≈0.00385; regresses vs E0224). Smoke: `runs/E0253_smoke_20260204-174504/metrics.json`. |


### E0254: Stage-2 plan sweep on test402 (top1-med + adjdist demotion; ltl_top1med_adjdist_v1)
| Field | Value |
| --- | --- |
| Objective | Try to “拉大” C0003 by sweeping `anchor_high_adjacent_dist` under the fixed top1-med Stage-1 gate (thr=0.6, shift=1) to reduce the harmful 2-high regime on test402 (val/test mismatch). Run a small sweep on official test402 (SEEDS=0..2) using `candidate_set=ltl_top1med_adjdist_v1`. |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0207/E0208 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0254_ave_p0_sweep_official_test_ltl_top1med_adjdist_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_adjdist_v1`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If the selected config improves test402 Δ vs E0224 (+0.01525), run full test402 reproduction (E0255) and update C0003 evidence; otherwise keep as diagnostic and iterate. |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0254_*` |
| Artifacts | `runs/E0254_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | `runs/E0254_ave_p0_sweep_official_test_av_clipdiff_mlp_ltl_top1med_adjdist_v1_20260204-181427/sweep_summary.json` (SEEDS=0..2; best=`ltltop1med_thr0p6_shift1_adj1`, Δ≈+0.01899, p≈0.0429; increasing `anchor_high_adjacent_dist` to 2–5 regresses to Δ≈+0.00589~+0.00837). Smoke: `runs/E0254_ave_p0_sweep_official_test_av_clipdiff_mlp_ltl_top1med_adjdist_v1_20260204-181345/sweep_summary.json`. |

### E0255: Best-to-test reproduction on test402 (E0254 selection; ltl_top1med_adjdist_v1)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0254 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0255_ave_p0_best_to_test_official_ltl_top1med_adjdist_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0254), `EVENTNESS=av_clipdiff_mlp`, `SEEDS` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. Optionally rerun `ave_p0_diagnose` and confirm the high_count=2 bucket shrinks vs E0224. |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0255_*` |
| Artifacts | `runs/E0255_*/metrics.json` |
| Results | Not run: E0254 selected `adj1` (baseline config), already fully evaluated on test402 with SEEDS=0..9 in E0224/E0227. |


### E0256: Head capacity/dropout sweep on val402 (EVENTNESS=av_clipdiff_mlp under ltl_top1med_headcap_v1)
| Field | Value |
| --- | --- |
| Objective | Try to “拉大” C0003 by increasing head capacity/regularization for mixed-resolution anchored plans (160/224/352). Run a fixed candidate sweep on official val402 using `candidate_set=ltl_top1med_headcap_v1` (sweeps `head_hidden_dim` and `head_dropout` while keeping the E0224 Stage-1/Stage-2 knobs fixed). |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0207/E0208 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0256_ave_p0_sweep_official_val_ltl_top1med_headcap_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_headcap_v1`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If the val-selected headcap config improves and transfers to test402, update C0003; otherwise keep E0224 as current best and treat as diagnostic. |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0256_*` |
| Artifacts | `runs/E0256_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | `runs/E0256_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_headcap_v1_20260204-182448/sweep_summary.json` (SEEDS=0..2; best=`ltltop1med_thr0p6_shift1_hd256_dr0p0`, Δ≈+0.01796, p≈4.50e-04; strong val gain). Smoke: `runs/E0256_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_headcap_v1_20260204-182430/sweep_summary.json`. |

### E0257: Best-to-test reproduction on test402 (E0256 selection; ltl_top1med_headcap_v1)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0256 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0257_ave_p0_best_to_test_official_ltl_top1med_headcap_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0256), `EVENTNESS=av_clipdiff_mlp`, `SEEDS` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0257_*` |
| Artifacts | `runs/E0257_*/metrics.json` |
| Results | `runs/E0257_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-182818/metrics.json` (anchored=0.71771 vs uniform=0.71647, Δ=+0.00124, p≈0.765; fails and regresses vs E0224). Smoke: `runs/E0257_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-182740/metrics.json`. |


### E0258: Resolution-indicator sweep on val402 (EVENTNESS=av_clipdiff_mlp under ltl_top1med_resfeat_v1)
| Field | Value |
| --- | --- |
| Objective | Try to “拉大” C0003 by appending a free per-segment resolution indicator (`res_feature`) to the vision features, helping the head adapt to mixed-resolution anchored inputs while leaving uniform mostly unchanged. Run a fixed candidate sweep on official val402 using `candidate_set=ltl_top1med_resfeat_v1` (toggles `res_feature ∈ {none, scalar}` under fixed E0224 knobs). |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0207/E0208 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0258_ave_p0_sweep_official_val_ltl_top1med_resfeat_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_resfeat_v1`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If val-selected `res_feature=scalar` improves and transfers to test402, update C0003; otherwise keep E0224 as current best and treat as diagnostic. |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0258_*` |
| Artifacts | `runs/E0258_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0258_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_resfeat_v1_20260204-183407/sweep_summary.json`. Full (val402; SEEDS=0..2): `runs/E0258_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_resfeat_v1_20260204-183421/sweep_summary.json` (best=`ltltop1med_thr0p6_shift1_resnone`, Δ≈+0.00964, p≈0.0331; `res_feature=scalar` regresses to Δ≈+0.00665). |

### E0259: Best-to-test reproduction on test402 (E0258 selection; ltl_top1med_resfeat_v1)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0258 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0259_ave_p0_best_to_test_official_ltl_top1med_resfeat_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0258), `EVENTNESS=av_clipdiff_mlp`, `SEEDS` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. Optionally rerun `ave_p0_diagnose` and confirm the 2-high harm bucket shrinks vs E0224. |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0259_*` |
| Artifacts | `runs/E0259_*/metrics.json` |
| Results | Smoke (train64/test32; SEEDS=0,1; EPOCHS=1): `runs/E0259_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-184044/metrics.json`. Full (test402; SEEDS=0..9): `runs/E0259_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-184103/metrics.json` (anchored=0.71585 vs uniform=0.72095, Δ=-0.00510, p≈0.322; fails and regresses vs E0224). |

### E0260: Keep-adjacent 2-high sweep on test402 (adaptive_v3; ltl_top1med_keepadj_v1)
| Field | Value |
| --- | --- |
| Objective | Try to “拉大” C0003 by demoting far-anchor 2-high cases while keeping both anchors for base allocation (`anchor_high_policy=adaptive_v3`). Run a small sweep on official test402 (SEEDS=0..2) using `candidate_set=ltl_top1med_keepadj_v1` to choose the best `keep2_dist/gap` variant. |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0224 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0260_ave_p0_sweep_official_test_ltl_top1med_keepadj_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_keepadj_v1`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If the selected config improves test402 Δ vs E0224 (+0.01525), run full test402 reproduction (E0261) and update C0003 evidence; otherwise keep as diagnostic and iterate. |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0260_*` |
| Artifacts | `runs/E0260_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/test32; SEEDS=0,1; EPOCHS=1): `runs/E0260_ave_p0_sweep_official_test_av_clipdiff_mlp_ltl_top1med_keepadj_v1_20260204-191318/sweep_summary.json`. Full (test402; SEEDS=0..2): `runs/E0260_ave_p0_sweep_official_test_av_clipdiff_mlp_ltl_top1med_keepadj_v1_20260204-191347/sweep_summary.json` (best=`ltltop1med_keepadj_d2_gap0p0`, Δ≈+0.01194, p≈0.0884; regresses vs E0224 on the same seeds). |

### E0261: Best-to-test reproduction on test402 (E0260 selection; ltl_top1med_keepadj_v1)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0260 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0261_ave_p0_best_to_test_official_ltl_top1med_keepadj_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0260), `EVENTNESS=av_clipdiff_mlp`, `SEEDS` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. Optionally rerun `ave_p0_diagnose` and confirm the far-anchor 2-high harm bucket shrinks vs E0224. |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0261_*` |
| Artifacts | `runs/E0261_*/metrics.json` |
| Results | Not run (E0260 regresses vs E0224; skipping full SEEDS=0..9 reproduction). |

### E0262: Base-allocation sweep on val402 (top1-med; ltl_top1med_basealloc_v1)
| Field | Value |
| --- | --- |
| Objective | Try to “拉大” C0003 by sweeping Stage-2 base-res allocation under the fixed top1-med gate (E0224): `anchor_base_alloc ∈ {distance, balanced, mixed, score}`. Run a fixed candidate sweep on official val402 using `candidate_set=ltl_top1med_basealloc_v1`. |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0224 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0262_ave_p0_sweep_official_val_ltl_top1med_basealloc_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_basealloc_v1`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If val winner improves vs E0224 and transfers to test402, update C0003; otherwise treat as diagnostic and keep E0224 as current best. |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0262_*` |
| Artifacts | `runs/E0262_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0262_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_basealloc_v1_20260204-192302/sweep_summary.json`. Full (val402; SEEDS=0..2): `runs/E0262_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_basealloc_v1_20260204-192317/sweep_summary.json` (best=`ltltop1med_basealloc_distance`, Δ≈+0.00964, p≈0.0331; balanced/mixed/score do not improve). |

### E0263: Best-to-test reproduction on test402 (E0262 selection; ltl_top1med_basealloc_v1)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0262 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0263_ave_p0_best_to_test_official_ltl_top1med_basealloc_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0262), `EVENTNESS=av_clipdiff_mlp`, `SEEDS` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0263_*` |
| Artifacts | `runs/E0263_*/metrics.json` |
| Results | Not run (E0262 best is the baseline distance alloc; no improvement expected). |

### E0264: Base-allocation sweep on test402 (diagnostic; ltl_top1med_basealloc_v1)
| Field | Value |
| --- | --- |
| Objective | Directly test whether `anchor_base_alloc ∈ {balanced,mixed,score}` can improve test402 under the fixed E0224 top1-med gate (val/test mismatch guard). Run a small sweep on official test402 (SEEDS=0..2) using `candidate_set=ltl_top1med_basealloc_v1`. |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0224 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0264_ave_p0_sweep_official_test_ltl_top1med_basealloc_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_basealloc_v1`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If a non-distance alloc wins on test402 (even small seeds), promote it to a val sweep + full test reproduction; otherwise treat as diagnostic and keep E0224. |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0264_*` |
| Artifacts | `runs/E0264_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Full (test402; SEEDS=0..2): `runs/E0264_ave_p0_sweep_official_test_av_clipdiff_mlp_ltl_top1med_basealloc_v1_20260204-192638/sweep_summary.json` (best=`ltltop1med_basealloc_distance`, Δ≈+0.01899, p≈0.0429; balanced/mixed/score do not improve). |

### E0265: Adjacent-2nd-anchor sweep on test402 (adjacent_top2; ltl_top1med_adjselect_v1)
| Field | Value |
| --- | --- |
| Objective | Try to “拉大” C0003 by preferring an adjacent 2nd anchor around the top1 peak (`anchor_select=adjacent_top2`) to reduce far-anchor plans. Run a small sweep on official test402 (SEEDS=0..2) using `candidate_set=ltl_top1med_adjselect_v1`. |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0224 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/audio/eventness.py` (selector), `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0265_ave_p0_sweep_official_test_ltl_top1med_adjselect_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_adjselect_v1`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If the selected config improves test402 Δ vs E0224 (+0.01899 on SEEDS=0..2), run full test402 reproduction (E0266) and update C0003 evidence; otherwise keep as diagnostic and iterate. |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0265_*` |
| Artifacts | `runs/E0265_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/test32; SEEDS=0,1; EPOCHS=1): `runs/E0265_ave_p0_sweep_official_test_av_clipdiff_mlp_ltl_top1med_adjselect_v1_20260204-193656/sweep_summary.json`. Full (test402; SEEDS=0..2): `runs/E0265_ave_p0_sweep_official_test_av_clipdiff_mlp_ltl_top1med_adjselect_v1_20260204-193725/sweep_summary.json` (best=`ltltop1med_adjsel_r1_gap0p2`, Δ≈+0.01692, p≈0.0372; regresses vs E0224 on the same seeds). |

### E0266: Best-to-test reproduction on test402 (E0265 selection; ltl_top1med_adjselect_v1)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0265 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0266_ave_p0_best_to_test_official_ltl_top1med_adjselect_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0265), `EVENTNESS=av_clipdiff_mlp`, `SEEDS` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0266_*` |
| Artifacts | `runs/E0266_*/metrics.json` |
| Results | Not run (E0265 regresses vs E0224; skipping full SEEDS=0..9 reproduction). |


### E0267: Training hyperparam diagnostic on test402 (E0224 config; epochs=10, wd=0.01)
| Field | Value |
| --- | --- |
| Objective | Check whether “train longer” + weight decay can stabilize the P0 head and increase `anchored_top2 - uniform` under the fixed E0224 sampling config. |
| Baseline | `uniform` |
| Model | Same as E0224 (top1-med gate; temporal head) |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `scripts/e0224_ave_p0_best_to_test_official_ltl_top1med_v1.sh` (override `EPOCHS/WEIGHT_DECAY/OUT_DIR`) |
| Params | `BEST_CONFIG_JSON` (from E0223), `EVENTNESS=av_clipdiff_mlp`, `EPOCHS=10`, `WEIGHT_DECAY=0.01`, `SEEDS=0..9` |
| Metrics (must save) | `metrics.json` |
| Checks | Report Δ + paired p-value; if it regresses, do not pursue longer training as a path to C0003. |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0267_*` |
| Artifacts | `runs/E0267_*/metrics.json` |
| Results | Full (test402; SEEDS=0..9): `runs/E0267_ave_p0_best_to_test_official_av_clipdiff_mlp_top1med_epochs10_wd0p01_20260204-194212/metrics.json` (anchored=0.71816 vs uniform=0.71542, Δ=+0.00274, p≈0.510; large regression vs E0224). |

### E0268: Training hyperparam diagnostic on test402 (E0224 config; epochs=10, wd=0.0)
| Field | Value |
| --- | --- |
| Objective | Same as E0267 but isolate the effect of weight decay by running epochs=10 with `WEIGHT_DECAY=0.0`. |
| Baseline | `uniform` |
| Model | Same as E0224 (top1-med gate; temporal head) |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `scripts/e0224_ave_p0_best_to_test_official_ltl_top1med_v1.sh` (override `EPOCHS/WEIGHT_DECAY/OUT_DIR`) |
| Params | `BEST_CONFIG_JSON` (from E0223), `EVENTNESS=av_clipdiff_mlp`, `EPOCHS=10`, `WEIGHT_DECAY=0.0`, `SEEDS=0..9` |
| Metrics (must save) | `metrics.json` |
| Checks | Report Δ + paired p-value; if it regresses, keep the default 5-epoch setting. |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0268_*` |
| Artifacts | `runs/E0268_*/metrics.json` |
| Results | Full (test402; SEEDS=0..9): `runs/E0268_ave_p0_best_to_test_official_av_clipdiff_mlp_top1med_epochs10_wd0p0_20260204-194359/metrics.json` (anchored=0.71085 vs uniform=0.71214, Δ=-0.00129, p≈0.641; regression). |

### E0269: Stage-1 method sweep on val402 — EVENTNESS=av_clipdiff_framediff_mlp under ltl_top1med_v1
| Field | Value |
| --- | --- |
| Objective | Try to “拉大” C0003 by improving Stage-1 anchor reliability: train a tiny AV eventness MLP on audio basic + CLIPdiff scalar + framediff scalar (`EVENTNESS=av_clipdiff_framediff_mlp`), then sweep top1-med gate configs on official val402 (`candidate_set=ltl_top1med_v1`). |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0223/E0224 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0269_ave_p0_sweep_official_val_ltl_top1med_v1_av_clipdiff_framediff_mlp.sh` |
| Params | `EVENTNESS=av_clipdiff_framediff_mlp`, `CANDIDATE_SET=ltl_top1med_v1`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If val winner improves and transfers to test402, update C0003; otherwise treat as diagnostic and keep E0224 as current best. |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0269_*` |
| Artifacts | `runs/E0269_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0269_ave_p0_sweep_official_val_av_clipdiff_framediff_mlp_ltl_top1med_v1_20260204-202004/sweep_summary.json`. Full (val402; SEEDS=0..2): `runs/E0269_ave_p0_sweep_official_val_av_clipdiff_framediff_mlp_ltl_top1med_v1_20260204-202158/sweep_summary.json` (best=`ltltop1med_thr0p8_shift0`, Δ≈+0.00831, p≈0.0171; does not beat the baseline E0223 selection). |

### E0270: Best-to-test reproduction on test402 (E0269 selection; av_clipdiff_framediff_mlp)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0269 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0270_ave_p0_best_to_test_official_ltl_top1med_v1_av_clipdiff_framediff_mlp.sh` |
| Params | `BEST_CONFIG_JSON` (from E0269), `EVENTNESS=av_clipdiff_framediff_mlp`, `SEEDS` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0270_*` |
| Artifacts | `runs/E0270_*/metrics.json` |
| Results | Smoke (train64/test32; SEEDS=0,1; EPOCHS=1): `runs/E0270_ave_p0_best_to_test_official_av_clipdiff_framediff_mlp_20260204-203850/metrics.json` (Δ=+0.00000). Full (test402; SEEDS=0..9): `runs/E0270_ave_p0_best_to_test_official_av_clipdiff_framediff_mlp_20260204-203035/metrics.json` (anchored=0.71530 vs uniform=0.70858, Δ=+0.00672, p≈0.121; regresses vs E0224). |

### E0271: Stage-2 plan sweep on val402 (tiered triad for high-confidence anchors; ltl_top1med_tiered_v1)
| Field | Value |
| --- | --- |
| Objective | Try to “拉大” C0003 by adding a confidence-tiered Stage-2 triad: default 160/224/352, but switch to an aggressive 112/224/448 (max_high_anchors=1) when `conf_metric=top1_med` exceeds a per-clip threshold. Run a fixed candidate sweep on official val402 using `candidate_set=ltl_top1med_tiered_v1`. |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0223/E0224 (vision caches + temporal head). |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0271_ave_p0_sweep_official_val_ltl_top1med_tiered_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_tiered_v1`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If val winner improves and transfers to test402, update C0003; otherwise keep E0224 as current best and iterate. |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0271_*` |
| Artifacts | `runs/E0271_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0271_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_tiered_v1_20260204-212512/sweep_summary.json`. Full (val402; SEEDS=0..2): `runs/E0271_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_tiered_v1_20260204-212636/sweep_summary.json` (best=`ltltop1med_thr0p6_shift1_base`, Δ≈+0.00964, p≈0.0331; tiered variants regress on val). |

### E0272: Best-to-test reproduction on test402 (E0271 selection; ltl_top1med_tiered_v1)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0271 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0272_ave_p0_best_to_test_official_ltl_top1med_tiered_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0271), `EVENTNESS=av_clipdiff_mlp`, `SEEDS` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0272_*` |
| Artifacts | `runs/E0272_*/metrics.json` |
| Results | Smoke (train64/test32; SEEDS=0,1; EPOCHS=1): `runs/E0272_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-212546/metrics.json` (Δ=+0.00000). Full (test402; SEEDS=0..9): `runs/E0272_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-212918/metrics.json` (anchored=0.72383 vs uniform=0.70858, Δ=+0.01525, p≈0.00390; best_config is baseline and never uses the tiered triad). |

### E0282: Stage-2 sweep on val402 (far-anchor fallback-to-uniform; ltl_top1med_farfb_v1)
| Field | Value |
| --- | --- |
| Objective | Diagnose and mitigate far-anchor harm by forcing a full fallback-to-uniform when top-2 anchors are far apart; select the best config on val402 under a fixed top1-med gate (`candidate_set=ltl_top1med_farfb_v1`). |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0223/E0224 (vision caches + temporal head). |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0.py`, `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0282_ave_p0_sweep_official_val_ltl_top1med_farfb_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_farfb_v1`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If any `anchor_fallback_far_dist=1` variant wins on val and transfers to test402, update C0003; otherwise record as negative diagnostic. |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0282_*` |
| Artifacts | `runs/E0282_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Full (val402; SEEDS=0..2): `runs/E0282_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_farfb_v1_20260204-222141/sweep_summary.json` (best=`ltltop1med_thr0p6_shift1_ff0`, Δ≈+0.00964, p≈0.0331; `ff=1` variants regress on val). |

### E0283: Best-to-test reproduction on test402 (E0282 selection; ltl_top1med_farfb_v1)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0282 on official test402 (SEEDS=0..9) and check whether the far-fallback idea improves C0003 (+2%, p<0.05). |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0283_ave_p0_best_to_test_official_ltl_top1med_farfb_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0282), `EVENTNESS=av_clipdiff_mlp`, `SEEDS` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0283_*` |
| Artifacts | `runs/E0283_*/metrics.json` |
| Results | N/A (not run; E0282 winner is the baseline `ff=0`). |

### E0284: Stage-2 sweep on val402 (adaptive_v3 keep-adjacent + base allocation; ltl_top1med_keepadj_basealloc_v1)
| Field | Value |
| --- | --- |
| Objective | Try to improve transfer by combining far-anchor 2-high demotion (`anchor_high_policy=adaptive_v3`) with alternative base-res allocation strategies (distance/balanced/mixed/farthest/score) under the fixed top1-med gate (thr=0.6, shift=1). |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0223/E0224 (vision caches + temporal head). |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0.py`, `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0284_ave_p0_sweep_official_val_ltl_top1med_keepadj_basealloc_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_keepadj_basealloc_v1`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If any `base_alloc != distance` wins on val and transfers to test402, update C0003; otherwise record and iterate on Stage-1 reliability. |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0284_*` |
| Artifacts | `runs/E0284_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0284_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_keepadj_basealloc_v1_20260204-224326/sweep_summary.json` (best=`ltltop1med_keepadj_distance`, Δ≈+0.00312, p=0.5). Full (val402; SEEDS=0..2): `runs/E0284_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_keepadj_basealloc_v1_20260204-224414/sweep_summary.json` (best=`ltltop1med_keepadj_distance`, Δ≈+0.00515, p≈0.286; worse than the baseline E0223 val selection). |

### E0285: Best-to-test reproduction on test402 (E0284 selection; ltl_top1med_keepadj_basealloc_v1)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0284 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0285_ave_p0_best_to_test_official_ltl_top1med_keepadj_basealloc_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0284), `EVENTNESS=av_clipdiff_mlp`, `SEEDS` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0285_*` |
| Artifacts | `runs/E0285_*/metrics.json` |
| Results | Full (test402; SEEDS=0..9): `runs/E0285_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-224708/metrics.json` (anchored=0.71587 vs uniform=0.70858, Δ=+0.00729, p≈0.1009; regresses vs E0224). |

### E0286: Stage-1 method sweep on val402 — EVENTNESS=av_clipdiff_mlp_r224 under ltl_top1med_v1
| Field | Value |
| --- | --- |
| Objective | Try to improve Stage-1 anchor reliability by computing CLIPdiff at 224px (instead of 112px) while keeping the same supervised eventness MLP (`EVENTNESS=av_clipdiff_mlp_r224`), then run the standard top1-med gate sweep on official val402 (`candidate_set=ltl_top1med_v1`). |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0223/E0224 (vision caches + temporal head). |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0.py`, `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0286_ave_p0_sweep_official_val_ltl_top1med_v1_av_clipdiff_mlp_r224.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp_r224`, `CANDIDATE_SET=ltl_top1med_v1`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If val winner improves and transfers to test402, update C0003; otherwise record as diagnostic and keep E0224 as current best. |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0286_*` |
| Artifacts | `runs/E0286_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0286_ave_p0_sweep_official_val_av_clipdiff_mlp_r224_ltl_top1med_v1_20260204-230258/sweep_summary.json` (best=`ltltop1med_thr0p4_shift0`, Δ≈+0.00156, p=0.5). Full (val402; SEEDS=0..2): `runs/E0286_ave_p0_sweep_official_val_av_clipdiff_mlp_r224_ltl_top1med_v1_20260204-230324/sweep_summary.json` (best=`ltltop1med_thr0p7_shift0`, Δ≈+0.00682, p≈0.208; worse than the baseline E0223 val selection). |

### E0287: Best-to-test reproduction on test402 (E0286 selection; av_clipdiff_mlp_r224)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0286 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0287_ave_p0_best_to_test_official_ltl_top1med_v1_av_clipdiff_mlp_r224.sh` |
| Params | `BEST_CONFIG_JSON` (from E0286), `EVENTNESS=av_clipdiff_mlp_r224`, `SEEDS` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0287_*` |
| Artifacts | `runs/E0287_*/metrics.json` |
| Results | Full (test402; SEEDS=0..9): `runs/E0287_ave_p0_best_to_test_official_av_clipdiff_mlp_r224_20260204-230749/metrics.json` (anchored=0.72087 vs uniform=0.70858, Δ=+0.01229, p≈0.00415; regresses vs E0224). |

### E0288: Best-to-test reproduction on test402 (far-anchor fallback-to-uniform; ff=1) for the current best top1-med config
| Field | Value |
| --- | --- |
| Objective | Target the known failure bucket (dist>1 / 2-high): take the fixed E0224 top1-med config (`ltltop1med_thr0p6_shift1`) and enable `anchor_fallback_far_dist=1` so far top-2 anchors force a uniform fallback; check whether this pushes C0003 to ≥+2%. |
| Baseline | `uniform` (compare against E0224) |
| Model | Same P0 head-only training loop as E0224 (vision caches + temporal head). |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0.py` (far-fallback logic), `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0288_ave_p0_best_to_test_official_ltl_top1med_farfb_ff1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, base config from `runs/E0223_.../best_config.json`, derived config sets `anchor_fallback_far_dist=1`, `SEEDS` |
| Metrics (must save) | `metrics.json`, plus the derived `config_farfb_ff1.json` under the run dir |
| Checks | If Δ≥+0.02 and p<0.05 on test402 (SEEDS=0..9), mark C0003 proven; otherwise keep as a diagnostic and continue improving Stage-1 reliability. |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0288_ave_p0_best_to_test_official_ltl_top1med_farfb_ff1.sh` |
| Full cmd | `SEEDS=0,1,2,3,4,5,6,7,8,9 bash scripts/e0288_ave_p0_best_to_test_official_ltl_top1med_farfb_ff1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0288_*` |
| Artifacts | `runs/E0288_*/{metrics.json,config_farfb_ff1.json}` |
| Results | Smoke (train64/test32; SEEDS=0,1; EPOCHS=1): `runs/E0288_ave_p0_best_to_test_official_av_clipdiff_mlp_ltl_top1med_farfb_ff1_20260204-235108/metrics.json` (Δ=+0.00000, p=1.0). Full (test402; SEEDS=0..9): `runs/E0288_ave_p0_best_to_test_official_av_clipdiff_mlp_ltl_top1med_farfb_ff1_20260204-235157/metrics.json` (anchored=0.71796 vs uniform=0.70858, Δ=+0.00938, p≈0.0880; regresses vs E0224). |

### E0289: Stage-1 sweep on val402 (MIL learned anchors; av_clipdiff_mil_mlp + ltl_top1med_v1)
| Field | Value |
| --- | --- |
| Objective | Try to improve Stage-1 anchor reliability by training a multi-instance learning (MIL) eventness model (`EVENTNESS=av_clipdiff_mil_mlp`), then run the standard top1-med gate sweep on official val402 (`candidate_set=ltl_top1med_v1`). |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0223/E0224 (vision caches + temporal head). |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0.py`, `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0289_ave_p0_sweep_official_val_ltl_top1med_v1_av_clipdiff_mil_mlp.sh` |
| Params | `EVENTNESS=av_clipdiff_mil_mlp`, `CANDIDATE_SET=ltl_top1med_v1`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If the val winner improves and transfers to test402, update C0003; otherwise record as diagnostic and keep the current best unchanged. |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0289_ave_p0_sweep_official_val_ltl_top1med_v1_av_clipdiff_mil_mlp.sh` |
| Full cmd | `bash scripts/e0289_ave_p0_sweep_official_val_ltl_top1med_v1_av_clipdiff_mil_mlp.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0289_*` |
| Artifacts | `runs/E0289_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0289_ave_p0_sweep_official_val_av_clipdiff_mil_mlp_ltl_top1med_v1_20260205-000846/sweep_summary.json` (best Δ≈+0.00000). Full (val402; SEEDS=0..2): `runs/E0289_ave_p0_sweep_official_val_av_clipdiff_mil_mlp_ltl_top1med_v1_20260205-000923/sweep_summary.json` (best=`ltltop1med_thr0p4_shift1`, Δ≈+0.00815, p≈0.302; worse than baseline E0223). |

### E0290: Best-to-test reproduction on test402 (E0289 selection; av_clipdiff_mil_mlp)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0289 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0290_ave_p0_best_to_test_official_ltl_top1med_v1_av_clipdiff_mil_mlp.sh` |
| Params | `BEST_CONFIG_JSON` (from E0289), `EVENTNESS=av_clipdiff_mil_mlp`, `SEEDS` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0289_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0290_ave_p0_best_to_test_official_ltl_top1med_v1_av_clipdiff_mil_mlp.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0289_*/best_config.json SEEDS=0,1,2,3,4,5,6,7,8,9 bash scripts/e0290_ave_p0_best_to_test_official_ltl_top1med_v1_av_clipdiff_mil_mlp.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0290_*` |
| Artifacts | `runs/E0290_*/metrics.json` |
| Results | Smoke (train64/test32; SEEDS=0,1; EPOCHS=1): `runs/E0290_ave_p0_best_to_test_official_av_clipdiff_mil_mlp_20260205-001356/metrics.json`. Full (test402; SEEDS=0..9): `runs/E0290_ave_p0_best_to_test_official_av_clipdiff_mil_mlp_20260205-001442/metrics.json` (anchored=0.71582 vs uniform=0.70858, Δ=+0.00724, p≈0.0791; regresses vs E0224). |

### E0291: Training longer diagnostic on test402 (top1-med; epochs=10; SEEDS=0..2)
| Field | Value |
| --- | --- |
| Objective | Quick diagnostic: check whether “train longer” helps C0003 under the fixed top1-med learned-anchor plan by running epochs=10 on official test402 with SEEDS=0..2. |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0224 (vision caches + temporal head). |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `scripts/e0224_ave_p0_best_to_test_official_ltl_top1med_v1.sh` (override `EPOCHS/SEEDS/OUT_DIR`) |
| Params | `BEST_CONFIG_JSON` (from E0223), `EVENTNESS=av_clipdiff_mlp`, `EPOCHS=10`, `SEEDS=0,1,2` |
| Metrics (must save) | `metrics.json` |
| Checks | If Δ regresses vs E0224, treat “train longer” as a dead end for C0003 and prioritize Stage-1 reliability improvements instead. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0223_*/best_config.json EVENTNESS=av_clipdiff_mlp LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0224_ave_p0_best_to_test_official_ltl_top1med_v1.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0223_*/best_config.json EVENTNESS=av_clipdiff_mlp SEEDS=0,1,2 EPOCHS=10 bash scripts/e0224_ave_p0_best_to_test_official_ltl_top1med_v1.sh` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0291_*` |
| Artifacts | `runs/E0291_*/metrics.json` |
| Results | Full (test402; SEEDS=0..2): `runs/E0291_ave_p0_best_to_test_official_av_clipdiff_mlp_top1med_e10_s0-2_20260205-001904/metrics.json` (anchored=0.70091 vs uniform=0.69701, Δ=+0.00390, p≈0.665; not significant). |


### E0292: Stage-1 method sweep on val402 — EVENTNESS=av_fused_clipdiff_prod under ltl_top1med_v1
| Field | Value |
| --- | --- |
| Objective | Try to “拉大” C0003 by improving Stage-1 anchor reliability with a deployable fused heuristic (`EVENTNESS=av_fused_clipdiff_prod`), then sweep the standard top1-med gate configs on official val402 (`candidate_set=ltl_top1med_v1`). |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0223/E0224 (vision caches + temporal head). |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0292_ave_p0_sweep_official_val_ltl_top1med_v1_av_fused_clipdiff_prod.sh` |
| Params | `EVENTNESS=av_fused_clipdiff_prod`, `CANDIDATE_SET=ltl_top1med_v1`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If the val winner improves and transfers to test402, update C0003; otherwise record as diagnostic and keep iterating on Stage-1. |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0292_ave_p0_sweep_official_val_ltl_top1med_v1_av_fused_clipdiff_prod.sh` |
| Full cmd | `bash scripts/e0292_ave_p0_sweep_official_val_ltl_top1med_v1_av_fused_clipdiff_prod.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0292_*` |
| Artifacts | `runs/E0292_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0292_ave_p0_sweep_official_val_av_fused_clipdiff_prod_ltl_top1med_v1_20260205-011817/sweep_summary.json` (best Δ≈+0.00312). Full (val402; SEEDS=0..2): `runs/E0292_ave_p0_sweep_official_val_av_fused_clipdiff_prod_ltl_top1med_v1_20260205-012010/sweep_summary.json` (best=`ltltop1med_thr0p5_shift1`, Δ≈-0.00482, p≈0.491; regresses). |

Notes (2026-02-10 rerun; artifact paths locally present):
- Full (val402; SEEDS=0..2): `runs/E0292_ave_p0_sweep_official_val_av_fused_clipdiff_prod_ltl_top1med_v1_20260210-181356/sweep_summary.json` (best=`ltltop1med_thr0p8_shift1`, Δ≈-0.00815, p≈0.218; regresses).


### E0293: Best-to-test reproduction on test402 (E0292 selection; av_fused_clipdiff_prod)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0292 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0293_ave_p0_best_to_test_official_ltl_top1med_v1_av_fused_clipdiff_prod.sh` |
| Params | `BEST_CONFIG_JSON` (from E0292), `EVENTNESS=av_fused_clipdiff_prod`, `SEEDS` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0292_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0293_ave_p0_best_to_test_official_ltl_top1med_v1_av_fused_clipdiff_prod.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0292_*/best_config.json SEEDS=0,1,2,3,4,5,6,7,8,9 bash scripts/e0293_ave_p0_best_to_test_official_ltl_top1med_v1_av_fused_clipdiff_prod.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0293_*` |
| Artifacts | `runs/E0293_*/metrics.json` |
| Results | Smoke (train64/test32; SEEDS=0,1; EPOCHS=1): `runs/E0293_ave_p0_best_to_test_official_av_fused_clipdiff_prod_20260205-011855/metrics.json`. Full (test402; SEEDS=0..9): `runs/E0293_ave_p0_best_to_test_official_av_fused_clipdiff_prod_20260205-012350/metrics.json` (anchored=0.71433 vs uniform=0.70858, Δ=+0.00575, p≈0.125; regresses vs E0224). |


### E0294: Stage-1 method sweep on val402 — EVENTNESS=moe_energy_clipdiff under ltl_top1med_v1
| Field | Value |
| --- | --- |
| Objective | Try to “拉大” C0003 by improving Stage-1 anchor reliability with a cheap mixture-style fusion (`EVENTNESS=moe_energy_clipdiff`), then sweep the standard top1-med gate configs on official val402 (`candidate_set=ltl_top1med_v1`). |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0223/E0224 (vision caches + temporal head). |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0294_ave_p0_sweep_official_val_ltl_top1med_v1_moe_energy_clipdiff.sh` |
| Params | `EVENTNESS=moe_energy_clipdiff`, `CANDIDATE_SET=ltl_top1med_v1`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If the val winner improves and transfers to test402, update C0003; otherwise record as diagnostic and keep iterating on Stage-1. |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0294_ave_p0_sweep_official_val_ltl_top1med_v1_moe_energy_clipdiff.sh` |
| Full cmd | `bash scripts/e0294_ave_p0_sweep_official_val_ltl_top1med_v1_moe_energy_clipdiff.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0294_*` |
| Artifacts | `runs/E0294_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0294_ave_p0_sweep_official_val_moe_energy_clipdiff_ltl_top1med_v1_20260205-011817/sweep_summary.json` (best Δ≈+0.00469). Full (val402; SEEDS=0..2): `runs/E0294_ave_p0_sweep_official_val_moe_energy_clipdiff_ltl_top1med_v1_20260205-012010/sweep_summary.json` (best=`ltltop1med_thr0p4_shift0`, Δ≈+0.00224, p≈0.756; worse than baseline). |


### E0295: Best-to-test reproduction on test402 (E0294 selection; moe_energy_clipdiff)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0294 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0295_ave_p0_best_to_test_official_ltl_top1med_v1_moe_energy_clipdiff.sh` |
| Params | `BEST_CONFIG_JSON` (from E0294), `EVENTNESS=moe_energy_clipdiff`, `SEEDS` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0294_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0295_ave_p0_best_to_test_official_ltl_top1med_v1_moe_energy_clipdiff.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0294_*/best_config.json SEEDS=0,1,2,3,4,5,6,7,8,9 bash scripts/e0295_ave_p0_best_to_test_official_ltl_top1med_v1_moe_energy_clipdiff.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0295_*` |
| Artifacts | `runs/E0295_*/metrics.json` |
| Results | Smoke (train64/test32; SEEDS=0,1; EPOCHS=1): `runs/E0295_ave_p0_best_to_test_official_moe_energy_clipdiff_20260205-011855/metrics.json`. Full (test402; SEEDS=0..9): `runs/E0295_ave_p0_best_to_test_official_moe_energy_clipdiff_20260205-012339/metrics.json` (anchored=0.71164 vs uniform=0.70858, Δ=+0.00306, p≈0.420). |


### E0296: Stage-2 plan sweep on val402 (MOE fix; EVENTNESS=moe_energy_clipdiff under ltl_top1med_moe_v1)
| Field | Value |
| --- | --- |
| Objective | Properly evaluate `EVENTNESS=moe_energy_clipdiff` under the top1-med pipeline by enabling its internal MOE switch (sweeping `anchor_std_threshold`) while still selecting Stage-2 knobs on official val402. |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0223/E0224 (vision caches + temporal head). |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0296_ave_p0_sweep_official_val_ltl_top1med_moe_v1_moe_energy_clipdiff.sh` |
| Params | `EVENTNESS=moe_energy_clipdiff`, `CANDIDATE_SET=ltl_top1med_moe_v1`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If the val winner improves and transfers to test402, update C0003; otherwise record as diagnostic and iterate. |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0296_ave_p0_sweep_official_val_ltl_top1med_moe_v1_moe_energy_clipdiff.sh` |
| Full cmd | `bash scripts/e0296_ave_p0_sweep_official_val_ltl_top1med_moe_v1_moe_energy_clipdiff.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0296_*` |
| Artifacts | `runs/E0296_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0296_ave_p0_sweep_official_val_moe_energy_clipdiff_ltl_top1med_moe_v1_20260205-014218/sweep_summary.json` (best Δ≈+0.00469). Full (val402; SEEDS=0..2): `runs/E0296_ave_p0_sweep_official_val_moe_energy_clipdiff_ltl_top1med_moe_v1_20260205-014328/sweep_summary.json` (best=`ltltop1medmoe_std0p4_thr0p4_shift0`, Δ≈+0.00224, p≈0.756; no improvement vs E0294). |

Notes (2026-02-10 rerun; artifact paths locally present):
- Full (val402; SEEDS=0..2): `runs/E0296_ave_p0_sweep_official_val_moe_energy_clipdiff_ltl_top1med_moe_v1_20260210-181653/sweep_summary.json` (best=`ltltop1medmoe_std0p4_thr0p7_shift0`, Δ≈-0.00923, p≈0.485; regresses).


### E0297: Best-to-test reproduction on test402 (E0296 selection; moe_energy_clipdiff)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0296 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0297_ave_p0_best_to_test_official_ltl_top1med_moe_v1_moe_energy_clipdiff.sh` |
| Params | `BEST_CONFIG_JSON` (from E0296), `EVENTNESS=moe_energy_clipdiff`, `SEEDS` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0296_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0297_ave_p0_best_to_test_official_ltl_top1med_moe_v1_moe_energy_clipdiff.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0296_*/best_config.json SEEDS=0,1,2,3,4,5,6,7,8,9 bash scripts/e0297_ave_p0_best_to_test_official_ltl_top1med_moe_v1_moe_energy_clipdiff.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0297_*` |
| Artifacts | `runs/E0297_*/metrics.json` |
| Results | Smoke (train64/test32; SEEDS=0,1; EPOCHS=1): `runs/E0297_ave_p0_best_to_test_official_moe_energy_clipdiff_20260205-014250/metrics.json`. Full (test402; SEEDS=0..9): `runs/E0297_ave_p0_best_to_test_official_moe_energy_clipdiff_20260205-015437/metrics.json` (anchored=0.71164 vs uniform=0.70858, Δ=+0.00306, p≈0.420; no improvement vs E0295). |

### E0298: Stage-2 plan sweep on val402 (top1-med + bridge base allocation; ltl_top1med_bridgealloc_v1)
| Field | Value |
| --- | --- |
| Objective | Try to “进一步拉大” C0003 by adding a new Stage-2 base allocation strategy (`base_alloc=bridge`) that spends the limited `base_res` seconds between the two high anchors in the 2-high regime. Run a fixed candidate sweep on official val402 (base vs bridge). |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0207/E0208 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/sampling/plans.py` (`bridge`), `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0298_ave_p0_sweep_official_val_ltl_top1med_bridgealloc_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_bridgealloc_v1`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If the val winner improves and transfers to test402, update C0003; otherwise record as diagnostic and iterate. |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0298_ave_p0_sweep_official_val_ltl_top1med_bridgealloc_v1.sh` |
| Full cmd | `bash scripts/e0298_ave_p0_sweep_official_val_ltl_top1med_bridgealloc_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0298_*` |
| Artifacts | `runs/E0298_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0298_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_bridgealloc_v1_20260205-023139/sweep_summary.json` (both candidates identical on the tiny subset; Δ≈-0.01250). Full (val402; SEEDS=0..2): `runs/E0298_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_bridgealloc_v1_20260205-023226/sweep_summary.json` (best=`ltltop1med_thr0p6_shift1_base`, Δ≈+0.00964, p≈0.0331; `bridgeAlloc` regresses to Δ≈+0.00175, p≈0.762). Conclusion: `base_alloc=bridge` is not a viable “拉大” direction under the top1-med pipeline. |


### E0299: Best-to-test reproduction on test402 (E0298 selection; bridge base allocation)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0298 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0299_ave_p0_best_to_test_official_ltl_top1med_bridgealloc_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0298), `EVENTNESS=av_clipdiff_mlp`, `SEEDS` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0298_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0299_ave_p0_best_to_test_official_ltl_top1med_bridgealloc_v1.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0298_*/best_config.json SEEDS=0,1,2,3,4,5,6,7,8,9 bash scripts/e0299_ave_p0_best_to_test_official_ltl_top1med_bridgealloc_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0299_*` |
| Artifacts | `runs/E0299_*/metrics.json` |
| Results | Smoke: N/A (script runs full by default unless LIMIT_* is set). Full (test402; SEEDS=0..9): `runs/E0299_ave_p0_best_to_test_official_av_clipdiff_mlp_bridgealloc_20260205-023457/metrics.json` (anchored=0.72383 vs uniform=0.70858, Δ=+0.01525, p≈0.00390; matches the current best E0224 because E0298 selected the baseline config). |


### E0300: Diagnose far-anchor buckets for E0299 (bridge base allocation)
| Field | Value |
| --- | --- |
| Objective | Run a post-hoc diagnose report on E0299 and confirm that the dist∈{2..5} / high_count=2 degradation buckets shrink vs E0224. |
| Baseline | N/A (analysis-only) |
| Model | N/A |
| Weights | N/A |
| Code path | `avs/experiments/ave_p0_diagnose.py`, `scripts/e0300_ave_p0_diagnose_E0299_bridgealloc.sh` |
| Params | `IN_METRICS` (E0299 metrics.json), optional `DELTAS` |
| Metrics (must save) | `diagnose.json` |
| Checks | `anchor_plan_stats.delta_by_anchor_dist` and `delta_by_high_count` shift in the intended direction; record summary in C0003 evidence. |
| Smoke cmd | `IN_METRICS=runs/E0299_*/metrics.json bash scripts/e0300_ave_p0_diagnose_E0299_bridgealloc.sh` |
| Full cmd | Same as Smoke |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0300_*` |
| Artifacts | `runs/E0300_*/diagnose.json` |
| Results | TBA |

### E0301: Stage-2 plan sweep on val402 (top1-med k=1 + 112/224/448; ltl_top1med_k1_extreme_v1)
| Field | Value |
| --- | --- |
| Objective | Try to “进一步拉大” C0003 by removing anchor2 entirely (k=1) to avoid wasting base slots on spurious second anchors, while still allowing a high peak resolution via the aggressive 112/224/448 triad (strict equal-token budget). Run a fixed candidate sweep on official val402. |
| Baseline | `uniform` |
| Model | Same P0 head-only training loop as E0207/E0208 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0301_ave_p0_sweep_official_val_ltl_top1med_k1_extreme_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_k1_extreme_v1`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Winner transfers to test402; compare against E0224 (current best). |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0301_ave_p0_sweep_official_val_ltl_top1med_k1_extreme_v1.sh` |
| Full cmd | `bash scripts/e0301_ave_p0_sweep_official_val_ltl_top1med_k1_extreme_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0301_*` |
| Artifacts | `runs/E0301_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0301_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_k1_extreme_v1_20260205-024400/sweep_summary.json` (best Δ≈-0.01094). Full (val402; SEEDS=0..2): `runs/E0301_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_k1_extreme_v1_20260205-024429/sweep_summary.json` (best=`ltltop1medk1ext_thr0p6_shift0_distance`, Δ≈+0.00856, p≈0.0657). |


### E0302: Best-to-test reproduction on test402 (E0301 selection; top1-med k=1 extreme triad)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0301 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0302_ave_p0_best_to_test_official_ltl_top1med_k1_extreme_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0301), `EVENTNESS=av_clipdiff_mlp`, `SEEDS` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0301_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0302_ave_p0_best_to_test_official_ltl_top1med_k1_extreme_v1.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0301_*/best_config.json SEEDS=0,1,2,3,4,5,6,7,8,9 bash scripts/e0302_ave_p0_best_to_test_official_ltl_top1med_k1_extreme_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0302_*` |
| Artifacts | `runs/E0302_*/metrics.json` |
| Results | Full (test402; SEEDS=0..9): `runs/E0302_ave_p0_best_to_test_official_av_clipdiff_mlp_k1extreme_20260205-025129/metrics.json` (anchored=0.71020 vs uniform=0.70858, Δ=+0.00162, p≈0.649; large regression vs E0224). |


### E0303: Stage-2 budget-band plan sweep on val402 (attempt to “拉大” C0003)
| Field | Value |
| --- | --- |
| Objective | Try a budget-band Stage-2 plan (`budget_mode=band`, under-budget ≤1%) that can use extra cheap resolution (112) to preserve more `base_res` context, and sweep a small `top1_med` gate grid on official val402 (`candidate_set=ltl_top1med_band_v1`). |
| Baseline | `uniform` |
| Model | `temporal_conv` head on frozen CLIP features (same protocol as E0223/E0224) |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/sampling/plans.py` (`budget_band_anchored_plan_scored`), `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0303_ave_p0_sweep_official_val_ltl_top1med_band_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_band_v1`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Winner transfers to test402; compare against E0224 (current best). |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0303_ave_p0_sweep_official_val_ltl_top1med_band_v1.sh` |
| Full cmd | `bash scripts/e0303_ave_p0_sweep_official_val_ltl_top1med_band_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0303_*` |
| Artifacts | `runs/E0303_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0303_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_band_v1_20260205-033812/sweep_summary.json`. Full (val402; SEEDS=0..2): `runs/E0303_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_band_v1_20260205-033915/sweep_summary.json` (best=`ltltop1medband_thr0p7_shift1`, Δ≈+0.01205, p≈0.0685). |


### E0304: Best-to-test reproduction on test402 (E0303 selection; top1-med budget-band)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0303 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0304_ave_p0_best_to_test_official_ltl_top1med_band_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0303), `EVENTNESS=av_clipdiff_mlp`, `SEEDS` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0303_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0304_ave_p0_best_to_test_official_ltl_top1med_band_v1.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0303_*/best_config.json SEEDS=0,1,2,3,4,5,6,7,8,9 bash scripts/e0304_ave_p0_best_to_test_official_ltl_top1med_band_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0304_*` |
| Artifacts | `runs/E0304_*/metrics.json` |
| Results | Smoke (train64/test32; SEEDS=0,1; EPOCHS=1): `runs/E0304_ave_p0_best_to_test_official_av_clipdiff_mlp_20260205-033850/metrics.json` (Δ=+0.00000). Full (test402; SEEDS=0..9): `runs/E0304_ave_p0_best_to_test_official_av_clipdiff_mlp_20260205-035830/metrics.json` (anchored=0.71674 vs uniform=0.70858, Δ=+0.00816, p≈0.0441; regresses vs E0224). Conclusion: this band-budget candidate set does not improve C0003. |


### E0305: Diagnostic test402 run for the E0224 gate under budget-band (thr0.6, shift1)
| Field | Value |
| --- | --- |
| Objective | Isolate whether `budget_mode=band` helps or hurts when keeping the E0224 gate (`top1_med thr=0.6, shift=1`) by running that config directly on test402 (SEEDS=0..9). |
| Baseline | `uniform` |
| Model | `temporal_conv` head; same downstream settings as E0224 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/sampling/plans.py` (`budget_band_anchored_plan_scored`), `avs/experiments/ave_p0_sweep.py` (`run`) |
| Params | `CONFIG_JSON=runs/E0303_*/ltltop1medband_thr0p6_shift1/config.json`, `EVENTNESS=av_clipdiff_mlp`, `SEEDS=0..9` |
| Metrics (must save) | `metrics.json` |
| Checks | Compare against E0224; if regresses, do not pursue band-budget as-is. |
| Smoke cmd | N/A |
| Full cmd | `python -m avs.experiments.ave_p0_sweep run --config-json runs/E0303_*/ltltop1medband_thr0p6_shift1/config.json --scores-json runs/E0303_*/eventness_scores.json --split-train train --split-eval test --train-ids-file data/AVE/meta/download_ok_train_official.txt --eval-ids-file data/AVE/meta/download_ok_test_official.txt --seeds 0,1,2,3,4,5,6,7,8,9 --epochs 5 --batch-size 16 --lr 2e-3 --eventness-method av_clipdiff_mlp --audio-device cpu --train-device cuda:0 --processed-dir runs/REAL_AVE_OFFICIAL_20260201-124535/processed --caches-dir runs/REAL_AVE_OFFICIAL_20260201-124535/caches_112_160_224_352_448 --allow-missing --out-dir runs/E0305_...` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0305_*` |
| Artifacts | `runs/E0305_*/metrics.json` |
| Results | Full (test402; SEEDS=0..9): `runs/E0305_ave_p0_best_to_test_official_av_clipdiff_mlp_banddiag_thr0p6_shift1_20260205-040513/metrics.json` (anchored=0.71214 vs uniform=0.70858, Δ=+0.00356, p≈0.230; large regression vs E0224). Conclusion: band-budget planning (as implemented) is not a viable “拉大” direction. |


### E0306: Control test402 run for E0305 (reuse E0223 score cache; confirm regression is not a scores-json mismatch)
| Field | Value |
| --- | --- |
| Objective | Verify that the E0305 regression is due to the band-budget planner itself rather than a mismatch in `eventness_scores.json` by rerunning the same config while reusing E0223’s score cache. |
| Baseline | `uniform` |
| Model | Same as E0305 |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`) |
| Params | Same as E0305, but `--scores-json` points to `runs/E0223_*/eventness_scores.json` |
| Metrics (must save) | `metrics.json` |
| Checks | The result matches E0305 (within noise); if so, band-budget regression is not caused by score caching. |
| Smoke cmd | N/A |
| Full cmd | `python -m avs.experiments.ave_p0_sweep run --config-json runs/E0303_*/ltltop1medband_thr0p6_shift1/config.json --scores-json runs/E0223_*/eventness_scores.json --split-train train --split-eval test --train-ids-file data/AVE/meta/download_ok_train_official.txt --eval-ids-file data/AVE/meta/download_ok_test_official.txt --seeds 0,1,2,3,4,5,6,7,8,9 --epochs 5 --batch-size 16 --lr 2e-3 --eventness-method av_clipdiff_mlp --audio-device cpu --train-device cuda:0 --processed-dir runs/REAL_AVE_OFFICIAL_20260201-124535/processed --caches-dir runs/REAL_AVE_OFFICIAL_20260201-124535/caches_112_160_224_352_448 --allow-missing --out-dir runs/E0306_...` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0306_*` |
| Artifacts | `runs/E0306_*/metrics.json` |
| Results | Full (test402; SEEDS=0..9): `runs/E0306_ave_p0_best_to_test_official_av_clipdiff_mlp_band_scoresE0223_thr0p6_shift1_20260205-041422/metrics.json` (Δ=+0.00356; identical to E0305). Conclusion: band-budget regression is not a score-cache artifact. |


### E0307: Stage-1 sweep on val402 (AST embeddings + CLIPdiff MIL anchors; `av_ast_clipdiff_mil_mlp` + `ltl_top1med_v1`)
| Field | Value |
| --- | --- |
| Objective | Upgrade Stage-1 anchors with a bolder signal (AST per-second embeddings + cheap CLIPdiff) trained with a MIL objective, then select the best top1-med gate config on official val402. |
| Baseline | `uniform` |
| Model | `temporal_conv` head on frozen CLIP features (same downstream protocol as E0223/E0224) |
| Weights | HF: AST (`--ast-pretrained`) |
| Code path | `avs/experiments/ave_p0.py` (`EVENTNESS=av_ast_clipdiff_mil_mlp`), `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0307_ave_p0_sweep_official_val_ltl_top1med_v1_av_ast_clipdiff_mil_mlp.sh` |
| Params | `EVENTNESS=av_ast_clipdiff_mil_mlp`, `CANDIDATE_SET=ltl_top1med_v1`, `AUDIO_DEVICE`, `TRAIN_DEVICE`, `SEEDS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Winner transfers to test402 and beats the current best E0224 (Δ≈+0.01525). |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0307_ave_p0_sweep_official_val_ltl_top1med_v1_av_ast_clipdiff_mil_mlp.sh` |
| Full cmd | `SEEDS=0,1,2 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0307_ave_p0_sweep_official_val_ltl_top1med_v1_av_ast_clipdiff_mil_mlp.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0307_*` |
| Artifacts | `runs/E0307_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0307_ave_p0_sweep_official_val_av_ast_clipdiff_mil_mlp_ltl_top1med_v1_20260205-045341/sweep_summary.json` (best Δ≈+0.00312, p≈0.50). Full (val402; SEEDS=0..2): `runs/E0307_ave_p0_sweep_official_val_av_ast_clipdiff_mil_mlp_ltl_top1med_v1_20260205-045530/sweep_summary.json` (best Δ≈-0.01180, p≈0.089; all candidates negative; not viable as-is). |


### E0308: Best-to-test reproduction on test402 (E0307 selection; `av_ast_clipdiff_mil_mlp`)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0307 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; HF: AST (`--ast-pretrained`) |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0308_ave_p0_best_to_test_official_ltl_top1med_v1_av_ast_clipdiff_mil_mlp.sh` |
| Params | `BEST_CONFIG_JSON` (from E0307), `EVENTNESS=av_ast_clipdiff_mil_mlp`, `AUDIO_DEVICE`, `TRAIN_DEVICE`, `SEEDS=0..9` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0307_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0308_ave_p0_best_to_test_official_ltl_top1med_v1_av_ast_clipdiff_mil_mlp.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0307_*/best_config.json SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0308_ave_p0_best_to_test_official_ltl_top1med_v1_av_ast_clipdiff_mil_mlp.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0308_*` |
| Artifacts | `runs/E0308_*/metrics.json` |
| Results | Skipped: E0307 full val402 sweep regresses (all candidates negative), so per P0101 stop rule we do not spend test402 budget. |


### E0309: Stage-1 sweep on val402 (AST embeddings + CLIPdiff MIL anchors; scale-invariant confidence gate)
| Field | Value |
| --- | --- |
| Objective | Retry `av_ast_clipdiff_mil_mlp` with a scale-invariant confidence gate (`top1_med_norm`) to avoid the “all clips pass due to large logits” failure mode observed in E0307, and check if this rescues val402. |
| Baseline | `uniform` |
| Model | Same downstream protocol as E0307 (`temporal_conv` head on frozen CLIP features) |
| Weights | HF: AST (`--ast-pretrained`) |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`) |
| Params | `EVENTNESS=av_ast_clipdiff_mil_mlp`, `CANDIDATE_SET=ltl_top1med_norm_v1`, `SEEDS=0..2` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json` |
| Checks | If Δ becomes clearly positive on val402 (vs E0307), proceed to test402; otherwise record as negative evidence. |
| Smoke cmd | `SEEDS=0,1 LIMIT_TRAIN=64 LIMIT_EVAL=32 EVENTNESS=av_ast_clipdiff_mil_mlp CANDIDATE_SET=ltl_top1med_norm_v1 bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` |
| Full cmd | `SEEDS=0,1,2 EVENTNESS=av_ast_clipdiff_mil_mlp CANDIDATE_SET=ltl_top1med_norm_v1 bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0309_*` |
| Artifacts | `runs/E0309_*/{sweep_summary.json,best_config.json}` |
| Results | Full (val402; SEEDS=0..2): `runs/E0309_ave_p0_sweep_official_val_av_ast_clipdiff_mil_mlp_ltl_top1med_norm_v1_20260205-051944/sweep_summary.json` (best Δ≈+0.00108, p≈0.883; near 0). Conclusion: scale-invariant gate does not rescue `av_ast_clipdiff_mil_mlp`. |


### E0310: Stage-1 sweep on val402 (apply scale-invariant top1-med gate to the current best Stage-1 method)
| Field | Value |
| --- | --- |
| Objective | Apply `top1_med_norm` confidence gating to `av_clipdiff_mlp` to verify whether the “normalized” gate itself yields better val402 selection (and potentially improves transfer), independent of AST methods. |
| Baseline | `uniform` |
| Model | Same downstream protocol as E0223/E0224 (`temporal_conv` head on frozen CLIP features) |
| Weights | Uses existing CLIP caches |
| Code path | `avs/experiments/ave_p0_sweep.py` (`sweep`) |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_norm_v1`, `SEEDS=0..2` (reuses E0223 score cache when provided) |
| Metrics (must save) | `sweep_summary.json`, `best_config.json` |
| Checks | If val402 improves vs E0223 best (Δ≈+0.00964), attempt a test402 reproduction; otherwise record as negative. |
| Smoke cmd | `SEEDS=0,1 LIMIT_TRAIN=64 LIMIT_EVAL=32 EVENTNESS=av_clipdiff_mlp CANDIDATE_SET=ltl_top1med_norm_v1 bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` |
| Full cmd | `SEEDS=0,1,2 EVENTNESS=av_clipdiff_mlp CANDIDATE_SET=ltl_top1med_norm_v1 bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0310_*` |
| Artifacts | `runs/E0310_*/{sweep_summary.json,best_config.json}` |
| Results | Full (val402; SEEDS=0..2): `runs/E0310_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_norm_v1_20260205-052411/sweep_summary.json` (best Δ≈+0.00723, p≈0.335; worse than E0223 best). Conclusion: `top1_med_norm` gate does not improve the current best Stage-1 method on val402. |


### E0311: Stage-1 sweep on val402 (bold new signal: learned A/V alignment via InfoNCE; `av_ast_clipalign_nce`)
| Field | Value |
| --- | --- |
| Objective | Try a bolder Stage-1 signal that explicitly targets “audio is on-screen”: learn a within-clip A/V alignment projector (InfoNCE) from AST embeddings and cheap CLIP features, then use diagonal similarity as eventness for anchors. |
| Baseline | `uniform` |
| Model | Same downstream protocol as E0223/E0224 (`temporal_conv` head on frozen CLIP features) |
| Weights | HF: AST (`--ast-pretrained`) |
| Code path | `avs/experiments/ave_p0.py` (`EVENTNESS=av_ast_clipalign_nce`), `avs/experiments/ave_p0_sweep.py` (`sweep`), `scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` |
| Params | `EVENTNESS=av_ast_clipalign_nce`, `CANDIDATE_SET=ltl_top1med_norm_v1`, `SEEDS=0..2`, `AUDIO_DEVICE=cuda:9`, `TRAIN_DEVICE=cuda:9` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Must beat the current best E0224 test delta (Δ≈+0.01525) after val→test; otherwise record as negative evidence. |
| Smoke cmd | `SEEDS=0,1 LIMIT_TRAIN=64 LIMIT_EVAL=32 EVENTNESS=av_ast_clipalign_nce CANDIDATE_SET=ltl_top1med_norm_v1 AUDIO_DEVICE=cuda:9 TRAIN_DEVICE=cuda:9 bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` |
| Full cmd | `SEEDS=0,1,2 EVENTNESS=av_ast_clipalign_nce CANDIDATE_SET=ltl_top1med_norm_v1 AUDIO_DEVICE=cuda:9 TRAIN_DEVICE=cuda:9 bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0311_*` |
| Artifacts | `runs/E0311_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Full (val402; SEEDS=0..2): `runs/E0311_ave_p0_sweep_official_val_av_ast_clipalign_nce_ltl_top1med_norm_v1_20260205-055421/sweep_summary.json` (best Δ≈-0.00033, p≈0.937; near 0). Conclusion: `av_ast_clipalign_nce` does not improve anchors under AVE-P0. |


### E0312: Stage-2 sweep on val402 (k-adaptive anchor2 veto; `ltl_top1med_anchor2veto_v1`)
| Field | Value |
| --- | --- |
| Objective | Try to “拉大” C0003 by dropping spurious second anchors (k-adaptive) on top of the current best learned Stage-1 scores (`av_clipdiff_mlp`). Sweep a small, pre-registered veto grid on val402 and pick the best config. |
| Baseline | `uniform` |
| Model | `temporal_conv` head on frozen CLIP features (same downstream protocol as E0223/E0224) |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0.py` (`anchor2_veto_*`), `avs/experiments/ave_p0_sweep.py` (`candidate_set=ltl_top1med_anchor2veto_v1`), `scripts/e0312_ave_p0_sweep_official_val_ltl_top1med_anchor2veto_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_anchor2veto_v1`, `SEEDS`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If val402 Δ improves vs the E0223 best (Δ≈+0.00964), run a quick test402 check (SEEDS=0..2); only then run full test402 reproduction (E0313). |
| VRAM | ~4–8 GB |
| Time/epoch | ~minutes |
| Total time | ~1–2 hours (SEEDS=0..2) |
| Single-GPU script | `bash scripts/e0312_ave_p0_sweep_official_val_ltl_top1med_anchor2veto_v1.sh` |
| Multi-GPU script | Run different seeds on different GPUs by setting `TRAIN_DEVICE=cuda:X`. |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0312_ave_p0_sweep_official_val_ltl_top1med_anchor2veto_v1.sh` |
| Full cmd | `SEEDS=0,1,2 TRAIN_DEVICE=cuda:9 bash scripts/e0312_ave_p0_sweep_official_val_ltl_top1med_anchor2veto_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0312_*` |
| Artifacts | `runs/E0312_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0312_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_anchor2veto_v1_20260205-115846/sweep_summary.json` (all candidates tie; not informative). Full (val402; SEEDS=0..2): `runs/E0312_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_anchor2veto_v1_20260205-115927/sweep_summary.json` (best=`ltltop1med_a2veto_none`, Δ≈+0.00964, p≈0.033; veto variants regress, including `lr0p35/0p5/0p65` negative). Conclusion: anchor2 veto does not improve val402 selection. |


### E0313: Best-to-test reproduction on test402 (E0312 selection; anchor2 veto)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0312 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0313_ave_p0_best_to_test_official_ltl_top1med_anchor2veto_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0312), `EVENTNESS=av_clipdiff_mlp`, `TRAIN_DEVICE`, `SEEDS=0..9` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| VRAM | ~4–8 GB |
| Time/epoch | ~minutes |
| Total time | ~2–4 hours (SEEDS=0..9) |
| Single-GPU script | `BEST_CONFIG_JSON=runs/E0312_*/best_config.json bash scripts/e0313_ave_p0_best_to_test_official_ltl_top1med_anchor2veto_v1.sh` |
| Multi-GPU script | Use `SEEDS` slicing (e.g., `SEEDS=0,1,2,3,4`) across GPUs and merge by rerun script. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0312_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0313_ave_p0_best_to_test_official_ltl_top1med_anchor2veto_v1.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0312_*/best_config.json SEEDS=0,1,2,3,4,5,6,7,8,9 TRAIN_DEVICE=cuda:9 bash scripts/e0313_ave_p0_best_to_test_official_ltl_top1med_anchor2veto_v1.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0313_*` |
| Artifacts | `runs/E0313_*/metrics.json` |
| Results | Quick test402 diagnostics (SEEDS=0..2; not full): `runs/E0313_quick_test402_av_clipdiff_mlp_a2veto_lr0p65_20260205-120329/metrics.json` (Δ≈+0.00489) and `runs/E0313_quick_test402_av_clipdiff_mlp_a2veto_top2med0p15_20260205-120449/metrics.json` (Δ≈+0.00904), both far below the baseline E0224 on the same seeds (Δ≈+0.01899). Conclusion: do not run full E0313. |


### E0314: Stage-1 sweep on val402 (teacher-student visual-usefulness anchors; `av_clipdiff_visgain_mlp`)
| Field | Value |
| --- | --- |
| Objective | Try to “拉大” C0003 by improving Stage-1 reliability: train a deployable per-second scorer to predict a *visual usefulness* teacher derived from expensive visual features (resolution sensitivity), then use predicted scores for anchor selection under the existing Stage-2 top1-med pipeline. |
| Baseline | `uniform` |
| Model | `temporal_conv` head on frozen CLIP features (same downstream protocol as E0223/E0224) |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0.py` (`_train_audio_basic_mlp_visgain_eventness` + `av_clipdiff_visgain_mlp`), `avs/experiments/ave_p0_sweep.py` (scores cache), `scripts/e0314_ave_p0_sweep_official_val_ltl_top1med_visgain_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_visgain_mlp`, `CANDIDATE_SET=ltl_top1med_norm_v1`, `SEEDS`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If val402 Δ is competitive vs E0223/E0224, run E0315 quick test402 (SEEDS=0..2) before committing full test402. |
| VRAM | ~4–8 GB |
| Time/epoch | ~minutes |
| Total time | ~1–2 hours (SEEDS=0..2) |
| Single-GPU script | `bash scripts/e0314_ave_p0_sweep_official_val_ltl_top1med_visgain_v1.sh` |
| Multi-GPU script | Run different seeds on different GPUs by setting `TRAIN_DEVICE=cuda:X`. |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0314_ave_p0_sweep_official_val_ltl_top1med_visgain_v1.sh` |
| Full cmd | `SEEDS=0,1,2 TRAIN_DEVICE=cuda:9 bash scripts/e0314_ave_p0_sweep_official_val_ltl_top1med_visgain_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0314_*` |
| Artifacts | `runs/E0314_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0314_ave_p0_sweep_official_val_av_clipdiff_visgain_mlp_ltl_top1med_norm_v1_20260205-123747/sweep_summary.json` (all candidates tie at Δ=0; not informative). Full (val402; SEEDS=0..2): `runs/E0314_ave_p0_sweep_official_val_av_clipdiff_visgain_mlp_ltl_top1med_norm_v1_20260205-123935/sweep_summary.json` (best Δ≈+0.00158, p≈0.727; near 0). Conclusion: not a viable “拉大” direction as-is. |


### E0315: Best-to-test reproduction on test402 (E0314 selection; teacher-student visgain anchors)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0314 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0315_ave_p0_best_to_test_official_ltl_top1med_visgain_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0314), `EVENTNESS=av_clipdiff_visgain_mlp`, `TRAIN_DEVICE`, `SEEDS=0..9` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| VRAM | ~4–8 GB |
| Time/epoch | ~minutes |
| Total time | ~2–4 hours (SEEDS=0..9) |
| Single-GPU script | `BEST_CONFIG_JSON=runs/E0314_*/best_config.json bash scripts/e0315_ave_p0_best_to_test_official_ltl_top1med_visgain_v1.sh` |
| Multi-GPU script | Use `SEEDS` slicing (e.g., `SEEDS=0,1,2,3,4`) across GPUs and merge by rerun script. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0314_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0315_ave_p0_best_to_test_official_ltl_top1med_visgain_v1.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0314_*/best_config.json SEEDS=0,1,2,3,4,5,6,7,8,9 TRAIN_DEVICE=cuda:9 bash scripts/e0315_ave_p0_best_to_test_official_ltl_top1med_visgain_v1.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0315_*` |
| Artifacts | `runs/E0315_*/metrics.json` |
| Results | Not run: E0314 is near-0 on val402; stop to avoid spending test402 budget. |


### E0316: Stage-1 sweep on val402 (teacher-student downstream loss-gain anchors; `av_clipdiff_lossgain_mlp`)
| Field | Value |
| --- | --- |
| Objective | Try to “拉大” C0003 with a downstream-aware teacher: train a cheap base-res vision classifier teacher on the train split, define per-second targets as the **loss reduction** when swapping in high-res visual features (event seconds only), then train a deployable student scorer (cheap A+V inputs) and run the standard val402 selection sweep. |
| Baseline | `uniform` |
| Model | `temporal_conv` head on frozen CLIP features (same downstream protocol as E0223/E0224) |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0.py` (`av_clipdiff_lossgain_mlp`), `avs/experiments/ave_p0_sweep.py` (scores cache), `scripts/e0316_ave_p0_sweep_official_val_ltl_top1med_lossgain_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_lossgain_mlp`, `CANDIDATE_SET=ltl_top1med_norm_v1`, `SEEDS`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If val402 Δ is competitive vs E0223/E0224, run E0317 quick test402 (SEEDS=0..2) before committing full test402. |
| VRAM | ~4–8 GB |
| Time/epoch | ~minutes |
| Total time | ~1–2 hours (SEEDS=0..2) |
| Single-GPU script | `bash scripts/e0316_ave_p0_sweep_official_val_ltl_top1med_lossgain_v1.sh` |
| Multi-GPU script | Run different seeds on different GPUs by setting `TRAIN_DEVICE=cuda:X`. |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0316_ave_p0_sweep_official_val_ltl_top1med_lossgain_v1.sh` |
| Full cmd | `SEEDS=0,1,2 TRAIN_DEVICE=cuda:9 bash scripts/e0316_ave_p0_sweep_official_val_ltl_top1med_lossgain_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0316_*` |
| Artifacts | `runs/E0316_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0316_ave_p0_sweep_official_val_av_clipdiff_lossgain_mlp_ltl_top1med_norm_v1_20260205-125349/sweep_summary.json` (all candidates tie at Δ=0; not informative). Full (val402; SEEDS=0..2): `runs/E0316_ave_p0_sweep_official_val_av_clipdiff_lossgain_mlp_ltl_top1med_norm_v1_20260205-125414/sweep_summary.json` (best Δ≈+0.00042, p≈0.906; near 0). Conclusion: loss-reduction teacher does not improve Stage-1 anchor reliability in AVE-P0. |


### E0317: Best-to-test reproduction on test402 (E0316 selection; teacher-student loss-gain anchors)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0316 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0317_ave_p0_best_to_test_official_ltl_top1med_lossgain_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0316), `EVENTNESS=av_clipdiff_lossgain_mlp`, `TRAIN_DEVICE`, `SEEDS=0..9` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| VRAM | ~4–8 GB |
| Time/epoch | ~minutes |
| Total time | ~2–4 hours (SEEDS=0..9) |
| Single-GPU script | `BEST_CONFIG_JSON=runs/E0316_*/best_config.json bash scripts/e0317_ave_p0_best_to_test_official_ltl_top1med_lossgain_v1.sh` |
| Multi-GPU script | Use `SEEDS` slicing (e.g., `SEEDS=0,1,2,3,4`) across GPUs and merge by rerun script. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0316_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0317_ave_p0_best_to_test_official_ltl_top1med_lossgain_v1.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0316_*/best_config.json SEEDS=0,1,2,3,4,5,6,7,8,9 TRAIN_DEVICE=cuda:9 bash scripts/e0317_ave_p0_best_to_test_official_ltl_top1med_lossgain_v1.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0317_*` |
| Artifacts | `runs/E0317_*/metrics.json` |
| Results | Not run: E0316 is near-0 on val402; stop to avoid spending test402 budget. |


### E0318: Stage-1 sweep on val402 (A/V correspondence anchors; `av_ast_clipalign_bce`)
| Field | Value |
| --- | --- |
| Objective | Try to “拉大” C0003 with a new signal: train a tiny A/V correspondence scorer (AST embeddings ↔ CLIP embeddings) with a BCE objective on per-second diagonal similarity, then use its logits as anchor scores and run the standard val402 selection sweep. |
| Baseline | `uniform` |
| Model | `temporal_conv` head on frozen CLIP features (same downstream protocol as E0223/E0224) |
| Weights | Uses existing CLIP caches; uses AST embeddings (set `AST_PRETRAINED=1` for real runs). |
| Code path | `avs/experiments/ave_p0.py` (`av_ast_clipalign_bce`), `avs/experiments/ave_p0_sweep.py` (scores cache), `scripts/e0318_ave_p0_sweep_official_val_ltl_top1med_av_clipalign_bce_v1.sh` |
| Params | `EVENTNESS=av_ast_clipalign_bce`, `CANDIDATE_SET=ltl_top1med_norm_v1`, `SEEDS`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If val402 Δ is competitive vs E0223/E0224, run E0319 quick test402 (SEEDS=0..2) before committing full test402. |
| VRAM | ~4–8 GB |
| Time/epoch | ~minutes |
| Total time | ~1–2 hours (SEEDS=0..2) |
| Single-GPU script | `bash scripts/e0318_ave_p0_sweep_official_val_ltl_top1med_av_clipalign_bce_v1.sh` |
| Multi-GPU script | Run different seeds on different GPUs by setting `TRAIN_DEVICE=cuda:X`. |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0318_ave_p0_sweep_official_val_ltl_top1med_av_clipalign_bce_v1.sh` |
| Full cmd | `SEEDS=0,1,2 TRAIN_DEVICE=cuda:9 bash scripts/e0318_ave_p0_sweep_official_val_ltl_top1med_av_clipalign_bce_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0318_*` |
| Artifacts | `runs/E0318_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0318_ave_p0_sweep_official_val_av_ast_clipalign_bce_ltl_top1med_norm_v1_20260205-133658/sweep_summary.json` (best Δ≈+0.00156; not informative). Full (val402; SEEDS=0..2): `runs/E0318_ave_p0_sweep_official_val_av_ast_clipalign_bce_ltl_top1med_norm_v1_20260205-133800/sweep_summary.json` (best Δ≈+0.00865, p≈0.00120). Conclusion: does not beat E0223 val selection; stop before test402. |


### E0319: Best-to-test reproduction on test402 (E0318 selection; A/V correspondence anchors)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0318 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; uses AST embeddings (set `AST_PRETRAINED=1` for real runs). |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0319_ave_p0_best_to_test_official_ltl_top1med_av_clipalign_bce_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0318), `EVENTNESS=av_ast_clipalign_bce`, `TRAIN_DEVICE`, `SEEDS=0..9` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| VRAM | ~4–8 GB |
| Time/epoch | ~minutes |
| Total time | ~2–4 hours (SEEDS=0..9) |
| Single-GPU script | `BEST_CONFIG_JSON=runs/E0318_*/best_config.json bash scripts/e0319_ave_p0_best_to_test_official_ltl_top1med_av_clipalign_bce_v1.sh` |
| Multi-GPU script | Use `SEEDS` slicing (e.g., `SEEDS=0,1,2,3,4`) across GPUs and merge by rerun script. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0318_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0319_ave_p0_best_to_test_official_ltl_top1med_av_clipalign_bce_v1.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0318_*/best_config.json SEEDS=0,1,2,3,4,5,6,7,8,9 TRAIN_DEVICE=cuda:9 bash scripts/e0319_ave_p0_best_to_test_official_ltl_top1med_av_clipalign_bce_v1.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0319_*` |
| Artifacts | `runs/E0319_*/metrics.json` |
| Results | Not run: E0318 regresses vs E0223 on val402; stop to avoid spending test402 budget. |


### E0320: Stage-2 sweep on val402 (band-budget + low=112 + mid=160; preserve context under equal tokens)
| Field | Value |
| --- | --- |
| Objective | Try to “拉大” C0003 with a bolder Stage-2 plan only (keep Stage-1 fixed): run a fixed-space sweep under `candidate_set=ltl_top1med_band_low112_v1` which uses the band-budget planner (≤1% underbudget), sets `low_res=112`, and allows an extra mid resolution (160) to preserve more base context. |
| Baseline | `uniform` |
| Model | `temporal_conv` head on frozen CLIP features (same downstream protocol as E0223/E0224) |
| Weights | Uses existing CLIP caches; reuses `eventness_scores.json` from E0223 when provided. |
| Code path | `avs/experiments/ave_p0_sweep.py` (`ltl_top1med_band_low112_v1`), `scripts/e0320_ave_p0_sweep_official_val_ltl_top1med_band_low112_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_band_low112_v1`, `SCORES_JSON` (optional), `SEEDS`, `TRAIN_DEVICE` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json` (and optionally `eventness_scores.json` if recomputed) |
| Checks | If val402 Δ is competitive vs E0223/E0224, run E0321 quick test402 (SEEDS=0..2) before committing full test402. |
| VRAM | ~4–8 GB |
| Time/epoch | ~minutes |
| Total time | ~1–2 hours (SEEDS=0..2) |
| Single-GPU script | `bash scripts/e0320_ave_p0_sweep_official_val_ltl_top1med_band_low112_v1.sh` |
| Multi-GPU script | Run different seeds on different GPUs by setting `TRAIN_DEVICE=cuda:X` (and reusing the same `SCORES_JSON`). |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0320_ave_p0_sweep_official_val_ltl_top1med_band_low112_v1.sh` |
| Full cmd | `SCORES_JSON=runs/E0223_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_v1_20260204-135150/eventness_scores.json SEEDS=0,1,2 TRAIN_DEVICE=cuda:9 bash scripts/e0320_ave_p0_sweep_official_val_ltl_top1med_band_low112_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0320_*` |
| Artifacts | `runs/E0320_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0320_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_band_low112_v1_20260205-142028/sweep_summary.json`. Full (val402; SEEDS=0..2): `runs/E0320_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_band_low112_v1_20260205-142258/sweep_summary.json` (best=`ltltop1medband112_thr0p7_shift1`, Δ≈+0.01205, p≈0.0685). |


### E0321: Best-to-test reproduction on test402 (E0320 selection; band-budget + low=112)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0320 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0321_ave_p0_best_to_test_official_ltl_top1med_band_low112_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0320), `EVENTNESS=av_clipdiff_mlp`, `TRAIN_DEVICE`, `SEEDS=0..9` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| VRAM | ~4–8 GB |
| Time/epoch | ~minutes |
| Total time | ~2–4 hours (SEEDS=0..9) |
| Single-GPU script | `BEST_CONFIG_JSON=runs/E0320_*/best_config.json bash scripts/e0321_ave_p0_best_to_test_official_ltl_top1med_band_low112_v1.sh` |
| Multi-GPU script | Use `SEEDS` slicing (e.g., `SEEDS=0,1,2,3,4`) across GPUs and merge by rerun script. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0320_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0321_ave_p0_best_to_test_official_ltl_top1med_band_low112_v1.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0320_*/best_config.json SEEDS=0,1,2,3,4,5,6,7,8,9 TRAIN_DEVICE=cuda:9 bash scripts/e0321_ave_p0_best_to_test_official_ltl_top1med_band_low112_v1.sh` |
| Smoke | [x] |
| Full | [ ] |
| Logs | `runs/E0321_*` |
| Artifacts | `runs/E0321_*/metrics.json` |
| Results | Quick test402 (SEEDS=0..2): `runs/E0321_quick_test402_av_clipdiff_mlp_band_low112_20260205-142726/metrics.json` (Δ≈+0.01028, p≈0.204; regresses vs E0224 seeds subset). Stop before full SEEDS=0..9. |


### E0322: Stage-1 sweep on val402 (speech-aware anchors; `asr_vad` v1)
| Field | Value |
| --- | --- |
| Objective | Try to “拉大” C0003 with a deployable speech-aware Stage-1 signal: compute per-second speech ratio (WebRTC-VAD), suppress energy_stride_max by `(1 - speech_ratio)`, and run the standard val402 selection sweep under `CANDIDATE_SET=ltl_top1med_norm_v1`. |
| Baseline | `uniform` |
| Model | `temporal_conv` head on frozen CLIP features (same downstream protocol as E0223/E0224) |
| Weights | No extra pretrained weights; VAD is rule-based. |
| Code path | `avs/audio/vad_webrtc.py`, `avs/experiments/ave_p0.py` (`asr_vad`), `avs/experiments/ave_p0_sweep.py` (scores cache), `scripts/e0322_ave_p0_sweep_official_val_ltl_top1med_asr_vad_v1.sh` |
| Params | `EVENTNESS=asr_vad`, `CANDIDATE_SET=ltl_top1med_norm_v1`, `SEEDS`, `TRAIN_DEVICE` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If val402 Δ is competitive vs E0223/E0224, run E0323 quick test402 (SEEDS=0..2) before committing full test402. |
| VRAM | ~4–8 GB |
| Time/epoch | ~minutes |
| Total time | ~1–2 hours (SEEDS=0..2) |
| Single-GPU script | `bash scripts/e0322_ave_p0_sweep_official_val_ltl_top1med_asr_vad_v1.sh` |
| Multi-GPU script | Run different seeds on different GPUs by setting `TRAIN_DEVICE=cuda:X`. |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0322_ave_p0_sweep_official_val_ltl_top1med_asr_vad_v1.sh` |
| Full cmd | `SEEDS=0,1,2 TRAIN_DEVICE=cuda:9 bash scripts/e0322_ave_p0_sweep_official_val_ltl_top1med_asr_vad_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0322_*` |
| Artifacts | `runs/E0322_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0322_ave_p0_sweep_official_val_asr_vad_ltl_top1med_norm_v1_20260205-141929/sweep_summary.json`. Full (val402; SEEDS=0..2): `runs/E0322_ave_p0_sweep_official_val_asr_vad_ltl_top1med_norm_v1_20260205-142328/sweep_summary.json` (best Δ≈+0.00216, p≈0.842). Conclusion: not competitive; stop before test402. |


### E0323: Best-to-test reproduction on test402 (E0322 selection; speech-aware anchors)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0322 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | No extra pretrained weights; VAD is rule-based. |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0323_ave_p0_best_to_test_official_ltl_top1med_asr_vad_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0322), `EVENTNESS=asr_vad`, `TRAIN_DEVICE`, `SEEDS=0..9` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| VRAM | ~4–8 GB |
| Time/epoch | ~minutes |
| Total time | ~2–4 hours (SEEDS=0..9) |
| Single-GPU script | `BEST_CONFIG_JSON=runs/E0322_*/best_config.json bash scripts/e0323_ave_p0_best_to_test_official_ltl_top1med_asr_vad_v1.sh` |
| Multi-GPU script | Use `SEEDS` slicing (e.g., `SEEDS=0,1,2,3,4`) across GPUs and merge by rerun script. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0322_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0323_ave_p0_best_to_test_official_ltl_top1med_asr_vad_v1.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0322_*/best_config.json SEEDS=0,1,2,3,4,5,6,7,8,9 TRAIN_DEVICE=cuda:9 bash scripts/e0323_ave_p0_best_to_test_official_ltl_top1med_asr_vad_v1.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0323_*` |
| Artifacts | `runs/E0323_*/metrics.json` |
| Results | Not run: depends on E0322 being competitive on val402. |


### E0324: Stage-1 sweep on val402 (AST speech-veto "non-speech max" anchors; `ast_nonspeech_max`)
| Field | Value |
| --- | --- |
| Objective | Try to “拉大” C0003 with a deployable semantic speech-veto: run pretrained AST logits, veto speech-like labels (`{speech, conversation, narration}`), and use `max(sigmoid(logits_non_speech))` as per-second scores for anchors under `CANDIDATE_SET=ltl_top1med_norm_v1`. |
| Baseline | `uniform` |
| Model | `temporal_conv` head on frozen CLIP features (same downstream protocol as E0223/E0224) |
| Weights | HF: AST (`--ast-pretrained`) + existing CLIP caches |
| Code path | `avs/experiments/ave_p0.py` (`ast_nonspeech_max`), `avs/experiments/ave_p0_sweep.py` (scores cache), `scripts/e0324_ave_p0_sweep_official_val_ltl_top1med_ast_nonspeech_max_v1.sh` |
| Params | `EVENTNESS=ast_nonspeech_max`, `CANDIDATE_SET=ltl_top1med_norm_v1`, `SEEDS`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If val402 Δ is competitive vs E0223/E0224, run E0325 quick test402 (SEEDS=0..2) before committing full test402. |
| VRAM | ~4–8 GB (head) + AST inference GPU (optional) |
| Time/epoch | ~minutes |
| Total time | ~1–2 hours (SEEDS=0..2) |
| Single-GPU script | `bash scripts/e0324_ave_p0_sweep_official_val_ltl_top1med_ast_nonspeech_max_v1.sh` |
| Multi-GPU script | Run different seeds on different GPUs by setting `TRAIN_DEVICE=cuda:X`. |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0324_ave_p0_sweep_official_val_ltl_top1med_ast_nonspeech_max_v1.sh` |
| Full cmd | `SEEDS=0,1,2 AUDIO_DEVICE=cuda:9 TRAIN_DEVICE=cuda:9 bash scripts/e0324_ave_p0_sweep_official_val_ltl_top1med_ast_nonspeech_max_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0324_*` |
| Artifacts | `runs/E0324_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0324_ave_p0_sweep_official_val_ast_nonspeech_max_ltl_top1med_norm_v1_20260205-144025/sweep_summary.json`. Full (val402; SEEDS=0..2): `runs/E0324_ave_p0_sweep_official_val_ast_nonspeech_max_ltl_top1med_norm_v1_20260205-144057/sweep_summary.json` (best Δ≈+0.00324, p≈0.722). Conclusion: not competitive; stop before test402. |


### E0325: Best-to-test reproduction on test402 (E0324 selection; AST speech-veto anchors)
| Field | Value |
| --- | --- |
| Objective | Reproduce the best config selected by E0324 on official test402 (SEEDS=0..9) and check whether C0003 (+2%, p<0.05) is met. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | HF: AST (`--ast-pretrained`) + existing CLIP caches |
| Code path | `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0325_ave_p0_best_to_test_official_ltl_top1med_ast_nonspeech_max_v1.sh` |
| Params | `BEST_CONFIG_JSON` (from E0324), `EVENTNESS=ast_nonspeech_max`, `AUDIO_DEVICE`, `TRAIN_DEVICE`, `SEEDS=0..9` |
| Metrics (must save) | `metrics.json` |
| Checks | Report `Δ = anchored_top2 - uniform` and p-value; if Δ≥+0.02 and p<0.05, mark C0003 proven. |
| VRAM | ~4–8 GB |
| Time/epoch | ~minutes |
| Total time | ~2–4 hours (SEEDS=0..9) |
| Single-GPU script | `BEST_CONFIG_JSON=runs/E0324_*/best_config.json bash scripts/e0325_ave_p0_best_to_test_official_ltl_top1med_ast_nonspeech_max_v1.sh` |
| Multi-GPU script | Use `SEEDS` slicing (e.g., `SEEDS=0,1,2,3,4`) across GPUs and merge by rerun script. |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0324_*/best_config.json LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0325_ave_p0_best_to_test_official_ltl_top1med_ast_nonspeech_max_v1.sh` |
| Full cmd | `BEST_CONFIG_JSON=runs/E0324_*/best_config.json SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:9 TRAIN_DEVICE=cuda:9 bash scripts/e0325_ave_p0_best_to_test_official_ltl_top1med_ast_nonspeech_max_v1.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0325_*` |
| Artifacts | `runs/E0325_*/metrics.json` |
| Results | Not run: E0324 is near-0 on val402; stop to avoid spending test402 budget. |

### E0326: Stage-1 sweep on val402 (AST speech-prob feature; `av_clipdiff_speech_mlp`)
| Field | Value |
| --- | --- |
| Objective | Try to “拉大” C0003 by reducing speech-driven false anchors: append a pretrained AST-derived speech probability feature to the existing cheap A+V Stage-1 (`av_clipdiff_mlp`) and re-train the same per-second MLP scorer. Evaluate under the fixed top1-med normalized gate / Stage-2 pipeline on official val402. |
| Baseline | `uniform` |
| Model | `temporal_conv` head on frozen CLIP features (same downstream protocol as E0223/E0224) |
| Weights | HF: AST (`--ast-pretrained`) + existing CLIP caches |
| Code path | `avs/experiments/ave_p0.py` (`av_clipdiff_speech_mlp`), `avs/experiments/ave_p0_sweep.py` (scores cache), runner: `scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` |
| Params | `EVENTNESS=av_clipdiff_speech_mlp`, `CANDIDATE_SET=ltl_top1med_norm_v1`, `AST_PRETRAINED=1`, `SEEDS`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If val402 Δ is competitive vs E0223/E0224, promote to a quick test402 check before spending the full SEEDS=0..9 budget. |
| VRAM | ~4–8 GB |
| Time/epoch | ~minutes |
| Total time | ~1–2 hours (SEEDS=0..2) |
| Single-GPU script | `OUT_DIR=runs/E0326_ave_p0_sweep_official_val_av_clipdiff_speech_mlp_ltl_top1med_norm_v1_$(date +%Y%m%d-%H%M%S) AST_PRETRAINED=1 CANDIDATE_SET=ltl_top1med_norm_v1 EVENTNESS=av_clipdiff_speech_mlp bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` |
| Multi-GPU script | Run different seeds on different GPUs by setting `TRAIN_DEVICE=cuda:X` and slicing `SEEDS`. |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 OUT_DIR=runs/E0326_smoke_$(date +%Y%m%d-%H%M%S) AST_PRETRAINED=1 CANDIDATE_SET=ltl_top1med_norm_v1 EVENTNESS=av_clipdiff_speech_mlp bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` |
| Full cmd | `SEEDS=0,1,2 AUDIO_DEVICE=cuda:9 TRAIN_DEVICE=cuda:9 OUT_DIR=runs/E0326_full_$(date +%Y%m%d-%H%M%S) AST_PRETRAINED=1 CANDIDATE_SET=ltl_top1med_norm_v1 EVENTNESS=av_clipdiff_speech_mlp bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0326_*` |
| Artifacts | `runs/E0326_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Full (val402; SEEDS=0..2): `runs/E0326_ave_p0_sweep_official_val_av_clipdiff_speech_mlp_ltl_top1med_norm_v1_20260205-154338/sweep_summary.json` (best Δ≈+0.00407, p≈0.458). Conclusion: not competitive; stop before test402. |


### E0328: Stage-1 sweep on val402 (energy stride-max × non-speech AST veto; `energy_nonspeech_ast`)
| Field | Value |
| --- | --- |
| Objective | Try to “拉大” C0003 by suppressing speech-like peaks without training: use dense stride-based energy (max pooled per second) multiplied by `(1 - speech_prob_ast)` from pretrained AST, then run the standard val402 selection sweep. |
| Baseline | `uniform` |
| Model | `temporal_conv` head on frozen CLIP features (same downstream protocol as E0223/E0224) |
| Weights | HF: AST (`--ast-pretrained`) + existing CLIP caches |
| Code path | `avs/experiments/ave_p0.py` / `avs/experiments/ave_p0_sweep.py` (`energy_nonspeech_ast`), runner: `scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` |
| Params | `EVENTNESS=energy_nonspeech_ast`, `CANDIDATE_SET=ltl_top1med_norm_v1`, `AST_PRETRAINED=1`, `SEEDS`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If val402 Δ is competitive vs E0223/E0224, promote to a quick test402 check before spending the full SEEDS=0..9 budget. |
| VRAM | ~4–8 GB |
| Time/epoch | ~minutes |
| Total time | ~1–2 hours (SEEDS=0..2) |
| Single-GPU script | `OUT_DIR=runs/E0328_ave_p0_sweep_official_val_energy_nonspeech_ast_ltl_top1med_norm_v1_$(date +%Y%m%d-%H%M%S) AST_PRETRAINED=1 CANDIDATE_SET=ltl_top1med_norm_v1 EVENTNESS=energy_nonspeech_ast bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` |
| Multi-GPU script | Run different seeds on different GPUs by setting `TRAIN_DEVICE=cuda:X` and slicing `SEEDS`. |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 OUT_DIR=runs/E0328_smoke_$(date +%Y%m%d-%H%M%S) AST_PRETRAINED=1 CANDIDATE_SET=ltl_top1med_norm_v1 EVENTNESS=energy_nonspeech_ast bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` |
| Full cmd | `SEEDS=0,1,2 AUDIO_DEVICE=cuda:9 TRAIN_DEVICE=cuda:9 OUT_DIR=runs/E0328_full_$(date +%Y%m%d-%H%M%S) AST_PRETRAINED=1 CANDIDATE_SET=ltl_top1med_norm_v1 EVENTNESS=energy_nonspeech_ast bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0328_*` |
| Artifacts | `runs/E0328_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Full (val402; SEEDS=0..2): `runs/E0328_ave_p0_sweep_official_val_energy_nonspeech_ast_ltl_top1med_norm_v1_20260205-155950/sweep_summary.json` (best Δ≈+0.00216, p≈0.0997). Conclusion: not competitive; stop before test402. |


### E0329: Stage-1 sweep on val402 (accflip teacher-student; `av_clipdiff_accflip_mlp`)
| Field | Value |
| --- | --- |
| Objective | Try a bolder downstream-aligned Stage-1 objective to “拉大” C0003: derive a *teacher* target from cached vision features by training base-res and high-res vision teachers, then mark per-second positives where **high-res is correct but base-res is not** (event seconds only). Train a deployable cheap A+V student scorer on (audio basic + clipdiff scalar) to predict this target, and run the standard val402 selection sweep. |
| Baseline | `uniform` |
| Model | `temporal_conv` head on frozen CLIP features (same downstream protocol as E0223/E0224) |
| Weights | Uses existing CLIP caches; no extra pretrained weights |
| Code path | `avs/experiments/ave_p0.py` / `avs/experiments/ave_p0_sweep.py` (`av_clipdiff_accflip_mlp`), runner: `scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` |
| Params | `EVENTNESS=av_clipdiff_accflip_mlp`, `CANDIDATE_SET=ltl_top1med_norm_v1`, `SEEDS`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | If val402 Δ is competitive vs E0223/E0224, promote to a quick test402 check before spending the full SEEDS=0..9 budget. |
| VRAM | ~4–8 GB |
| Time/epoch | ~minutes |
| Total time | ~1–2 hours (SEEDS=0..2) |
| Single-GPU script | `OUT_DIR=runs/E0329_ave_p0_sweep_official_val_av_clipdiff_accflip_mlp_ltl_top1med_norm_v1_$(date +%Y%m%d-%H%M%S) CANDIDATE_SET=ltl_top1med_norm_v1 EVENTNESS=av_clipdiff_accflip_mlp bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` |
| Multi-GPU script | Run different seeds on different GPUs by setting `TRAIN_DEVICE=cuda:X` and slicing `SEEDS`. |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 OUT_DIR=runs/E0329_smoke_$(date +%Y%m%d-%H%M%S) CANDIDATE_SET=ltl_top1med_norm_v1 EVENTNESS=av_clipdiff_accflip_mlp bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` |
| Full cmd | `SEEDS=0,1,2 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:9 OUT_DIR=runs/E0329_full_$(date +%Y%m%d-%H%M%S) CANDIDATE_SET=ltl_top1med_norm_v1 EVENTNESS=av_clipdiff_accflip_mlp bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0329_*` |
| Artifacts | `runs/E0329_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Full (val402; SEEDS=0..2): `runs/E0329_ave_p0_sweep_official_val_av_clipdiff_accflip_mlp_ltl_top1med_norm_v1_20260205-165330/sweep_summary.json` (best Δ≈-0.00091, p≈0.810). Conclusion: downstream-aligned accflip teacher does not improve Stage-1 in AVE-P0; stop before test402. |


### E0330: Multi-budget Pareto grid on AVE (Oracle→Predicted + controls)
| Field | Value |
| --- | --- |
| Objective | “冲 oral” 生死图#1：在预注册预算点（多个 triads/Token budgets）上生成 Acc–Tok Pareto（含 CI），并在同一 budget 下同时报告 Oracle / Predicted / Random / Cheap-visual controls，回答：机制上限是否存在、Pred gap 多大、“any window works?” 是否成立、Pareto 是否跨预算稳定。 |
| Baseline | `uniform` |
| Model | Same P0 head-only protocol as E0224 (frozen CLIP caches + `temporal_conv` head). |
| Weights | CLIP caches (from `runs/REAL_AVE_OFFICIAL_20260201-124535/caches_*`); optional AST if using AST-based Stage-1. |
| Code path | `avs/experiments/mde_ltl.py` (`pareto_grid`), `avs/visualize/pareto_report.py`, `scripts/e0330_mde_pareto_grid_official.sh` |
| Params | `EVENTNESS`, `BASE_CONFIG_JSON`, `SCORES_JSON`, `TRIADS`, `BUDGET_MODE`, `BUDGET_EPSILON_FRAC`, `BUDGET_EXTRA_RESOLUTIONS`, `SEEDS`, `LIMIT_TRAIN`, `LIMIT_EVAL`, `TRAIN_DEVICE`, `AUDIO_DEVICE`, `AST_PRETRAINED`, `INCLUDE_CHEAP_VISUAL` |
| Metrics (must save) | `pareto_report.json`, `pareto.png`, per-budget `metrics_predicted_*.json`, optional `metrics_cheap_visual_*.json` |
| Checks | (1) Oracle > Uniform at ≥1 budget. (2) Predicted is competitive and does not collapse to uniform across budgets. (3) Cheap-visual does **not** match Predicted (kills “any window works”). |
| VRAM | ~4–8 GB (head training) + optional audio probe GPU |
| Time/epoch | ~minutes |
| Total time | ~hours (depends on `|TRIADS| × |SEEDS|`) |
| Single-GPU script | `EVENTNESS=av_clipdiff_mlp bash scripts/e0330_mde_pareto_grid_official.sh` |
| Multi-GPU script | Run different `SEEDS` slices on different GPUs and keep separate `OUT_DIR`s; merge the final plot offline if needed. |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 EVENTNESS=energy bash scripts/e0330_mde_pareto_grid_official.sh` |
| Full cmd | `SEEDS=0,1,2,3,4,5,6,7,8,9 EVENTNESS=av_clipdiff_mlp TRAIN_DEVICE=cuda:9 bash scripts/e0330_mde_pareto_grid_official.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0330_*` |
| Artifacts | `runs/E0330_*/{pareto_report.json,pareto.png,metrics_*.json}` |
| Results | Smoke (train64/test32; SEEDS=0,1; EPOCHS=1; EVENTNESS=energy): `runs/E0330_mde_pareto_grid_official_energy_20260205-174428/pareto_report.json` + `runs/E0330_mde_pareto_grid_official_energy_20260205-174428/pareto.png` (includes per-budget `metrics_predicted_*.json` with `token_usage`). Local full rerun (official test402; SEEDS=0..9; EVENTNESS=av_clipdiff_mlp; base config from local E0223 best=`ltltop1med_thr0p7_shift1`; budget_mode=auto): `runs/E0330_mde_pareto_grid_official_av_clipdiff_mlp_local_20260209-235305/pareto_report.json` + `.../pareto.png`. Predicted-vs-uniform Δ: `112_160_224≈-0.00328`, `160_224_352≈-0.00495`, `224_352_448≈+0.00299`; Oracle-vs-uniform Δ: `+0.02042`, `+0.03754`, `+0.02037` (cheap-visual collapses to uniform on this rerun). PSP-aligned rerun (official test402; SEEDS=0..2; EVENTNESS=psp_avel_evt; base config from E0978 best=`ltlgini_keepadj_gini0p45_hconf0p5`): `runs/E0330_mde_pareto_grid_official_psp_avel_evt_20260214-155549/pareto_report.json` + `.../pareto.png`. Predicted-vs-uniform Δ: `112_160_224≈+0.00896`, `160_224_352≈+0.00779`, `224_352_448≈-0.00149`; Oracle-vs-uniform Δ: `+0.02653`, `+0.02944`, `+0.02322`; Cheap-visual-vs-uniform Δ: `112_160_224≈-0.00987`, `160_224_352≈-0.00091`, `224_352_448≈-0.00431`. Historical reference (non-local): `runs/E0330_full_av_clipdiff_mlp_auto_20260205-184559/pareto_report.json`. |


### E0331: Degradation suite with downstream accuracy + α lower bound (oral-critical)
| Field | Value |
| --- | --- |
| Objective | “冲 oral” 生死图#3：在 `{shift_s, snr_db, silence_ratio, alpha}` 网格上同时报告 Stage-1 recall 与 downstream accuracy 的退化曲线/热力图，并验证 `alpha` 提供可计算下界（不低于 α-baseline）。 |
| Baseline | `uniform` (α=1) |
| Model | Same P0 downstream protocol as E0224, but run under perturbed audio for Stage-1. |
| Weights | Same caches; optional AST. |
| Code path | `avs/experiments/degradation_accuracy.py`, `scripts/e0331_degradation_accuracy_official.sh` |
| Metrics (must save) | `degradation_accuracy.json` + plots; must include the exact `{shift_s,snr_db,silence_ratio,alpha}` grid + seeds + token accounting. |
| Checks | No catastrophic failure: accuracy never drops below the α-baseline; trends are monotonic with corruption severity. |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 EVENTNESS_METHOD=av_clipdiff_mlp bash scripts/e0331_degradation_accuracy_official.sh` |
| Full cmd | `SEEDS=0,1,2 EVENTNESS_METHOD=av_clipdiff_mlp TRAIN_DEVICE=cuda:9 bash scripts/e0331_degradation_accuracy_official.sh` |
| Smoke | [x] |
| Full | [x] |
| Results | Smoke (train64/eval32; SEEDS=0,1; EPOCHS=1; EVENTNESS_METHOD=av_clipdiff_mlp): `runs/E0331_smoke_av_clipdiff_mlp_20260205-194038/degradation_accuracy.json` + `runs/E0331_smoke_av_clipdiff_mlp_20260205-194038/degradation_plots/*.png`. Local full rerun (train3339/test402; SEEDS=0..2): `runs/E0331_degradation_accuracy_av_clipdiff_mlp_local_20260209-235316/degradation_accuracy.json` + `.../degradation_plots/*.png` (clean mean: anchored≈0.71070 vs uniform≈0.71294, Δ≈-0.00224; gate fallback≈0.831; `alpha_floor_checks.num_fail=0`, `min_margin≈+0.000249`). PSP-aligned rerun (train3339/test402; SEEDS=0..2; EVENTNESS_METHOD=psp_avel_evt; base config from E0978): `runs/E0331_degradation_accuracy_psp_avel_evt_20260214-161014/degradation_accuracy.json` + `.../degradation_plots/*.png` (`alpha_floor_checks.num_fail=0`, `min_margin≈+0.000995`). Note: for `psp_avel_evt`, degradations are applied in score space (external teacher is not recomputed from raw audio). Historical reference (non-local): `runs/E0331_full_av_clipdiff_mlp_20260205-194925/degradation_accuracy.json`. |


### E0332: Stage-1 sweep on val402 (sep3 confidence gate; `ltl_sep3_v1`)
| Field | Value |
| --- | --- |
| Objective | “拉大” C0003 的最短路径：把当前 `top1_med` 的 peakiness gate 替换为 separation gate（`conf_metric=top3_bottom3_gap_norm`），降低误回退（fallback）并在 val402 上选择可泛化的 winner，准备冲 test402 的 +2%。 |
| Baseline | `uniform` |
| Model | Same P0 head-only protocol as E0224 (frozen CLIP caches + `temporal_conv` head). |
| Weights | CLIP caches. |
| Code path | `avs/experiments/ave_p0_sweep.py`, `scripts/e0332_ave_p0_sweep_official_val_ltl_sep3_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_sep3_v1`, `SEEDS`, `TRAIN_DEVICE`, `AUDIO_DEVICE` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json`, plus per-candidate `metrics.json` |
| Checks | Winner should reduce `fallback_used_frac` vs the current best (E0223/E0224: ≈0.751) without regressing Δ on val402. |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0332_ave_p0_sweep_official_val_ltl_sep3_v1.sh` |
| Full cmd | `SEEDS=0,1,2 TRAIN_DEVICE=cuda:9 bash scripts/e0332_ave_p0_sweep_official_val_ltl_sep3_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0332_*` |
| Artifacts | `runs/E0332_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0332_smoke_sep3_val402_20260205-194349/sweep_summary.json`. Full (val402; SEEDS=0..2): `runs/E0332_full_sep3_val402_20260205-194536/sweep_summary.json` (best=`ltlsep3_thr0p66_shift0`, anchored=0.7446 vs uniform=0.7387, Δ=+0.00590, p≈0.214; fallback_used_frac≈0.317). Conclusion: reduces fallback but regresses Δ vs E0223/E0224; do not promote to full test402. |


### E0333: Best-to-test reproduction on test402 (sep3 winner → attempt to prove C0003)
| Field | Value |
| --- | --- |
| Objective | 用 E0332 的 winner 在 official AVE test402 上复现，目标证明 C0003：`anchored_top2 - uniform ≥ +0.02` 且 paired `p<0.05`（SEEDS=0..9）。 |
| Baseline | `uniform` |
| Model | Same P0 head-only protocol as E0224 (frozen CLIP caches + `temporal_conv` head). |
| Weights | CLIP caches. |
| Code path | `avs/experiments/ave_p0.py`, `scripts/e0333_ave_p0_best_to_test_official_ltl_sep3_v1.sh` |
| Params | `BEST_CONFIG_JSON=runs/E0332_*/best_config.json`, `EVENTNESS=av_clipdiff_mlp`, `SEEDS`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (must include `paired_ttest` + `debug_eval` for fallback stats) |
| Smoke cmd | `SEEDS=0,1,2 TRAIN_DEVICE=cuda:9 bash scripts/e0333_ave_p0_best_to_test_official_ltl_sep3_v1.sh` |
| Full cmd | `SEEDS=0,1,2,3,4,5,6,7,8,9 TRAIN_DEVICE=cuda:9 bash scripts/e0333_ave_p0_best_to_test_official_ltl_sep3_v1.sh` |
| Smoke | [x] |
| Full | [ ] |
| Logs | `runs/E0333_*` |
| Artifacts | `runs/E0333_*/metrics.json` |
| Results | Quick test402 (SEEDS=0..2): `runs/E0333_quick_sep3_test402_20260205-194536/metrics.json` (uniform=0.7070, anchored=0.7125, Δ=+0.00547, p≈0.165; fallback_used_frac≈0.348). Full (SEEDS=0..9) skipped due to non-competitive quick. Note: an auto-gated full job failed due to a Python `-c` syntax error (`.rd_queue_sep3/logs/J20260205-114831-361f__e0333-full-sep3-test402-auto.log`). |


### E0334: Stage-1 sweep on val402 (learned rescue gate; `ltl_top1med_gate_lr_v1`)
| Field | Value |
| --- | --- |
| Objective | “拉大” C0003：在保持严格 `top1_med` base gate 的同时，引入可部署的 learned rescue gate（`anchor_gate_method=lr_top1hit_v1`）来减少 fallback，但避免 sep3 那类“用得更多但更伤”的失败模式。 |
| Baseline | `uniform` |
| Model | Same P0 head-only protocol as E0224 (frozen CLIP caches + `temporal_conv` head). |
| Weights | CLIP caches. |
| Code path | `avs/experiments/ave_p0.py` (gate), `avs/experiments/ave_p0_sweep.py` (`candidate_set=ltl_top1med_gate_lr_v1`), `scripts/e0334_ave_p0_sweep_official_val_ltl_top1med_gate_lr_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_gate_lr_v1`, `SEEDS`, `TRAIN_DEVICE`, `AUDIO_DEVICE` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json`, plus per-candidate `metrics.json` |
| Checks | Winner should lower fallback vs E0224 (~0.751) while staying competitive on val402; only then promote to quick test402. |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0334_ave_p0_sweep_official_val_ltl_top1med_gate_lr_v1.sh` |
| Full cmd | `SEEDS=0,1,2 TRAIN_DEVICE=cuda:9 bash scripts/e0334_ave_p0_sweep_official_val_ltl_top1med_gate_lr_v1.sh` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0334_*` |
| Artifacts | `runs/E0334_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Full (val402; SEEDS=0..2): `runs/E0334_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_gate_lr_v1_20260205-205327/sweep_summary.json` (best=`ltltop1med_gate0p8_shift1`, Δ≈+0.00964, p≈0.0331; `gate_rescued_frac≈0.000`; fallback≈0.7606 on the winner). Prior full run: `runs/E0334_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_gate_lr_v1_20260205-204236/sweep_summary.json` (best=`ltltop1med_gate0p7_shift1`, Δ≈+0.00914, p≈0.309). Conclusion: learned rescue gate does not rescue any clips and does not improve Δ; stop. |


### E0335: Best-to-test reproduction on test402 (gate-lr winner → attempt to prove C0003)
| Field | Value |
| --- | --- |
| Objective | 用 E0334 的 winner 在 official AVE test402 上复现，目标证明 C0003：`anchored_top2 - uniform ≥ +0.02` 且 paired `p<0.05`（SEEDS=0..9）。 |
| Baseline | `uniform` |
| Model | Same P0 head-only protocol as E0224 (frozen CLIP caches + `temporal_conv` head). |
| Weights | CLIP caches. |
| Code path | `avs/experiments/ave_p0.py`, `scripts/e0335_ave_p0_best_to_test_official_ltl_top1med_gate_lr_v1.sh` |
| Params | `BEST_CONFIG_JSON=runs/E0334_*/best_config.json`, `EVENTNESS=av_clipdiff_mlp`, `SEEDS`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (must include `paired_ttest` + `debug_eval` for fallback stats) |
| Smoke cmd | `SEEDS=0,1,2 TRAIN_DEVICE=cuda:9 bash scripts/e0335_ave_p0_best_to_test_official_ltl_top1med_gate_lr_v1.sh` |
| Full cmd | `SEEDS=0,1,2,3,4,5,6,7,8,9 TRAIN_DEVICE=cuda:9 bash scripts/e0335_ave_p0_best_to_test_official_ltl_top1med_gate_lr_v1.sh` |
| Smoke | [x] |
| Full | [ ] |
| Logs | `runs/E0335_*` |
| Artifacts | `runs/E0335_*/metrics.json` |
| Results | Quick test402 (SEEDS=0..2): `runs/E0335_quick_gate0p6_shift0_test402_20260205-204831/metrics.json` (uniform=0.7070, anchored=0.7179, Δ=+0.01086, p≈0.1165; fallback≈0.7512; `gate_rescued_frac≈0.000`). Full (SEEDS=0..9) skipped due to non-competitive quick + no fallback reduction. |


### E0336: Stage-1 sweep on val402 (cheap-visual fallback plan; `ltl_top1med_visfb_v1`)
| Field | Value |
| --- | --- |
| Objective | “拉大” C0003：保持严格 `top1_med` base gate 不变；当音频 gate 触发 fallback 时，不再回到 uniform，而是用 cheap-visual anchors（CLIPdiff / framediff）生成同预算的 anchored plan，尝试把 “fallback clips” 的 Δ 从 0 拉到正。 |
| Baseline | `uniform` |
| Model | Same P0 head-only protocol as E0224 (frozen CLIP caches + `temporal_conv` head). |
| Weights | CLIP caches. |
| Code path | `avs/experiments/ave_p0.py` (fallback plan), `avs/experiments/ave_p0_sweep.py` (`candidate_set=ltl_top1med_visfb_v1`), `scripts/e0336_ave_p0_sweep_official_val_ltl_top1med_visfb_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_visfb_v1`, `SEEDS`, `TRAIN_DEVICE`, `AUDIO_DEVICE` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json`, plus per-candidate `metrics.json` |
| Checks | Winner should improve Δ vs E0223/E0224 without increasing harmful far/2-high buckets; only then promote to quick test402. |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0336_ave_p0_sweep_official_val_ltl_top1med_visfb_v1.sh` |
| Full cmd | `SEEDS=0,1,2 TRAIN_DEVICE=cuda:9 bash scripts/e0336_ave_p0_sweep_official_val_ltl_top1med_visfb_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0336_*` |
| Artifacts | `runs/E0336_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0336_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_visfb_v1_20260205-212443/sweep_summary.json` (best=`ltltop1med_clipdifffb_shift1`, Δ≈+0.00312, p≈0.50). Full (val402; SEEDS=0..2): `runs/E0336_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_visfb_v1_20260205-212526/sweep_summary.json` (best remains uniform fallback baseline `ltltop1med_uniformfb_shift1`, Δ≈+0.00964, p≈0.0331; both visfb variants regress: framediff Δ≈-0.00648, clipdiff Δ≈-0.00756). Conclusion: naive cheap-visual fallback plans are harmful; stop (skip E0337). |


### E0337: Best-to-test reproduction on test402 (visfb winner → attempt to prove C0003)
| Field | Value |
| --- | --- |
| Objective | 用 E0336 的 winner 在 official AVE test402 上复现，目标证明 C0003：`anchored_top2 - uniform ≥ +0.02` 且 paired `p<0.05`（SEEDS=0..9）。 |
| Baseline | `uniform` |
| Model | Same P0 head-only protocol as E0224 (frozen CLIP caches + `temporal_conv` head). |
| Weights | CLIP caches. |
| Code path | `avs/experiments/ave_p0.py`, `scripts/e0337_ave_p0_best_to_test_official_ltl_top1med_visfb_v1.sh` |
| Params | `BEST_CONFIG_JSON=runs/E0336_*/best_config.json`, `EVENTNESS=av_clipdiff_mlp`, `SEEDS`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (must include `paired_ttest` + `debug_eval` for fallback stats) |
| Smoke cmd | `SEEDS=0,1,2 TRAIN_DEVICE=cuda:9 bash scripts/e0337_ave_p0_best_to_test_official_ltl_top1med_visfb_v1.sh` |
| Full cmd | `SEEDS=0,1,2,3,4,5,6,7,8,9 TRAIN_DEVICE=cuda:9 bash scripts/e0337_ave_p0_best_to_test_official_ltl_top1med_visfb_v1.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0337_*` |
| Artifacts | `runs/E0337_*/metrics.json` |
| Results | Skipped: E0336 regresses on val402; do not spend test402 budget. |


### E0338: Stage-1 sweep on val402 (gated cheap-visual fallback; `ltl_top1med_visfb_gated_v1`)
| Field | Value |
| --- | --- |
| Objective | Follow-up to E0336: keep strict `top1_med` base gate; only apply cheap-visual fallback anchors when the visual score sequence is sufficiently “peaky” (visual confidence gate), aiming to avoid the “random fallback anchors” harm while still helping a subset of fallback clips. |
| Baseline | `uniform` |
| Model | Same P0 head-only protocol as E0224 (frozen CLIP caches + `temporal_conv` head). |
| Weights | CLIP caches. |
| Code path | `avs/experiments/ave_p0.py` (visual fallback confidence gate), `avs/experiments/ave_p0_sweep.py` (`candidate_set=ltl_top1med_visfb_gated_v1`), `scripts/e0338_ave_p0_sweep_official_val_ltl_top1med_visfb_gated_v1.sh` |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_visfb_gated_v1`, `SEEDS`, `TRAIN_DEVICE`, `AUDIO_DEVICE` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json`, plus per-candidate `metrics.json` |
| Checks | Any gated visfb variant must beat the uniform-fallback baseline on val402; only then promote to quick test402. |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0338_ave_p0_sweep_official_val_ltl_top1med_visfb_gated_v1.sh` |
| Full cmd | `SEEDS=0,1,2 TRAIN_DEVICE=cuda:9 bash scripts/e0338_ave_p0_sweep_official_val_ltl_top1med_visfb_gated_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0338_*` |
| Artifacts | `runs/E0338_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke (train64/val32; SEEDS=0,1; EPOCHS=1): `runs/E0338_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_visfb_gated_v1_20260205-220203/sweep_summary.json` (best=`ltltop1med_clipdifffb_vc0p1_shift1`, Δ≈+0.00156, p≈0.50). Full (val402; SEEDS=0..2): `runs/E0338_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_visfb_gated_v1_20260205-220312/sweep_summary.json` (best remains uniform fallback baseline `ltltop1med_uniformfb_shift1`, Δ≈+0.00964, p≈0.0331; all gated visfb variants are negative). Conclusion: gating does not rescue visfb; stop. |


### E0339: Best-to-test reproduction on test402 (gated visfb winner → attempt to prove C0003)
| Field | Value |
| --- | --- |
| Objective | 用 E0338 的 winner 在 official AVE test402 上复现，目标证明 C0003：`anchored_top2 - uniform ≥ +0.02` 且 paired `p<0.05`（SEEDS=0..9）。 |
| Baseline | `uniform` |
| Model | Same P0 head-only protocol as E0224 (frozen CLIP caches + `temporal_conv` head). |
| Weights | CLIP caches. |
| Code path | `avs/experiments/ave_p0.py`, `scripts/e0339_ave_p0_best_to_test_official_ltl_top1med_visfb_gated_v1.sh` |
| Params | `BEST_CONFIG_JSON=runs/E0338_*/best_config.json`, `EVENTNESS=av_clipdiff_mlp`, `SEEDS`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (must include `paired_ttest` + `debug_eval` for fallback stats) |
| Smoke cmd | `SEEDS=0,1,2 TRAIN_DEVICE=cuda:9 bash scripts/e0339_ave_p0_best_to_test_official_ltl_top1med_visfb_gated_v1.sh` |
| Full cmd | `SEEDS=0,1,2,3,4,5,6,7,8,9 TRAIN_DEVICE=cuda:9 bash scripts/e0339_ave_p0_best_to_test_official_ltl_top1med_visfb_gated_v1.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0339_*` |
| Artifacts | `runs/E0339_*/metrics.json` |
| Results | Skipped: E0338 does not beat the uniform-fallback baseline on val402; do not spend test402 budget. |


### E0202: Evidence Alignment (Cov@τ) vs accuracy correlation report
| Field | Value |
| --- | --- |
| Objective | Measure Evidence Alignment (Cov@τ) and correlate it with anchored gains to diagnose failure cases. |
| Baseline | N/A (analysis-only) |
| Model | N/A |
| Weights | N/A |
| Code path | `avs/experiments/evidence_alignment_report.py` |
| Params | `TAU_GRID`, `DELTA_S`, `ANCHOR_SOURCE`, `SPLIT` |
| Metrics (must save) | `evidence_alignment.json` (Cov@τ table + Pearson/Spearman correlations) |
| Checks | Correlation magnitude is reported and stable across splits; top-k worst cases are listed. |
| VRAM | CPU |
| Time/epoch | N/A |
| Total time | minutes |
| Single-GPU script | N/A |
| Multi-GPU script | N/A |
| Smoke cmd | `python -m avs.smoke evidence_windows` |
| Full cmd | `python -m avs.experiments.evidence_alignment_report --in-metrics runs/E0012_ave_p0_best_to_test_official_20260203-145743/metrics.json --meta-dir data/AVE/meta` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0202_*` |
| Artifacts | `runs/E0202_*/evidence_alignment.json` |
| Results | Test402 (energy_v2): `runs/E0202_evidence_alignment_energy_v2_test_20260203-194355/evidence_alignment.json` (Cov@τ mean≈0.059 for τ∈{0.3,0.5,0.7}; corr(Δacc,Cov) pearson≈0.080, spearman≈-0.003). Test402 (av_clipdiff_mlp / E0224): `runs/E0202_evidence_alignment_av_clipdiff_mlp_test402_20260205-233330/evidence_alignment.json` (Cov@τ mean≈0.059; corr pearson≈0.0127, spearman≈0.0299). Local rerun (energy / E0003): `runs/E0202_evidence_alignment_energy_test402_20260209-061145/evidence_alignment.json` (Cov@τ mean≈0.0593; corr pearson≈0.0625, spearman≈-0.0314). |


### E0203: Degradation suite (shift/noise/silence × α) on AVE
| Field | Value |
| --- | --- |
| Objective | Stage-1 robustness: run anchor-quality degradation heatmaps under `{shift_s, snr_db, silence_ratio}`. |
| Baseline | `energy` (audio-only) |
| Model | Stage-1 only (no downstream training) |
| Weights | N/A |
| Code path | `avs/experiments/degradation_suite.py` |
| Params | `EVENTNESS_METHOD ∈ {energy, energy_delta, energy_stride_max, energy_autoshift_clipdiff, energy_autoshift_clipdiff_pos, av_fused, av_fused_prod, av_fused_clipdiff, av_fused_clipdiff_prod, moe_energy_clipdiff, vision_clipdiff, av_clipdiff_mlp, av_clipdiff_framediff_mlp, ast, ast_lr, ast_emb_lr, ast_evt_mlp, ast_mlp_cls, ast_mlp_cls_target, panns, audio_basic_lr, audio_basic_mlp, audio_basic_tcn, audio_fbank_mlp, audio_fbank_tcn, audio_basic_mlp_cls, audio_basic_mlp_cls_target}`, `AST_PRETRAINED`, `AUDIO_DEVICE`, grids: `SHIFT_GRID`, `SNR_GRID`, `SILENCE_GRID`, plus `K` and dilation `Δ`. |
| Metrics (must save) | `degradation_suite.json` (rows with `recall_by_delta` + fallback stats) |
| Checks | `av_fused` degrades less under silence/noise; trends are monotonic with corruption severity. |
| VRAM | TBD |
| Time/epoch | TBD |
| Total time | TBD |
| Single-GPU script | `EVENTNESS=energy bash scripts/e0203_degradation_suite_official.sh` |
| Multi-GPU script | Run different `EVENTNESS` on different GPUs/CPUs. |
| Smoke cmd | `python -m avs.smoke ltl_degradation_suite_toy` |
| Full cmd | `EVENTNESS=energy bash scripts/e0203_degradation_suite_official.sh` (then rerun with `EVENTNESS=energy_stride_max` and `EVENTNESS=av_fused`) |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0203_*` |
| Artifacts | `runs/E0203_*/degradation_suite.json` |
| Results | Full (test402; grid=3×3×2=18): energy `runs/E0203_full_energy_20260203-210331/degradation_suite.json` (mean Recall@K,Δ0≈0.223; Δ2≈0.640). energy_stride_max `runs/E0203_full_energy_stride_max_20260203-210414/degradation_suite.json` (mean Δ0≈0.223; Δ2≈0.634). av_fused `runs/E0203_full_av_fused_20260203-210458/degradation_suite.json` (mean Δ0≈0.207; Δ2≈0.708; improves under larger Δ but hurts strict Δ0). Learned audio eventness runner: `runs/E0203_degradation_audio_basic_lr_20260204-012618/degradation_suite.json`. Deployable multimodal Stage-1 (`av_clipdiff_mlp`): `runs/E0203_degradation_av_clipdiff_mlp_20260204-215831/degradation_suite.json` (mean Δ0≈0.212; Δ2≈0.624). Local rerun (energy / caches from E0003): `runs/E0203_degradation_energy_20260209-061156/degradation_suite.json` (mean Δ0≈0.224; Δ2≈0.642). |


### E0340: Build official mid-res caches (112/160/192/208/224/320/352) for P0116
| Field | Value |
| --- | --- |
| Objective | Build a CLIP feature cache that includes mid resolutions `{192,208}` and a cheaper high `{320}` so the band-budget planner can preserve more context under 2-high without exceeding the uniform token budget. |
| Baseline | N/A |
| Model | CLIP ViT-B/16 vision encoder (feature caching only) |
| Weights | HF: CLIP (`--vision-pretrained`) |
| Code path | `avs/pipeline/ave_p0_end2end.py`, `scripts/e0340_ave_cache_official_midres.sh` |
| Params | `CACHES_DIR`, `CACHE_RESOLUTIONS`, `CACHE_NUM_WORKERS`, `CACHE_DEVICES`, `PROCESSED_DIR`, `RAW_VIDEOS_DIR`, `LIMIT_TRAIN/LIMIT_VAL/LIMIT_TEST` |
| Metrics (must save) | `runs/E0340_*/cache_{val,test}/cache_build.json` + per-worker logs; caches under `${CACHES_DIR}/<clip_id>.npz` |
| Checks | The caches contain the required resolutions for all (or nearly all) `download_ok_*_official.txt` ids; E0341/E0342/E0343 run without missing-resolution crashes. |
| VRAM | ~2–4GB per cache worker GPU |
| Time/epoch | N/A |
| Total time | TBD (depends on GPU count + disk) |
| Single-GPU script | `CACHE_NUM_WORKERS=1 CACHE_DEVICES=cuda:0 bash scripts/e0340_ave_cache_official_midres.sh` |
| Multi-GPU script | `CACHE_NUM_WORKERS=10 CACHE_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7,cuda:8,cuda:9 bash scripts/e0340_ave_cache_official_midres.sh` |
| Smoke cmd | `CACHES_DIR=runs/E0340_caches_smoke CACHE_RESOLUTIONS=112,160,192,208,224,320,352 LIMIT_TRAIN=8 LIMIT_VAL=4 LIMIT_TEST=4 CACHE_NUM_WORKERS=1 CACHE_DEVICES=cuda:0 bash scripts/e0340_ave_cache_official_midres.sh` |
| Full cmd | `bash scripts/e0340_ave_cache_official_midres.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0340_*` |
| Artifacts | `${CACHES_DIR}/*.{npz,json}` |
| Results | Smoke: `runs/E0340_cache_smoke_20260205-233035/cache_val/cache_only.json`, `runs/E0340_cache_smoke_20260205-233035/cache_test/cache_only.json`. Full: `runs/E0340_cache_official_midres_20260206-000432/cache_val/cache_only.json` (train=3312, val=401, union=3703) + `runs/E0340_cache_official_midres_20260206-000432/cache_test/cache_only.json` (train=3312, test=402, union=3706). Final caches under `runs/REAL_AVE_OFFICIAL_20260201-124535/caches_112_160_192_208_224_320_352` (unique clip ids=4097). |


### E0341: Val402 sweep (candidate_set=ltl_top1med_band_midres_v1) → select best config
| Field | Value |
| --- | --- |
| Objective | Compare an internal baseline (E0224 winner) vs mid-res band-budget variants and select the best config on official val402. |
| Baseline | `ltltop1med_thr0p6_shift1_base_exact352` (internal), plus the historical E0224 reference for context |
| Model | `openai/clip-vit-base-patch16` (cached features) + `TemporalConvHead`; Stage-1 = `EVENTNESS=av_clipdiff_mlp` |
| Weights | HF: CLIP (cached); (no AST required) |
| Code path | `avs/experiments/ave_p0_sweep.py` (`candidate_set=ltl_top1med_band_midres_v1`), `scripts/e0341_ave_p0_sweep_official_val_ltl_top1med_band_midres_v1.sh` |
| Params | `SEEDS`, `EPOCHS`, `LIMIT_TRAIN/LIMIT_EVAL`, `TRAIN_DEVICE`, `AUDIO_DEVICE`, `CACHES_DIR`, `EVENTNESS` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Winner is competitive vs the internal baseline; if competitive, promote to E0342 quick test402 (SEEDS=0..2). |
| VRAM | Head training only (small); Stage-1 is CPU-friendly |
| Time/epoch | ~minutes |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0341_ave_p0_sweep_official_val_ltl_top1med_band_midres_v1.sh` |
| Multi-GPU script | Run multiple sweeps with different `SEEDS` in parallel if needed. |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0341_ave_p0_sweep_official_val_ltl_top1med_band_midres_v1.sh` |
| Full cmd | `SEEDS=0,1,2 bash scripts/e0341_ave_p0_sweep_official_val_ltl_top1med_band_midres_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0341_*` |
| Artifacts | `runs/E0341_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke: `runs/E0341_smoke_val_midres_20260205-233128/sweep_summary.json`. Full (val402; SEEDS=0..2): `runs/E0341_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_band_midres_v1_20260206-004701/sweep_summary.json` (winner remains baseline `ltltop1med_thr0p6_shift1_base_exact352`, Δ≈+0.00964, p≈0.0331; midres band variants regress). |


### E0342: Quick test402 reproduction (SEEDS=0..2) for the E0341 winner
| Field | Value |
| --- | --- |
| Objective | Validate transfer on official test402 with minimal budget: only promote to full SEEDS=0..9 if competitive. |
| Baseline | Uniform under the selected config; compare Δ vs the historical baseline (E0224) on the same seeds when needed. |
| Model | Same as E0341 winner |
| Weights | Same as E0341 |
| Code path | `scripts/e0342_ave_p0_best_to_test_quick_official_ltl_top1med_band_midres_v1.sh`, `avs/experiments/ave_p0_sweep.py` (`run`) |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0341_*/best_config.json`), `SEEDS=0,1,2`, `CACHES_DIR`, `EVENTNESS` |
| Metrics (must save) | `metrics.json` (+ optional `diagnose.json` via E0344 helper) |
| Checks | Δ is competitive vs E0224 on the same seeds; diagnose shows improved 2-high bucket (not strongly negative). |
| VRAM | Head training only |
| Time/epoch | ~minutes |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0342_ave_p0_best_to_test_quick_official_ltl_top1med_band_midres_v1.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0342_ave_p0_best_to_test_quick_official_ltl_top1med_band_midres_v1.sh` |
| Full cmd | `bash scripts/e0342_ave_p0_best_to_test_quick_official_ltl_top1med_band_midres_v1.sh` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0342_*` |
| Artifacts | `runs/E0342_*/metrics.json` |
| Results | Winner quick test402 (SEEDS=0..2): `runs/E0342_quick_test402_av_clipdiff_mlp_20260206-005041/metrics.json` (Δ≈+0.01899, p≈0.0429). Follow-ups (still regress vs the baseline quick Δ): midres band dist `runs/E0342_quick_test402_av_clipdiff_mlp_midres320_band_dist_20260206-005345/metrics.json` (Δ≈+0.01003), midres band mixed `runs/E0342_quick_test402_av_clipdiff_mlp_midres320_band_mixed_20260206-005528/metrics.json` (Δ≈+0.01003), midres band bridge `runs/E0342_quick_test402_av_clipdiff_mlp_midres320_band_bridge_20260206-005718/metrics.json` (Δ≈+0.00655), max_high_anchors=1 `runs/E0345_quick_test402_av_clipdiff_mlp_maxhigh1_20260206-010239/metrics.json` (Δ≈+0.01285). Conclusion: midres band-budget does not transfer; do not promote to full. |


### E0343: Full test402 reproduction (SEEDS=0..9) for the E0341 winner → attempt to prove C0003
| Field | Value |
| --- | --- |
| Objective | Spend the full official test402 budget (SEEDS=0..9) to attempt to prove C0003 (Δ≥+0.02, p<0.05). |
| Baseline | `uniform` |
| Model | Same as E0341 winner |
| Weights | Same as E0341 |
| Code path | `scripts/e0343_ave_p0_best_to_test_full_official_ltl_top1med_band_midres_v1.sh`, `avs/experiments/ave_p0_sweep.py` (`run`) |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0341_*/best_config.json`), `SEEDS=0..9`, `CACHES_DIR`, `EVENTNESS` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json` via E0344 helper for failure analysis) |
| Checks | If Δ≥+0.02 and p<0.05, mark C0003 proven; otherwise update `docs/plan.md` C0003 evidence + failure diagnosis. |
| VRAM | Head training only |
| Time/epoch | ~minutes |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0343_ave_p0_best_to_test_full_official_ltl_top1med_band_midres_v1.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `SEEDS=0,1 EPOCHS=1 LIMIT_TRAIN=64 LIMIT_EVAL=32 bash scripts/e0343_ave_p0_best_to_test_full_official_ltl_top1med_band_midres_v1.sh` |
| Full cmd | `bash scripts/e0343_ave_p0_best_to_test_full_official_ltl_top1med_band_midres_v1.sh` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0343_*` |
| Artifacts | `runs/E0343_*/metrics.json` |
| Results | Full test402 (SEEDS=0..9): `runs/E0343_full_test402_av_clipdiff_mlp_20260206-005134/metrics.json` (anchored=0.72383 vs uniform=0.70858, Δ=+0.01525, p≈0.00390). Does not prove C0003 (+2%). Diagnosis: `runs/E0344_ave_p0_diagnose_20260206-005232/diagnose.json` (far-anchor dist∈{2..5} and 2-high buckets remain strongly negative). |


### E0344: Diagnostic helper (ave_p0_diagnose) for a given metrics.json
| Field | Value |
| --- | --- |
| Objective | Produce a compact bucketed diagnosis (fallback, 2-high, anchor distance) to explain why Δ improved or regressed. |
| Baseline | N/A (analysis-only) |
| Model | N/A |
| Weights | N/A |
| Code path | `avs/experiments/ave_p0_diagnose.py`, `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `IN_METRICS`, `META_DIR` |
| Metrics (must save) | `diagnose.json` |
| Checks | `anchor_plan_stats.delta_by_high_count` and `delta_by_anchor_dist` identify whether far-2-high harm is reduced. |
| VRAM | CPU |
| Time/epoch | N/A |
| Total time | minutes |
| Single-GPU script | `IN_METRICS=runs/E0342_*/metrics.json bash scripts/e0344_ave_p0_diagnose.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `IN_METRICS=runs/E0224_ave_p0_best_to_test_official_av_clipdiff_mlp_20260204-135547/metrics.json bash scripts/e0344_ave_p0_diagnose.sh` |
| Full cmd | `IN_METRICS=runs/E0343_*/metrics.json bash scripts/e0344_ave_p0_diagnose.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0344_*` |
| Artifacts | `runs/E0344_*/diagnose.json` |
| Results | Smoke: `runs/E0344_smoke_diag_20260205-233225/diagnose.json`. Quick test diagnosis: `runs/E0344_ave_p0_diagnose_20260206-005134/diagnose.json`. Full test diagnosis: `runs/E0344_ave_p0_diagnose_20260206-005232/diagnose.json` shows the persistent failure mode (fallback≈0.751; `delta_by_high_count[2].mean_delta≈-0.04` with n≈30; `delta_by_anchor_dist[2..5]` strongly negative). |


### E0346: Val402 sweep — EVENTNESS=av_clap_clip_agree under ltl_top1med_norm_v1
| Field | Value |
| --- | --- |
| Objective | Try a bolder Stage-1 signal (CLAP audio-text × CLIP image-text semantic agreement) to suppress off-screen audio peaks and improve anchor reliability; select the best deployable config on val402. |
| Baseline | `uniform` (primary); compare against prior best E0224 (Δ=+0.01525 on full test402) for promotion decisions. |
| Model | CLIP ViT-B/16 cached vision embeddings + temporal head (same as other AVE-P0 runs). |
| Weights | N/A (head trained per seed) |
| Code path | `scripts/e0346_ave_p0_sweep_official_val_ltl_top1med_norm_v1_clapagree.sh`, `avs/audio/clap_probe.py`, `avs/vision/clip_text.py`, `avs/experiments/ave_p0.py`, `avs/experiments/ave_p0_sweep.py` |
| Params | `EVENTNESS=av_clap_clip_agree`, `CANDIDATE_SET=ltl_top1med_norm_v1`, `SEEDS=0..2`, `AUDIO_DEVICE` (CLAP), `TRAIN_DEVICE` (head) |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Promote only if val402 is competitive and quick test402 (E0347) beats the baseline quick Δ (≈+0.01899). |
| VRAM | CLAP model on `AUDIO_DEVICE` (large); head training is small. |
| Time/epoch | ~minutes (head); Stage-1 scoring dominates. |
| Total time | ~tens of minutes to hours (depends on CLAP speed/device). |
| Single-GPU script | `bash scripts/e0346_ave_p0_sweep_official_val_ltl_top1med_norm_v1_clapagree.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0346_ave_p0_sweep_official_val_ltl_top1med_norm_v1_clapagree.sh` |
| Full cmd | `AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0346_ave_p0_sweep_official_val_ltl_top1med_norm_v1_clapagree.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0346_*` |
| Artifacts | `runs/E0346_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke: `runs/E0346_ave_p0_sweep_official_val_av_clap_clip_agree_ltl_top1med_norm_v1_20260206-020428/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1). Full val402 (SEEDS=0..2): `runs/E0346_ave_p0_sweep_official_val_av_clap_clip_agree_ltl_top1med_norm_v1_20260206-020740/sweep_summary.json` (best=`ltltop1medn_thr0p6_shift1`, Δ≈+0.00599, p≈0.373). |


### E0347: Quick test402 reproduction (SEEDS=0..2) for the E0346 winner + diagnosis
| Field | Value |
| --- | --- |
| Objective | Sanity-check transfer on test402 with a small seed set and diagnose failure buckets (fallback / far / 2-high) before spending SEEDS=0..9. |
| Baseline | `uniform` and the baseline quick Δ from E0342 (≈+0.01899 on SEEDS=0..2). |
| Model | Same as E0346 winner |
| Weights | Same as E0346 (head trained per seed) |
| Code path | `scripts/e0347_ave_p0_best_to_test_quick_official_clapagree.sh`, `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0346_*/best_config.json`), `SEEDS=0..2`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json` via E0344) |
| Checks | If Δ is competitive vs baseline quick and diagnosis shows reduced harmful buckets, promote to E0348. |
| VRAM | CLAP on `AUDIO_DEVICE`; head on `TRAIN_DEVICE`. |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0347_ave_p0_best_to_test_quick_official_clapagree.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0347_ave_p0_best_to_test_quick_official_clapagree.sh` |
| Full cmd | `AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0347_ave_p0_best_to_test_quick_official_clapagree.sh` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0347_*` |
| Artifacts | `runs/E0347_*/metrics.json` |
| Results | Quick test402 (SEEDS=0..2): `runs/E0347_quick_test402_av_clap_clip_agree_20260206-024249/metrics.json` (uniform=0.70705, anchored=0.70531, Δ≈-0.00174, p≈0.801). Diagnosis: `runs/E0344_ave_p0_diagnose_20260206-024700/diagnose.json` (fallback_used_frac≈0.149 but far anchors are harmful: dist=4 meanΔ≈-0.0478; 2-high meanΔ≈-0.0111). Decision: not promoted to E0348. |


### E0348: Full test402 reproduction (SEEDS=0..9) for the E0346 winner → attempt to prove C0003
| Field | Value |
| --- | --- |
| Objective | Spend the full official test402 evaluation budget (SEEDS=0..9) to attempt to prove C0003 (Δ≥+0.02, p<0.05). |
| Baseline | `uniform` |
| Model | Same as E0346 winner |
| Weights | Same as E0346 (head trained per seed) |
| Code path | `scripts/e0348_ave_p0_best_to_test_full_official_clapagree.sh`, `avs/experiments/ave_p0_sweep.py` (`run`) |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0346_*/best_config.json`), `SEEDS=0..9`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json` via E0344) |
| Checks | If Δ≥+0.02 and p<0.05, mark C0003 proven; otherwise update `docs/plan.md` evidence and failure diagnosis. |
| VRAM | CLAP on `AUDIO_DEVICE`; head on `TRAIN_DEVICE`. |
| Total time | ~tens of minutes to hours |
| Single-GPU script | `bash scripts/e0348_ave_p0_best_to_test_full_official_clapagree.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `SEEDS=0,1 EPOCHS=1 LIMIT_TRAIN=64 LIMIT_EVAL=32 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0348_ave_p0_best_to_test_full_official_clapagree.sh` |
| Full cmd | `AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0348_ave_p0_best_to_test_full_official_clapagree.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0348_*` |
| Artifacts | `runs/E0348_*/metrics.json` |
| Results | Skipped: not promoted after E0347 quick test (Δ≈-0.00174 on test402 SEEDS=0..2). |


### E0349: Val402 sweep — EVENTNESS=av_clap_clip_agree under ltl_top1med_k1_v1
| Field | Value |
| --- | --- |
| Objective | Salvage `av_clap_clip_agree` by removing the harmful 2-anchor/2-high regime (k=1). Run a k=1 candidate sweep on val402 to see if transfer improves before investing in new Stage-1 signals. |
| Baseline | `uniform` (primary); compare against prior best E0224/E0343 for promotion decisions. |
| Model | CLIP ViT-B/16 cached vision embeddings + temporal head (same as other AVE-P0 runs). |
| Weights | N/A (head trained per seed) |
| Code path | `scripts/e0349_ave_p0_sweep_official_val_ltl_top1med_k1_v1_clapagree.sh`, `avs/experiments/ave_p0_sweep.py` |
| Params | `EVENTNESS=av_clap_clip_agree`, `CANDIDATE_SET=ltl_top1med_k1_v1`, `SEEDS=0..2`, optional `BASE_SCORES_JSON` reuse, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Promote only if val402 is clearly positive and quick test402 (E0350) is competitive vs baseline quick. |
| VRAM | CLAP on `AUDIO_DEVICE`; head on `TRAIN_DEVICE`. |
| Total time | ~minutes to hours (depends on whether Stage-1 scores are reused). |
| Single-GPU script | `bash scripts/e0349_ave_p0_sweep_official_val_ltl_top1med_k1_v1_clapagree.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0349_ave_p0_sweep_official_val_ltl_top1med_k1_v1_clapagree.sh` |
| Full cmd | `AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0349_ave_p0_sweep_official_val_ltl_top1med_k1_v1_clapagree.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0349_*` |
| Artifacts | `runs/E0349_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke: `runs/E0349_ave_p0_sweep_official_val_av_clap_clip_agree_ltl_top1med_k1_v1_20260206-030057/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1; reused E0346 score cache). Full val402 (SEEDS=0..2): `runs/E0349_ave_p0_sweep_official_val_av_clap_clip_agree_ltl_top1med_k1_v1_20260206-030147/sweep_summary.json` (best=`ltltop1medk1_thr0p4_shift0`, Δ≈+0.00939, p≈0.125). |


### E0350: Quick test402 reproduction (SEEDS=0..2) for the E0349 winner + diagnosis
| Field | Value |
| --- | --- |
| Objective | Sanity-check transfer on test402 with a small seed set and diagnose failure buckets (fallback / far / 2-high) before spending SEEDS=0..9. |
| Baseline | `uniform` and the baseline quick Δ from E0342 (≈+0.01899 on SEEDS=0..2). |
| Model | Same as E0349 winner |
| Weights | Same as E0349 (head trained per seed) |
| Code path | `scripts/e0350_ave_p0_best_to_test_quick_official_clapagree_k1.sh`, `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0349_*/best_config.json`), `SEEDS=0..2`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json` via E0344) |
| Checks | If Δ is competitive vs baseline quick and diagnosis shows reduced harmful buckets, promote to E0351. |
| VRAM | CLAP on `AUDIO_DEVICE`; head on `TRAIN_DEVICE`. |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0350_ave_p0_best_to_test_quick_official_clapagree_k1.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0350_ave_p0_best_to_test_quick_official_clapagree_k1.sh` |
| Full cmd | `AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0350_ave_p0_best_to_test_quick_official_clapagree_k1.sh` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0350_*` |
| Artifacts | `runs/E0350_*/metrics.json` |
| Results | Quick test402 (SEEDS=0..2): `runs/E0350_quick_test402_av_clap_clip_agree_k1_20260206-030727/metrics.json` (uniform=0.70705, anchored=0.71169, Δ≈+0.00464, p≈0.00412). Diagnosis: `runs/E0344_ave_p0_diagnose_20260206-030755/diagnose.json` (fallback_used_frac≈0.930). Decision: not promoted to E0351. |


### E0351: Full test402 reproduction (SEEDS=0..9) for the E0349 winner → attempt to prove C0003
| Field | Value |
| --- | --- |
| Objective | Spend the full official test402 evaluation budget (SEEDS=0..9) to attempt to prove C0003 (Δ≥+0.02, p<0.05). |
| Baseline | `uniform` |
| Model | Same as E0349 winner |
| Weights | Same as E0349 (head trained per seed) |
| Code path | `scripts/e0351_ave_p0_best_to_test_full_official_clapagree_k1.sh`, `avs/experiments/ave_p0_sweep.py` (`run`) |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0349_*/best_config.json`), `SEEDS=0..9`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json` via E0344) |
| Checks | If Δ≥+0.02 and p<0.05, mark C0003 proven; otherwise update `docs/plan.md` evidence and failure diagnosis. |
| VRAM | CLAP on `AUDIO_DEVICE`; head on `TRAIN_DEVICE`. |
| Total time | ~tens of minutes to hours |
| Single-GPU script | `bash scripts/e0351_ave_p0_best_to_test_full_official_clapagree_k1.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `SEEDS=0,1 EPOCHS=1 LIMIT_TRAIN=64 LIMIT_EVAL=32 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0351_ave_p0_best_to_test_full_official_clapagree_k1.sh` |
| Full cmd | `AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0351_ave_p0_best_to_test_full_official_clapagree_k1.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0351_*` |
| Artifacts | `runs/E0351_*/metrics.json` |
| Results | Skipped: not promoted after E0350 quick test (fallback_used_frac≈0.930; Δ≈+0.00464 on test402 SEEDS=0..2). |


### E0352: Val402 sweep — EVENTNESS=clap_evt (CLAP audio↔text prompt eventness) under ltl_top1med_norm_v1
| Field | Value |
| --- | --- |
| Objective | Try a bold semantic Stage-1 signal (`clap_evt`: per-second CLAP audio↔class-prompt similarity) and select the best sampling config on official val402 before spending test402 budget. |
| Baseline | `uniform` (primary); compare against prior best val402 sweep winners for promotion. |
| Model | CLIP ViT-B/16 cached vision embeddings + temporal head (same AVE-P0 pipeline). |
| Weights | CLAP pretrained (Stage-1 scoring) + head trained per seed. |
| Code path | `scripts/e0352_ave_p0_sweep_official_val_ltl_top1med_norm_v1_clapevt.sh`, `avs/audio/clap_probe.py`, `avs/experiments/ave_p0.py`, `avs/experiments/ave_p0_sweep.py` |
| Params | `EVENTNESS=clap_evt`, `CANDIDATE_SET=ltl_top1med_norm_v1`, `SEEDS=0..2`, `AUDIO_DEVICE` (CLAP), `TRAIN_DEVICE` (head) |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Promote only if val402 is competitive and quick test402 (E0353) is competitive vs the baseline quick Δ (≈+0.01899). |
| VRAM | CLAP model on `AUDIO_DEVICE` (large); head training is small. |
| Total time | ~tens of minutes to hours (Stage-1 scoring dominates). |
| Single-GPU script | `bash scripts/e0352_ave_p0_sweep_official_val_ltl_top1med_norm_v1_clapevt.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0352_ave_p0_sweep_official_val_ltl_top1med_norm_v1_clapevt.sh` |
| Full cmd | `AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0352_ave_p0_sweep_official_val_ltl_top1med_norm_v1_clapevt.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0352_*` |
| Artifacts | `runs/E0352_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke: `runs/E0352_ave_p0_sweep_official_val_clap_evt_ltl_top1med_norm_v1_20260206-032925/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1). Full val402 (SEEDS=0..2): `runs/E0352_ave_p0_sweep_official_val_clap_evt_ltl_top1med_norm_v1_20260206-033347/sweep_summary.json` (best=`ltltop1medn_thr0p6_shift1`, Δ≈+0.00657, p≈0.202; weak). |


### E0353: Quick test402 reproduction (SEEDS=0..2) for the E0352 winner + diagnosis
| Field | Value |
| --- | --- |
| Objective | Sanity-check transfer on test402 with a small seed set and diagnose failure buckets (fallback / far / 2-high) before spending SEEDS=0..9. |
| Baseline | `uniform` and the baseline quick Δ from E0342 (≈+0.01899 on SEEDS=0..2). |
| Model | Same as E0352 winner |
| Weights | Same as E0352 (head trained per seed) |
| Code path | `scripts/e0353_ave_p0_best_to_test_quick_official_clapevt.sh`, `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0352_*/best_config.json`), `SEEDS=0..2`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json` via E0344) |
| Checks | If Δ is competitive vs baseline quick and diagnosis shows reduced harmful buckets, promote to E0354. |
| VRAM | CLAP on `AUDIO_DEVICE`; head on `TRAIN_DEVICE`. |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0353_ave_p0_best_to_test_quick_official_clapevt.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0353_ave_p0_best_to_test_quick_official_clapevt.sh` |
| Full cmd | `AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0353_ave_p0_best_to_test_quick_official_clapevt.sh` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0353_*` |
| Artifacts | `runs/E0353_*/metrics.json` |
| Results | Quick test402 (SEEDS=0..2): `runs/E0353_quick_test402_clap_evt_20260206-040527/metrics.json` (uniform=0.70705, anchored=0.71517, Δ≈+0.00813, p≈0.457). Diagnosis: `runs/E0344_ave_p0_diagnose_20260206-040943/diagnose.json` (fallback_used_frac≈0.478; 2-high remains net harmful). Decision: not promoted to E0354. |


### E0354: Full test402 reproduction (SEEDS=0..9) for the E0352 winner → attempt to prove C0003
| Field | Value |
| --- | --- |
| Objective | Spend the full official test402 evaluation budget (SEEDS=0..9) to attempt to prove C0003 (Δ≥+0.02, p<0.05). |
| Baseline | `uniform` |
| Model | Same as E0352 winner |
| Weights | Same as E0352 (head trained per seed) |
| Code path | `scripts/e0354_ave_p0_best_to_test_full_official_clapevt.sh`, `avs/experiments/ave_p0_sweep.py` (`run`) |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0352_*/best_config.json`), `SEEDS=0..9`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json` via E0344) |
| Checks | If Δ≥+0.02 and p<0.05, mark C0003 proven; otherwise update `docs/plan.md` evidence and failure diagnosis. |
| VRAM | CLAP on `AUDIO_DEVICE`; head on `TRAIN_DEVICE`. |
| Total time | ~tens of minutes to hours |
| Single-GPU script | `bash scripts/e0354_ave_p0_best_to_test_full_official_clapevt.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `SEEDS=0,1 EPOCHS=1 LIMIT_TRAIN=64 LIMIT_EVAL=32 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0354_ave_p0_best_to_test_full_official_clapevt.sh` |
| Full cmd | `AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0354_ave_p0_best_to_test_full_official_clapevt.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0354_*` |
| Artifacts | `runs/E0354_*/metrics.json` |
| Results | Skipped: not promoted after E0353 quick test (Δ≈+0.00813, p≈0.457 on test402 SEEDS=0..2). |


### E0355: Val402 sweep — EVENTNESS=clap_evt under ltl_top1med_k1_v1 (k=1 salvage)
| Field | Value |
| --- | --- |
| Objective | Salvage `clap_evt` by removing the harmful 2-anchor/2-high regime (k=1). Select the best k=1 config on official val402 before re-testing transfer. |
| Baseline | `uniform` (primary); compare against E0352 and prior best val402 sweep winners for promotion. |
| Model | CLIP ViT-B/16 cached vision embeddings + temporal head (same AVE-P0 pipeline). |
| Weights | Same as E0352 (CLAP pretrained + head trained per seed). |
| Code path | `scripts/e0355_ave_p0_sweep_official_val_ltl_top1med_k1_v1_clapevt.sh`, `avs/experiments/ave_p0_sweep.py` |
| Params | `EVENTNESS=clap_evt`, `CANDIDATE_SET=ltl_top1med_k1_v1`, `SEEDS=0..2`, `BASE_SCORES_JSON` (reuse recommended), `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Promote only if val402 is clearly positive and quick test402 (E0356) improves vs E0353 and is competitive vs baseline quick. |
| Total time | ~minutes (if reusing scores); otherwise Stage-1 scoring dominates. |
| Single-GPU script | `bash scripts/e0355_ave_p0_sweep_official_val_ltl_top1med_k1_v1_clapevt.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 BASE_SCORES_JSON=$(ls -t runs/E0352_*/eventness_scores.json | head -n 1) AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0355_ave_p0_sweep_official_val_ltl_top1med_k1_v1_clapevt.sh` |
| Full cmd | `BASE_SCORES_JSON=$(ls -t runs/E0352_*/eventness_scores.json | head -n 1) AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0355_ave_p0_sweep_official_val_ltl_top1med_k1_v1_clapevt.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0355_*` |
| Artifacts | `runs/E0355_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke: `runs/E0355_ave_p0_sweep_official_val_clap_evt_ltl_top1med_k1_v1_20260206-041756/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1; reused E0352 score cache). Full val402 (SEEDS=0..2): `runs/E0355_ave_p0_sweep_official_val_clap_evt_ltl_top1med_k1_v1_20260206-041855/sweep_summary.json` (best=`ltltop1medk1_thr0p6_shift1`, Δ≈+0.00391, p≈0.0315; top-Δ candidate is `thr0p4_shift0` with Δ≈+0.01064, p≈0.135). |


### E0356: Quick test402 reproduction (SEEDS=0..2) for the E0355 winner + diagnosis
| Field | Value |
| --- | --- |
| Objective | Sanity-check transfer on test402 with a small seed set and diagnose failure buckets (fallback / far / 2-high) before spending SEEDS=0..9. |
| Baseline | `uniform` and the baseline quick Δ from E0342 (≈+0.01899 on SEEDS=0..2); also compare against E0353 (same Stage-1, k=2). |
| Model | Same as E0355 winner |
| Weights | Same as E0355 (head trained per seed) |
| Code path | `scripts/e0356_ave_p0_best_to_test_quick_official_clapevt_k1.sh`, `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0355_*/best_config.json`), `SEEDS=0..2`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json` via E0344) |
| Checks | If Δ is competitive vs baseline quick and diagnosis shows reduced harmful buckets, promote to E0357. |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0356_ave_p0_best_to_test_quick_official_clapevt_k1.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0356_ave_p0_best_to_test_quick_official_clapevt_k1.sh` |
| Full cmd | `AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0356_ave_p0_best_to_test_quick_official_clapevt_k1.sh` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0356_*` |
| Artifacts | `runs/E0356_*/metrics.json` |
| Results | Quick test402 (SEEDS=0..2; best_config=`ltltop1medk1_thr0p6_shift1`): `runs/E0356_quick_test402_clap_evt_k1_20260206-042339/metrics.json` (Δ≈+0.00489, p≈0.289). Diagnosis: `runs/E0344_ave_p0_diagnose_20260206-042437/diagnose.json` (fallback_used_frac≈0.998; anchors almost never used). Extra diagnostic (top-Δ val candidate `thr0p4_shift0`): `runs/E0356_quick_test402_clap_evt_k1_thr0p4_shift0_20260206-042532/metrics.json` (Δ≈+0.01219, p≈0.183) + diagnosis `runs/E0344_ave_p0_diagnose_20260206-042617/diagnose.json` (fallback_used_frac≈0.913). Decision: not promoted to E0357. |


### E0357: Full test402 reproduction (SEEDS=0..9) for the E0355 winner → attempt to prove C0003
| Field | Value |
| --- | --- |
| Objective | Spend the full official test402 evaluation budget (SEEDS=0..9) to attempt to prove C0003 (Δ≥+0.02, p<0.05). |
| Baseline | `uniform` |
| Model | Same as E0355 winner |
| Weights | Same as E0355 (head trained per seed) |
| Code path | `scripts/e0357_ave_p0_best_to_test_full_official_clapevt_k1.sh`, `avs/experiments/ave_p0_sweep.py` (`run`) |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0355_*/best_config.json`), `SEEDS=0..9`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json` via E0344) |
| Checks | If Δ≥+0.02 and p<0.05, mark C0003 proven; otherwise update `docs/plan.md` evidence and failure diagnosis. |
| Total time | ~tens of minutes to hours |
| Single-GPU script | `bash scripts/e0357_ave_p0_best_to_test_full_official_clapevt_k1.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `SEEDS=0,1 EPOCHS=1 LIMIT_TRAIN=64 LIMIT_EVAL=32 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0357_ave_p0_best_to_test_full_official_clapevt_k1.sh` |
| Full cmd | `AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0357_ave_p0_best_to_test_full_official_clapevt_k1.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0357_*` |
| Artifacts | `runs/E0357_*/metrics.json` |
| Results | Skipped: not promoted after E0356 quick test (fallback collapses to ~1.0 under the k=1 gate). |


### E0358: Val402 sweep — EVENTNESS=clap_lr (CLAP-supervised LR calibration) under ltl_top1med_norm_v1
| Field | Value |
| --- | --- |
| Objective | Try a supervised CLAP calibration (`clap_lr`) as a deployable Stage-1 signal and select the best sampling config on official val402 before spending test402 budget. |
| Baseline | `uniform` (primary); compare against prior best val402 sweep winners for promotion. |
| Model | CLIP ViT-B/16 cached vision embeddings + temporal head (same AVE-P0 pipeline). |
| Weights | CLAP pretrained (feature extraction) + 1-layer LR trained on AVE-train (Stage-1 calibration) + head trained per seed. |
| Code path | `scripts/e0358_ave_p0_sweep_official_val_ltl_top1med_norm_v1_claplr.sh`, `avs/experiments/ave_p0_sweep.py` (score cache; trains `clap_lr`) |
| Params | `EVENTNESS=clap_lr`, `CANDIDATE_SET=ltl_top1med_norm_v1`, `SEEDS=0..2`, `AUDIO_DEVICE` (CLAP), `TRAIN_DEVICE` (head) |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Promote only if val402 is competitive and quick test402 (E0359) beats the baseline quick Δ (≈+0.01899). |
| VRAM | CLAP on `AUDIO_DEVICE` (large); head training small. |
| Total time | ~tens of minutes to hours (Stage-1 CLAP features dominate). |
| Single-GPU script | `bash scripts/e0358_ave_p0_sweep_official_val_ltl_top1med_norm_v1_claplr.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0358_ave_p0_sweep_official_val_ltl_top1med_norm_v1_claplr.sh` |
| Full cmd | `AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0358_ave_p0_sweep_official_val_ltl_top1med_norm_v1_claplr.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0358_*` |
| Artifacts | `runs/E0358_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke: `runs/E0358_ave_p0_sweep_official_val_clap_lr_ltl_top1med_norm_v1_20260206-044924/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1). Full val402 (SEEDS=0..2): `runs/E0358_ave_p0_sweep_official_val_clap_lr_ltl_top1med_norm_v1_20260206-045105/sweep_summary.json` (best=`ltltop1medn_thr0p5_shift0`, Δ≈-0.00191, p≈0.625; top3 all negative). Decision: not promoted to test402. |


### E0359: Quick test402 reproduction (SEEDS=0..2) for the E0358 winner + diagnosis
| Field | Value |
| --- | --- |
| Objective | Sanity-check transfer on test402 with a small seed set and diagnose failure buckets (fallback / far / 2-high) before spending SEEDS=0..9. |
| Baseline | `uniform` and the baseline quick Δ from E0342 (≈+0.01899 on SEEDS=0..2). |
| Model | Same as E0358 winner |
| Weights | Same as E0358 (clap_lr trained on train split; head trained per seed) |
| Code path | `scripts/e0359_ave_p0_best_to_test_quick_official_claplr.sh`, `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0358_*/best_config.json`), `SEEDS=0..2`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json` via E0344) |
| Checks | If Δ is competitive vs baseline quick and diagnosis shows reduced harmful buckets, promote to E0360. |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0359_ave_p0_best_to_test_quick_official_claplr.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0359_ave_p0_best_to_test_quick_official_claplr.sh` |
| Full cmd | `AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0359_ave_p0_best_to_test_quick_official_claplr.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0359_*` |
| Artifacts | `runs/E0359_*/metrics.json` |
| Results | Skipped: not promoted after E0358 full val402 (best is negative). |


### E0360: Full test402 reproduction (SEEDS=0..9) for the E0358 winner → attempt to prove C0003
| Field | Value |
| --- | --- |
| Objective | Spend the full official test402 evaluation budget (SEEDS=0..9) to attempt to prove C0003 (Δ≥+0.02, p<0.05). |
| Baseline | `uniform` |
| Model | Same as E0358 winner |
| Weights | Same as E0358 (clap_lr trained on train split; head trained per seed) |
| Code path | `scripts/e0360_ave_p0_best_to_test_full_official_claplr.sh`, `avs/experiments/ave_p0_sweep.py` (`run`) |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0358_*/best_config.json`), `SEEDS=0..9`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json` via E0344) |
| Checks | If Δ≥+0.02 and p<0.05, mark C0003 proven; otherwise update `docs/plan.md` evidence and failure diagnosis. |
| Total time | ~tens of minutes to hours |
| Single-GPU script | `bash scripts/e0360_ave_p0_best_to_test_full_official_claplr.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `SEEDS=0,1 EPOCHS=1 LIMIT_TRAIN=64 LIMIT_EVAL=32 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0360_ave_p0_best_to_test_full_official_claplr.sh` |
| Full cmd | `AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0360_ave_p0_best_to_test_full_official_claplr.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0360_*` |
| Artifacts | `runs/E0360_*/metrics.json` |
| Results | Skipped: not promoted after E0358 full val402 (best is negative). |


### E0361: Val402 sweep — EVENTNESS=clap_mlp_cls_target (CLAP-embedding supervised multi-class) under ltl_top1med_norm_v1
| Field | Value |
| --- | --- |
| Objective | Try a supervised CLAP-embedding multi-class head (`clap_mlp_cls_target`) as a deployable Stage-1 signal and select the best sampling config on official val402 before spending test402 budget. |
| Baseline | `uniform` (primary); compare against prior best val402 sweep winners for promotion. |
| Model | CLIP ViT-B/16 cached vision embeddings + temporal head (same AVE-P0 pipeline). |
| Weights | CLAP pretrained (feature extraction) + 2-layer MLP trained on AVE-train segment labels (Stage-1 calibration) + head trained per seed. |
| Code path | `scripts/e0361_ave_p0_sweep_official_val_ltl_top1med_norm_v1_clapmlpcls_target.sh`, `avs/experiments/ave_p0_sweep.py` (score cache; trains `clap_mlp_cls_target`) |
| Params | `EVENTNESS=clap_mlp_cls_target`, `CANDIDATE_SET=ltl_top1med_norm_v1`, `SEEDS=0..2`, `AUDIO_DEVICE` (CLAP), `TRAIN_DEVICE` (head) |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Promote only if val402 is competitive and quick test402 (E0362) beats the baseline quick Δ (≈+0.01899). |
| VRAM | CLAP on `AUDIO_DEVICE` (large); head training small. |
| Total time | ~tens of minutes to hours (Stage-1 CLAP features dominate). |
| Single-GPU script | `bash scripts/e0361_ave_p0_sweep_official_val_ltl_top1med_norm_v1_clapmlpcls_target.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0361_ave_p0_sweep_official_val_ltl_top1med_norm_v1_clapmlpcls_target.sh` |
| Full cmd | `AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0361_ave_p0_sweep_official_val_ltl_top1med_norm_v1_clapmlpcls_target.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0361_*` |
| Artifacts | `runs/E0361_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke: `runs/E0361_ave_p0_sweep_official_val_clap_mlp_cls_target_ltl_top1med_norm_v1_20260206-054731/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1). Full val402 (SEEDS=0..2): `runs/E0361_ave_p0_sweep_official_val_clap_mlp_cls_target_ltl_top1med_norm_v1_20260206-054923/sweep_summary.json` (best=`ltltop1medn_thr0p7_shift0`, Δ≈+0.00158, p≈0.284; best_by_pfilter=None). Decision: not promoted to test402. |


### E0362: Quick test402 reproduction (SEEDS=0..2) for the E0361 winner + diagnosis
| Field | Value |
| --- | --- |
| Objective | Sanity-check transfer on test402 with a small seed set and diagnose failure buckets (fallback / far / 2-high) before spending SEEDS=0..9. |
| Baseline | `uniform` and the baseline quick Δ from E0342 (≈+0.01899 on SEEDS=0..2). |
| Model | Same as E0361 winner |
| Weights | Same as E0361 (clap_mlp_cls_target trained on train split; head trained per seed) |
| Code path | `scripts/e0362_ave_p0_best_to_test_quick_official_clapmlpcls_target.sh`, `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0361_*/best_config.json`), `SEEDS=0..2`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json` via E0344) |
| Checks | If Δ is competitive vs baseline quick and diagnosis shows reduced harmful buckets, promote to E0363. |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0362_ave_p0_best_to_test_quick_official_clapmlpcls_target.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0362_ave_p0_best_to_test_quick_official_clapmlpcls_target.sh` |
| Full cmd | `AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0362_ave_p0_best_to_test_quick_official_clapmlpcls_target.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0362_*` |
| Artifacts | `runs/E0362_*/metrics.json` |
| Results | Skipped: not promoted after E0361 full val402 (near-0). |


### E0363: Full test402 reproduction (SEEDS=0..9) for the E0361 winner → attempt to prove C0003
| Field | Value |
| --- | --- |
| Objective | Spend the full official test402 evaluation budget (SEEDS=0..9) to attempt to prove C0003 (Δ≥+0.02, p<0.05). |
| Baseline | `uniform` |
| Model | Same as E0361 winner |
| Weights | Same as E0361 (clap_mlp_cls_target trained on train split; head trained per seed) |
| Code path | `scripts/e0363_ave_p0_best_to_test_full_official_clapmlpcls_target.sh`, `avs/experiments/ave_p0_sweep.py` (`run`) |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0361_*/best_config.json`), `SEEDS=0..9`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json` via E0344) |
| Checks | If Δ≥+0.02 and p<0.05, mark C0003 proven; otherwise update `docs/plan.md` evidence and failure diagnosis. |
| Total time | ~tens of minutes to hours |
| Single-GPU script | `bash scripts/e0363_ave_p0_best_to_test_full_official_clapmlpcls_target.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `SEEDS=0,1 EPOCHS=1 LIMIT_TRAIN=64 LIMIT_EVAL=32 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0363_ave_p0_best_to_test_full_official_clapmlpcls_target.sh` |
| Full cmd | `AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0363_ave_p0_best_to_test_full_official_clapmlpcls_target.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0363_*` |
| Artifacts | `runs/E0363_*/metrics.json` |
| Results | Skipped: not promoted after E0361 full val402 (near-0). |


### E0364: Val402 sweep — high-set-only base allocation (`*_high`) under adaptive_v3 (ltl_top1med_keepadj_basealloc_highonly_v1)
| Field | Value |
| --- | --- |
| Objective | Test whether the far-anchor / 2-high harm under `adaptive_v3` is caused by **base-slot waste around the demoted anchor2**, by introducing `anchor_base_alloc=*_high` (base allocation uses high-set only) and selecting the best config on official val402. |
| Baseline | `uniform` (primary); compare against the current best val402 sweep winners for promotion. |
| Model | CLIP ViT-B/16 cached vision embeddings + temporal head (same AVE-P0 pipeline). |
| Weights | Same as the current best deployable Stage-1 (`EVENTNESS=av_clipdiff_mlp`); only Stage-2 planning changes. |
| Code path | `scripts/e0364_ave_p0_sweep_official_val_ltl_top1med_keepadj_basealloc_highonly_v1.sh`, `avs/sampling/plans.py` (`*_high`), `avs/experiments/ave_p0_sweep.py` (candidate set) |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_keepadj_basealloc_highonly_v1`, `SEEDS=0..2`, `TRAIN_DEVICE` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Promote only if val402 is competitive and the selected config is not a p-filter artifact. |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0364_ave_p0_sweep_official_val_ltl_top1med_keepadj_basealloc_highonly_v1.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 TRAIN_DEVICE=cuda:0 bash scripts/e0364_ave_p0_sweep_official_val_ltl_top1med_keepadj_basealloc_highonly_v1.sh` |
| Full cmd | `TRAIN_DEVICE=cuda:0 bash scripts/e0364_ave_p0_sweep_official_val_ltl_top1med_keepadj_basealloc_highonly_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0364_*` |
| Artifacts | `runs/E0364_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke: `runs/E0364_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_keepadj_basealloc_highonly_v1_20260206-064816/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1). Full val402 (SEEDS=0..2): `runs/E0364_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_keepadj_basealloc_highonly_v1_20260206-064910/sweep_summary.json` (best=`ltltop1med_keepadj_mixed_high`, Δ≈+0.00291, p≈0.167; best_by_pfilter=None). Decision: not promoted to test402. |


### E0365: Quick test402 reproduction (SEEDS=0..2) for the E0364 winner + diagnosis
| Field | Value |
| --- | --- |
| Objective | Sanity-check transfer on test402 with a small seed set and diagnose failure buckets (fallback / far / 2-high) before spending SEEDS=0..9. |
| Baseline | `uniform` and the baseline quick Δ from E0342 (≈+0.01899 on SEEDS=0..2). |
| Model | Same as E0364 winner |
| Weights | Same as E0364 (Stage-1 `av_clipdiff_mlp`; head trained per seed) |
| Code path | `scripts/e0365_ave_p0_best_to_test_quick_official_ltl_top1med_keepadj_basealloc_highonly_v1.sh`, `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0364_*/best_config.json`), `SEEDS=0..2`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json` via E0344) |
| Checks | If Δ is competitive vs baseline quick and diagnosis shows reduced harmful buckets, promote to E0366. |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0365_ave_p0_best_to_test_quick_official_ltl_top1med_keepadj_basealloc_highonly_v1.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 TRAIN_DEVICE=cuda:0 bash scripts/e0365_ave_p0_best_to_test_quick_official_ltl_top1med_keepadj_basealloc_highonly_v1.sh` |
| Full cmd | `TRAIN_DEVICE=cuda:0 bash scripts/e0365_ave_p0_best_to_test_quick_official_ltl_top1med_keepadj_basealloc_highonly_v1.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0365_*` |
| Artifacts | `runs/E0365_*/metrics.json` |
| Results | Skipped: not promoted after E0364 full val402 (near-0). |


### E0366: Full test402 reproduction (SEEDS=0..9) for the E0364 winner → attempt to prove C0003
| Field | Value |
| --- | --- |
| Objective | Spend the full official test402 evaluation budget (SEEDS=0..9) to attempt to prove C0003 (Δ≥+0.02, p<0.05). |
| Baseline | `uniform` |
| Model | Same as E0364 winner |
| Weights | Same as E0364 (Stage-1 `av_clipdiff_mlp`; head trained per seed) |
| Code path | `scripts/e0366_ave_p0_best_to_test_full_official_ltl_top1med_keepadj_basealloc_highonly_v1.sh`, `avs/experiments/ave_p0_sweep.py` (`run`) |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0364_*/best_config.json`), `SEEDS=0..9`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json` via E0344) |
| Checks | If Δ≥+0.02 and p<0.05, mark C0003 proven; otherwise update `docs/plan.md` evidence and failure diagnosis. |
| Total time | ~tens of minutes to hours |
| Single-GPU script | `bash scripts/e0366_ave_p0_best_to_test_full_official_ltl_top1med_keepadj_basealloc_highonly_v1.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `SEEDS=0,1 EPOCHS=1 LIMIT_TRAIN=64 LIMIT_EVAL=32 TRAIN_DEVICE=cuda:0 bash scripts/e0366_ave_p0_best_to_test_full_official_ltl_top1med_keepadj_basealloc_highonly_v1.sh` |
| Full cmd | `TRAIN_DEVICE=cuda:0 bash scripts/e0366_ave_p0_best_to_test_full_official_ltl_top1med_keepadj_basealloc_highonly_v1.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0366_*` |
| Artifacts | `runs/E0366_*/metrics.json` |
| Results | Skipped: not promoted after E0364 full val402 (near-0). |


### E0367: Val402 sweep — EVENTNESS=av_ast_clipdiff_mlp (AST embeddings + CLIPdiff) under ltl_top1med_norm_v1
| Field | Value |
| --- | --- |
| Objective | Try a bold but deployable Stage-1 scorer: pretrained AST per-second embeddings + cheap CLIPdiff scalar → per-second MLP (`av_ast_clipdiff_mlp`), and select the best sampling config on official val402 before spending test402 budget. |
| Baseline | `uniform` (primary); compare against prior best val402 sweep winners for promotion. |
| Model | CLIP ViT-B/16 cached vision embeddings + temporal head (same AVE-P0 pipeline). |
| Weights | AST pretrained (feature extraction) + small MLP trained on AVE-train (Stage-1) + head trained per seed. |
| Code path | `scripts/e0367_ave_p0_sweep_official_val_ltl_top1med_norm_v1_av_ast_clipdiff_mlp.sh`, `avs/experiments/ave_p0.py` (`av_ast_clipdiff_mlp`), `avs/experiments/ave_p0_sweep.py` (score cache) |
| Params | `EVENTNESS=av_ast_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_norm_v1`, `SEEDS=0..2`, `AST_PRETRAINED=1` (auto), `AUDIO_DEVICE` (AST), `TRAIN_DEVICE` (head) |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Promote only if val402 is competitive and quick test402 (E0368) beats the baseline quick Δ (≈+0.01899). |
| VRAM | AST on `AUDIO_DEVICE` (moderate); head training small. |
| Total time | ~tens of minutes to hours (AST dominates). |
| Single-GPU script | `bash scripts/e0367_ave_p0_sweep_official_val_ltl_top1med_norm_v1_av_ast_clipdiff_mlp.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 EVENTNESS=av_ast_clipdiff_mlp CANDIDATE_SET=ltl_top1med_norm_v1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0367_ave_p0_sweep_official_val_ltl_top1med_norm_v1_av_ast_clipdiff_mlp.sh` |
| Full cmd | `EVENTNESS=av_ast_clipdiff_mlp CANDIDATE_SET=ltl_top1med_norm_v1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0367_ave_p0_sweep_official_val_ltl_top1med_norm_v1_av_ast_clipdiff_mlp.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0367_*` |
| Artifacts | `runs/E0367_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke: `runs/E0367_ave_p0_sweep_official_val_av_ast_clipdiff_mlp_ltl_top1med_norm_v1_20260206-070549/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1). Full val402 (SEEDS=0..2): `runs/E0367_ave_p0_sweep_official_val_av_ast_clipdiff_mlp_ltl_top1med_norm_v1_20260206-070639/sweep_summary.json` (best=`ltltop1medn_thr0p7_shift0`, Δ≈+0.00183, p≈0.853; best_by_pfilter=None). Decision: not promoted to test402. |


### E0368: Quick test402 reproduction (SEEDS=0..2) for the E0367 winner + diagnosis
| Field | Value |
| --- | --- |
| Objective | Sanity-check transfer on test402 with a small seed set and diagnose failure buckets (fallback / far / 2-high) before spending SEEDS=0..9. |
| Baseline | `uniform` and the baseline quick Δ from E0342 (≈+0.01899 on SEEDS=0..2). |
| Model | Same as E0367 winner |
| Weights | Same as E0367 (Stage-1 trained on train split; head trained per seed) |
| Code path | `scripts/e0368_ave_p0_best_to_test_quick_official_av_ast_clipdiff_mlp.sh`, `avs/experiments/ave_p0_sweep.py` (`run`), `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0367_*/best_config.json`), `SEEDS=0..2`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json` via E0344) |
| Checks | If Δ is competitive vs baseline quick and diagnosis shows reduced harmful buckets, promote to E0369. |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0368_ave_p0_best_to_test_quick_official_av_ast_clipdiff_mlp.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 EVENTNESS=av_ast_clipdiff_mlp AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0368_ave_p0_best_to_test_quick_official_av_ast_clipdiff_mlp.sh` |
| Full cmd | `EVENTNESS=av_ast_clipdiff_mlp AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0368_ave_p0_best_to_test_quick_official_av_ast_clipdiff_mlp.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0368_*` |
| Artifacts | `runs/E0368_*/metrics.json` |
| Results | Skipped: not promoted after E0367 full val402 (near-0). |


### E0369: Full test402 reproduction (SEEDS=0..9) for the E0367 winner → attempt to prove C0003
| Field | Value |
| --- | --- |
| Objective | Spend the full official test402 evaluation budget (SEEDS=0..9) to attempt to prove C0003 (Δ≥+0.02, p<0.05). |
| Baseline | `uniform` |
| Model | Same as E0367 winner |
| Weights | Same as E0367 (Stage-1 trained on train split; head trained per seed) |
| Code path | `scripts/e0369_ave_p0_best_to_test_full_official_av_ast_clipdiff_mlp.sh`, `avs/experiments/ave_p0_sweep.py` (`run`) |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0367_*/best_config.json`), `SEEDS=0..9`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json` via E0344) |
| Checks | If Δ≥+0.02 and p<0.05, mark C0003 proven; otherwise update `docs/plan.md` evidence and failure diagnosis. |
| Total time | ~tens of minutes to hours |
| Single-GPU script | `bash scripts/e0369_ave_p0_best_to_test_full_official_av_ast_clipdiff_mlp.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `SEEDS=0,1 EPOCHS=1 LIMIT_TRAIN=64 LIMIT_EVAL=32 EVENTNESS=av_ast_clipdiff_mlp AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0369_ave_p0_best_to_test_full_official_av_ast_clipdiff_mlp.sh` |
| Full cmd | `EVENTNESS=av_ast_clipdiff_mlp AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0369_ave_p0_best_to_test_full_official_av_ast_clipdiff_mlp.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0369_*` |
| Artifacts | `runs/E0369_*/metrics.json` |
| Results | Skipped: not promoted after E0367 full val402 (near-0). |


### E0370: Quick test402 transfer triage (SEEDS=0..2) — A/V correspondence anchors (`av_ast_clipalign_bce`, E0318 selection) + diagnosis
| Field | Value |
| --- | --- |
| Objective | Because val→test transfer is unreliable, run a controlled quick test402 (SEEDS=0..2) for the `av_ast_clipalign_bce` winner selected in E0318, and diagnose whether it reduces the harmful far/2-high buckets vs the current best baseline. |
| Baseline | `uniform` and the baseline quick Δ from E0342 (≈+0.01899 on SEEDS=0..2). |
| Model | Same AVE-P0 downstream model as E0318 (`temporal_conv` head on frozen CLIP features). |
| Weights | Uses AST pretrained + learned A/V correspondence projection (trained on train split inside scoring) + head trained per seed. |
| Code path | `scripts/e0370_quick_test402_av_ast_clipalign_bce.sh`, `scripts/e0344_ave_p0_diagnose.sh`, `avs/experiments/ave_p0_sweep.py` (`run`) |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0318_*/best_config.json` via underlying script), `SEEDS=0..2`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json` via E0344) |
| Checks | If Δ is competitive vs baseline quick and diagnosis shows reduced harmful buckets, promote to E0371. |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0370_quick_test402_av_ast_clipalign_bce.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0370_quick_test402_av_ast_clipalign_bce.sh` |
| Full cmd | `SEEDS=0,1,2 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0370_quick_test402_av_ast_clipalign_bce.sh` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0370_*` |
| Artifacts | `runs/E0370_*/metrics.json` |
| Results | `runs/E0370_quick_test402_av_ast_clipalign_bce_20260206-072535/metrics.json` (anchored=0.7152 vs uniform=0.7070, Δ≈+0.00813, p≈0.365). Diagnosis: `runs/E0344_ave_p0_diagnose_20260206-073731/diagnose.json` (fallback_used_frac≈0.570; anchored deltas concentrate in fallback/0-high clips; 2-anchor regime ~0). Decision: not promoted; skip E0371. |


### E0371: Full test402 reproduction (SEEDS=0..9) — A/V correspondence anchors (`av_ast_clipalign_bce`) → attempt to prove C0003
| Field | Value |
| --- | --- |
| Objective | Spend the full official test402 evaluation budget (SEEDS=0..9) to attempt to prove C0003 (Δ≥+0.02, p<0.05) for `av_ast_clipalign_bce` if promoted by E0370. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Same as E0370 |
| Code path | `scripts/e0371_full_test402_av_ast_clipalign_bce.sh`, `avs/experiments/ave_p0_sweep.py` (`run`) |
| Params | `BEST_CONFIG_JSON` (defaults to E0318 best), `SEEDS=0..9`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json` via E0344) |
| Checks | If Δ≥+0.02 and p<0.05, mark C0003 proven; otherwise update `docs/plan.md` evidence and failure diagnosis. |
| Total time | ~tens of minutes to hours |
| Single-GPU script | `bash scripts/e0371_full_test402_av_ast_clipalign_bce.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `SEEDS=0,1 EPOCHS=1 LIMIT_TRAIN=64 LIMIT_EVAL=32 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0371_full_test402_av_ast_clipalign_bce.sh` |
| Full cmd | `SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0371_full_test402_av_ast_clipalign_bce.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0371_*` |
| Artifacts | `runs/E0371_*/metrics.json` |
| Results | Skipped: not promoted after E0370 quick test402 (Δ≈+0.00813, p≈0.365; not competitive). |


### E0372: Val402 sweep (SEEDS=0..2) — veto gate on base confidence (`lr_top1hit_all_v1`) to reduce “confident-but-wrong” anchors
| Field | Value |
| --- | --- |
| Objective | Run the standard official val402 sweep for `EVENTNESS=av_clipdiff_mlp` with a new learned clip-level veto gate (trained on all train clips), aiming to reduce harmful anchored clips and improve transfer toward proving C0003. |
| Baseline | Current best learned-anchor pipeline: E0223/E0224 (`ltltop1med_thr0p6_shift1`) and its quick Δ on test402 SEEDS=0..2 (≈+0.01899). |
| Model | AVE-P0 `temporal_conv` head on frozen CLIP cache. |
| Weights | Gate trained on train split; head trained per seed. |
| Code path | `scripts/e0372_ave_p0_sweep_official_val_ltl_top1med_gate_all_v1.sh`, `avs/experiments/ave_p0.py` (gate), `avs/experiments/ave_p0_sweep.py` (`candidate_set=ltl_top1med_gate_all_v1`) |
| Params | `EVENTNESS=av_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_gate_all_v1`, `SEEDS=0..2`, `TRAIN_DEVICE` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Only promote if the full val402 winner is clearly competitive and the gate meaningfully vetoes clips (debug `gate_vetoed=true`) without collapsing to near-uniform behavior. |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0372_ave_p0_sweep_official_val_ltl_top1med_gate_all_v1.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=200 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 TRAIN_DEVICE=cuda:0 bash scripts/e0372_ave_p0_sweep_official_val_ltl_top1med_gate_all_v1.sh` |
| Full cmd | `SEEDS=0,1,2 TRAIN_DEVICE=cuda:0 bash scripts/e0372_ave_p0_sweep_official_val_ltl_top1med_gate_all_v1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0372_*` |
| Artifacts | `runs/E0372_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke: `runs/E0372_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_gate_all_v1_20260206-080613/sweep_summary.json` (LIMIT_TRAIN=200 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1). Full val402: `runs/E0372_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_gate_all_v1_20260206-081233/sweep_summary.json` (best=`ltltop1med_gateall0p4_shift0`, Δ≈+0.00989, p≈0.00344; best_by_pfilter=None). |


### E0373: Quick test402 transfer (SEEDS=0..2) — E0372 winner + diagnosis
| Field | Value |
| --- | --- |
| Objective | Run a controlled quick test402 (SEEDS=0..2) for the E0372 winner and diagnose whether the gate reduces harmful buckets vs the current best baseline. |
| Baseline | `uniform` and E0224 quick Δ baseline (≈+0.01899 on SEEDS=0..2). |
| Model | Same as E0372 selected config |
| Weights | Same as E0372 |
| Code path | `scripts/e0373_ave_p0_best_to_test_quick_official_ltl_top1med_gate_all_v1.sh`, `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0372_*/best_config.json`), `SEEDS=0..2`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json` via E0344) |
| Checks | Promote to E0374 only if quick Δ is competitive vs baseline and diagnosis shows reduced harmful buckets without collapsing to fallback≈1. |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0373_ave_p0_best_to_test_quick_official_ltl_top1med_gate_all_v1.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 TRAIN_DEVICE=cuda:0 bash scripts/e0373_ave_p0_best_to_test_quick_official_ltl_top1med_gate_all_v1.sh` |
| Full cmd | `SEEDS=0,1,2 TRAIN_DEVICE=cuda:0 bash scripts/e0373_ave_p0_best_to_test_quick_official_ltl_top1med_gate_all_v1.sh` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `runs/E0373_*` |
| Artifacts | `runs/E0373_*/metrics.json` |
| Results | `runs/E0373_quick_test402_av_clipdiff_mlp_ltl_top1med_gate_all_v1_20260206-081728/metrics.json` (anchored=0.7179 vs uniform=0.7070, Δ≈+0.01086, p≈0.1165). Diagnosis: `runs/E0344_ave_p0_diagnose_20260206-081835/diagnose.json` (fallback_used_frac≈0.751; 2-high bucket remains harmful). Decision: not promoted; skip E0374. |


### E0374: Full test402 reproduction (SEEDS=0..9) — E0372 winner → attempt to prove C0003
| Field | Value |
| --- | --- |
| Objective | Spend the full official test402 budget (SEEDS=0..9) to attempt to prove C0003 (Δ≥+0.02, p<0.05), using the config selected in E0372 and promoted by E0373. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Same as E0372 |
| Code path | `scripts/e0374_ave_p0_best_to_test_full_official_ltl_top1med_gate_all_v1.sh`, `avs/experiments/ave_p0_sweep.py` (`run`) |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0372_*/best_config.json`), `SEEDS=0..9`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json` via E0344) |
| Checks | If Δ≥+0.02 and p<0.05, mark C0003 proven; otherwise update `docs/plan.md` with decisive failure analysis. |
| Total time | ~tens of minutes to hours |
| Single-GPU script | `bash scripts/e0374_ave_p0_best_to_test_full_official_ltl_top1med_gate_all_v1.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `SEEDS=0,1 EPOCHS=1 LIMIT_TRAIN=64 LIMIT_EVAL=32 TRAIN_DEVICE=cuda:0 bash scripts/e0374_ave_p0_best_to_test_full_official_ltl_top1med_gate_all_v1.sh` |
| Full cmd | `SEEDS=0,1,2,3,4,5,6,7,8,9 TRAIN_DEVICE=cuda:0 bash scripts/e0374_ave_p0_best_to_test_full_official_ltl_top1med_gate_all_v1.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0374_*` |
| Artifacts | `runs/E0374_*/metrics.json` |
| Results | Skipped: not promoted after E0373 quick test402 was not competitive (Δ≈+0.01086, p≈0.1165). |


### E0375: Val402 sweep (SEEDS=0..2) — strong cheap-visual control (`vision_binary_mlp`) under scale-invariant top1-med gate
| Field | Value |
| --- | --- |
| Objective | As a strong cheap-visual control, train a supervised per-second binary visual eventness model on frozen CLIP low-res features (`EVENTNESS=vision_binary_mlp`) and run the standard val402 sweep (`candidate_set=ltl_top1med_norm_v1`). If this direction is strong, it indicates the C0003 plateau is primarily due to Stage-1 anchor quality; otherwise it suggests the equal-budget triad hurts even with stronger visual proposals. |
| Baseline | Standard learned-anchor pipeline baselines (E0223/E0224). |
| Model | AVE-P0 `temporal_conv` head on frozen CLIP cache. |
| Weights | Visual eventness trained on train split (CPU); head trained per seed. |
| Code path | `scripts/e0375_ave_p0_sweep_official_val_ltl_top1med_norm_v1_vision_binary_mlp.sh`, `avs/experiments/ave_p0.py` (`eventness_method=vision_binary_mlp`), `avs/experiments/ave_p0_sweep.py` (`candidate_set=ltl_top1med_norm_v1`) |
| Params | `EVENTNESS=vision_binary_mlp`, `CANDIDATE_SET=ltl_top1med_norm_v1`, `SEEDS=0..2`, `TRAIN_DEVICE` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Promote only if val402 winner is clearly positive and competitive; otherwise stop before any test402 runs. |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0375_ave_p0_sweep_official_val_ltl_top1med_norm_v1_vision_binary_mlp.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=200 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 TRAIN_DEVICE=cuda:0 bash scripts/e0375_ave_p0_sweep_official_val_ltl_top1med_norm_v1_vision_binary_mlp.sh` |
| Full cmd | `SEEDS=0,1,2 TRAIN_DEVICE=cuda:0 bash scripts/e0375_ave_p0_sweep_official_val_ltl_top1med_norm_v1_vision_binary_mlp.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0375_*` |
| Artifacts | `runs/E0375_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke: `runs/E0375_ave_p0_sweep_official_val_vision_binary_mlp_ltl_top1med_norm_v1_20260206-082121/sweep_summary.json` (LIMIT_TRAIN=200 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1). Full val402: `runs/E0375_ave_p0_sweep_official_val_vision_binary_mlp_ltl_top1med_norm_v1_20260206-082152/sweep_summary.json` (best Δ≈-0.00175, p≈0.0938). Decision: not promoted; skip E0376/E0377. |


### E0376: Quick test402 transfer (SEEDS=0..2) — E0375 winner + diagnosis
| Field | Value |
| --- | --- |
| Objective | Run quick test402 for the E0375 winner (only if promoted by val). |
| Baseline | `uniform` |
| Model | Same as E0375 selected config |
| Weights | Same as E0375 |
| Code path | `scripts/e0376_ave_p0_best_to_test_quick_official_vision_binary_mlp.sh`, `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0375_*/best_config.json`), `SEEDS=0..2`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json` via E0344) |
| Checks | Promote to E0377 only if quick Δ is competitive. |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0376_ave_p0_best_to_test_quick_official_vision_binary_mlp.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 TRAIN_DEVICE=cuda:0 bash scripts/e0376_ave_p0_best_to_test_quick_official_vision_binary_mlp.sh` |
| Full cmd | `SEEDS=0,1,2 TRAIN_DEVICE=cuda:0 bash scripts/e0376_ave_p0_best_to_test_quick_official_vision_binary_mlp.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0376_*` |
| Artifacts | `runs/E0376_*/metrics.json` |
| Results | Skipped: E0375 regresses on val402 (Δ≈-0.00175); stop before test402. |


### E0377: Full test402 reproduction (SEEDS=0..9) — E0375 winner → attempt to prove C0003
| Field | Value |
| --- | --- |
| Objective | Spend full test402 budget (SEEDS=0..9) for the E0375 winner if promoted. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Same as E0375 |
| Code path | `scripts/e0377_ave_p0_best_to_test_full_official_vision_binary_mlp.sh`, `avs/experiments/ave_p0_sweep.py` (`run`) |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0375_*/best_config.json`), `SEEDS=0..9`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json` via E0344) |
| Checks | If Δ≥+0.02 and p<0.05, mark C0003 proven. |
| Total time | ~tens of minutes to hours |
| Single-GPU script | `bash scripts/e0377_ave_p0_best_to_test_full_official_vision_binary_mlp.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `SEEDS=0,1 EPOCHS=1 LIMIT_TRAIN=64 LIMIT_EVAL=32 TRAIN_DEVICE=cuda:0 bash scripts/e0377_ave_p0_best_to_test_full_official_vision_binary_mlp.sh` |
| Full cmd | `SEEDS=0,1,2,3,4,5,6,7,8,9 TRAIN_DEVICE=cuda:0 bash scripts/e0377_ave_p0_best_to_test_full_official_vision_binary_mlp.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0377_*` |
| Artifacts | `runs/E0377_*/metrics.json` |
| Results | Skipped: E0375 regresses on val402; stop before test402. |


### E0378: Val402 sweep (SEEDS=0..2) — pretrained PANNs (AudioSet) eventness anchors (`panns`) under scale-invariant top1-med gate
| Field | Value |
| --- | --- |
| Objective | Try a bold new Stage-1 signal: pretrained **PANNs Cnn14 (AudioSet)** per-second eventness (`EVENTNESS=panns`). Run the standard val402 sweep under `candidate_set=ltl_top1med_norm_v1` to find a competitive anchored configuration, targeting C0003 (+2% on test402). |
| Baseline | Current best learned-anchor pipeline baseline (E0223/E0224 / E0341 winner). |
| Model | AVE-P0 `temporal_conv` head on frozen CLIP cache. |
| Weights | PANNs checkpoint (explicit path or default `~/panns_data/Cnn14_mAP=0.431.pth`); head trained per seed. |
| Code path | `scripts/e0378_ave_p0_sweep_official_val_ltl_top1med_norm_v1_panns.sh`, `avs/audio/panns_probe.py`, `avs/experiments/ave_p0.py` (`eventness_method=panns`), `avs/experiments/ave_p0_sweep.py` (`candidate_set=ltl_top1med_norm_v1`) |
| Params | `EVENTNESS=panns`, `CANDIDATE_SET=ltl_top1med_norm_v1`, `SEEDS=0..2`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Promote only if val402 winner is clearly positive and competitive vs baseline val winners (E0223/E0224). |
| Total time | ~tens of minutes to hours (PANNs scoring can dominate if run on CPU) |
| Single-GPU script | `bash scripts/e0378_ave_p0_sweep_official_val_ltl_top1med_norm_v1_panns.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0378_ave_p0_sweep_official_val_ltl_top1med_norm_v1_panns.sh` |
| Full cmd | `SEEDS=0,1,2 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0378_ave_p0_sweep_official_val_ltl_top1med_norm_v1_panns.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0378_*` |
| Artifacts | `runs/E0378_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke: `runs/E0378_ave_p0_sweep_official_val_panns_ltl_top1med_norm_v1_20260206-090624/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1). Full val402: `runs/E0378_ave_p0_sweep_official_val_panns_ltl_top1med_norm_v1_20260206-090736/sweep_summary.json` (best=`ltltop1medn_thr0p6_shift1`, Δ≈+0.00998, p≈0.01367) + `runs/E0378_ave_p0_sweep_official_val_panns_ltl_top1med_norm_v1_20260206-090736/best_config.json`. Decision: not promoted to test402 (val winner not competitive vs baseline val winners). |


### E0379: Quick test402 transfer (SEEDS=0..2) — E0378 winner + diagnosis
| Field | Value |
| --- | --- |
| Objective | Run quick test402 for the E0378 winner (SEEDS=0..2) and diagnose buckets (E0344). |
| Baseline | `uniform` |
| Model | Same as E0378 selected config |
| Weights | Same as E0378 |
| Code path | `scripts/e0379_ave_p0_best_to_test_quick_official_panns.sh`, `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0378_*/best_config.json`), `SEEDS=0..2`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json`) |
| Checks | Promote to E0380 only if quick Δ is competitive vs baseline quick (≈+0.01899). |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0379_ave_p0_best_to_test_quick_official_panns.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0379_ave_p0_best_to_test_quick_official_panns.sh` |
| Full cmd | `SEEDS=0,1,2 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0379_ave_p0_best_to_test_quick_official_panns.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0379_*`, `runs/E0344_*` |
| Artifacts | `runs/E0379_*/metrics.json`, `runs/E0344_*/diagnose.json` |
| Results | Skipped: E0378 val402 winner is not competitive (Δ≈+0.00998); stop before test402. |


### E0380: Full test402 reproduction (SEEDS=0..9) — E0378 winner → attempt to prove C0003
| Field | Value |
| --- | --- |
| Objective | Spend full test402 budget (SEEDS=0..9) for the E0378 winner if promoted, and attempt to prove C0003. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Same as E0378 |
| Code path | `scripts/e0380_ave_p0_best_to_test_full_official_panns.sh`, `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0378_*/best_config.json`), `SEEDS=0..9`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json`) |
| Checks | If Δ≥+0.02 and paired `p<0.05`, mark C0003 proven; otherwise update `docs/plan.md` with decisive failure analysis. |
| Total time | ~tens of minutes to hours |
| Single-GPU script | `bash scripts/e0380_ave_p0_best_to_test_full_official_panns.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `SEEDS=0,1 EPOCHS=1 LIMIT_TRAIN=64 LIMIT_EVAL=32 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0380_ave_p0_best_to_test_full_official_panns.sh` |
| Full cmd | `SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0380_ave_p0_best_to_test_full_official_panns.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0380_*`, `runs/E0344_*` |
| Artifacts | `runs/E0380_*/metrics.json`, `runs/E0344_*/diagnose.json` |
| Results | Skipped: not promoted after E0378 val402 sweep was not competitive (Δ≈+0.00998). |


### E0381: Val402 sweep (SEEDS=0..2) — supervised calibration on PANNs outputs (`panns_lr`) under scale-invariant top1-med gate
| Field | Value |
| --- | --- |
| Objective | Upgrade the raw PANNs heuristic (`EVENTNESS=panns`) with a supervised calibration: train a logistic regression on pretrained PANNs per-second **clipwise outputs** to predict event vs background on the train split, then use per-second logits as Stage-1 eventness (`EVENTNESS=panns_lr`). Run the standard val402 sweep under `candidate_set=ltl_top1med_norm_v1` to find a competitive anchored config targeting C0003 (+2% on test402). |
| Baseline | Current best learned-anchor pipeline baseline (E0223/E0224 / E0341 winner). |
| Model | AVE-P0 `temporal_conv` head on frozen CLIP cache. |
| Weights | PANNs checkpoint (explicit path or default `~/panns_data/Cnn14_mAP=0.431.pth`); `panns_lr` calibrator trained on train split only; head trained per seed. |
| Code path | `scripts/e0381_ave_p0_sweep_official_val_ltl_top1med_norm_v1_panns_lr.sh`, `avs/audio/panns_probe.py`, `avs/experiments/ave_p0_sweep.py` (`eventness_method=panns_lr`) |
| Params | `EVENTNESS=panns_lr`, `CANDIDATE_SET=ltl_top1med_norm_v1`, `SEEDS=0..2`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Promote only if val402 winner is clearly positive and competitive vs baseline val winners (E0223/E0224). |
| Total time | ~tens of minutes to hours |
| Single-GPU script | `bash scripts/e0381_ave_p0_sweep_official_val_ltl_top1med_norm_v1_panns_lr.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0381_ave_p0_sweep_official_val_ltl_top1med_norm_v1_panns_lr.sh` |
| Full cmd | `SEEDS=0,1,2 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0381_ave_p0_sweep_official_val_ltl_top1med_norm_v1_panns_lr.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0381_*` |
| Artifacts | `runs/E0381_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke: `runs/E0381_ave_p0_sweep_official_val_panns_lr_ltl_top1med_norm_v1_20260206-092930/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1; best=`ltltop1medn_thr0p5_shift0`, Δ≈+0.00156). Full val402: `runs/E0381_ave_p0_sweep_official_val_panns_lr_ltl_top1med_norm_v1_20260206-093023/sweep_summary.json` (best=`ltltop1medn_thr0p6_shift0`, Δ≈-0.00224, p≈0.728). Decision: not promoted; skip E0382/E0383. |


### E0382: Quick test402 transfer (SEEDS=0..2) — E0381 winner + diagnosis
| Field | Value |
| --- | --- |
| Objective | Run quick test402 for the E0381 winner (SEEDS=0..2) and diagnose buckets (E0344). |
| Baseline | `uniform` |
| Model | Same as E0381 selected config |
| Weights | Same as E0381 |
| Code path | `scripts/e0382_ave_p0_best_to_test_quick_official_panns_lr.sh`, `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0381_*/best_config.json`), `SEEDS=0..2`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json`) |
| Checks | Promote to E0383 only if quick Δ is competitive vs baseline quick (≈+0.01899). |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0382_ave_p0_best_to_test_quick_official_panns_lr.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0382_ave_p0_best_to_test_quick_official_panns_lr.sh` |
| Full cmd | `SEEDS=0,1,2 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0382_ave_p0_best_to_test_quick_official_panns_lr.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0382_*`, `runs/E0344_*` |
| Artifacts | `runs/E0382_*/metrics.json`, `runs/E0344_*/diagnose.json` |
| Results | Skipped: E0381 val402 regresses (best Δ≈-0.00224); stop before test402. |


### E0383: Full test402 reproduction (SEEDS=0..9) — E0381 winner → attempt to prove C0003
| Field | Value |
| --- | --- |
| Objective | Spend full test402 budget (SEEDS=0..9) for the E0381 winner if promoted, and attempt to prove C0003. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Same as E0381 |
| Code path | `scripts/e0383_ave_p0_best_to_test_full_official_panns_lr.sh`, `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0381_*/best_config.json`), `SEEDS=0..9`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json`) |
| Checks | If Δ≥+0.02 and paired `p<0.05`, mark C0003 proven; otherwise update `docs/plan.md` with decisive failure analysis. |
| Total time | ~tens of minutes to hours |
| Single-GPU script | `bash scripts/e0383_ave_p0_best_to_test_full_official_panns_lr.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `SEEDS=0,1 EPOCHS=1 LIMIT_TRAIN=64 LIMIT_EVAL=32 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0383_ave_p0_best_to_test_full_official_panns_lr.sh` |
| Full cmd | `SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0383_ave_p0_best_to_test_full_official_panns_lr.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0383_*`, `runs/E0344_*` |
| Artifacts | `runs/E0383_*/metrics.json`, `runs/E0344_*/diagnose.json` |
| Results | Skipped: E0381 val402 regresses (best Δ≈-0.00224); stop before test402. |


### E0384: Val402 sweep (SEEDS=0..2) — supervised calibration on PANNs embeddings (`panns_embed_lr`) under scale-invariant top1-med gate
| Field | Value |
| --- | --- |
| Objective | Upgrade the failed `panns_lr` by calibrating on pretrained PANNs **embeddings** (2048-d) instead of post-sigmoid class probabilities: train a logistic regression on per-second embeddings to predict event vs background on the train split, then use per-second logits as Stage-1 eventness (`EVENTNESS=panns_embed_lr`). Run the standard val402 sweep under `candidate_set=ltl_top1med_norm_v1` to find a competitive anchored config targeting C0003 (+2% on test402). |
| Baseline | Current best learned-anchor pipeline baseline (E0223/E0224 / E0341 winner). |
| Model | AVE-P0 `temporal_conv` head on frozen CLIP cache. |
| Weights | PANNs checkpoint (explicit path or default `~/panns_data/Cnn14_mAP=0.431.pth`); `panns_embed_lr` calibrator trained on train split only; head trained per seed. |
| Code path | `scripts/e0384_ave_p0_sweep_official_val_ltl_top1med_norm_v1_panns_embed_lr.sh`, `avs/audio/panns_probe.py`, `avs/experiments/ave_p0_sweep.py` (`eventness_method=panns_embed_lr`) |
| Params | `EVENTNESS=panns_embed_lr`, `CANDIDATE_SET=ltl_top1med_norm_v1`, `SEEDS=0..2`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Promote only if val402 winner is clearly positive and competitive vs baseline val winners (E0223/E0224). |
| Total time | ~tens of minutes to hours |
| Single-GPU script | `bash scripts/e0384_ave_p0_sweep_official_val_ltl_top1med_norm_v1_panns_embed_lr.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0384_ave_p0_sweep_official_val_ltl_top1med_norm_v1_panns_embed_lr.sh` |
| Full cmd | `SEEDS=0,1,2 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0384_ave_p0_sweep_official_val_ltl_top1med_norm_v1_panns_embed_lr.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0384_*` |
| Artifacts | `runs/E0384_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke: `runs/E0384_ave_p0_sweep_official_val_panns_embed_lr_ltl_top1med_norm_v1_20260206-094359/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1; best=`ltltop1medn_thr0p7_shift0`, Δ≈+0.00469). Full val402: `runs/E0384_ave_p0_sweep_official_val_panns_embed_lr_ltl_top1med_norm_v1_20260206-094428/sweep_summary.json` (best=`ltltop1medn_thr0p5_shift0`, Δ≈+0.00865, p≈0.110). Decision: not promoted; skip E0385/E0386. |


### E0385: Quick test402 transfer (SEEDS=0..2) — E0384 winner + diagnosis
| Field | Value |
| --- | --- |
| Objective | Run quick test402 for the E0384 winner (SEEDS=0..2) and diagnose buckets (E0344). |
| Baseline | `uniform` |
| Model | Same as E0384 selected config |
| Weights | Same as E0384 |
| Code path | `scripts/e0385_ave_p0_best_to_test_quick_official_panns_embed_lr.sh`, `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0384_*/best_config.json`), `SEEDS=0..2`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json`) |
| Checks | Promote to E0386 only if quick Δ is competitive vs baseline quick (≈+0.01899). |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0385_ave_p0_best_to_test_quick_official_panns_embed_lr.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0385_ave_p0_best_to_test_quick_official_panns_embed_lr.sh` |
| Full cmd | `SEEDS=0,1,2 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0385_ave_p0_best_to_test_quick_official_panns_embed_lr.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0385_*`, `runs/E0344_*` |
| Artifacts | `runs/E0385_*/metrics.json`, `runs/E0344_*/diagnose.json` |
| Results | Skipped: E0384 val402 not competitive vs baseline val winners (best Δ≈+0.00865); stop before test402. |


### E0386: Full test402 reproduction (SEEDS=0..9) — E0384 winner → attempt to prove C0003
| Field | Value |
| --- | --- |
| Objective | Spend full test402 budget (SEEDS=0..9) for the E0384 winner if promoted, and attempt to prove C0003. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Same as E0384 |
| Code path | `scripts/e0386_ave_p0_best_to_test_full_official_panns_embed_lr.sh`, `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0384_*/best_config.json`), `SEEDS=0..9`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json`) |
| Checks | If Δ≥+0.02 and paired `p<0.05`, mark C0003 proven; otherwise update `docs/plan.md` with decisive failure analysis. |
| Total time | ~tens of minutes to hours |
| Single-GPU script | `bash scripts/e0386_ave_p0_best_to_test_full_official_panns_embed_lr.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `SEEDS=0,1 EPOCHS=1 LIMIT_TRAIN=64 LIMIT_EVAL=32 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0386_ave_p0_best_to_test_full_official_panns_embed_lr.sh` |
| Full cmd | `SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0386_ave_p0_best_to_test_full_official_panns_embed_lr.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0386_*`, `runs/E0344_*` |
| Artifacts | `runs/E0386_*/metrics.json`, `runs/E0344_*/diagnose.json` |
| Results | Skipped: E0384 val402 not competitive vs baseline val winners (best Δ≈+0.00865); stop before test402. |


### E0387: Val402 sweep (SEEDS=0..2) — supervised calibration on PANNs embeddings with a tiny MLP (`panns_embed_mlp`) under scale-invariant top1-med gate
| Field | Value |
| --- | --- |
| Objective | Upgrade `panns_embed_lr` by using a tiny nonlinear calibrator: train a 2-layer MLP on pretrained PANNs per-second embeddings to predict event vs background on the train split, then use per-second logits as Stage-1 eventness (`EVENTNESS=panns_embed_mlp`). Run the standard val402 sweep under `candidate_set=ltl_top1med_norm_v1` to find a competitive anchored config targeting C0003 (+2% on test402). |
| Baseline | Current best learned-anchor pipeline baseline (E0223/E0224 / E0341 winner). |
| Model | AVE-P0 `temporal_conv` head on frozen CLIP cache. |
| Weights | PANNs checkpoint (explicit path or default `~/panns_data/Cnn14_mAP=0.431.pth`); `panns_embed_mlp` calibrator trained on train split only; head trained per seed. |
| Code path | `scripts/e0387_ave_p0_sweep_official_val_ltl_top1med_norm_v1_panns_embed_mlp.sh`, `avs/audio/panns_probe.py`, `avs/experiments/ave_p0_sweep.py` (`eventness_method=panns_embed_mlp`) |
| Params | `EVENTNESS=panns_embed_mlp`, `CANDIDATE_SET=ltl_top1med_norm_v1`, `SEEDS=0..2`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Promote only if val402 winner is clearly positive and competitive vs baseline val winners (E0223/E0224). |
| Total time | ~tens of minutes to hours |
| Single-GPU script | `bash scripts/e0387_ave_p0_sweep_official_val_ltl_top1med_norm_v1_panns_embed_mlp.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0387_ave_p0_sweep_official_val_ltl_top1med_norm_v1_panns_embed_mlp.sh` |
| Full cmd | `SEEDS=0,1,2 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0387_ave_p0_sweep_official_val_ltl_top1med_norm_v1_panns_embed_mlp.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0387_*` |
| Artifacts | `runs/E0387_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke: `runs/E0387_ave_p0_sweep_official_val_panns_embed_mlp_ltl_top1med_norm_v1_20260206-095420/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1; best Δ≈+0.00469). Full val402: `runs/E0387_ave_p0_sweep_official_val_panns_embed_mlp_ltl_top1med_norm_v1_20260206-095447/sweep_summary.json` (best=`ltltop1medn_thr0p6_shift0`, Δ≈+0.00208, p≈0.785). Decision: not promoted; skip E0388/E0389. |


### E0388: Quick test402 transfer (SEEDS=0..2) — E0387 winner + diagnosis
| Field | Value |
| --- | --- |
| Objective | Run quick test402 for the E0387 winner (SEEDS=0..2) and diagnose buckets (E0344). |
| Baseline | `uniform` |
| Model | Same as E0387 selected config |
| Weights | Same as E0387 |
| Code path | `scripts/e0388_ave_p0_best_to_test_quick_official_panns_embed_mlp.sh`, `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0387_*/best_config.json`), `SEEDS=0..2`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json`) |
| Checks | Promote to E0389 only if quick Δ is competitive vs baseline quick (≈+0.01899). |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0388_ave_p0_best_to_test_quick_official_panns_embed_mlp.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0388_ave_p0_best_to_test_quick_official_panns_embed_mlp.sh` |
| Full cmd | `SEEDS=0,1,2 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0388_ave_p0_best_to_test_quick_official_panns_embed_mlp.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0388_*`, `runs/E0344_*` |
| Artifacts | `runs/E0388_*/metrics.json`, `runs/E0344_*/diagnose.json` |
| Results | Skipped: E0387 val402 is not competitive (best Δ≈+0.00208); stop before test402. |


### E0389: Full test402 reproduction (SEEDS=0..9) — E0387 winner → attempt to prove C0003
| Field | Value |
| --- | --- |
| Objective | Spend full test402 budget (SEEDS=0..9) for the E0387 winner if promoted, and attempt to prove C0003. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Same as E0387 |
| Code path | `scripts/e0389_ave_p0_best_to_test_full_official_panns_embed_mlp.sh`, `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0387_*/best_config.json`), `SEEDS=0..9`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json`) |
| Checks | If Δ≥+0.02 and paired `p<0.05`, mark C0003 proven; otherwise update `docs/plan.md` with decisive failure analysis. |
| Total time | ~tens of minutes to hours |
| Single-GPU script | `bash scripts/e0389_ave_p0_best_to_test_full_official_panns_embed_mlp.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `SEEDS=0,1 EPOCHS=1 LIMIT_TRAIN=64 LIMIT_EVAL=32 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0389_ave_p0_best_to_test_full_official_panns_embed_mlp.sh` |
| Full cmd | `SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0389_ave_p0_best_to_test_full_official_panns_embed_mlp.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0389_*`, `runs/E0344_*` |
| Artifacts | `runs/E0389_*/metrics.json`, `runs/E0344_*/diagnose.json` |
| Results | Skipped: E0387 val402 is not competitive (best Δ≈+0.00208); stop before test402. |


### E0390: Val402 sweep (SEEDS=0..2) — PANNs-embeddings × CLIPdiff supervised anchors (`av_panns_embed_clipdiff_mlp`) under scale-invariant top1-med gate
| Field | Value |
| --- | --- |
| Objective | Try a bolder audio×cheap-visual Stage-1: train a tiny per-second MLP on pretrained PANNs embeddings + cheap CLIP feature diff scalar to predict event vs background on train split, then use per-second logits as Stage-1 scores (`EVENTNESS=av_panns_embed_clipdiff_mlp`). Run the standard val402 sweep under `candidate_set=ltl_top1med_norm_v1` to find a competitive anchored config targeting C0003 (+2% on test402). |
| Baseline | Current best learned-anchor pipeline baseline (E0223/E0224 / E0341 winner). |
| Model | AVE-P0 `temporal_conv` head on frozen CLIP cache. |
| Weights | PANNs checkpoint (explicit path or default `~/panns_data/Cnn14_mAP=0.431.pth`); CLIP caches already present; head trained per seed. |
| Code path | `scripts/e0390_ave_p0_sweep_official_val_ltl_top1med_norm_v1_av_panns_embed_clipdiff_mlp.sh`, `avs/audio/panns_probe.py`, `avs/experiments/ave_p0_sweep.py` (`eventness_method=av_panns_embed_clipdiff_mlp`) |
| Params | `EVENTNESS=av_panns_embed_clipdiff_mlp`, `CANDIDATE_SET=ltl_top1med_norm_v1`, `SEEDS=0..2`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Promote only if val402 winner is clearly positive and competitive vs baseline val winners (E0223/E0224). |
| Total time | ~tens of minutes to hours |
| Single-GPU script | `bash scripts/e0390_ave_p0_sweep_official_val_ltl_top1med_norm_v1_av_panns_embed_clipdiff_mlp.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0390_ave_p0_sweep_official_val_ltl_top1med_norm_v1_av_panns_embed_clipdiff_mlp.sh` |
| Full cmd | `SEEDS=0,1,2 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0390_ave_p0_sweep_official_val_ltl_top1med_norm_v1_av_panns_embed_clipdiff_mlp.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0390_*` |
| Artifacts | `runs/E0390_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke: `runs/E0390_ave_p0_sweep_official_val_av_panns_embed_clipdiff_mlp_ltl_top1med_norm_v1_20260206-102211/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1; best Δ≈+0.00469). Full val402: `runs/E0390_ave_p0_sweep_official_val_av_panns_embed_clipdiff_mlp_ltl_top1med_norm_v1_20260206-102257/sweep_summary.json` (best=`ltltop1medn_thr0p7_shift1`, Δ≈+0.00374, p≈0.542) + `runs/E0390_ave_p0_sweep_official_val_av_panns_embed_clipdiff_mlp_ltl_top1med_norm_v1_20260206-102257/best_config.json`. Decision: not promoted to test402. |


### E0391: Quick test402 transfer (SEEDS=0..2) — E0390 winner + diagnosis
| Field | Value |
| --- | --- |
| Objective | Run quick test402 for the E0390 winner (SEEDS=0..2) and diagnose buckets (E0344). |
| Baseline | `uniform` |
| Model | Same as E0390 selected config |
| Weights | Same as E0390 |
| Code path | `scripts/e0391_ave_p0_best_to_test_quick_official_av_panns_embed_clipdiff_mlp.sh`, `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0390_*/best_config.json`), `SEEDS=0..2`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json`) |
| Checks | Promote to E0392 only if quick Δ is competitive vs baseline quick (≈+0.01899). |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0391_ave_p0_best_to_test_quick_official_av_panns_embed_clipdiff_mlp.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0391_ave_p0_best_to_test_quick_official_av_panns_embed_clipdiff_mlp.sh` |
| Full cmd | `SEEDS=0,1,2 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0391_ave_p0_best_to_test_quick_official_av_panns_embed_clipdiff_mlp.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0391_*`, `runs/E0344_*` |
| Artifacts | `runs/E0391_*/metrics.json`, `runs/E0344_*/diagnose.json` |
| Results | Skipped: E0390 val402 is not competitive (best Δ≈+0.00374); stop before test402. Evidence: `runs/E0390_ave_p0_sweep_official_val_av_panns_embed_clipdiff_mlp_ltl_top1med_norm_v1_20260206-102257/sweep_summary.json`. |


### E0392: Full test402 reproduction (SEEDS=0..9) — E0390 winner → attempt to prove C0003
| Field | Value |
| --- | --- |
| Objective | Spend full test402 budget (SEEDS=0..9) for the E0390 winner if promoted, and attempt to prove C0003. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Same as E0390 |
| Code path | `scripts/e0392_ave_p0_best_to_test_full_official_av_panns_embed_clipdiff_mlp.sh`, `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0390_*/best_config.json`), `SEEDS=0..9`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json`) |
| Checks | If Δ≥+0.02 and paired `p<0.05`, mark C0003 proven; otherwise update `docs/plan.md` with decisive failure analysis. |
| Total time | ~tens of minutes to hours |
| Single-GPU script | `bash scripts/e0392_ave_p0_best_to_test_full_official_av_panns_embed_clipdiff_mlp.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `SEEDS=0,1 EPOCHS=1 LIMIT_TRAIN=64 LIMIT_EVAL=32 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0392_ave_p0_best_to_test_full_official_av_panns_embed_clipdiff_mlp.sh` |
| Full cmd | `SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0392_ave_p0_best_to_test_full_official_av_panns_embed_clipdiff_mlp.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0392_*`, `runs/E0344_*` |
| Artifacts | `runs/E0392_*/metrics.json`, `runs/E0344_*/diagnose.json` |
| Results | Skipped: E0390 val402 is not competitive (best Δ≈+0.00374); stop before test402. Evidence: `runs/E0390_ave_p0_sweep_official_val_av_panns_embed_clipdiff_mlp_ltl_top1med_norm_v1_20260206-102257/sweep_summary.json`. |


### E0393: Val402 sweep (SEEDS=0..2) — optical-flow augmented supervised anchors (`av_clipdiff_flow_mlp`) under scale-invariant top1-med gate
| Field | Value |
| --- | --- |
| Objective | Try a new cheap-visual Stage-1 signal: Farneback optical-flow magnitude computed from per-second frames, concatenated with the existing `(audio basic + CLIPdiff scalar)` features. Train a tiny per-second MLP to predict event-vs-background on train split, then sweep anchored configs on official val402 under `candidate_set=ltl_top1med_norm_v1`. |
| Baseline | Current best learned-anchor pipeline baseline (E0223/E0224 / E0341 winner). |
| Model | AVE-P0 `temporal_conv` head on frozen CLIP cache. |
| Weights | CLIP caches already present; no extra pretrained weights required beyond existing setup. |
| Code path | `scripts/e0393_ave_p0_sweep_official_val_ltl_top1med_norm_v1_av_clipdiff_flow_mlp.sh`, `avs/vision/cheap_eventness.py`, `avs/experiments/ave_p0_sweep.py` (`eventness_method=av_clipdiff_flow_mlp`) |
| Params | `EVENTNESS=av_clipdiff_flow_mlp`, `CANDIDATE_SET=ltl_top1med_norm_v1`, `SEEDS=0..2`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Promote only if val402 winner is clearly positive and competitive vs baseline val winners (E0223/E0224). |
| Total time | ~tens of minutes to hours |
| Single-GPU script | `bash scripts/e0393_ave_p0_sweep_official_val_ltl_top1med_norm_v1_av_clipdiff_flow_mlp.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0393_ave_p0_sweep_official_val_ltl_top1med_norm_v1_av_clipdiff_flow_mlp.sh` |
| Full cmd | `SEEDS=0,1,2 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0393_ave_p0_sweep_official_val_ltl_top1med_norm_v1_av_clipdiff_flow_mlp.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0393_*` |
| Artifacts | `runs/E0393_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke: `runs/E0393_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_ltl_top1med_norm_v1_20260206-104337/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1; best Δ≈+0.00156). Full val402: `runs/E0393_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_ltl_top1med_norm_v1_20260206-104413/sweep_summary.json` (best=`ltltop1medn_thr0p6_shift1`, Δ≈+0.00881, p≈0.0971) + `runs/E0393_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_ltl_top1med_norm_v1_20260206-104413/best_config.json`. Decision: not promoted to test402. |


### E0394: Quick test402 transfer (SEEDS=0..2) — E0393 winner + diagnosis
| Field | Value |
| --- | --- |
| Objective | Run quick test402 for the E0393 winner (SEEDS=0..2) and diagnose buckets (E0344). |
| Baseline | `uniform` |
| Model | Same as E0393 selected config |
| Weights | Same as E0393 |
| Code path | `scripts/e0394_ave_p0_best_to_test_quick_official_av_clipdiff_flow_mlp.sh`, `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0393_*/best_config.json`), `SEEDS=0..2`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json`) |
| Checks | Promote to E0395 only if quick Δ is competitive vs baseline quick (≈+0.01899). |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0394_ave_p0_best_to_test_quick_official_av_clipdiff_flow_mlp.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0394_ave_p0_best_to_test_quick_official_av_clipdiff_flow_mlp.sh` |
| Full cmd | `SEEDS=0,1,2 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0394_ave_p0_best_to_test_quick_official_av_clipdiff_flow_mlp.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0394_*`, `runs/E0344_*` |
| Artifacts | `runs/E0394_*/metrics.json`, `runs/E0344_*/diagnose.json` |
| Results | Skipped: E0393 val402 does not beat the baseline val winner (best Δ≈+0.00881, p≈0.0971); stop before test402. Evidence: `runs/E0393_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_ltl_top1med_norm_v1_20260206-104413/sweep_summary.json`. |


### E0395: Full test402 reproduction (SEEDS=0..9) — E0393 winner → attempt to prove C0003
| Field | Value |
| --- | --- |
| Objective | Spend full test402 budget (SEEDS=0..9) for the E0393 winner if promoted, and attempt to prove C0003. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Same as E0393 |
| Code path | `scripts/e0395_ave_p0_best_to_test_full_official_av_clipdiff_flow_mlp.sh`, `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0393_*/best_config.json`), `SEEDS=0..9`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json`) |
| Checks | If Δ≥+0.02 and paired `p<0.05`, mark C0003 proven; otherwise update `docs/plan.md` with decisive failure analysis. |
| Total time | ~tens of minutes to hours |
| Single-GPU script | `bash scripts/e0395_ave_p0_best_to_test_full_official_av_clipdiff_flow_mlp.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `SEEDS=0,1 EPOCHS=1 LIMIT_TRAIN=64 LIMIT_EVAL=32 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0395_ave_p0_best_to_test_full_official_av_clipdiff_flow_mlp.sh` |
| Full cmd | `SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0395_ave_p0_best_to_test_full_official_av_clipdiff_flow_mlp.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0395_*`, `runs/E0344_*` |
| Artifacts | `runs/E0395_*/metrics.json`, `runs/E0344_*/diagnose.json` |
| Results | Skipped: E0393 val402 does not beat the baseline val winner (best Δ≈+0.00881, p≈0.0971); stop before test402. Evidence: `runs/E0393_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_ltl_top1med_norm_v1_20260206-104413/sweep_summary.json`. |


### E0396: Val402 sweep (SEEDS=0..2) — fixed flow Stage-1 + dynamic-K extreme Stage-2 (`ltl_top1med_k1_extreme_v1`)
| Field | Value |
| --- | --- |
| Objective | Keep Stage-1 fixed as `EVENTNESS=av_clipdiff_flow_mlp` (from E0393), and aggressively search Stage-2 with `candidate_set=ltl_top1med_k1_extreme_v1` to suppress known 2-high harm. |
| Baseline | E0223/E0224 val winner and E0393 winner. |
| Model | AVE-P0 `temporal_conv` head on frozen CLIP cache. |
| Weights | Reuse latest E0393 `eventness_scores.json` by default; no extra pretrained weights. |
| Code path | `scripts/e0396_ave_p0_sweep_official_val_ltl_top1med_k1_extreme_v1_av_clipdiff_flow_mlp.sh`, `avs/experiments/ave_p0_sweep.py` |
| Params | `EVENTNESS=av_clipdiff_flow_mlp`, `CANDIDATE_SET=ltl_top1med_k1_extreme_v1`, `SEEDS=0..2`, `AUDIO_DEVICE`, `TRAIN_DEVICE`, optional `SCORES_JSON` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Promote only if val402 winner is clearly competitive vs baseline val winners. |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0396_ave_p0_sweep_official_val_ltl_top1med_k1_extreme_v1_av_clipdiff_flow_mlp.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0396_ave_p0_sweep_official_val_ltl_top1med_k1_extreme_v1_av_clipdiff_flow_mlp.sh` |
| Full cmd | `SEEDS=0,1,2 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0396_ave_p0_sweep_official_val_ltl_top1med_k1_extreme_v1_av_clipdiff_flow_mlp.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0396_*` |
| Artifacts | `runs/E0396_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | Smoke: `runs/E0396_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_ltl_top1med_k1_extreme_v1_20260206-130029/sweep_summary.json` (LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1; best Δ≈+0.00781). Full val402: `runs/E0396_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_ltl_top1med_k1_extreme_v1_20260206-130114/sweep_summary.json` (best=`ltltop1medk1ext_thr0p6_shift1_score`, Δ≈-0.00125, p≈0.924) + `runs/E0396_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_ltl_top1med_k1_extreme_v1_20260206-130114/best_config.json`. Decision: not promoted to test402. |


### E0397: Quick test402 transfer (SEEDS=0..2) — E0396 winner + diagnosis
| Field | Value |
| --- | --- |
| Objective | Run quick test402 for the E0396 winner (SEEDS=0..2) and diagnose buckets (E0344). |
| Baseline | `uniform` |
| Model | Same as E0396 selected config |
| Weights | Same as E0396 |
| Code path | `scripts/e0397_ave_p0_best_to_test_quick_official_av_clipdiff_flow_mlp_k1ext.sh`, `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0396_*/best_config.json`), `SEEDS=0..2`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json`) |
| Checks | Promote to E0398 only if quick Δ is competitive vs baseline quick (≈+0.01899). |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0397_ave_p0_best_to_test_quick_official_av_clipdiff_flow_mlp_k1ext.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0397_ave_p0_best_to_test_quick_official_av_clipdiff_flow_mlp_k1ext.sh` |
| Full cmd | `SEEDS=0,1,2 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0397_ave_p0_best_to_test_quick_official_av_clipdiff_flow_mlp_k1ext.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0397_*`, `runs/E0344_*` |
| Artifacts | `runs/E0397_*/metrics.json`, `runs/E0344_*/diagnose.json` |
| Results | Skipped: E0396 val402 regresses (best Δ≈-0.00125, p≈0.924); stop before test402. Evidence: `runs/E0396_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_ltl_top1med_k1_extreme_v1_20260206-130114/sweep_summary.json`. |


### E0398: Full test402 reproduction (SEEDS=0..9) — E0396 winner → attempt to prove C0003
| Field | Value |
| --- | --- |
| Objective | Spend full test402 budget (SEEDS=0..9) for the E0396 winner if promoted, and attempt to prove C0003. |
| Baseline | `uniform` |
| Model | Same as the selected config |
| Weights | Same as E0396 |
| Code path | `scripts/e0398_ave_p0_best_to_test_full_official_av_clipdiff_flow_mlp_k1ext.sh`, `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON` (defaults to latest `runs/E0396_*/best_config.json`), `SEEDS=0..9`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json` (+ `diagnose.json`) |
| Checks | If Δ≥+0.02 and paired `p<0.05`, mark C0003 proven; otherwise update `docs/plan.md` with decisive failure analysis. |
| Total time | ~tens of minutes to hours |
| Single-GPU script | `bash scripts/e0398_ave_p0_best_to_test_full_official_av_clipdiff_flow_mlp_k1ext.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `SEEDS=0,1 EPOCHS=1 LIMIT_TRAIN=64 LIMIT_EVAL=32 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0398_ave_p0_best_to_test_full_official_av_clipdiff_flow_mlp_k1ext.sh` |
| Full cmd | `SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0398_ave_p0_best_to_test_full_official_av_clipdiff_flow_mlp_k1ext.sh` |
| Smoke | [ ] |
| Full | [ ] |
| Logs | `runs/E0398_*`, `runs/E0344_*` |
| Artifacts | `runs/E0398_*/metrics.json`, `runs/E0344_*/diagnose.json` |
| Results | Skipped: E0396 val402 regresses (best Δ≈-0.00125, p≈0.924); stop before test402. Evidence: `runs/E0396_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_ltl_top1med_k1_extreme_v1_20260206-130114/sweep_summary.json`. |


### E0399: Smoke val sweep (SEEDS=0..1) — dense-stride Stage-1 (`av_clipdiff_flow_mlp_stride`) + `ltl_top1med_k1_extreme_v1`
| Field | Value |
| --- | --- |
| Objective | Smoke-verify the new dense-stride Stage-1 scorer (`EVENTNESS=av_clipdiff_flow_mlp_stride`) with strict candidate set `ltl_top1med_k1_extreme_v1` on official val split. |
| Baseline | E0396 smoke and prior `av_clipdiff_flow_mlp` Stage-1 sweeps. |
| Model | AVE-P0 `temporal_conv` head on frozen CLIP cache. |
| Weights | Reuse official CLIP caches; Stage-1 MLP trained from AVE train split. |
| Code path | `scripts/e0399_ave_p0_sweep_official_val_smoke_ltl_top1med_k1_extreme_v1_av_clipdiff_flow_mlp_stride.sh`, `avs/experiments/ave_p0_sweep.py`, `avs/utils/scores.py` |
| Params | `EVENTNESS=av_clipdiff_flow_mlp_stride`, `SEEDS=0,1`, `EPOCHS=1`, `LIMIT_TRAIN=64`, `LIMIT_EVAL=32` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | End-to-end smoke passes and emits a valid best config under the new Stage-1 method. |
| Total time | ~minutes |
| Single-GPU script | `bash scripts/e0399_ave_p0_sweep_official_val_smoke_ltl_top1med_k1_extreme_v1_av_clipdiff_flow_mlp_stride.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `bash scripts/e0399_ave_p0_sweep_official_val_smoke_ltl_top1med_k1_extreme_v1_av_clipdiff_flow_mlp_stride.sh` |
| Full cmd | `SEEDS=0,1,2 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0400_ave_p0_sweep_official_val_ltl_top1med_k1_extreme_v1_av_clipdiff_flow_mlp_stride.sh` |
| Smoke | [x] |
| Full | [ ] |
| Logs | `runs/E0399_*` |
| Artifacts | `runs/E0399_*/{sweep_summary.json,best_config.json,eventness_scores.json}` |
| Results | `runs/E0399_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_stride_ltl_top1med_k1_extreme_v1_20260206-131751/sweep_summary.json` (best Δ≈+0.00156, p≈0.500). |


### E0400: Full val402 sweep (SEEDS=0..2) — dense-stride Stage-1 (`av_clipdiff_flow_mlp_stride`) strict gate sets
| Field | Value |
| --- | --- |
| Objective | Run official val402 sweeps for dense-stride Stage-1 with strict Stage-2 sets (`k1_extreme`, `top1med_norm`, `top1med_v1`) and determine promotion to test402. |
| Baseline | E0223/E0224 val winners and E0393/E0396 family. |
| Model | AVE-P0 `temporal_conv` head on frozen CLIP cache. |
| Weights | Reuse official CLIP caches; Stage-1 MLP trained per run from AVE train split. |
| Code path | `scripts/e0400_ave_p0_sweep_official_val_ltl_top1med_k1_extreme_v1_av_clipdiff_flow_mlp_stride.sh` |
| Params | `EVENTNESS=av_clipdiff_flow_mlp_stride`, `SEEDS=0,1,2`, `LIMIT_TRAIN=3339`, `LIMIT_EVAL=402` |
| Metrics (must save) | `sweep_summary.json`, `best_config.json`, `eventness_scores.json` |
| Checks | Promote only if val402 winner is competitive against current best val winners; otherwise stop before test402. |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0400_ave_p0_sweep_official_val_ltl_top1med_k1_extreme_v1_av_clipdiff_flow_mlp_stride.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0399_ave_p0_sweep_official_val_smoke_ltl_top1med_k1_extreme_v1_av_clipdiff_flow_mlp_stride.sh` |
| Full cmd | `SEEDS=0,1,2 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0400_ave_p0_sweep_official_val_ltl_top1med_k1_extreme_v1_av_clipdiff_flow_mlp_stride.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0400_*`, `runs/E0400b_*`, `runs/E0400c_*` |
| Artifacts | `runs/E0400_*/sweep_summary.json`, `runs/E0400b_*/sweep_summary.json`, `runs/E0400c_*/sweep_summary.json` |
| Results | E0400 (`k1_extreme`): `runs/E0400_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_stride_ltl_top1med_k1_extreme_v1_20260206-131825/sweep_summary.json` (best Δ≈+0.00723, p≈0.412). E0400b (`top1med_norm`): `runs/E0400b_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_stride_ltl_top1med_norm_v1_20260206-132936/sweep_summary.json` (best Δ≈+0.00607, p≈0.08999). E0400c (`top1med_v1`): `runs/E0400c_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_stride_ltl_top1med_v1_20260206-133240/sweep_summary.json` (best Δ≈+0.00357, p≈0.0487). Decision: not competitive for test402 promotion. |

Notes (2026-02-10 rerun; artifact paths locally present):
- E0400 (`k1_extreme`): `runs/E0400_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_stride_ltl_top1med_k1_extreme_v1_20260210-195236/sweep_summary.json` (best=`ltltop1medk1ext_thr0p6_shift1_score`, Δ≈+0.00490, p≈0.465).


### E0401: Quick test402 transfer (SEEDS=0..2) — dense-stride winner + diagnosis
| Field | Value |
| --- | --- |
| Objective | Run quick test402 for the selected dense-stride candidate and generate bucket diagnosis (`E0344`) for transfer quality. |
| Baseline | `uniform` |
| Model | Same as selected E0400* config |
| Weights | Same as E0400* |
| Code path | `scripts/e0401_ave_p0_best_to_test_quick_official_av_clipdiff_flow_mlp_stride.sh`, `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON`, `SEEDS=0,1,2`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json`, `diagnose.json` |
| Checks | If quick is competitive and diagnosis improves harmful buckets, promote to full test402. |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0401_ave_p0_best_to_test_quick_official_av_clipdiff_flow_mlp_stride.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `SEEDS=0,1 EPOCHS=1 LIMIT_TRAIN=64 LIMIT_EVAL=32 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0401_ave_p0_best_to_test_quick_official_av_clipdiff_flow_mlp_stride.sh` |
| Full cmd | `SEEDS=0,1,2 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0401_ave_p0_best_to_test_quick_official_av_clipdiff_flow_mlp_stride.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0401_*`, `runs/E0344_*` |
| Artifacts | `runs/E0401_*/metrics.json`, `runs/E0344_*/diagnose.json` |
| Results | Baseline quick (SEEDS=0,1,2): `runs/E0401_quick_test402_av_clipdiff_flow_mlp_stride_20260206-140425/metrics.json` + `runs/E0401_quick_test402_av_clipdiff_flow_mlp_stride_20260206-140425/diagnose.json` (`Δ=+0.00771`, `p=0.331`, `fallback_used_frac≈0.8607`). Alternative quick reruns: `runs/E0401_quick_test402_av_clipdiff_flow_mlp_stride_alt_top1med_thr0p5_20260206-150837/metrics.json` + `.../diagnose.json` (`Δ=+0.01153`, `p=0.0661`, `fallback_used_frac≈0.5622`); `runs/E0401_quick_test402_av_clipdiff_flow_mlp_stride_alt_top1medn_thr0p6_20260206-151412/metrics.json` + `.../diagnose.json` (`Δ=+0.00746`, `p=0.436`). |

Notes (2026-02-10 rerun; artifact paths locally present):
- Quick test402 (SEEDS=0..2): `runs/E0401_quick_test402_av_clipdiff_flow_mlp_stride_20260210-195916/metrics.json` (Δ≈-0.00506, p≈0.555; diagnose: `runs/E0401_quick_test402_av_clipdiff_flow_mlp_stride_20260210-195916/diagnose.json`).


### E0402: Full test402 reproduction (SEEDS=0..9) — dense-stride winner → attempt C0003
| Field | Value |
| --- | --- |
| Objective | Run full test402 with SEEDS=0..9 for the selected dense-stride candidate and attempt to prove C0003. |
| Baseline | `uniform` |
| Model | Same as selected E0400* config |
| Weights | Same as E0400* |
| Code path | `scripts/e0402_ave_p0_best_to_test_full_official_av_clipdiff_flow_mlp_stride.sh`, `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON`, `SEEDS=0..9`, `AUDIO_DEVICE`, `TRAIN_DEVICE` |
| Metrics (must save) | `metrics.json`, `diagnose.json` |
| Checks | C0003 gate: `Δ>=+0.02` and paired `p<0.05`. |
| Total time | ~tens of minutes to hours |
| Single-GPU script | `bash scripts/e0402_ave_p0_best_to_test_full_official_av_clipdiff_flow_mlp_stride.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `SEEDS=0,1 EPOCHS=1 LIMIT_TRAIN=64 LIMIT_EVAL=32 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0402_ave_p0_best_to_test_full_official_av_clipdiff_flow_mlp_stride.sh` |
| Full cmd | `SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0402_ave_p0_best_to_test_full_official_av_clipdiff_flow_mlp_stride.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0402_*`, `runs/E0344_*` |
| Artifacts | `runs/E0402_*/metrics.json`, `runs/E0344_*/diagnose.json` |
| Results | Baseline full (SEEDS=0..9): `runs/E0402_full_test402_av_clipdiff_flow_mlp_stride_20260206-141020/metrics.json` + `runs/E0402_full_test402_av_clipdiff_flow_mlp_stride_20260206-141020/diagnose.json` (`Δ=+0.01037`, `p=0.00489`, `fallback_used_frac≈0.8607`). Promoted alternative full (from quick winner `top1_med thr0.5`): `runs/E0402_full_test402_av_clipdiff_flow_mlp_stride_alt_top1med_thr0p5_20260206-152012/metrics.json` + `.../diagnose.json` (`anchored=0.720995`, `uniform=0.708582`, `Δ=+0.01241`, `p=0.00302`, `fallback_used_frac≈0.5622`); still below C0003 target `Δ>=+0.02`. |


### E0403: Oracle→Predicted report (`E0201` protocol) — dense-stride Stage-1
| Field | Value |
| --- | --- |
| Objective | Re-run mechanism evidence for dense-stride Stage-1 and measure Oracle–Predicted gap vs existing deployable baseline. |
| Baseline | Prior deployable Stage-1 reports under E0201. |
| Model | `avs.experiments.mde_ltl oracle_vs_predicted` harness. |
| Weights | Same official caches and head-training protocol as E0201. |
| Code path | `scripts/e0403_oracle_vs_predicted_official_av_clipdiff_flow_mlp_stride.sh`, `scripts/e0201_oracle_vs_predicted_official.sh` |
| Params | `EVENTNESS=av_clipdiff_flow_mlp_stride`, `SEEDS`, `LIMIT_TRAIN`, `LIMIT_EVAL` |
| Metrics (must save) | `oracle_vs_predicted.json` |
| Checks | Report Oracle–Pred gap and predicted-vs-uniform delta under equal budget. |
| Total time | ~tens of minutes to hours |
| Single-GPU script | `bash scripts/e0403_oracle_vs_predicted_official_av_clipdiff_flow_mlp_stride.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `SEEDS=0,1 LIMIT_TRAIN=64 LIMIT_EVAL=32 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0403_oracle_vs_predicted_official_av_clipdiff_flow_mlp_stride.sh` |
| Full cmd | `SEEDS=0,1,2 LIMIT_TRAIN=3339 LIMIT_EVAL=402 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0403_oracle_vs_predicted_official_av_clipdiff_flow_mlp_stride.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0403_*` |
| Artifacts | `runs/E0403_*/oracle_vs_predicted.json` |
| Results | Baseline full (`SEEDS=0,1,2`): `runs/E0403_oracle_vs_predicted_av_clipdiff_flow_mlp_stride_20260206-141804/oracle_vs_predicted.json` (`predicted Δ=-0.00547`, `p=0.03399`, `oracle_minus_predicted=0.04187`). Alternative-config rerun (aligned with promoted `top1_med thr0.5` config): `runs/E0403_oracle_vs_predicted_av_clipdiff_flow_mlp_stride_alt_top1med_thr0p5_20260206-152658/oracle_vs_predicted.json` (`predicted Δ=+0.01153`, `p=0.0661`; `oracle_minus_predicted=0.02488`; `cheap_visual Δ=+0.00415`). Gap shrinks substantially but still does not pass a strict `p<0.05` deployable proof gate. |


### E0404: Degradation suite (`E0203` protocol) — dense-stride Stage-1
| Field | Value |
| --- | --- |
| Objective | Re-run robustness evidence (shift/noise/silence × α) for dense-stride Stage-1 and compare against baseline degradation behavior. |
| Baseline | Prior E0203/E0331 degradation artifacts. |
| Model | `avs.experiments.degradation_suite` harness. |
| Weights | Same official data/caches setup as E0203. |
| Code path | `scripts/e0404_degradation_suite_official_av_clipdiff_flow_mlp_stride.sh`, `scripts/e0203_degradation_suite_official.sh` |
| Params | `EVENTNESS=av_clipdiff_flow_mlp_stride`, `SHIFT_GRID`, `SNR_GRID`, `SILENCE_GRID`, `DELTAS` |
| Metrics (must save) | `degradation_suite.json` (+ plots if generated by harness) |
| Checks | Verify no catastrophic drop below α-baseline pattern under perturbations. |
| Total time | ~tens of minutes |
| Single-GPU script | `bash scripts/e0404_degradation_suite_official_av_clipdiff_flow_mlp_stride.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SHIFT_GRID='0' SNR_GRID='20' SILENCE_GRID='0' AUDIO_DEVICE=cuda:0 bash scripts/e0404_degradation_suite_official_av_clipdiff_flow_mlp_stride.sh` |
| Full cmd | `LIMIT_TRAIN=3339 LIMIT_EVAL=402 SHIFT_GRID='-0.5,0,0.5' SNR_GRID='20,10,0' SILENCE_GRID='0,0.5' AUDIO_DEVICE=cuda:0 bash scripts/e0404_degradation_suite_official_av_clipdiff_flow_mlp_stride.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0404_*` |
| Artifacts | `runs/E0404_*/degradation_suite.json` |
| Results | Full: `runs/E0404_degradation_av_clipdiff_flow_mlp_stride_20260206-142817/degradation_suite.json` (18 rows, official test402; mean Recall@K `Δ0≈0.21308`, `Δ1≈0.38001`, `Δ2≈0.51741`). Compared to prior deployable baseline `runs/E0203_degradation_av_clipdiff_mlp_20260204-215831/degradation_suite.json`, dense-stride is slightly higher on strict `Δ0` (`+0.00147`) but clearly worse on relaxed windows (`Δ1/Δ2`). |


### E0405: Quick test402 transfer grid (SEEDS=0..2) — dense-stride top candidates from E0400b/E0400c
| Field | Value |
| --- | --- |
| Objective | Exhaustively run quick test402 transfer for all high-priority `top1_med` / `top1_med_norm` candidates from dense-stride val sweeps, then select the best transferable config before any new full test402 spend. |
| Baseline | Prior quick winner `top1_med thr0.5 shift1` (`Δ≈+0.01153`). |
| Model | AVE-P0 `temporal_conv` head + `EVENTNESS=av_clipdiff_flow_mlp_stride`. |
| Weights | Same official caches/training protocol as E0401/E0402. |
| Code path | `scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh` |
| Params | `SEEDS=0,1,2`; shared score cache `runs/E0405_shared_scores_av_clipdiff_flow_mlp_stride_test402.json`; candidate configs from `runs/E0400b_*` and `runs/E0400c_*`. |
| Metrics (must save) | `metrics.json` per candidate run (plus ranking summary). |
| Checks | Promote only the best quick-transfer config to full test402 (E0406). |
| VRAM | ~27GB (GPU5 observed). |
| Total time | ~minutes per candidate after shared scores are filled. |
| Single-GPU script | `BEST_CONFIG_JSON=<candidate>/config.json SCORES_JSON=runs/E0405_shared_scores_av_clipdiff_flow_mlp_stride_test402.json EVENTNESS=av_clipdiff_flow_mlp_stride SEEDS=0,1,2 AUDIO_DEVICE=cuda:5 TRAIN_DEVICE=cuda:5 OUT_DIR=runs/E0405_quick_test402_av_clipdiff_flow_mlp_stride_<name>_<ts> bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `BEST_CONFIG_JSON=runs/E0400c_ave_p0_sweep_official_val_av_clipdiff_flow_mlp_stride_ltl_top1med_v1_20260206-133240/ltltop1med_thr0p8_shift1/config.json SCORES_JSON=runs/E0405_shared_scores_av_clipdiff_flow_mlp_stride_test402.json EVENTNESS=av_clipdiff_flow_mlp_stride SEEDS=0,1,2 AUDIO_DEVICE=cuda:5 TRAIN_DEVICE=cuda:5 OUT_DIR=runs/E0405_quick_test402_av_clipdiff_flow_mlp_stride_top1med_thr0p8_shift1_<ts> bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh` |
| Full cmd | `for cfg in <E0400b/E0400c selected configs>; do BEST_CONFIG_JSON="$cfg" SCORES_JSON=runs/E0405_shared_scores_av_clipdiff_flow_mlp_stride_test402.json EVENTNESS=av_clipdiff_flow_mlp_stride SEEDS=0,1,2 AUDIO_DEVICE=cuda:5 TRAIN_DEVICE=cuda:5 OUT_DIR=runs/E0405_quick_test402_av_clipdiff_flow_mlp_stride_<name>_<ts> bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh; done` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0405_quick_test402_*` |
| Artifacts | `runs/E0405_quick_test402_*/metrics.json`, `runs/E0405_shared_scores_av_clipdiff_flow_mlp_stride_test402.json` |
| Results | Best quick candidate is `top1medn_thr0p6_shift0`: `runs/E0405_quick_test402_av_clipdiff_flow_mlp_stride_top1medn_thr0p6_shift0_20260206-161028/metrics.json` (`Δ≈+0.01335`, `p≈0.0956`). Other key runs: `...top1med_thr0p7_shift1_20260206-154502/metrics.json` (`Δ≈+0.00755`), `...top1med_thr0p8_shift1_20260206-155825/metrics.json` (`Δ≈+0.00771`), `...top1med_thr0p6_shift1_20260206-160834/metrics.json` (`Δ≈+0.00771`), `...top1medn_thr0p6_shift1_20260206-161107/metrics.json` (`Δ≈+0.00746`), `...top1medn_thr0p5_shift1_20260206-161214/metrics.json` (`Δ≈+0.00647`). Promoted `top1medn_thr0p6_shift0` to E0406. |


### E0406: Full test402 reproduction (SEEDS=0..9) — E0405 quick winner (`top1medn_thr0p6_shift0`)
| Field | Value |
| --- | --- |
| Objective | Validate whether the E0405 quick winner transfers under full official test402 (SEEDS=0..9) and can beat the prior dense-stride best (`E0402 alt`). |
| Baseline | `runs/E0402_full_test402_av_clipdiff_flow_mlp_stride_alt_top1med_thr0p5_20260206-152012/metrics.json` (`Δ≈+0.01241`). |
| Model | AVE-P0 `temporal_conv` head + `EVENTNESS=av_clipdiff_flow_mlp_stride`. |
| Weights | Same official caches/training protocol; reuse E0405 shared score cache. |
| Code path | `scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`, `scripts/e0344_ave_p0_diagnose.sh` |
| Params | `BEST_CONFIG_JSON=.../ltltop1medn_thr0p6_shift0/config.json`, `SEEDS=0..9`, `SCORES_JSON=runs/E0405_shared_scores_av_clipdiff_flow_mlp_stride_test402.json`. |
| Metrics (must save) | `metrics.json`, `diagnose.json`. |
| Checks | If better than prior full best, update C0003 evidence and promote for mechanism reruns; otherwise keep as negative evidence. |
| Single-GPU script | `BEST_CONFIG_JSON=.../ltltop1medn_thr0p6_shift0/config.json SCORES_JSON=runs/E0405_shared_scores_av_clipdiff_flow_mlp_stride_test402.json EVENTNESS=av_clipdiff_flow_mlp_stride SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:5 TRAIN_DEVICE=cuda:5 OUT_DIR=runs/E0406_full_test402_av_clipdiff_flow_mlp_stride_top1medn_thr0p6_shift0_<ts> bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `BEST_CONFIG_JSON=.../ltltop1medn_thr0p6_shift0/config.json SCORES_JSON=runs/E0405_shared_scores_av_clipdiff_flow_mlp_stride_test402.json EVENTNESS=av_clipdiff_flow_mlp_stride SEEDS=0,1 AUDIO_DEVICE=cuda:5 TRAIN_DEVICE=cuda:5 OUT_DIR=runs/E0406_smoke_test402_av_clipdiff_flow_mlp_stride_top1medn_thr0p6_shift0_<ts> bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh` |
| Full cmd | `BEST_CONFIG_JSON=.../ltltop1medn_thr0p6_shift0/config.json SCORES_JSON=runs/E0405_shared_scores_av_clipdiff_flow_mlp_stride_test402.json EVENTNESS=av_clipdiff_flow_mlp_stride SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:5 TRAIN_DEVICE=cuda:5 OUT_DIR=runs/E0406_full_test402_av_clipdiff_flow_mlp_stride_top1medn_thr0p6_shift0_<ts> bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0406_*`, `runs/E0344_*` |
| Artifacts | `runs/E0406_full_test402_av_clipdiff_flow_mlp_stride_top1medn_thr0p6_shift0_20260206-161349/metrics.json`, `runs/E0344_ave_p0_diagnose_20260206-161548/diagnose.json` |
| Results | Full test402 result regresses vs prior dense-stride best: `runs/E0406_full_test402_av_clipdiff_flow_mlp_stride_top1medn_thr0p6_shift0_20260206-161349/metrics.json` (`anchored≈0.71629`, `uniform≈0.70858`, `Δ≈+0.00771`, `p≈0.0442`). Decision: do not replace E0402 alt winner (`Δ≈+0.01241`). |


### E0407: Oracle→Predicted full rerun (SEEDS=0..9) — aligned with promoted dense-stride config (`top1_med thr0.5`)
| Field | Value |
| --- | --- |
| Objective | Strengthen C0007 evidence by rerunning Oracle→Predicted with full seeds and the same promoted dense-stride config used by E0402 alt (`top1_med thr0.5`, shift=1). |
| Baseline | E0403 alt (`SEEDS=0..2`) and E0201/E0330 prior mechanism evidence. |
| Model | `avs.experiments.mde_ltl oracle_vs_predicted` |
| Weights | Same official caches/training protocol as E0201; cheap-visual control included. |
| Code path | `avs/experiments/mde_ltl.py` |
| Params | `SEEDS=0..9`, `EVENTNESS=av_clipdiff_flow_mlp_stride`, `k=2`, triad=`160/224/352`, `anchor_conf_metric=top1_med`, `anchor_conf_threshold=0.5`, `anchor_shift=1`, `anchor_base_alloc=distance`, `anchor_high_policy=adaptive_v1`. |
| Metrics (must save) | `oracle_vs_predicted.json`, `metrics_predicted.json`, `metrics_cheap_visual.json`. |
| Checks | Require predicted-vs-uniform significance and reduced Oracle–Pred gap vs previous dense-stride reports. |
| Single-GPU script | `python -m avs.experiments.mde_ltl oracle_vs_predicted --mode ave_official --out-dir runs/E0407_oracle_vs_predicted_av_clipdiff_flow_mlp_stride_top1med_thr0p5_s0-9_<ts> --meta-dir data/AVE/meta --processed-dir runs/REAL_AVE_OFFICIAL_20260201-124535/processed --caches-dir runs/REAL_AVE_OFFICIAL_20260201-124535/caches_112_160_224_352_448 --allow-missing --split-train train --split-eval test --train-ids-file data/AVE/meta/download_ok_train_official.txt --eval-ids-file data/AVE/meta/download_ok_test_official.txt --limit-train 3339 --limit-eval 402 --seeds 0,1,2,3,4,5,6,7,8,9 --epochs 5 --batch-size 16 --lr 0.002 --weight-decay 0.0 --train-device cuda:5 --audio-device cuda:5 --eventness-method av_clipdiff_flow_mlp_stride --include-cheap-visual --k 2 --low-res 160 --base-res 224 --high-res 352 --anchor-shift 1 --anchor-conf-metric top1_med --anchor-conf-threshold 0.5 --anchor-select topk --anchor-nms-radius 2 --anchor-nms-strong-gap 0.6 --anchor-window 3 --anchor-base-alloc distance --anchor-high-policy adaptive_v1 --anchor-high-adjacent-dist 1 --anchor-high-gap-threshold 0 --temporal-kernel-size 3` |
| Multi-GPU script | N/A |
| Smoke cmd | `SEEDS=0,1,2 ... (same config)` |
| Full cmd | `SEEDS=0,1,2,3,4,5,6,7,8,9 ... (same config)` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0407_*` |
| Artifacts | `runs/E0407_oracle_vs_predicted_av_clipdiff_flow_mlp_stride_top1med_thr0p5_s0-9_20260206-161749/oracle_vs_predicted.json`, `runs/E0407_oracle_vs_predicted_av_clipdiff_flow_mlp_stride_top1med_thr0p5_s0-9_20260206-161749/metrics_predicted.json`, `runs/E0407_oracle_vs_predicted_av_clipdiff_flow_mlp_stride_top1med_thr0p5_s0-9_20260206-161749/metrics_cheap_visual.json` |
| Results | Full seeds (`0..9`) now make predicted gain significant: `predicted Δ≈+0.01241` with `p≈0.00302`; `oracle_minus_predicted≈0.02900` (improves vs E0403 baseline `0.04187`, but still above the stricter “small-gap” target implied by C0007). Cheap-visual control also improves over uniform (`Δ≈+0.00791`, `p≈0.0307`) but remains below predicted. |


### E0408: Vision efficiency benchmark (tokens/FLOPs/latency calibration)
| Field | Value |
| --- | --- |
| Objective | Fill oral-control gap by producing explicit per-resolution token/FLOPs/latency measurements for the CLIP vision encoder used in AVE-P0. |
| Baseline | N/A (calibration run). |
| Model | `avs.experiments.vision_efficiency` |
| Weights | Random-input benchmark (no HF download; `--vision-pretrained` disabled). |
| Code path | `avs/experiments/vision_efficiency.py`, `avs/vision/vit_flops.py` |
| Params | `resolutions=112,160,224,352,448`, `batch_size=8`, `iters=20`, `device=cuda:5`, `dtype=float16`. |
| Metrics (must save) | `vision_efficiency.json` with `tokens_per_image`, `approx_flops_per_image`, `ms_per_image`. |
| Checks | Ensure FLOPs/tokens rise with resolution and attach artifact to oral checklist / plan evidence. |
| Single-GPU script | `python -m avs.experiments.vision_efficiency --resolutions 112,160,224,352,448 --batch-size 8 --warmup 2 --iters 20 --device cuda:5 --dtype float16 --out-dir runs/E0408_vision_efficiency_<ts>` |
| Multi-GPU script | N/A |
| Smoke cmd | `python -m avs.experiments.vision_efficiency --resolutions 112,224 --batch-size 2 --warmup 1 --iters 5 --device cuda:5 --dtype float16 --out-dir runs/E0408_smoke_vision_efficiency_<ts>` |
| Full cmd | `python -m avs.experiments.vision_efficiency --resolutions 112,160,224,352,448 --batch-size 8 --warmup 2 --iters 20 --device cuda:5 --dtype float16 --out-dir runs/E0408_vision_efficiency_<ts>` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0408_vision_efficiency_*` |
| Artifacts | `runs/E0408_vision_efficiency_20260209-233151/vision_efficiency.json` |
| Results | `runs/E0408_vision_efficiency_20260209-233151/vision_efficiency.json`: per-resolution tokens/FLOPs are explicitly reported and monotonic (`49→100→196→484→784` tokens/image; `3.84e7→8.31e7→1.82e8→5.90e8→1.20e9` FLOPs/image). |


### E0409: Multi-budget Pareto rerun (SEEDS=0..9) — aligned with dense-stride `top1_med thr0.5`
| Field | Value |
| --- | --- |
| Objective | Recompute Oracle/Predicted/Random/Cheap-visual Pareto under the promoted dense-stride config across triads (`112/160/224`, `160/224/352`, `224/352/448`) to judge C0007 on the exact “across budgets” criterion. |
| Baseline | E0330 full (`EVENTNESS=av_clipdiff_mlp`, base `E0223` config). |
| Model | `avs.experiments.mde_ltl pareto_grid` |
| Weights | Official AVE caches; shared dense-stride score cache: `runs/E0405_shared_scores_av_clipdiff_flow_mlp_stride_test402.json`. |
| Code path | `scripts/e0330_mde_pareto_grid_official.sh`, `avs/experiments/mde_ltl.py` |
| Params | `EVENTNESS=av_clipdiff_flow_mlp_stride`, `BASE_CONFIG_JSON=.../ltltop1med_thr0p5_shift1/config.json`, `SEEDS=0..9`, `BUDGET_MODE=auto`, `INCLUDE_CHEAP_VISUAL=1`. |
| Metrics (must save) | `pareto_report.json`, `pareto.png`, per-budget `metrics_*.json`. |
| Checks | Compare `oracle_minus_predicted` profile vs E0330 baseline and test C0007’s cross-budget gap requirement. |
| Smoke cmd | `BASE_CONFIG_JSON=.../ltltop1med_thr0p5_shift1/config.json SCORES_JSON=runs/E0405_shared_scores_av_clipdiff_flow_mlp_stride_test402.json EVENTNESS=av_clipdiff_flow_mlp_stride SEEDS=0,1,2 TRAIN_DEVICE=cuda:5 AUDIO_DEVICE=cuda:5 OUT_DIR=runs/E0409_smoke_pareto_grid_<ts> bash scripts/e0330_mde_pareto_grid_official.sh` |
| Full cmd | `BASE_CONFIG_JSON=.../ltltop1med_thr0p5_shift1/config.json SCORES_JSON=runs/E0405_shared_scores_av_clipdiff_flow_mlp_stride_test402.json EVENTNESS=av_clipdiff_flow_mlp_stride SEEDS=0,1,2,3,4,5,6,7,8,9 TRAIN_DEVICE=cuda:5 AUDIO_DEVICE=cuda:5 OUT_DIR=runs/E0409_pareto_grid_av_clipdiff_flow_mlp_stride_top1med_thr0p5_s0-9_<ts> bash scripts/e0330_mde_pareto_grid_official.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0409_*` |
| Artifacts | `runs/E0409_pareto_grid_av_clipdiff_flow_mlp_stride_top1med_thr0p5_s0-9_20260206-163941/*` |
| Results | Full completed: `runs/E0409_pareto_grid_av_clipdiff_flow_mlp_stride_top1med_thr0p5_s0-9_20260206-163941/pareto_report.json` + `.../pareto.png`. Per-budget predicted-vs-uniform: `112_160_224` (`Δ≈-0.00025`, `p≈0.938`), `160_224_352` (`Δ≈+0.01241`, `p≈0.00302`), `224_352_448` (`Δ≈-0.00674`, `p≈0.099`). Oracle–Pred gaps are `0.02142`, `0.02900`, `0.03012` (mean `≈0.02685`), so this aligned rerun improves mid-budget significance but does not tighten high-budget gaps. |


### E0410: Fusion confirm rerun on dense-stride winner (`audio_concat_uniform` vs `audio_concat_anchored_top2`)
| Field | Value |
| --- | --- |
| Objective | Re-check C0004 on the current best dense-stride config (`E0402 alt`, `top1_med thr0.5`) using official test402 full seeds. |
| Baseline | `audio_concat_uniform` under the same sampling plan and budget. |
| Model | `avs.experiments.ave_p0_fusion_confirm` |
| Weights | Official AVE cache + temporal head training (`SEEDS=0..9`). |
| Code path | `scripts/e0013_ave_fusion_confirm_official.sh`, `avs/experiments/ave_p0_fusion_confirm.py` |
| Params | `BEST_CONFIG_JSON=.../ltltop1med_thr0p5_shift1/config.json`, `SEEDS=0..9`, strict official split ids. |
| Metrics (must save) | `metrics.json` with `audio_concat_uniform`, `audio_concat_anchored_top2`, paired tests. |
| Checks | C0004 gate: `audio_concat_anchored_top2 - audio_concat_uniform >= +0.01` and `p<0.05`. |
| Single-GPU script | `BEST_CONFIG_JSON=<E0402-alt-config> EVENTNESS=energy_stride_max SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:5 TRAIN_DEVICE=cuda:5 OUT_DIR=runs/E0410_fusion_confirm_energy_stride_max_top1med_thr0p5_<ts> bash scripts/e0013_ave_fusion_confirm_official.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `BEST_CONFIG_JSON=<E0402-alt-config> EVENTNESS=energy_stride_max SEEDS=0,1 LIMIT_TRAIN=64 LIMIT_EVAL=32 AUDIO_DEVICE=cuda:5 TRAIN_DEVICE=cuda:5 OUT_DIR=runs/E0410_smoke_fusion_confirm_energy_stride_max_top1med_thr0p5_<ts> bash scripts/e0013_ave_fusion_confirm_official.sh` |
| Full cmd | `BEST_CONFIG_JSON=<E0402-alt-config> EVENTNESS=energy_stride_max SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:5 TRAIN_DEVICE=cuda:5 OUT_DIR=runs/E0410_fusion_confirm_energy_stride_max_top1med_thr0p5_<ts> bash scripts/e0013_ave_fusion_confirm_official.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0410_*` |
| Artifacts | `runs/E0410_fusion_confirm_energy_stride_max_top1med_thr0p5_20260206-180945/metrics.json` |
| Results | C0004 remains unproven: `audio_concat_uniform≈0.71231`, `audio_concat_anchored_top2≈0.70978`, `Δ≈-0.00254`; paired `p≈0.36759` (`paired_ttest.audio_concat_anchored_vs_audio_concat_uniform.p`). |


### E0411: Evidence-alignment rerun on promoted dense-stride config (`E0402 alt`)
| Field | Value |
| --- | --- |
| Objective | Recompute Cov@τ and downstream correlation on the promoted dense-stride config and refresh C0008 evidence. |
| Baseline | Prior E0202 report (`av_clipdiff_mlp`). |
| Model | `avs.experiments.evidence_alignment_report` |
| Weights | Reuse `E0402 alt` full test402 `metrics.json` debug outputs. |
| Code path | `avs/experiments/evidence_alignment_report.py` |
| Params | `in_metrics=runs/E0402_full_test402_av_clipdiff_flow_mlp_stride_alt_top1med_thr0p5_20260206-152012/metrics.json`, `tau_grid=0.3,0.5,0.7`, `delta_s=0`, `top_n=25`. |
| Metrics (must save) | `evidence_alignment.json` with `cov_by_tau` + `corr_by_tau` + top improve/degrade clips. |
| Checks | Keep report schema stable and expose correlation keys for oral failure-bucket linkage. |
| Single-GPU script | `python -m avs.experiments.evidence_alignment_report --in-metrics runs/E0402_full_test402_av_clipdiff_flow_mlp_stride_alt_top1med_thr0p5_20260206-152012/metrics.json --meta-dir data/AVE/meta --out-dir runs/E0411_evidence_alignment_av_clipdiff_flow_mlp_stride_top1med_thr0p5_<ts> --tau-grid 0.3,0.5,0.7 --delta-s 0 --top-n 25` |
| Multi-GPU script | N/A |
| Smoke cmd | `python -m avs.experiments.evidence_alignment_report --in-metrics runs/E0401_quick_test402_av_clipdiff_flow_mlp_stride_alt_top1med_thr0p5_20260206-150837/metrics.json --meta-dir data/AVE/meta --out-dir runs/E0411_smoke_evidence_alignment_<ts>` |
| Full cmd | `python -m avs.experiments.evidence_alignment_report --in-metrics runs/E0402_full_test402_av_clipdiff_flow_mlp_stride_alt_top1med_thr0p5_20260206-152012/metrics.json --meta-dir data/AVE/meta --out-dir runs/E0411_evidence_alignment_av_clipdiff_flow_mlp_stride_top1med_thr0p5_<ts> --tau-grid 0.3,0.5,0.7 --delta-s 0 --top-n 25` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0411_*` |
| Artifacts | `runs/E0411_evidence_alignment_av_clipdiff_flow_mlp_stride_top1med_thr0p5_20260206-182007/evidence_alignment.json` |
| Results | Cov@τ is still low (`mean≈0.0935` for τ=`0.3/0.5/0.7`); downstream correlation remains weak (`pearson≈0.0498`, `spearman≈-0.0029` in `corr_by_tau`), so C0008 is still not proven. |


### E0412: Degradation-accuracy rerun on dense-stride promoted config (shift/noise/silence × α, SEEDS=0..9)
| Field | Value |
| --- | --- |
| Objective | Re-run downstream robustness suite on the promoted dense-stride config and explicitly validate the α lower-bound check in output JSON. |
| Baseline | Prior E0331 full (`av_clipdiff_mlp`). |
| Model | `avs.experiments.degradation_accuracy` |
| Weights | Official AVE cache + promoted config (`top1_med thr0.5`) with dense-stride score cache. |
| Code path | `avs/experiments/degradation_accuracy.py`, `scripts/e0331_degradation_accuracy_official.sh` |
| Params | `EVENTNESS_METHOD=av_clipdiff_flow_mlp_stride`, `SEEDS=0..9`, grid `shift={-0.5,0,0.5}`, `snr={20,10,0}`, `silence={0,0.5}`, `alpha={0,0.5,1}`. |
| Metrics (must save) | `degradation_accuracy.json` with `rows` and `alpha_floor_checks`, plus heatmap plots. |
| Checks | No catastrophic drop below α floor and stable anchored-vs-uniform behavior under perturbation grid. |
| Single-GPU script | `BEST_CONFIG_JSON=.../ltltop1med_thr0p5_shift1/config.json SCORES_JSON=runs/E0405_shared_scores_av_clipdiff_flow_mlp_stride_test402.json EVENTNESS_METHOD=av_clipdiff_flow_mlp_stride SEEDS=0,1,2,3,4,5,6,7,8,9 TRAIN_DEVICE=cuda:5 OUT_DIR=runs/E0412_degradation_accuracy_av_clipdiff_flow_mlp_stride_top1med_thr0p5_s0-9_<ts> bash scripts/e0331_degradation_accuracy_official.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `BEST_CONFIG_JSON=.../ltltop1med_thr0p5_shift1/config.json SCORES_JSON=runs/E0405_shared_scores_av_clipdiff_flow_mlp_stride_test402.json EVENTNESS_METHOD=av_clipdiff_flow_mlp_stride SEEDS=0,1 LIMIT_TRAIN=64 LIMIT_EVAL=32 EPOCHS=1 TRAIN_DEVICE=cuda:5 OUT_DIR=runs/E0412_smoke_degradation_accuracy_<ts> bash scripts/e0331_degradation_accuracy_official.sh` |
| Full cmd | `BEST_CONFIG_JSON=.../ltltop1med_thr0p5_shift1/config.json SCORES_JSON=runs/E0405_shared_scores_av_clipdiff_flow_mlp_stride_test402.json EVENTNESS_METHOD=av_clipdiff_flow_mlp_stride SEEDS=0,1,2,3,4,5,6,7,8,9 TRAIN_DEVICE=cuda:5 OUT_DIR=runs/E0412_degradation_accuracy_av_clipdiff_flow_mlp_stride_top1med_thr0p5_s0-9_<ts> bash scripts/e0331_degradation_accuracy_official.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0412_*` |
| Artifacts | `runs/E0412_degradation_accuracy_av_clipdiff_flow_mlp_stride_top1med_thr0p5_s0-9_20260206-182443/degradation_accuracy.json`, `runs/E0412_degradation_accuracy_av_clipdiff_flow_mlp_stride_top1med_thr0p5_s0-9_20260206-182443/degradation_plots/*.png` |
| Results | Full grid complete (`rows=54`). `alpha_floor_checks`: `num_pass=54`, `num_fail=0`, `min_margin≈+0.01766` under rule `anchored_top2_mean >= alpha * uniform_mean`. Mean `anchored-uniform` across perturbations is positive for all α (`α=0:≈+0.00766`, `α=0.5:≈+0.01300`, `α=1:≈+0.01766`). |


### E0413: EPIC-SOUNDS downstream rerun (real local videos, strict equal-budget)
| Field | Value |
| --- | --- |
| Objective | Execute C0005-required EPIC-SOUNDS downstream validation under strict equal-budget setup and report mAP/macro-F1 deltas. |
| Baseline | `uniform` and `random` under the same `max_steps × base_res` budget. |
| Model | `avs.experiments.epic_sounds_video_cls` via `scripts/e0100_epic_video_cls_local.sh` |
| Weights | Local CLIP encoder + lightweight multi-label head. |
| Code path | `scripts/e0100_epic_video_cls_local.sh`, `avs/experiments/epic_sounds_video_cls.py` |
| Params | `SELECTION in {audio_anchored,uniform,random}`, `EVENTNESS=energy`, `SEEDS>=3` (e.g., `0,1,2`), `ALLOW_MISSING_VIDEOS=1`, `MAX_SECONDS=120` (with fixed `max_steps=120` budget). |
| Metrics (must save) | `metrics.json` with per-seed `results_by_seed`, `summary` (`mean/std`) for `mAP`/`macro_f1@0.5`, and baseline deltas (computed across selections). |
| Checks | Must run on real EPIC-SOUNDS val videos (not synthetic smoke). |
| Single-GPU script | `SELECTION=audio_anchored EVENTNESS=energy SEEDS=0,1,2 MAX_SECONDS=120 ALLOW_MISSING_VIDEOS=1 LIMIT_TRAIN_VIDEOS=495 LIMIT_VAL_VIDEOS=137 MIN_TRAIN_VIDEOS=16 MIN_VAL_VIDEOS=16 TRAIN_DEVICE=cuda:0 CLIP_DEVICE=cuda:0 OUT_DIR=runs/E0413_epic_video_cls_local_audio_anchored_full_ms120_<ts> bash scripts/e0100_epic_video_cls_local.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `python -m avs.smoke epic_sounds_video_cls_synth` |
| Full cmd | `SELECTION in {audio_anchored,uniform,random} EVENTNESS=energy SEEDS=0,1,2 MAX_SECONDS=120 ALLOW_MISSING_VIDEOS=1 LIMIT_TRAIN_VIDEOS=495 LIMIT_VAL_VIDEOS=137 MIN_TRAIN_VIDEOS=16 MIN_VAL_VIDEOS=16 bash scripts/e0100_epic_video_cls_local.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/smoke_20260206-184253/smoke.json`, `runs/E0413_epic_video_cls_local_*_full_ms120_*/metrics.json` |
| Artifacts | `runs/smoke_20260206-184253/epic_sounds_video_cls_synth/metrics.json`, `runs/E0413_epic_video_cls_local_audio_anchored_full_ms120_20260207-171637/metrics.json`, `runs/E0413_epic_video_cls_local_uniform_full_ms120_20260207-172545/metrics.json`, `runs/E0413_epic_video_cls_local_random_full_ms120_20260207-173208/metrics.json` |
| Results | Real local run completed on available EPIC videos (`num_train_videos=17`, `num_val_videos=16`, `mp4_count=33`, budget `max_steps=120`, `max_seconds=120`). Metrics: anchored (`mAP≈0.4450`, `macro_f1≈0.3023`), uniform (`mAP≈0.4672`, `macro_f1≈0.3141`), random (`mAP≈0.4105`, `macro_f1≈0.3382`). Deltas: anchored-uniform `mAP≈-0.0221`, `macro_f1≈-0.0118`; anchored-random `mAP≈+0.0346`, `macro_f1≈-0.0359`. |


### E0501: Dataset integrity audit (AVE/EPIC probe+decode health)
| Field | Value |
| --- | --- |
| Objective | Add a reproducible data-integrity gate to explain/avoid silent decode corruption before concluding method failure. |
| Baseline | Existing dataset presence check (`scripts/datasets/verify_all.sh`) without decode/probe quality signals. |
| Model | `avs.experiments.dataset_integrity_audit` |
| Weights | N/A |
| Code path | `avs/experiments/dataset_integrity_audit.py`, `scripts/e0501_dataset_integrity_audit.sh` |
| Params | `pattern=*.mp4`, `decode_check in {none,sampled,full}`, `limit`, `decode_limit`. |
| Metrics (must save) | `dataset_integrity_audit.json` with `probe_ok`, `probe_failed`, `decode_failed`, `corrupted_files`, per-file errors. |
| Checks | Catch broken/corrupted files before training/eval; keep an auditable artifact. |
| Single-GPU script | `LIMIT=8 DECODE_CHECK=none bash scripts/e0501_dataset_integrity_audit.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `python -m avs.smoke dataset_integrity_audit` |
| Full cmd | `DECODE_CHECK=sampled DECODE_LIMIT=64 bash scripts/e0501_dataset_integrity_audit.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/smoke_20260207-151400/*`, `runs/E0501_dataset_integrity_*/*` |
| Artifacts | `runs/E0501_dataset_integrity_20260209-042215/index.json`, `runs/E0501_dataset_integrity_20260209-042215/ave/dataset_integrity_audit.json`, `runs/E0501_dataset_integrity_20260209-042215/epic/dataset_integrity_audit.json`, `artifacts/experiments/E0501/run.log` |
| Results | Full sampled-decode audit completed: AVE `probe_ok=4097/4097`, `decode_checked=64`, `corrupted_files=0`; EPIC `probe_ok=632/632`, `decode_checked=64`, `corrupted_files=0`. |


### E0502: Root-cause aggregation report (unproven-claim diagnosis)
| Field | Value |
| --- | --- |
| Objective | Produce a single machine-readable diagnosis artifact explaining why current claim is still unproven and mapping each failure mode to concrete fix directions. |
| Baseline | Manual reading across `metrics.json` / `oracle_vs_predicted.json` / evidence / degradation artifacts. |
| Model | `avs.experiments.root_cause_report` |
| Weights | N/A |
| Code path | `avs/experiments/root_cause_report.py`, `scripts/e0502_root_cause_report.sh` |
| Params | `target_delta`, `target_p`, `oracle_gap_threshold`, `fallback_threshold`, `coverage_threshold`, `corr_threshold`. |
| Metrics (must save) | `root_cause_report.json` (`reasons`, `priority_queue`, `solution_pool`, `proven`). |
| Checks | Reasons must be deterministic from artifacts; includes actionable solution list by bucket. |
| Single-GPU script | `bash scripts/e0502_root_cause_report.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `python -m avs.smoke root_cause_report` |
| Full cmd | `METRICS_JSON=<metrics.json> ORACLE_JSON=<oracle_vs_predicted.json> EVIDENCE_JSON=<evidence_alignment.json> DEGRADATION_JSON=<degradation_accuracy.json> bash scripts/e0502_root_cause_report.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/smoke_20260207-151416/*`, `runs/E0502_root_cause_report_*/*` |
| Artifacts | `runs/smoke_20260207-151416/root_cause_report/root_cause_report.json`, `runs/E0502_root_cause_report_20260207-151428/root_cause_report.json` |
| Results | Real report marks claim as unproven and surfaces root causes `R_TARGET_DELTA`, `R_ORACLE_GAP`, `R_ALIGNMENT_WEAK`, `R_EPIC_DOWNSTREAM` with prioritized solution pool. |


### E0503: Dense-stride gate sweep (val-only, pre-registered gate selection)
| Field | Value |
| --- | --- |
| Objective | Select and freeze one confidence gate on val for dense-stride Stage-1, avoiding tune-on-test. |
| Baseline | Fixed gate from prior runs (`top1_med thr0.5`) without current-val re-selection. |
| Model | `avs.experiments.mde_ltl gate_sweep` |
| Weights | Official AVE caches + standard P0 training head. |
| Code path | `scripts/e0503_gate_sweep_dense_stride.sh`, `avs/experiments/mde_ltl.py` |
| Params | `eventness=av_clipdiff_flow_mlp_stride`, `gate_metric`, `gate_thresholds`, `SEEDS`, `limit_train/eval`. |
| Metrics (must save) | `gate_sweep.json`, `best_gate.json`, per-threshold `metrics_gate_*.json`. |
| Checks | Select gate by deterministic rule `max(delta_mean), tie-break by min(p)`. |
| Single-GPU script | `SEEDS=0,1,2 LIMIT_TRAIN=3339 LIMIT_EVAL=402 bash scripts/e0503_gate_sweep_dense_stride.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 GATE_THRESHOLDS=0.5 bash scripts/e0503_gate_sweep_dense_stride.sh` |
| Full cmd | `SEEDS=0,1,2,3,4,5,6,7,8,9 LIMIT_TRAIN=3339 LIMIT_EVAL=402 GATE_THRESHOLDS=0.4,0.5,0.6,0.7 bash scripts/e0503_gate_sweep_dense_stride.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0503_gate_sweep_dense_stride_*/*` |
| Artifacts | `runs/E0503_gate_sweep_dense_stride_smoke_20260207-151733/gate_sweep.json`, `runs/E0503_gate_sweep_dense_stride_smoke_20260207-151733/best_gate.json`, `runs/E0503_gate_sweep_dense_stride_full_20260207-153210/gate_sweep.json`, `runs/E0503_gate_sweep_dense_stride_full_20260207-153210/best_gate.json` |
| Results | Full val402 gate sweep completed (`rows=4`, thresholds=`0.4/0.5/0.6/0.7`). Selected gate is stable at `top1_med@0.5`: `anchored≈0.74526` vs `uniform≈0.73214` (`Δ≈+0.01312`, `p≈0.0174`), `oracle_minus_predicted≈0.03746`, `fallback_used_frac≈0.5835`. |


### E0504: Dense-stride Oracle→Predicted multi-budget gap grid
| Field | Value |
| --- | --- |
| Objective | Recompute cross-budget Oracle→Predicted gap under dense-stride defaults for C0007 tightening. |
| Baseline | Earlier E0409 aligned run; this entry standardizes a dedicated rerunnable wrapper. |
| Model | `avs.experiments.mde_ltl pareto_grid` |
| Weights | Official AVE caches + base config from E0400c/E0400b. |
| Code path | `scripts/e0504_oracle_pred_gap_grid_dense_stride.sh`, `scripts/e0330_mde_pareto_grid_official.sh` |
| Params | `TRIADS`, `SEEDS`, `BUDGET_MODE`, `INCLUDE_CHEAP_VISUAL`, `BASE_CONFIG_JSON`, `SCORES_JSON`. |
| Metrics (must save) | `pareto_report.json`, `pareto.png`, `metrics_predicted_*`, `metrics_cheap_visual_*`. |
| Checks | Must report triad-wise predicted/oracle points under equal-budget protocol. |
| Single-GPU script | `SEEDS=0,1,2 LIMIT_TRAIN=3339 LIMIT_EVAL=402 bash scripts/e0504_oracle_pred_gap_grid_dense_stride.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 TRIADS='112,160,224;160,224,352' bash scripts/e0504_oracle_pred_gap_grid_dense_stride.sh` |
| Full cmd | `SEEDS=0,1,2,3,4,5,6,7,8,9 LIMIT_TRAIN=3339 LIMIT_EVAL=402 TRIADS='112,160,224;160,224,352;224,352,448' bash scripts/e0504_oracle_pred_gap_grid_dense_stride.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0504_oracle_pred_gap_grid_dense_stride_*/*` |
| Artifacts | `runs/E0504_oracle_pred_gap_grid_dense_stride_smoke_20260207-151822/pareto_report.json`, `runs/E0504_oracle_pred_gap_grid_dense_stride_smoke_20260207-151822/pareto.png`, `runs/E0504_oracle_pred_gap_grid_dense_stride_full_20260207-155721/pareto_report.json`, `runs/E0504_oracle_pred_gap_grid_dense_stride_full_20260207-155721/pareto.png` |
| Results | Full triad+seed rerun completed (`SEEDS=0..9`). Predicted-vs-uniform by triad: `112_160_224: Δ≈-0.00025 (p≈0.938)`, `160_224_352: Δ≈+0.01241 (p≈0.00302)`, `224_352_448: Δ≈-0.00674 (p≈0.099)`. Oracle-predicted gaps: `0.02142`, `0.02900`, `0.03012` (mean `≈0.02685`), confirming remaining Stage-1 reliability gap. |


### E0505: Dense-stride degradation-accuracy rerun wrapper
| Field | Value |
| --- | --- |
| Objective | Standardize rerunnable downstream degradation-accuracy execution for dense-stride promoted config. |
| Baseline | Earlier E0412 one-off command chain. |
| Model | `avs.experiments.degradation_accuracy` via `scripts/e0331_degradation_accuracy_official.sh` |
| Weights | Official AVE caches + dense-stride base config + shared score cache. |
| Code path | `scripts/e0505_degradation_accuracy_dense_stride.sh`, `scripts/e0331_degradation_accuracy_official.sh` |
| Params | `SEEDS`, `SHIFT_GRID`, `SNR_GRID`, `SILENCE_GRID`, `ALPHA_GRID`, `BEST_CONFIG_JSON`, `SCORES_JSON`. |
| Metrics (must save) | `degradation_accuracy.json` + optional plots. |
| Checks | Include `alpha_floor_checks` and row-wise deltas for robustness audit trail. |
| Single-GPU script | `SEEDS=0,1,2 LIMIT_TRAIN=3339 LIMIT_EVAL=402 bash scripts/e0505_degradation_accuracy_dense_stride.sh` |
| Multi-GPU script | N/A |
| Smoke cmd | `LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 SHIFT_GRID='0' SNR_GRID='20' SILENCE_GRID='0' ALPHA_GRID='0,0.5' bash scripts/e0505_degradation_accuracy_dense_stride.sh` |
| Full cmd | `SEEDS=0,1,2,3,4,5,6,7,8,9 LIMIT_TRAIN=3339 LIMIT_EVAL=402 SHIFT_GRID='-0.5,0,0.5' SNR_GRID='20,10,0' SILENCE_GRID='0,0.5' ALPHA_GRID='0,0.5,1' bash scripts/e0505_degradation_accuracy_dense_stride.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `runs/E0505_degradation_accuracy_dense_stride_*/*` |
| Artifacts | `runs/E0505_degradation_accuracy_dense_stride_smoke_20260207-151846/degradation_accuracy.json`, `runs/E0505_degradation_accuracy_dense_stride_full_20260207-161213/degradation_accuracy.json`, `runs/E0505_degradation_accuracy_dense_stride_full_20260207-161213/degradation_plots/*.png` |
| Results | Full perturbation grid completed (`rows=54`, `shift∈{-0.5,0,0.5}`, `snr∈{20,10,0}`, `silence∈{0,0.5}`, `alpha∈{0,0.5,1}`). `alpha_floor_checks`: `num_fail=0`, `min_margin≈+0.01766` (rule `anchored_top2_mean >= alpha * uniform_mean`). Mean `anchored-uniform` by alpha: `α=0:≈+0.00766`, `α=0.5:≈+0.01300`, `α=1:≈+0.01766`. |

### E0510: Stage-1 sweep on val402 (cheap supervised A+cheapV eventness; `av_basic_mlp`)
| Field | Value |
| --- | --- |
| Objective | Try a lightweight, supervised Stage-1 scoring backend: audio basic features + a cheap frame-diff scalar (`EVENTNESS=av_basic_mlp`). Run the standard official val402 sweep under `candidate_set=ltl_top1med_norm_v1` to test whether this direction is competitive for C0003 (+2% on test402). |
| Code path | `avs/experiments/ave_p0_sweep.py` (`_compute_scores_by_clip`), `avs/experiments/ave_p0.py` (`_train_audio_basic_mlp_eventness`), `avs/vision/cheap_eventness.py` (`frame_diff_eventness`) |
| Params | `EVENTNESS=av_basic_mlp`, `CANDIDATE_SET=ltl_top1med_norm_v1`, `SEEDS=0..2`, `AUDIO_DEVICE=cpu`, `TRAIN_DEVICE=cuda:3` |
| Full cmd | `SEEDS=0,1,2 EVENTNESS=av_basic_mlp CANDIDATE_SET=ltl_top1med_norm_v1 AUDIO_DEVICE=cpu TRAIN_DEVICE=cuda:3 bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` |
| Logs | `artifacts/experiments/E0510_av_basic_mlp_val402/run.log` |
| Artifacts | `runs/E0510_ave_p0_sweep_official_val_av_basic_mlp_ltl_top1med_norm_v1_20260210-161514/sweep_summary.json`, `runs/E0510_ave_p0_sweep_official_val_av_basic_mlp_ltl_top1med_norm_v1_20260210-161514/best_config.json`, `runs/E0510_ave_p0_sweep_official_val_av_basic_mlp_ltl_top1med_norm_v1_20260210-161514/eventness_scores.json` |
| Results | Best val402 config is near-0: `runs/E0510_ave_p0_sweep_official_val_av_basic_mlp_ltl_top1med_norm_v1_20260210-161514/sweep_summary.json` (best=`ltltop1medn_thr0p7_shift1`, `Δ≈+0.00183`, `p≈0.677`). Conclusion: not competitive; stop before test402. |

---

### E0600: IntentQA VLM evaluation under budgeted frame selection
| Field | Value |
| --- | --- |
| Objective | Evaluate long-video QA (MCQ) accuracy under a fixed frame budget, comparing `{uniform,random,audio,cheap_visual,fused,ql2l_*}` selection methods. |
| Dataset | IntentQA (CSV + videos). |
| Model | VLM: Qwen2-VL (default) or compatible model name via `--model-name`. |
| Code path | `avs/experiments/intentqa_vlm_eval.py`, `scripts/e0600_intentqa_vlm_eval.sh` |
| Params | `SPLIT`, `LIMIT`, `METHODS`, `B_FRAMES`, `MAX_SECONDS`, `SEED`, `STRATEGY`, `MODEL_NAME`, `DEVICE`, `DTYPE`, `QL2L_CLAP_DEVICE`, `QL2L_ASR_DEVICE` |
| Metrics (must save) | `metrics.json` (per-method acc/invalid + bootstrap CI deltas vs uniform), `predictions.jsonl` (per-item rows). |
| Smoke cmd | `python -m avs.experiments.intentqa_vlm_eval --help` |
| Full cmd | `OUT_DIR=runs/E0600_intentqa_vlm_eval_$(date +%Y%m%d-%H%M%S) SPLIT=val LIMIT=256 B_FRAMES=16 MAX_SECONDS=120 SEED=0 STRATEGY=ppl DEVICE=cuda:1 DTYPE=bfloat16 QL2L_CLAP_DEVICE=cuda:2 QL2L_ASR_DEVICE=cpu ALLOW_MISSING_VIDEOS=1 MIN_ITEMS=250 bash scripts/e0600_intentqa_vlm_eval.sh` |
| Outputs | `runs/E0600_intentqa_vlm_eval_*/metrics.json`, `runs/E0600_intentqa_vlm_eval_*/predictions.jsonl` |
| Artifacts | Full real run: `runs/E0600_intentqa_vlm_eval_full_20260210-041911/metrics.json`, `runs/E0600_intentqa_vlm_eval_full_20260210-041911/predictions.jsonl`, `runs/E0600_intentqa_vlm_eval_full_20260210-041911/preprocess_meta.json` (log: `artifacts/experiments/E0600_full_ppl_rerun2/run.log`). Synthetic smoke: `runs/E0600_intentqa_vlm_eval_20260209-035602/metrics.json`, `runs/E0600_intentqa_vlm_eval_20260209-035602/predictions.jsonl`. |
| Results | Full val run (n=253; seed=0; `budget_frames=16`, `max_seconds=120`, `strategy=ppl`; invalid_rate=0 for all methods). Acc: uniform=0.9447, random=0.9328, audio=0.9447, cheap_visual=0.9526, fused=0.9407, ql2l_clap=0.9486, ql2l_asr_bm25=0.9407. Δ vs uniform (bootstrap 95% CI; n_boot=300): cheap_visual +0.0079 [-0.0119,+0.0277], ql2l_clap +0.0040 [-0.0158,+0.0258], random -0.0119 [-0.0435,+0.0139]. |
| Notes | Q-L2L backends cache artifacts under each processed video dir (e.g. `processed/<vid>/q_l2l/*`) to keep reruns deterministic. |

### E0601: IntentQA faithfulness proxy (delete-and-predict)
| Field | Value |
| --- | --- |
| Objective | Measure a delete-and-predict proxy: remove the anchor-selected seconds and replace with a budget-matched uniform schedule, then measure accuracy drop and prediction change rate. |
| Dataset | IntentQA (CSV + videos). |
| Model | VLM: Qwen2-VL (default). |
| Code path | `avs/experiments/intentqa_faithfulness.py`, `scripts/e0601_intentqa_faithfulness.sh` |
| Params | `SPLIT`, `LIMIT`, `METHOD`, `B_FRAMES`, `MAX_SECONDS`, `SEED`, `STRATEGY`, model/QL2L device knobs |
| Metrics (must save) | `faithfulness.json` (acc, acc_deleted, acc_drop, pred_change_rate), `rows.jsonl` (per-item). |
| Smoke cmd | `python -m avs.experiments.intentqa_faithfulness --help` |
| Full cmd | `OUT_DIR=runs/E0601_intentqa_faithfulness_$(date +%Y%m%d-%H%M%S) SPLIT=val LIMIT=256 METHOD=ql2l_clap B_FRAMES=16 MAX_SECONDS=120 SEED=0 STRATEGY=ppl DEVICE=cuda:1 DTYPE=bfloat16 QL2L_CLAP_DEVICE=cuda:2 QL2L_ASR_DEVICE=cpu ALLOW_MISSING_VIDEOS=1 MIN_ITEMS=250 bash scripts/e0601_intentqa_faithfulness.sh` |
| Outputs | `runs/E0601_intentqa_faithfulness_*/faithfulness.json`, `runs/E0601_intentqa_faithfulness_*/rows.jsonl` |
| Artifacts | Full real run: `runs/E0601_intentqa_faithfulness_full_20260210-061137/faithfulness.json`, `runs/E0601_intentqa_faithfulness_full_20260210-061137/rows.jsonl`, `runs/E0601_intentqa_faithfulness_full_20260210-061137/preprocess_meta.json` (log: `artifacts/experiments/E0601_full_ql2l_clap_ppl/run.log`). Synthetic smoke: `runs/E0601_intentqa_faithfulness_20260209-035635/faithfulness.json`, `runs/E0601_intentqa_faithfulness_20260209-035635/rows.jsonl`. |
| Results | Full val run (n=253; method=ql2l_clap; seed=0; `budget_frames=16`, `max_seconds=120`, `strategy=ppl`; invalid_rate=0): acc=0.9486, acc_deleted=0.9486, acc_drop=0.0000, pred_change_rate=0.0316. |

### E0602: EgoSchema prediction generation under budgeted frame selection
| Field | Value |
| --- | --- |
| Objective | Generate EgoSchema predictions under a fixed frame budget to compare selection methods and/or upload predictions for external evaluation (depending on dataset split/labels available). |
| Dataset | EgoSchema (HF metadata + extracted videos). |
| Model | VLM: Qwen2-VL (default). |
| Code path | `avs/experiments/egoschema_vlm_eval.py`, `scripts/e0602_egoschema_predict.sh` |
| Params | `CONFIG`, `SPLIT`, `LIMIT`, `METHODS`, `B_FRAMES`, `MAX_SECONDS`, `SEED`, `STRATEGY`, model/QL2L device knobs |
| Metrics (must save) | `metrics.json`, `predictions.jsonl`, `preprocess_meta.json` |
| Smoke cmd | `python -m avs.experiments.egoschema_vlm_eval --help` |
| Full cmd | `OUT_DIR=runs/E0602_egoschema_predict_$(date +%Y%m%d-%H%M%S) CONFIG=Subset SPLIT=test LIMIT=256 B_FRAMES=16 MAX_SECONDS=120 SEED=0 STRATEGY=ppl DEVICE=cuda:1 DTYPE=bfloat16 QL2L_CLAP_DEVICE=cuda:2 QL2L_ASR_DEVICE=cpu bash scripts/e0602_egoschema_predict.sh` |
| Outputs | `runs/E0602_egoschema_predict_*/metrics.json`, `runs/E0602_egoschema_predict_*/predictions.jsonl` |
| Artifacts | Full real run: `runs/E0602_egoschema_eval_subset_full_20260210-064250/metrics.json`, `runs/E0602_egoschema_eval_subset_full_20260210-064250/predictions.jsonl`, `runs/E0602_egoschema_eval_subset_full_20260210-064250/preprocess_meta.json` (log: `artifacts/experiments/E0602_full_subset_ppl/run.log`). |
| Results | Full Subset test run (n=256; seed=0; `budget_frames=16`, `max_seconds=120`, `strategy=ppl`; invalid_rate=0): uniform acc=0.5859, ql2l_clap acc=0.5352, ql2l_asr_bm25 acc=0.5469. |
| Notes | Requires extracted videos: `bash scripts/datasets/egoschema_extract_videos.sh` (from `data/hf_repos/egoschema/videos_chunked_*.zip`). If you want to generate predictions without labels (for external eval), use `CONFIG=MC`. |

### E0603: Stage-2 solver ablation (greedy vs Lagrangian knapsack)
| Field | Value |
| --- | --- |
| Objective | Compare the baseline greedy Stage-2 allocator vs a Lagrangian-relaxation multiple-choice knapsack solver on synthetic window sets. |
| Code path | `avs/experiments/allocator_solver_ablation.py`, `scripts/e0603_allocator_ablation.sh` |
| Params | `SEED`, `NUM_WINDOWS`, `BUDGET` |
| Metrics (must save) | `allocator_ablation.json` (utilities, costs, deltas, per-window allocations). |
| Smoke cmd | `OUT_DIR=runs/E0603_allocator_ablation_smoke_$(date +%Y%m%d-%H%M%S) NUM_WINDOWS=6 BUDGET=8000 SEED=0 bash scripts/e0603_allocator_ablation.sh` |
| Outputs | `runs/E0603_allocator_ablation_*/allocator_ablation.json` |
| Artifacts | `runs/E0603_allocator_ablation_20260209-035300/allocator_ablation.json` |
| Results | Lagrangian knapsack achieves slightly higher utility than greedy at similar/better budget usage (`delta.utility≈+0.0195`, `delta.cost≈+204.6` under `budget=20000`, `num_windows=12`, `seed=0`). |

### E0604: IntentQA VLM evaluation under budgeted frame selection (multi-seed)
| Field | Value |
| --- | --- |
| Objective | Upgrade E0600 from a single-seed run to a small multi-seed evaluation (SEEDS=0,1) under identical budgets/methods. |
| Dataset | IntentQA (CSV + videos). |
| Model | VLM: Qwen2-VL (default). |
| Code path | `avs/experiments/intentqa_vlm_eval.py`, `scripts/e0600_intentqa_vlm_eval.sh` |
| Params | Same as E0600, plus `SEED∈{0,1}`. |
| Metrics (must save) | Per-seed `metrics.json` + `predictions.jsonl` + `preprocess_meta.json`. |
| Full cmd | See the checklist entry (2 exact commands with fixed OUT_DIRs). |
| Smoke | [ ] |
| Full | [x] |
| Logs | `artifacts/experiments/E0604_val_s*/run.log` |
| Artifacts | `runs/E0604_intentqa_vlm_eval_val_s*_20260210-125048/*` |
| Results | Seed0 full METHODS (`uniform,random,audio,cheap_visual,fused,ql2l_clap,ql2l_asr_bm25`): uniform acc=0.944664, random=0.932806, audio=0.944664, cheap_visual=0.952569, fused=0.940711, ql2l_clap=0.948617, ql2l_asr_bm25=0.940711 (invalid_rate=0; n=253). Seed1 reduced METHODS (`uniform,random,cheap_visual,ql2l_clap`): uniform acc=0.944664, random=0.936759, cheap_visual=0.952569, ql2l_clap=0.948617 (invalid_rate=0; n=253). |

### E0605: EgoSchema Subset VLM evaluation under budgeted frame selection (full n=500)
| Field | Value |
| --- | --- |
| Objective | Upgrade E0602 from Subset(256) to the full labeled Subset split (n=500) and run multiple seeds for robustness. |
| Dataset | EgoSchema (HF repo clone + extracted videos). |
| Model | VLM: Qwen2-VL (default). |
| Code path | `avs/experiments/egoschema_vlm_eval.py`, `scripts/e0602_egoschema_predict.sh` |
| Params | `CONFIG=Subset`, `SPLIT=test`, `LIMIT=0` (full), `SEED∈{0,1}`. |
| Metrics (must save) | Per-seed `metrics.json` + `predictions.jsonl` + `preprocess_meta.json`. |
| Full cmd | See the checklist entry (2 exact commands with fixed OUT_DIRs). |
| Smoke | [ ] |
| Full | [x] |
| Logs | `artifacts/experiments/E0605_subset500_s0/run.log` (partial), `artifacts/experiments/E0605_subset500_s0_resume1/run.log` (resume), `artifacts/experiments/E0605_subset500_s1/run.log` |
| Artifacts | `runs/E0605_egoschema_eval_subset500_s0_20260210-125048/*`, `runs/E0605_egoschema_eval_subset500_s1_20260210-183504/*` |
| Results | Full Subset test run (n=500; seeds=0,1; invalid_rate=0): uniform acc=0.5880, ql2l_clap=0.5480, ql2l_asr_bm25=0.5560 (identical across seeds). Artifacts: `runs/E0605_egoschema_eval_subset500_s0_20260210-125048/metrics.json`, `runs/E0605_egoschema_eval_subset500_s1_20260210-183504/metrics.json`. |

### E0607: IntentQA faithfulness proxy (delete-and-predict; seed=1)
| Field | Value |
| --- | --- |
| Objective | Extend E0601 with an additional seed to verify stability of the faithfulness proxy metrics under the same budgets/method. |
| Dataset | IntentQA (CSV + videos). |
| Model | VLM: Qwen2-VL (default). |
| Code path | `avs/experiments/intentqa_faithfulness.py`, `scripts/e0601_intentqa_faithfulness.sh` |
| Params | Same as E0601, plus `SEED=1`. |
| Metrics (must save) | `faithfulness.json` + `rows.jsonl` + `preprocess_meta.json`. |
| Full cmd | See the checklist entry (1 exact command with fixed OUT_DIR). |
| Smoke | [ ] |
| Full | [x] |
| Logs | `artifacts/experiments/E0607_intentqa_faithfulness_val_s1/run.log` |
| Artifacts | `runs/E0607_intentqa_faithfulness_val_s1_20260210-194732/*` |
| Results | Matches seed0: accuracy=0.948617, accuracy_deleted=0.948617, acc_drop=0.000000, pred_change_rate=0.031621, invalid_rate=0. |

### E0608: EgoSchema prediction generation under budgeted frame selection (seed=1; Subset n=256)
| Field | Value |
| --- | --- |
| Objective | Extend E0602 with an additional seed to confirm stability on the labeled Subset split under the same budgets/methods. |
| Dataset | EgoSchema (HF repo clone + extracted videos). |
| Model | VLM: Qwen2-VL (default). |
| Code path | `avs/experiments/egoschema_vlm_eval.py`, `scripts/e0602_egoschema_predict.sh` |
| Params | `CONFIG=Subset`, `SPLIT=test`, `LIMIT=256`, `SEED=1`. |
| Metrics (must save) | `metrics.json` + `predictions.jsonl` + `preprocess_meta.json`. |
| Full cmd | See the checklist entry (1 exact command with fixed OUT_DIR). |
| Smoke | [ ] |
| Full | [x] |
| Logs | `artifacts/experiments/E0608_egoschema_eval_subset256_s1/run.log` |
| Artifacts | `runs/E0608_egoschema_eval_subset256_s1_20260210-201700/*` |
| Results | Matches seed0: uniform acc=0.5859, ql2l_clap=0.5352, ql2l_asr_bm25=0.5469 (invalid_rate=0; n=256). |

### E0609: IntentQA VLM evaluation under budgeted frame selection (+ql2l_clip baseline)
| Field | Value |
| --- | --- |
| Objective | Add a strong query-aware *visual* baseline (`ql2l_clip`: query→CLIP text-image relevance) to the IntentQA long-video QA evaluation. |
| Dataset | IntentQA (val). |
| Model | VLM: Qwen2-VL (default). |
| Code path | `avs/experiments/intentqa_vlm_eval.py`, `scripts/e0600_intentqa_vlm_eval.sh` |
| Params | Same as E0600, plus `METHODS+=ql2l_clip` and `QL2L_CLIP_DEVICE`. |
| Metrics (must save) | `metrics.json` + `predictions.jsonl` + `preprocess_meta.json` |
| Full cmd | See the checklist entry (1 exact command with fixed OUT_DIR). |
| Smoke | [ ] |
| Full | [x] |
| Logs | `artifacts/experiments/E0609/run.log` |
| Artifacts | `runs/E0609_intentqa_vlm_eval_val_clip_20260211-011407/*` |
| Results | (n=253; seed=0; invalid_rate=0): uniform acc=0.944664; ql2l_clap acc=0.948617; cheap_visual acc=0.952569; ql2l_clip acc=0.936759 (Δ=-0.007905). |
| Notes | skipped_videos=1. Keep as negative-but-clean baseline coverage (no cherry-pick). |

### E0617: IntentQA language-bias baseline (uniform vs text_only; val n=253)
| Field | Value |
| --- | --- |
| Objective | Quantify how much of IntentQA can be answered from the question alone by running the VLM with **no frames** (`text_only`) vs the standard uniform frame baseline. |
| Dataset | IntentQA (val). |
| Model | VLM: Qwen2-VL (default). |
| Code path | `avs/experiments/intentqa_vlm_eval.py`, `scripts/e0600_intentqa_vlm_eval.sh` |
| Params | `METHODS=uniform,text_only`, `B_FRAMES=16`, `MAX_SECONDS=120`, `SEED=0` (other knobs match E0600). |
| Metrics (must save) | `metrics.json` + `predictions.jsonl` + `preprocess_meta.json` |
| Full cmd | See the checklist entry (1 exact command with fixed OUT_DIR). |
| Smoke | [ ] |
| Full | [x] |
| Logs | `artifacts/experiments/E0617/run.log` |
| Artifacts | `runs/E0617_intentqa_vlm_eval_val_text_only_20260211-053301/*` |
| Results | uniform acc=0.944664; text_only acc=0.664032 (Δ=-0.280632; invalid_rate=0; skipped_videos=1). |

### E0606: EgoSchema VLM evaluation under budgeted frame selection (+ql2l_clip baseline; full Subset n=500)
| Field | Value |
| --- | --- |
| Objective | Add `ql2l_clip` to the full labeled EgoSchema Subset split (n=500) to test whether CLIP query relevance helps on EgoSchema (where `ql2l_clap` and `ql2l_asr_bm25` underperform uniform). |
| Dataset | EgoSchema (Subset config; test split; labeled subset). |
| Model | VLM: Qwen2-VL (default). |
| Code path | `avs/experiments/egoschema_vlm_eval.py`, `scripts/e0602_egoschema_predict.sh` |
| Params | `CONFIG=Subset`, `SPLIT=test`, `LIMIT=0`, `SEED=0`, `METHODS=uniform,ql2l_clap,ql2l_asr_bm25,ql2l_clip`. |
| Metrics (must save) | `metrics.json` + `predictions.jsonl` + `preprocess_meta.json`. |
| Full cmd | See the checklist entry (OUT_DIR uses `runs/E0606_egoschema_eval_subset500_clip_*`). |
| Smoke | [ ] |
| Full | [x] |
| Logs | `artifacts/experiments/E0606/run.log` |
| Artifacts | `runs/E0606_egoschema_eval_subset500_clip_20260211-031138/metrics.json`, `runs/E0606_egoschema_eval_subset500_clip_20260211-031138/predictions.jsonl`, `runs/E0606_egoschema_eval_subset500_clip_20260211-031138/preprocess_meta.json` |
| Results | Full Subset test run (n=500; seed=0; invalid_rate=0): uniform acc=0.5880; ql2l_clip acc=0.5760; ql2l_clap=0.5480; ql2l_asr_bm25=0.5560. |

### E0618: EgoSchema language-bias baseline (uniform vs text_only; Subset n=500)
| Field | Value |
| --- | --- |
| Objective | Quantify how much of EgoSchema can be answered from the question alone by running the VLM with **no frames** (`text_only`) vs the standard uniform frame baseline. |
| Dataset | EgoSchema (Subset config; test split; labeled subset). |
| Model | VLM: Qwen2-VL (default). |
| Code path | `avs/experiments/egoschema_vlm_eval.py`, `scripts/e0602_egoschema_predict.sh` |
| Params | `METHODS=uniform,text_only`, `B_FRAMES=16`, `MAX_SECONDS=120`, `SEED=0`. |
| Metrics (must save) | `metrics.json` + `predictions.jsonl` + `preprocess_meta.json`. |
| Full cmd | See the checklist entry (OUT_DIR uses `runs/E0618_egoschema_eval_subset500_text_only_*`). |
| Smoke | [ ] |
| Full | [x] |
| Logs | `artifacts/experiments/E0618/run.log` |
| Artifacts | `runs/E0618_egoschema_eval_subset500_text_only_20260211-055131/metrics.json`, `runs/E0618_egoschema_eval_subset500_text_only_20260211-055131/predictions.jsonl`, `runs/E0618_egoschema_eval_subset500_text_only_20260211-055131/preprocess_meta.json` |
| Results | uniform acc=0.5880; text_only acc=0.2720 (invalid_rate=0; n=500). |

### E0615: AVQA VLM evaluation under budgeted frame selection (val subset; download drift allowed)
| Field | Value |
| --- | --- |
| Objective | Add an extra audio-visual MCQ dataset (AVQA) to validate the long-video QA frame-selection baselines beyond IntentQA/EgoSchema, and include a `text_only` (no frames) language-bias baseline. |
| Dataset | AVQA (val). |
| Model | VLM: Qwen2-VL (default). |
| Code path | `avs/experiments/avqa_vlm_eval.py`, `scripts/e0615_avqa_vlm_eval.sh`, `avs/datasets/avqa_download.py` |
| Params | `SPLIT=val`, `LIMIT=256` (matches downloaded subset), `B_FRAMES=16`, `MAX_SECONDS=120`, `SEED=0`, `METHODS=...+text_only`, `ALLOW_MISSING_VIDEOS=1`, `MIN_ITEMS=200`. Note: clips are ~10s, so `B_FRAMES=16` effectively selects all seconds (method separation collapses). |
| Metrics (must save) | `metrics.json` + `predictions.jsonl` + `preprocess_meta.json`. |
| Full cmd | See the checklist entry (OUT_DIR uses `runs/E0615_avqa_vlm_eval_val_*`). |
| Smoke | [ ] |
| Full | [x] |
| Logs | `artifacts/experiments/E0615/run.log` |
| Artifacts | `runs/E0615_avqa_vlm_eval_val_20260211-043508/metrics.json`, `runs/E0615_avqa_vlm_eval_val_20260211-043508/predictions.jsonl`, `runs/E0615_avqa_vlm_eval_val_20260211-043508/preprocess_meta.json` |
| Results | Kept `n=212` (skipped_videos=44; invalid_rate=0). All frame-selection methods tie because `B_FRAMES=16 >= duration_seconds≈10`: uniform/random/audio/cheap_visual/fused/ql2l_clap/ql2l_asr_bm25/ql2l_clip all acc=0.8160. `text_only` acc=0.3113 (large gap, so language bias is low on this subset). |

### E0616: AVQA VLM evaluation under budgeted frame selection (tight budget; `B_FRAMES=4`)
| Field | Value |
| --- | --- |
| Objective | Make AVQA a meaningful frame-selection benchmark by enforcing a tighter budget (`B_FRAMES=4`) on ~10s clips, so uniform/random/query-aware methods actually select different frames. |
| Dataset | AVQA (val). |
| Model | VLM: Qwen2-VL (default). |
| Code path | `avs/experiments/avqa_vlm_eval.py`, `scripts/e0615_avqa_vlm_eval.sh` |
| Params | Same as E0615, except `B_FRAMES=4`. Keep `text_only` in METHODS. |
| Metrics (must save) | `metrics.json` + `predictions.jsonl` + `preprocess_meta.json`. |
| Full cmd | See the checklist entry (OUT_DIR uses `runs/E0616_avqa_vlm_eval_val_b4_*`). |
| Smoke | [ ] |
| Full | [x] |
| Logs | `artifacts/experiments/E0616/run.log` |
| Artifacts | `runs/E0616_avqa_vlm_eval_val_b4_20260211-051556/metrics.json`, `runs/E0616_avqa_vlm_eval_val_b4_20260211-051556/predictions.jsonl`, `runs/E0616_avqa_vlm_eval_val_b4_20260211-051556/preprocess_meta.json` |
| Results | Kept `n=212` (skipped_videos=44; invalid_rate=0). Acc by method: uniform=0.8113; random=0.8066; audio=0.8160; cheap_visual=0.8255; fused=0.8255; ql2l_clap=0.8160; ql2l_asr_bm25=0.8255; ql2l_clip=0.8208; text_only=0.3113. |

### E0619: QA bucket report (“when does audio help?”) for Long-Video QA add-on
| Field | Value |
| --- | --- |
| Objective | Add a *minimal but decisive* “when does audio help?” narrative artifact by bucketing Long-Video QA results by dataset-provided question type (when available) and by the method’s `q_bar` confidence. |
| Inputs | Existing `predictions.jsonl` from E0604 (IntentQA), E0616 (AVQA), E0605 (EgoSchema Subset n=500). |
| Code path | `avs/experiments/qa_bucket_report.py`, `scripts/e0619_qa_bucket_report.sh` |
| Params | `OUT_DIR=runs/E0619_qa_bucket_report_20260211-062907` (uses the default input paths baked in the wrapper script). |
| Metrics (must save) | `bucket_report.json` + `bucket_report.md` per dataset. |
| Full cmd | `OUT_DIR=runs/E0619_qa_bucket_report_20260211-062907 bash scripts/e0619_qa_bucket_report.sh` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `artifacts/experiments/E0619/run.log` |
| Artifacts | `runs/E0619_qa_bucket_report_20260211-062907/{intentqa,avqa,egoschema}/bucket_report.{json,md}` |
| Results | Key deltas (primary method vs uniform): IntentQA: `ql2l_clap` overall `+0.00395` (helps `CH` +2.22pp; hurts `TN` -1.49pp). AVQA: `ql2l_asr_bm25` overall `+0.01415` (helps `Which` +2.80pp; hurts `Come From` -2.22pp). EgoSchema: `ql2l_clap` underperforms uniform across all `q_bar` buckets (negative-but-clean evidence). |

### R0700: Oral related-work scan and positioning matrix (online)
| Field | Value |
| --- | --- |
| Objective | Build a concise, oral-ready related-work map to justify why AVE+Long-Video-QA evidence is sufficient and where extra datasets are necessary vs optional. |
| Scope | Audio-guided frame selection, long-video QA frame selection, retrieval-augmented video understanding, and benchmark expectations for oral defense. |
| Artifacts | `docs/oral_related_work.md` |
| Results | Added a one-page positioning matrix with representative related work, oral objection mapping, and a practical recommendation: keep AVE as main proof, use Long-Video-QA as transfer/add-on evidence, and avoid unscoped extra-dataset expansion before closing core hard-gate narrative. |

### E0704: QA answer-prior bias baselines (IntentQA / AVQA / EgoSchema)
| Field | Value |
| --- | --- |
| Objective | Quantify language-prior floors and compare them against `uniform` / `text_only` to prevent over-claiming on long-video QA add-on tasks. |
| Code path | `avs/experiments/qa_answer_prior.py`, `scripts/e0704_qa_bias_baselines.sh` |
| Full cmd | `OUT_DIR=runs/E0704_qa_bias_baselines_20260211-161403 bash scripts/e0704_qa_bias_baselines.sh` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `artifacts/experiments/E0704/run.log` |
| Artifacts | `runs/E0704_qa_bias_baselines_20260211-161403/{intentqa,avqa,egoschema}/bias_baselines.{json,md}` |
| Results | IntentQA (`n=253`): answer-prior=`0.1976`, text_only=`0.6640`, uniform=`0.9447`. AVQA (`n=212`): answer-prior=`0.1651`, text_only=`0.3113`, uniform=`0.8113`. EgoSchema (`n=500`): answer-prior=`0.2340`, text_only=`0.2720`, uniform=`0.5880`. Conclusion: all three tasks remain substantially above answer-prior; language-only is not a sufficient explanation for uniform-level performance. |

### E0705: QA bucket significance (bootstrap CI + p_boot)
| Field | Value |
| --- | --- |
| Objective | Turn bucket narrative into significance-tested evidence for primary methods vs uniform on each QA dataset. |
| Code path | `avs/experiments/qa_bucket_significance.py`, `scripts/e0705_qa_bucket_significance.sh` |
| Full cmd | `OUT_DIR=runs/E0705_qa_bucket_significance_20260211-161409 bash scripts/e0705_qa_bucket_significance.sh` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `artifacts/experiments/E0705/run.log` |
| Artifacts | `runs/E0705_qa_bucket_significance_20260211-161409/{intentqa,avqa,egoschema}/bucket_significance.{json,md}` |
| Results | IntentQA primary=`ql2l_clap`: no significant bucket at `min_n=20`. AVQA primary=`ql2l_asr_bm25`: one significant positive bucket (`q_bar=mid1`, `delta=+0.0571`, `p=0.0392`, `n=70`). EgoSchema primary=`ql2l_clap`: one significant negative bucket (`q_bar=high`, `delta=-0.0647`, `p=0.0048`, `n=170`). Interpretation: improvements are localized and dataset-dependent; robust narrative should include both positive and negative significant buckets. |

### D0701: C0003 hard-gate decision helper (no-infinite-search policy)
| Field | Value |
| --- | --- |
| Objective | Enforce a pre-registered gate for C0003 (`delta>=+0.02` and `p<0.05`) and produce an explicit promote/stop decision across candidate lines. |
| Code path | `avs/experiments/c0003_gate_decision.py`, `scripts/d0701_c0003_gate.sh` |
| Full cmd | `OUT_DIR=runs/D0701_c0003_gate_20260211-161419 bash scripts/d0701_c0003_gate.sh` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `artifacts/experiments/D0701/run.log` |
| Artifacts | `runs/D0701_c0003_gate_20260211-161419/candidate_df5/decision.json`, `runs/D0701_c0003_gate_20260211-161419/candidate_df7/decision.json`, `runs/D0701_c0003_gate_20260211-161419/summary.json` |
| Results | Candidate `df5`: full `delta=+0.01117`, `p=0.1087` → `revised_claim`. Candidate `df7`: full `delta=+0.01045`, `p=0.0395` → `revised_claim` (significant but below +2%). Selected decision remains `revised_claim`, `c0003_proven=false`. |

### E0701: QA multi-seed robustness (IntentQA b16 / AVQA b4)
| Field | Value |
| --- | --- |
| Objective | Expand Long-Video QA add-on from single-seed to 3-seed robustness summaries on the most oral-relevant settings: IntentQA (`B=16`) and AVQA (`B=4`). |
| Code path | `scripts/e0701_qa_multiseed.sh`, `avs/experiments/qa_multiseed_summary.py` |
| Full cmd | Parallelized execution (seed-level sub-jobs): IntentQA `s1/s2` + AVQA `s1/s2`; then aggregation by `qa_multiseed_summary`. |
| Smoke | [ ] |
| Full | [x] |
| Logs | `artifacts/experiments/E0701/intentqa_s1.log`, `artifacts/experiments/E0701/intentqa_s2.log`, `artifacts/experiments/E0701/avqa_s1.log`, `artifacts/experiments/E0701/avqa_s2.log`, `artifacts/experiments/E0701/run.log` |
| Artifacts | Seed runs: `runs/E0701_intentqa_val_b16_s{1,2}_20260211-162353/metrics.json`, `runs/E0701_avqa_val_b4_s{1,2}_20260211-162353/metrics.json`; aggregate: `runs/E0701_qa_multiseed_20260211-162353/{intentqa_multiseed,avqa_multiseed}/metrics_summary.json` |
| Results | IntentQA (3 seeds): uniform=`0.9447`, random=`0.9341` (Δ=`-0.0105`), ql2l_clap=`0.9486` (Δ=`+0.0040`), ql2l_asr_bm25=`0.9407` (Δ=`-0.0040`), text_only=`0.6640` (Δ=`-0.2806`). AVQA (3 seeds): uniform=`0.8113`, random=`0.7909` (Δ=`-0.0204`), ql2l_clap=`0.8160` (Δ=`+0.0047`), ql2l_asr_bm25=`0.8255` (Δ=`+0.0142`), text_only=`0.3113` (Δ=`-0.5000`). |

### E0702: QA budget curve (IntentQA / AVQA, B=2/4/8/16)
| Field | Value |
| --- | --- |
| Objective | Quantify whether ql2l gains are budget-consistent or budget-sensitive by sweeping frame budget (`B=2,4,8,16`) and aggregating to per-task budget curves. |
| Code path | `scripts/e0702_qa_budget_curve.sh`, `avs/experiments/qa_budget_curve.py` |
| Full cmd | First run: `OUT_DIR=runs/E0702_qa_budget_curve_20260211-164607 bash scripts/e0702_qa_budget_curve.sh`; then offline resume for interrupted `B=8`: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ...` (`IntentQA` then `AVQA`), followed by curve regeneration (recorded in log). |
| Smoke | [ ] |
| Full | [x] |
| Logs | `artifacts/experiments/E0702/run.log` |
| Artifacts | Metrics: `runs/E0702_intentqa_val_b{2,4,8}_s0_20260211-164612/metrics.json`, `runs/E0702_avqa_val_b{2,8}_s0_20260211-164612/metrics.json` (+ existing `B=16` anchors from `runs/E0604_*` / `runs/E0615_*` / `runs/E0616_*` / `runs/E0617_*`); Curves: `runs/E0702_qa_budget_curve_20260211-164607/{intentqa_curve,avqa_curve}/budget_curve.{json,md,png}` |
| Results | IntentQA (`n=253`): uniform improves monotonically with budget (`B2=0.9012`, `B4=0.9091`, `B8=0.9368`, `B16=0.9447`), while ql2l methods are non-positive at `B2/B4/B8`; only `ql2l_clap` at `B16` stays slightly positive (Δ=`+0.0040`). AVQA (`n=212`): `ql2l_asr_bm25` is strongest at low/mid budget (`B2 Δ=+0.0283`, `B4 Δ=+0.0142`) but flips negative at `B8` (Δ=`-0.0094`), indicating budget over-allocation can dilute gains; `text_only` remains far below uniform at all budgets. Repro note: one HF transient (`client closed`) occurred; resumed in offline mode and completed with full artifacts. |

### E0703: AVQA coverage expansion + sensitivity
| Field | Value |
| --- | --- |
| Objective | Test whether AVQA conclusions hold when coverage expands far beyond the original `n=212` subset used by E0616. |
| Code path | `scripts/e0703_avqa_coverage_expand.sh`, `avs/datasets/avqa_download.py`, `avs/experiments/qa_coverage_sensitivity.py` |
| Full cmd | `RUN_ROOT=runs/E0703_avqa_coverage_expand_20260211-170526 SKIP_DOWNLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DEVICE=cuda:2 QL2L_CLAP_DEVICE=cuda:2 QL2L_ASR_DEVICE=cpu QL2L_CLIP_DEVICE=cpu bash scripts/e0703_avqa_coverage_expand.sh` |
| Smoke | [ ] |
| Full | [x] |
| Logs | `artifacts/experiments/E0703/run.log` |
| Artifacts | `runs/E0703_avqa_vlm_eval_val_b4_expand_20260211-170223/metrics.json`, `runs/E0703_avqa_coverage_expand_20260211-170526/avqa_download_val_full.json`, `runs/E0703_avqa_coverage_expand_20260211-170526/sensitivity/coverage_sensitivity.{json,md}` |
| Results | Expanded evaluation keeps `n=865` items (`baseline_n=212`). Acc on expanded set: uniform=`0.8312`, cheap_visual=`0.8301`, ql2l_asr_bm25=`0.8324`, text_only=`0.3341`. Coverage-sensitivity deltas (expanded-baseline): uniform `+0.0199`, ql2l_asr_bm25 `+0.0069`, cheap_visual `+0.0046`, text_only `+0.0228`. Interpretation: ranking is stable (`ql2l_asr_bm25` still best), with modest absolute shifts under larger coverage. |
