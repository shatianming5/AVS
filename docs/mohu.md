# Mohu

## 1) Not Implemented

## 2) Ambiguities

## Resolved (archive)
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
