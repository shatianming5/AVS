# Oral-Competitive Push Queue (C0003 +2% Attempt)

Date: 2026-02-12

Goal: turn the current oral-ready pack into an **oral-competitive** result by attempting to **prove C0003**:
`anchored_top2 - uniform >= +0.02` with paired `p < 0.05` on **official AVE test402** (`SEEDS=0..9`).

## 0) Known bottleneck (why current best is stuck at ~+1%)

Current best full test402:
- `runs/E0643_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_df7_officialids_s0-9_20260211-001604/metrics.json`
- `Δ=+0.01045`, `p≈0.0395` (significant but far from +2%)

Diagnosis suggests the main loss is **fallback dilution** + residual harmful buckets (see `diagnose.json`).

## 1) Hard constraints (avoid p-hacking)

- Promotion gate is fixed:
  1) `val402 sweep (SEEDS=0..2)` → pick 1 winner by *mean Δ*; tie-break by lower fallback.
  2) `quick test402 (SEEDS=0..2)` → only promote if Δ is competitive and diagnose is not worse than baseline.
  3) `full test402 (SEEDS=0..9)` → only here do we claim C0003 is proven.
- Only **2** new Stage-1 ideas are allowed in this push (ImageBind, WavLM) to keep the search bounded.

## 2) New Stage-1 ideas (high-upside, literature-aligned)

### Track B1: ImageBind AV-consistency eventness (training-free Stage-1)

Motivation:
- ImageBind provides a shared embedding space for audio and vision; an AV-consistency score can serve as eventness.
- Closely related to open-vocabulary audio-visual event localization using ImageBind.

Implementation tasks:
- Add a new `eventness_method=imagebind_av_sim` to `avs/experiments/ave_p0.py`:
  - per-second score `s(t) = cosine(emb_audio(t), emb_image(t))`
  - optional smoothing (mean/median) with fixed window (pre-registered)
  - cache scores into the existing `--scores-json` format (`eventness_scores.json`)
- Add a minimal wrapper under `avs/multimodal/`:
  - loads pretrained ImageBind
  - encodes 1s audio segments + middle-frame JPGs
  - returns per-second embeddings
- Add a smoke test: `python -m avs.smoke imagebind_eventness` (new)

Runs:
- Val sweep:
  - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=imagebind_av_sim CANDIDATE_SET=ltl_adaptive_keepadj_v2 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
  - Save: `runs/E08xx_.../{sweep_summary.json,best_config.json,eventness_scores.json}`
- Quick test402:
  - `PROCESSED_DIR=... CACHES_DIR=... BEST_CONFIG_JSON=runs/E08xx_.../best_config.json EVENTNESS=imagebind_av_sim SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
  - Then: `IN_METRICS=runs/E08xx_.../metrics.json OUT_DIR=runs/E08xx_... bash scripts/e0344_ave_p0_diagnose.sh`
- Full test402 (only if quick is promoted):
  - `PROCESSED_DIR=... CACHES_DIR=... BEST_CONFIG_JSON=... EVENTNESS=imagebind_av_sim SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`

Promotion rule (quick→full):
- Must satisfy both:
  - `Δ_quick >= +0.018` AND `p_quick <= 0.2`
  - fallback_used_frac does not increase vs the current best quick baseline (compare diagnose)

### Track B2: WavLM audio embeddings + learned eventness head (stronger audio Stage-1)

Motivation:
- WavLM is a strong audio representation; a small head trained on AVE train can produce sharper per-second eventness.

Implementation tasks:
- Add `eventness_method=wavlm_evt_mlp`:
  - WavLM encoder (frozen), per-second embedding extraction
  - train tiny MLP to predict the AVE segment label (multi-class; background dominates)
  - score `s(t)=max(logits[1:]) - logits[0]` as eventness
  - cache scores to `--scores-json`
- Add smoke: `python -m avs.smoke wavlm_eventness` (new)

Runs:
- Same sweep/quick/full pipeline as ImageBind, with `EVENTNESS=wavlm_evt_mlp` (and `WAVLM_*` envs).

Promotion rule (quick→full):
- Same as above.

## 3) Stop rules (avoid infinite loop)

- If both B1 and B2 fail full test402, stop the +2% chase and revert to the revised-claim oral pack.
- If either B1 or B2 succeeds (Δ>=+0.02 and p<0.05 on SEEDS=0..9):
  - rerun the key oral plots for the winning Stage-1 (E0330/E0331 equivalents as needed)
  - update `docs/evidence_matrix.md`, `docs/oral_narrative.md`, and export new slide assets under `docs/oral_assets/`.

## 4) Execution results (this push)

- Track B1 (ImageBind) result:
  - Val402 sweep (best): `runs/E0801_val402_imagebind_keepadjv2_20260212-035956/sweep_summary.json` (Δ≈-0.00008)
  - Quick test402: `runs/E0802_quick_test402_imagebind_20260212-040440/metrics.json` (Δ≈-0.00265; p≈0.754)
  - Decision: not promoted.

- Track B2 (WavLM) result:
  - Val402 sweep (best): `runs/E0810_val402_wavlm_20260212-041931/sweep_summary.json` (Δ≈-0.00424)
  - Quick test402: `runs/E0811_quick_test402_wavlm_20260212-042425/metrics.json` (Δ≈+0.00124; p≈0.918)
  - Decision: not promoted.

- Stop rule triggered:
  - With both Stage-1 ideas failing the promotion gate, we stop the +2% chase and stick to the revised-claim oral pack.

## 5) Track C (override stop rule): Oracle-distilled visual-usefulness Stage-1

User request (2026-02-12): keep pushing `C0003 (+2%)` and try a *qualitatively different* Stage-1 than “swap audio backbone / similarity”.

Idea:
- Distill an **oracle-style visual usefulness teacher**: per-second **loss-gain** when swapping in higher-res CLIP features vs base-res, measured by a cheap vision teacher classifier trained on train split.
- Train a deployable student to predict this loss-gain from **WavLM audio embeddings + low-res CLIP embeddings**.
- Use the student’s predicted gain scores as Stage-1 eventness scores.

Implementation:
- New Stage-1 backend: `EVENTNESS=av_wavlm_clip_lossgain_mlp` (see `avs/experiments/ave_p0_sweep.py`).

Runs (sequential; same promotion gate):
- Val402 sweep (SEEDS=0..2):
  - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_wavlm_clip_lossgain_mlp CANDIDATE_SET=ltl_adaptive_keepadj_v2 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0820_val402_wavlm_cliplossgain_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
- Quick test402 (SEEDS=0..2) + diagnose:
  - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0820_*/best_config.json EVENTNESS=av_wavlm_clip_lossgain_mlp SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0821_quick_test402_wavlm_cliplossgain_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
  - `IN_METRICS=runs/E0821_*/metrics.json OUT_DIR=runs/E0821_* bash scripts/e0344_ave_p0_diagnose.sh`
- Full test402 (SEEDS=0..9) only if quick is promoted:
  - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0820_*/best_config.json EVENTNESS=av_wavlm_clip_lossgain_mlp SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0822_full_test402_wavlm_cliplossgain_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
  - `IN_METRICS=runs/E0822_*/metrics.json OUT_DIR=runs/E0822_* bash scripts/e0344_ave_p0_diagnose.sh`

Results:
- Val402 sweep (best): `runs/E0820_val402_wavlm_cliplossgain_20260212-112651/sweep_summary.json` (best Δ=-0.00116; p=0.9163).
- Quick test402: `runs/E0821_quick_test402_wavlm_cliplossgain_20260212-113152/metrics.json` (Δ=+0.00149; p=0.9007; fallback_used_frac≈0.970).
- Decision: not promoted (skip full).

## 6) Track D: WavLM+CLIP MIL Stage-1 (peaky anchors; Top-K aligned)

Idea:
- Train a Stage-1 scorer with a **multi-instance learning (MIL)** objective so that at least one true event second ranks near the top.
- Use strong frozen backbones: **WavLM audio embeddings + low-res CLIP embeddings**.
- Hypothesis: MIL yields a *peakier* score distribution (higher per-clip score std), reducing fallback dilution and improving C0003 transfer.

Implementation:
- New Stage-1 backend: `EVENTNESS=av_wavlm_clip_mil_mlp` (see `avs/experiments/ave_p0_sweep.py`).

Runs (sequential; same promotion gate):
- Val402 sweep (SEEDS=0..2):
  - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_wavlm_clip_mil_mlp CANDIDATE_SET=ltl_adaptive_keepadj_v2 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0830_val402_wavlm_clipmil_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
- Quick test402 (SEEDS=0..2) + diagnose:
  - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0830_*/best_config.json EVENTNESS=av_wavlm_clip_mil_mlp SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0831_quick_test402_wavlm_clipmil_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
  - `IN_METRICS=runs/E0831_*/metrics.json OUT_DIR=runs/E0831_* bash scripts/e0344_ave_p0_diagnose.sh`
- Full test402 (SEEDS=0..9) only if quick is promoted:
  - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0830_*/best_config.json EVENTNESS=av_wavlm_clip_mil_mlp SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0832_full_test402_wavlm_clipmil_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
  - `IN_METRICS=runs/E0832_*/metrics.json OUT_DIR=runs/E0832_* bash scripts/e0344_ave_p0_diagnose.sh`

Results:
- Val402 sweep (best): `runs/E0830_val402_wavlm_clipmil_20260212-115213/sweep_summary.json` (best Δ=-0.00150; p=0.5286).
- Quick test402: `runs/E0831_quick_test402_wavlm_clipmil_20260212-115656/metrics.json` (Δ=-0.00141; p=0.8594; fallback_used_frac=0.0).
- Decision: not promoted (skip full).

## 7) Track E: WavLM+CLIP supervised eventness (BCE MLP)

Idea:
- Train a lightweight per-second eventness head on frozen **WavLM audio** + **low-res CLIP vision** to predict `(label!=0)`.
- Use logits as Stage-1 scores, hoping to reduce far-anchor errors without relying on similarity heuristics.

Implementation:
- New Stage-1 backend: `EVENTNESS=av_wavlm_clip_evt_mlp` (see `avs/experiments/ave_p0_sweep.py`).

Runs:
- Val402 sweep:
  - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 EVENTNESS=av_wavlm_clip_evt_mlp CANDIDATE_SET=ltl_top1med_norm_v1 SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0840_val402_wavlm_clipevt_mlp_$(date +%Y%m%d-%H%M%S) bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`
- Quick test402 + diagnose:
  - `PROCESSED_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed CACHES_DIR=runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_112_160_224_352_448 BEST_CONFIG_JSON=runs/E0840_*/best_config.json EVENTNESS=av_wavlm_clip_evt_mlp SEEDS=0,1,2 AUDIO_DEVICE=cuda:1 TRAIN_DEVICE=cuda:0 WAVLM_PRETRAINED=1 WAVLM_MODEL=microsoft/wavlm-base-plus WAVLM_BATCH_SIZE=16 OUT_DIR=runs/E0841_quick_test402_wavlm_clipevt_mlp_$(date +%Y%m%d-%H%M%S) bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`
  - `IN_METRICS=runs/E0841_*/metrics.json OUT_DIR=runs/E0841_* bash scripts/e0344_ave_p0_diagnose.sh`
- Full test402: skipped if not promoted.

Results:
- Val402 sweep (best): `runs/E0840_val402_wavlm_clipevt_mlp_20260212-121939/sweep_summary.json` (anchored=0.74023 vs uniform=0.74680; Δ=-0.00657; p=0.5425).
- Quick test402: `runs/E0841_quick_test402_wavlm_clipevt_mlp_20260212-122228/metrics.json` (anchored=0.71526 vs uniform=0.71294; Δ=+0.00232; p=0.243; fallback_used_frac≈0.483).
- Decision: not promoted (skip full).

## 8) Track F: WavLM+CLIP supervised eventness (TCN)

Idea:
- Same as Track E, but use a tiny temporal conv net (TCN) to leverage local temporal context.

Implementation:
- New Stage-1 backend: `EVENTNESS=av_wavlm_clip_evt_tcn` (see `avs/experiments/ave_p0_sweep.py`).

Results:
- Val402 sweep (best): `runs/E0850_val402_wavlm_clipevt_tcn_20260212-122411/sweep_summary.json` (anchored=0.74813 vs uniform=0.74680; Δ=+0.00133; p=0.900).
- Quick test402: `runs/E0851_quick_test402_wavlm_clipevt_tcn_20260212-122630/metrics.json` (anchored=0.70597 vs uniform=0.71294; Δ=-0.00697; p=0.673; fallback_used_frac≈0.679).
- Decision: not promoted (skip full).

## 9) Track G: Stage-2 sep3 gate on vec-MLP (sanity sweep)

Idea:
- Try the scale-invariant separation gate (`conf_metric=top3_bottom3_gap_norm`) as the Stage-2 confidence filter for `av_clipdiff_vec_mlp`.

Results:
- Val402 sweep: `runs/E0860_val402_vecmlp_sep3_20260212-122905/sweep_summary.json` (best Δ=-0.00540; p=0.253).
- Quick test402: `runs/E0861_quick_test402_vecmlp_sep3_20260212-123220/metrics.json` (Δ=-0.00680; p=0.599; fallback_used_frac≈0.231).
- Decision: harmful; skip full.

## 10) Track H: WavLM + CLIPdiff vector MLP (upgrade the current best family)

Idea:
- Replace `av_clipdiff_vec_mlp`’s basic audio features with frozen **WavLM embeddings** while keeping the strongest visual motion proxy (**CLIPdiff vector**).

Implementation:
- New Stage-1 backend: `EVENTNESS=av_wavlm_clipdiff_vec_mlp` (see `avs/experiments/ave_p0_sweep.py`).

Results:
- Val402 sweep: `runs/E0870_val402_wavlm_clipdiff_vecmlp_20260212-123504/sweep_summary.json` (best Δ=-0.00042; p=0.951).
- Quick test402: `runs/E0871_quick_test402_wavlm_clipdiff_vecmlp_20260212-123718/metrics.json` (anchored=0.71940 vs uniform=0.71294; Δ=+0.00647; p=0.513; fallback_used_frac≈0.510).
- Decision: not promoted (skip full).

## 11) Track I: WavLM+CLIP multi-class segment classifier (class-conditional Stage-1)

Idea:
- Train a lightweight **multi-class** per-second classifier on strong frozen features:
  - audio: `l2norm(WavLM emb[t])`
  - vision: `l2norm(low-res CLIP emb[t])`
- Use a class-conditional margin score as Stage-1 eventness:
  - infer a clip-level class `cls = argmax(mean_logits[1:])`
  - score per-second as `(logit[t, cls] - logit[t, bg])`
- Hypothesis: class-conditional scoring yields sharper anchors than binary `(label!=0)` eventness.

Implementation:
- New Stage-1 backend: `EVENTNESS=av_wavlm_clip_mlp_cls_target` (code: `avs/experiments/ave_p0_sweep.py`).
- Reliability: `avs/audio/wavlm_probe.py` now prefers HF local cache (`local_files_only=True`) before online fetches to avoid rerun failures.

Results:
- Val402 sweep (SEEDS=0..2): `runs/E0880_val402_wavlm_clip_cls_target_20260212-125727/sweep_summary.json`
  - best: `ltltop1medn_thr0p7_shift0` (Δ≈-0.00283; p≈0.828)
- Quick test402 (SEEDS=0..2): `runs/E0881_quick_test402_wavlm_clip_cls_target_20260212-130104/metrics.json` (Δ≈+0.00473; p≈0.617)
  - diagnose: `runs/E0881_quick_test402_wavlm_clip_cls_target_20260212-130104/diagnose.json` (fallback_used_frac≈0.933)
- Decision: not promoted (skip full).

Follow-up (same family, different scoring rule):
- Margin variant `EVENTNESS=av_wavlm_clip_mlp_cls`:
  - val402 sweep: `runs/E0886_val402_wavlm_clip_cls_margin_20260212-131558/sweep_summary.json` (best Δ≈+0.00166; p≈0.888)
  - quick test402: `runs/E0887_quick_test402_wavlm_clip_cls_margin_20260212-131757/metrics.json` (Δ≈+0.00232; p≈0.678)
    - diagnose: `runs/E0887_quick_test402_wavlm_clip_cls_margin_20260212-131757/diagnose.json` (fallback_used_frac≈0.923)
  - Decision: not promoted.

## 12) Track J: Vec-MLP Stage-2 max-high=1 sweep (preserve context; reduce 2-high harm)

Idea:
- Diagnose on the current best full test402 (`E0643`) shows the **2-high** regime is near-zero delta.
- Force `max_high_anchors=1` to preserve more `base_res` context under the same equal token budget.

Results:
- Val402 sweep (SEEDS=0..2): `runs/E0883_val402_vecmlp_maxhigh1_20260212-130225/sweep_summary.json`
  - best: `ltlmax1_thr0p45_balanced_window3` (Δ≈+0.00939; p≈0.275)
- Quick test402 (SEEDS=0..2): `runs/E0884_quick_test402_vecmlp_maxhigh1_20260212-130830/metrics.json` (Δ≈+0.01111; p≈0.323)
  - diagnose: `runs/E0884_quick_test402_vecmlp_maxhigh1_20260212-130830/diagnose.json` (fallback_used_frac≈0.311)
- Decision: not promoted (skip full).

## 13) Track K: MIL Stage-1 rerun with max-high=1 normalized gate

Motivation:
- We already tried `av_wavlm_clip_mil_mlp` (E0830/E0831) under a keepadj-style Stage-2 sweep, and it did not transfer.
- Rerun the same Stage-1 with a **scale-invariant gate** + **max-high=1** to see if this reduces (a) score-scale sensitivity and (b) 2-high harm.

Implementation:
- New candidate set: `CANDIDATE_SET=ltl_top1medn_maxhigh1_v1` (see `avs/experiments/ave_p0_sweep.py`).

Results:
- Val402 sweep (SEEDS=0..2): `runs/E0890_val402_wavlm_clip_mil_mlp_20260212-133043/sweep_summary.json`
  - best: `ltltop1mednmax1_thr0p5_shift0` (anchored=0.74214 vs uniform=0.74680; Δ=-0.00466; p=0.3436)
- Quick test402 (SEEDS=0..2): `runs/E0891_quick_test402_wavlm_clip_mil_mlp_20260212-133312/metrics.json`
  - anchored=0.70730 vs uniform=0.71294 (Δ=-0.00564; p=0.5272)
  - diagnose: `runs/E0891_quick_test402_wavlm_clip_mil_mlp_20260212-133312/diagnose.json` (`anchors_len_fallback_frac≈0.532`)
- Decision: harmful; skip full.

## 14) Track L: Stage-2 ablations on the current best df7 keepadj config (vec-MLP)

Goal:
- Reduce the known 2-high harm on test402, and check whether budget-band can preserve context in the 2-high regime.

### L1) Force `max_high_anchors=1` (remove 2-high)

- Quick test402 (SEEDS=0..2): `runs/E0893_quick_test402_vecmlp_df7_maxhigh1_20260212-133857/metrics.json`
  - anchored=0.72347 vs uniform=0.71294 (Δ=+0.01053; p=0.4780)
  - diagnose: `runs/E0893_quick_test402_vecmlp_df7_maxhigh1_20260212-133857/diagnose.json` (no 2-high; still fallback≈0.502)
- Decision: not better than the current best full-test result; not promoted.

### L2) Budget-band + extra low-res 112 (attempt to rescue 2-high)

- Quick test402 (SEEDS=0..2): `runs/E0895_quick_test402_vecmlp_df7_band112_20260212-134114/metrics.json`
  - anchored=0.72828 vs uniform=0.71294 (Δ=+0.01534; p=0.1997)
- Full test402 (SEEDS=0..9): `runs/E0896_full_test402_vecmlp_df7_band112_20260212-134215/metrics.json`
  - anchored=0.72410 vs uniform=0.71622 (Δ=+0.00789; p=0.2531)
  - diagnose shows 2-high is negative again: `runs/E0896_full_test402_vecmlp_df7_band112_20260212-134215/diagnose.json`
- Decision: quick looked good but does not hold on full; do not adopt.

### L3) Std-threshold sweep on the max-high=1 variant (try reducing fallback)

- std0.35 quick (SEEDS=0..2): `runs/E0898_quick_test402_vecmlp_df7_maxhigh1_std0p35_20260212-134530/metrics.json` (Δ=+0.00871; p=0.5286; fallback≈0.189)
- std0.45 quick (SEEDS=0..2): `runs/E0899_quick_test402_vecmlp_df7_maxhigh1_std0p45_20260212-134637/metrics.json` (Δ=+0.00498; p=0.7523; fallback≈0.311)
- std0.65 quick (SEEDS=0..2): `runs/E0900_quick_test402_vecmlp_df7_maxhigh1_std0p65_20260212-134717/metrics.json` (Δ=+0.00373; p=0.7991; fallback≈0.624)

### L4) `k=1` (single-anchor allocation; removes 2-anchor regime)

- Quick test402 (SEEDS=0..2): `runs/E0901_quick_test402_vecmlp_df7_k1_20260212-135506/metrics.json`
  - anchored=0.72504 vs uniform=0.71294 (Δ=+0.01211; p=0.3983)
  - diagnose: `runs/E0901_quick_test402_vecmlp_df7_k1_20260212-135506/diagnose.json` (`anchors_len_fallback_frac≈0.552`)
- Decision: modest quick gain, still far from +2; not promoted.
- Conclusion: lowering the std gate reduces fallback but hurts Δ on quick; the best remaining df7 family result is still `E0643` on full test402 (Δ=+0.01045; p≈0.0395).

## 15) Track M: Cross-time A↔V attention MIL Stage-1 (XAttn MIL)

Idea:
- Train an audio-conditioned cross-time A↔V attention model to produce per-second anchor scores with a MIL objective,
  aiming to (a) learn soft temporal alignment and (b) yield sharper anchor distributions (less fallback dilution).

Implementation:
- New Stage-1 backend: `EVENTNESS=av_wavlm_clip_xattn_mil` (code: `avs/experiments/ave_p0_sweep.py`).
- Variants are controlled via env vars:
  - `XATTN_VIS_RES` (e.g., 112/224/352)
  - `XATTN_VIS_FEATS` (`clip|clipdiff|clip+clipdiff`)
  - `XATTN_TRAIN_DEVICE` (default cpu; set to cuda for speed)

Results:
- Baseline val402 sweep (SEEDS=0..2; `candidate_set=ltl_top1medn_maxhigh1_v1`): `runs/E0902_val402_wavlm_clip_xattn_mil_20260212-140250/sweep_summary.json`
  - best: `ltltop1mednmax1_thr0p7_shift1` (anchored=0.74314 vs uniform=0.74680; Δ=-0.00366; p=0.257)
- r224 + clip+clipdiff val402 sweep (SEEDS=0..2): `runs/E0903_val402_wavlm_clip_xattn_mil_r224_clipdiff_20260212-141657/sweep_summary.json`
  - best: `ltltop1mednmax1_thr0p6_shift0` (anchored=0.74746 vs uniform=0.74680; Δ=+0.00066; p=0.911)
- keepadjv2 Stage-2 sweep using cached E0903 scores (SEEDS=0..2): `runs/E0904_val402_xattn_mil_r224_clipdiff_keepadjv2_20260212-141941/sweep_summary.json`
  - best: `ltlkeepadjv2_adj2_shift1_std0p25` (anchored=0.74123 vs uniform=0.74680; Δ=-0.00557; p=0.428)
- r352 + clip val402 sweep (SEEDS=0..2): `runs/E0905_val402_xattn_mil_r352_clip_20260212-142552/sweep_summary.json`
  - best: `ltltop1mednmax1_thr0p5_shift0` (anchored=0.74722 vs uniform=0.74680; Δ=+0.00042; p=0.950)

Decision:
- Not promotable: all variants are near-zero/negative on val402, so we stop this direction (no quick/full test402).

## 16) Track N: High-res vision-only Stage-1 (binary CLIP eventness @ r352)

Idea:
- As an aggressive Stage-1 attempt aligned with the `oracle_top2 = (label!=0)` teacher, train a binary visual eventness
  model on cached CLIP features at **r352** (`vision_binary_mlp_r352`) and use its logits as per-second scores.

Implementation:
- New Stage-1 backend: `EVENTNESS=vision_binary_mlp_r352` (code: `avs/experiments/ave_p0_sweep.py`).

Results:
- Val402 sweep (SEEDS=0..2; `candidate_set=ltl_top1medn_maxhigh1_v1`): `runs/E0906_val402_vision_binary_mlp_r352_20260212-143611/sweep_summary.json`
  - best: `ltltop1mednmax1_thr0p6_shift0` (anchored=0.73716 vs uniform=0.74680; Δ=-0.00964; p=0.116; fallback≈0.728 by `conf_below_threshold`)
- Val402 sweep (SEEDS=0..2; `candidate_set=ltl_gini_v2`; cached E0906 scores): `runs/E0907_val402_vision_binary_mlp_r352_gini_v2_20260212-143837/sweep_summary.json`
  - best: `ltlgini2_gini0p35_shift0` (anchored=0.74331 vs uniform=0.74680; Δ=-0.00349; p=0.707)

Decision:
- Harmful/negative on val402 even with gini gating; stop (no quick/full test402).

## 17) Track O: AVE-localizer-style Stage-1 (A+V fusion + BiLSTM cls-target)

Idea:
- Replace Stage-1 with a **true temporal event localizer** trained on AVE segment labels:
  - audio (WavLM per-second embeddings) + vision (cached CLIP / CLIPdiff features)
  - gated A+V fusion + BiLSTM over time
  - per-second event class logits → score via **class-conditional margin** (`logit[class*] - logit[bg]`)

Implementation:
- New Stage-1 backend: `EVENTNESS=av_wavlm_clip_avel_bilstm_cls_target` (code: `avs/experiments/ave_p0_sweep.py`).
- Controlled via env vars:
  - `AVEL_VIS_RES` (default 160; used 352)
  - `AVEL_VIS_FEATS` (`clip|clipdiff|clip+clipdiff`; used `clip+clipdiff`)
  - `AVEL_TRAIN_DEVICE`, `AVEL_EPOCHS`, `AVEL_BS`, `AVEL_LR`, etc.

Results:
- Baseline val402 sweep (SEEDS=0..2; `candidate_set=ltl_top1medn_maxhigh1_v1`): `runs/E0908_val402_avel_bilstm_cls_r352_clipdiff_20260212-205411/sweep_summary.json`
  - best: `ltltop1mednmax1_thr0p5_shift1` (anchored=0.74796 vs uniform=0.74680; Δ=+0.00116; p=0.922)
- Minmax-normalized score cache sanity check (SEEDS=0..2): `runs/E0909_val402_avel_bilstm_cls_r352_clipdiff_minmax_20260212-210652/sweep_summary.json`
  - identical to E0908 (ranking-only)
- Onset-like score transform (positive derivative + minmax; SEEDS=0..2): `runs/E0910_val402_avel_bilstm_cls_onset_deriv_pos_rerun_20260212-213925/sweep_summary.json`
  - best: `ltltop1mednmax1_thr0p7_shift1` (anchored≈uniform; Δ≈0.0; p≈1.0)
- Stage-2 candidate set variations using cached E0908 scores:
  - keepadjv2: `runs/E0911_val402_avel_bilstm_cls_keepadjv2_20260212-214325/sweep_summary.json` (best Δ≈-0.00698; p≈0.358)
  - gini gate: `runs/E0912_val402_avel_bilstm_cls_gini_v2_20260212-214729/sweep_summary.json` (best Δ≈+0.00208; p≈0.785)
  - adaptive_v2: `runs/E0913_val402_avel_bilstm_cls_adaptive_v2_20260212-215337/sweep_summary.json` (best Δ≈-0.00673; p≈0.464)

Decision:
- Not promotable: despite being a “real” temporal localizer, downstream anchored-vs-uniform gains remain near-zero or harmful on val402, so we stop (no quick/full test402).

## 18) Track P: Cross-time A↔V attention (supervised cls-target; XAttn-Cls)

Idea:
- Take Track M’s explicit A↔V alignment (10×10 attention) but train it **supervised** with AVE segment labels (multi-class CE)
  instead of MIL, hoping the supervision stabilizes anchor localization.

Implementation:
- New Stage-1 backend: `EVENTNESS=av_wavlm_clip_xattn_cls_target` (code: `avs/experiments/ave_p0_sweep.py`).
- Controls: reuse `XATTN_*` env vars (e.g., `XATTN_VIS_RES`, `XATTN_VIS_FEATS`, `XATTN_EPOCHS`, `XATTN_BS`).

Results:
- Val402 sweep (SEEDS=0..2; `candidate_set=ltl_top1medn_maxhigh1_v1`; r352; clip+clipdiff): `runs/E0914_val402_xattn_cls_target_r352_clipdiff_20260212-221441/sweep_summary.json`
  - best: `ltltop1mednmax1_thr0p5_shift1` (anchored=0.74422 vs uniform=0.74680; Δ=-0.00258; p=0.693)

Decision:
- Not promotable (negative on val402); no quick/full test402.

## 19) Track Q: Vision Backbone Swap via timm EVA02 (cache rebuild)

Idea:
- Try a qualitatively different lever than swapping Stage-1 heads: **change the vision backbone** used for caching
  (and optionally only for Stage-1 scoring) to improve anchor reliability and/or downstream robustness.

Implementation (repo changes; see the uncommitted diff in this workspace):
- `avs/vision/clip_vit.py`: add `timm:` backend (preprocess + `timm.create_model(..., num_classes=0)`).
- `avs/pipeline/ave_p0_end2end.py`: add `--vision-model-name` and write backbone metadata into `cache_build.json`.
- `avs/experiments/ave_p0_sweep.py`: allow `STAGE1_CACHES_DIR` override so Stage-1 can use a different cache than Stage-2.

Runs:
- EVA02 caches built (r∈{112,160,224,352}):
  - Train+val: `runs/E0915_build_cache_eva02_clip_p16_112_160_224_352_20260212-225043/cache_build.json` (3703 clips)
  - Test (incremental): `runs/E0915_build_cache_eva02_clip_p16_112_160_224_352_test_20260212-230913/cache_build.json` (394 clips)
  - Cache dir: `runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_eva02_base_patch16_clip_224_112_160_224_352/` (total `.npz`=4097)

Results:
- Stage-2 swapped to EVA02 caches (train+eval):
  - Val402 sweep: `runs/E0916_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_adaptive_keepadj_v1_eva02_20260212-231218/sweep_summary.json`
    - best: `ltlkeepadj_adj1_shift0_std0p6` (anchored=0.76110 vs uniform=0.75869; Δ=+0.00241; p=0.6538)
  - Interpretation: uniform rises a lot; selection gain collapses.

- Stage-1-only EVA02 caches (`STAGE1_CACHES_DIR`), Stage-2 stays baseline:
  - keepadj candidate set: `runs/E0917_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_adaptive_keepadj_v1_stage1eva02_20260212-231759/sweep_summary.json`
    - best: `ltlkeepadj_adj1_shift0_std0p45` (anchored=0.75004 vs uniform=0.74680; Δ=+0.00324; p=0.5702)
  - top1med_norm candidate set: `runs/E0918_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_top1med_norm_v1_stage1eva02_20260212-232240/sweep_summary.json`
    - best: `ltltop1medn_thr0p5_shift0` (anchored=0.74863 vs uniform=0.74680; Δ=+0.00183; p=0.8223)

Decision:
- Not competitive on val402 in any setup; do not promote to quick/full test402.

## 20) Track R: DINOv2 Stage-1-only caches (`STAGE1_CACHES_DIR`)

Idea:
- Try a non-CLIP self-supervised backbone for Stage-1 scoring (motion/change proxy via feature diffs),
  while keeping Stage-2 fixed to the official CLIP cache for comparability.

Runs:
- Cache build (r∈{112,160,224,352,448}):
  - Train+val: `runs/E0919_build_cache_dinov2_fill_trainval_20260213-001226/cache_build.json` (num_total_ids=3703)
  - Test: `runs/E0919_build_cache_dinov2_fill_test_20260213-002118/cache_build.json` (adds 394 test clips; total `.npz`=4097)
  - Cache dir: `runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch14_dinov2_112_160_224_352_448/`

Results:
- Val402 sweep (keepadj; SEEDS=0..2): `runs/E0920_val402_vecmlp_keepadj_stage1dinov2_20260213-001634/sweep_summary.json`
  - best: `ltlkeepadj_adj2_shift0_std0p55` (anchored=0.75520 vs uniform=0.74680; Δ=+0.00840; p=0.4031)
- Quick test402 (SEEDS=0..2): `runs/E0921_quick_test402_vecmlp_keepadj_stage1dinov2_20260213-002346/metrics.json`
  - anchored=0.72189 vs uniform=0.71294 (Δ=+0.00896; p=0.5825)
  - diagnose: `runs/E0921_quick_test402_vecmlp_keepadj_stage1dinov2_20260213-002346/diagnose.json`

Decision:
- Not promoted to full test402 (Δ is still far from +2% and not significant on quick).

## 21) Track S: DINOv2 Stage-1-only + Dense-Stride Stage-1 (`av_clipdiff_flow_mlp_stride`)

Idea:
- The dense-stride family (`av_clipdiff_flow_mlp_stride`) is one of the strongest non-oracle Stage-1s on official test402,
  but its Stage-1 proxy may be bottlenecked by CLIP features.
- Swap the Stage-1 cache to DINOv2 via `STAGE1_CACHES_DIR`, keep Stage-2 fixed to the official CLIP cache for comparability.

Runs:
- Val402 sweep (`candidate_set=ltl_top1med_k1_extreme_v1`; SEEDS=0..2): `runs/E0923_val402_flow_stride_stage1dinov2_20260213-003820/sweep_summary.json`
  - best: `ltltop1medk1ext_thr0p6_shift0_score` (Δ≈+0.00333; p≈0.551)
- Quick test402 (SEEDS=0..2): `runs/E0924_quick_test402_flow_stride_stage1dinov2_20260213-004520/metrics.json`
  - anchored=0.71401 vs uniform=0.71294 (Δ≈+0.00108; p≈0.922)
  - diagnose: `runs/E0924_quick_test402_flow_stride_stage1dinov2_20260213-004520/diagnose.json`

Decision:
- Not promotable (quick Δ collapses); skip full test402.

## 22) Track T: DINOv2 Stage-1-only + AVE-localizer BiLSTM (cls-target)

Idea:
- Re-run the AVE-localizer-style BiLSTM Stage-1 (`av_wavlm_clip_avel_bilstm_cls_target`) but swap the Stage-1 cache
  to DINOv2 via `STAGE1_CACHES_DIR` (Stage-2 remains baseline CLIP cache).

Runs:
- Val402 sweep (SEEDS=0..2; `candidate_set=ltl_top1medn_maxhigh1_v1`): `runs/E0926_val402_avel_bilstm_cls_stage1dinov2_20260213-005021/sweep_summary.json`
  - best: `ltltop1mednmax1_thr0p7_shift1` (Δ≈-0.00898; p≈0.2298)

Decision:
- Not promotable (negative on val402); no quick/full test402.

## 23) Track U: DINOv2 Stage-1-only + XAttn supervised cls-target

Idea:
- Same as Track P, but swap Stage-1 cache to DINOv2 to test whether stronger visual features make the
  cross-time A↔V attention Stage-1 usable.

Runs:
- Val402 sweep (SEEDS=0..2; `candidate_set=ltl_top1medn_maxhigh1_v1`): `runs/E0927_val402_xattn_cls_target_stage1dinov2_20260213-005232/sweep_summary.json`
  - best: `ltltop1mednmax1_thr0p5_shift1` (Δ≈+0.00324; p≈0.688)

Decision:
- Not promotable (small Δ on val402); no quick/full test402.

## 24) Track V: DINOv2 Stage-1 scores reuse + top1med_norm Stage-2 sweep

Idea:
- Reuse the DINOv2 Stage-1 score cache from Track R (`SCORES_JSON`) and resweep a different Stage-2 candidate set
  (`ltl_top1med_norm_v1`) to test whether a different gate transfers better than keepadj.

Runs:
- Val402 sweep (SEEDS=0..2): `runs/E0928_val402_vecmlp_top1medn_scores_dinov2_20260213-005513/sweep_summary.json`
  - best: `ltltop1medn_thr0p6_shift1` (Δ≈+0.00017; p≈0.984)

Decision:
- Not promotable; keepadj remains the only semi-competitive Stage-2 family under these scores.

## 25) Track W: SigLIP Stage-2 Backbone Swap (in progress)

Idea:
- Try a qualitatively stronger vision backbone that is (a) patch16 (token accounting compatible) and (b) known to be strong
  for vision-language pretraining: SigLIP ViT-B/16.
- Hypothesis: if the CLIP-frozen Stage-2 head is the bottleneck, swapping the Stage-2 backbone may increase the resolution
  sensitivity and widen anchored-vs-uniform Δ.

Implementation:
- Add a timm pooling fallback for models without a CLS token (SigLIP uses attention pooling by default):
  - `avs/vision/clip_vit.py`: if `global_pool='token'` is unsupported, fall back to the model default `global_pool` (e.g., `map`).

Queue:
- E0930: build SigLIP caches (train+val then test) under:
  - `runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/caches_vit_base_patch16_siglip_224_webli_112_160_224_352_448/`
- E0931→E0933: run the standard promotion gate (val402 → quick test402 → full test402 if promoted).
