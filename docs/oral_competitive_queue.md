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
