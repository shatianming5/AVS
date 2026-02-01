# AVS — Audio-Visual Synchronizer

AVS is a lightweight research codebase for **audio-guided temporal anchoring** and **equal-token-budget visual sampling**.

The core idea: audio is cheap and sparse in time, so we can use per-second audio “eventness” scores to propose a few
**anchor seconds**, then spend a fixed visual token budget more effectively by allocating higher resolution near anchors
while keeping the **total ViT token count exactly the same** as a uniform baseline.

This repo is designed to be:
- **Reproducible**: deterministic preprocessing (per-second frames + wav) and JSON/JSONL artifacts
- **Fast to iterate**: expensive vision encoding is cached; training is a small head on frozen features
- **Audit-friendly**: every run writes `metrics.json` plus debug fields (anchors/scores/plan) for inspection

If you want the canonical actionable plan and the experiment ledger, see:
- `docs/plan.md`
- `docs/experiment.md`

## Quick Start

### 0) Requirements

System tools:
- `ffmpeg` (required for video/audio extraction)

Python:
- Python 3.10+ recommended
- Install deps:

```bash
pip install -r requirements.txt
```

Optional but helpful:
- `scipy` (enables paired t-tests in `metrics.json`)

### 1) Run smoke checks

```bash
python -m avs.smoke all
```

Artifacts go under `runs/` by default (override with `AVS_RUNS_DIR=/path/to/runs`).

## Repository Layout

High-level folders:
- `avs/`: python package (datasets, preprocess, audio eventness, sampling plans, caching, experiments)
- `scripts/`: runnable shell scripts for common workflows (download/install, experiments)
- `docs/`: canonical plan + experiment logs

Runtime artifacts (ignored by git):
- `data/`: dataset cache (override with `AVS_DATA_DIR=/path/to/data`)
- `runs/`: experiment outputs (override with `AVS_RUNS_DIR=/path/to/runs`)

(`data/` and `runs/` are in `.gitignore`.)

## Core Concepts

### AVE protocol (10 seconds → 10 segments)

AVE is treated as a fixed-length 10s clip:
- Preprocessing extracts:
  - `audio.wav` (16kHz mono)
  - `frames/{0..9}.jpg` (one middle frame per second)

### Eventness → Anchors → Plan

1) Compute per-second eventness scores `s(t)` (length=10)
2) Select Top-K anchor seconds (e.g. `k=2`)
3) Create an **equal-token-budget** plan for 10 seconds:
   - `uniform`: all seconds use `base_res`
   - `anchored_top2`: anchors use `high_res`, others use `low_res`/`base_res` such that total tokens match `uniform`

Implementation:
- Audio eventness and anchor selection: `avs/audio/eventness.py`
- Equal-budget plan generation: `avs/sampling/plans.py`

### Baselines you’ll see in `metrics.json`

Produced by `avs/pipeline/ave_p0_end2end.py`:
- `uniform`: uniform sampling at `base_res`
- `uniform_low`: uniform at `low_res` (cheaper; for efficiency curve)
- `random_top2`: random anchors, equal budget
- `anchored_top2`: audio anchors, equal budget
- `oracle_top2`: uses ground-truth event seconds as anchors (upper bound)
- `audio_concat_uniform`: uniform sampling, but concatenate scalar audio eventness `s(t)` to the visual embedding
- `audio_concat_anchored_top2`: anchored sampling + concatenate `s(t)`
- `audio_feat_concat_uniform`: uniform sampling + concatenate simple per-second audio features
- `audio_feat_concat_anchored_top2`: anchored sampling + concatenate audio features

(`audio_concat_uniform` is *not* changing sampling; it is an audio-visual feature fusion baseline.)

## Experiments (AVE-P0)

The main “P0” experiment is a controlled study:
- Vision encoder: CLIP ViT-B/16 (frozen)
- Trainable model: a small head (`mlp` or `temporal_conv`)
- Comparison: different sampling plans under the **same token budget**

Entrypoints:
- End-to-end pipeline (download/preprocess/cache/run): `python -m avs.pipeline.ave_p0_end2end`
- Fast rerun on existing caches/audio/ids: `python -m avs.experiments.ave_p0_rerun`

### Install official AVE dataset (full 4143 clips)

This repo supports the official AVE dataset zip installer:

```bash
bash scripts/ave_install_official.sh
```

It installs to:
- `data/AVE/raw/videos/<video_id>.mp4`
- and writes availability lists to `data/AVE/meta/*.txt`

### Full official verification (recommended)

Once the official dataset is installed:

```bash
bash scripts/ave_verify_official_after_install.sh
```

This runs:
- E0002-style anchor eval on full val/test
- E0001-style AVE-P0 train→val and train→test on the full train split

### Run AVE-P0 on a smaller subset (single GPU)

Default uses `yt-dlp` to fetch a small subset (best-effort):

```bash
bash scripts/e0001_ave_p0_real.sh
```

If you already have local AVE mp4s in a directory:

```bash
MODE=local SRC_DIR=/path/to/ave/mp4s bash scripts/e0001_ave_p0_real.sh
```

### Run AVE-P0 with multi-GPU cache building

The expensive step is building multi-resolution feature caches. The script supports multi-process cache build
with different workers pinned to different devices:

```bash
CACHE_NUM_WORKERS=4 \
CACHE_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
bash scripts/e0001_ave_p0_real_multigpu.sh
```

### Run anchor quality eval (Recall@K / Recall@K,Δ)

```bash
bash scripts/e0002_anchor_eval_real.sh
```

Or on installed official videos:

```bash
MODE=local SRC_DIR=data/AVE/raw/videos SPLIT=test LIMIT=402 bash scripts/e0002_anchor_eval_real.sh
```

## Important knobs (anchored sampling)

These flags are supported in `avs/pipeline/ave_p0_end2end.py` and passed through by the scripts:

Audio eventness backend:
- `--eventness-method`:
  - `energy` (default; per-second log energy)
  - `energy_delta` (novelty/change-point)
  - `ast` / `panns` / `audiomae` (optional probes; may require checkpoints)
  - supervised probes for screening: `audio_basic_lr`, `audio_basic_mlp`, `audio_basic_mlp_cls`, ...

Anchor selection and robustness:
- `--k`: top-k anchors
- `--anchor-shift`: shift anchors to model A/V misalignment
- `--anchor-std-threshold`: if `std(scores)` is too small, fall back to uniform sampling
- `--anchor-select`: `topk` | `nms` | `nms_strong`
- `--anchor-nms-radius`: suppression radius for `nms`
- `--anchor-nms-strong-gap`: for `nms_strong`, accept a far anchor only if it is competitive

Equal-budget plan allocation:
- `--low-res`, `--base-res`, `--high-res`, `--patch-size`
- `--anchor-base-alloc`:
  - `distance`: allocate base-res by distance to anchors (legacy)
  - `score`: allocate base-res by high eventness score
  - `farthest`: allocate base-res far from anchors (context)
  - `mixed`: half near anchors + half far (context)
- `--anchor-high-policy`: `fixed` or `adaptive_v1` (demote 2nd high-res anchor when adjacent/weak)

## Reproducing current best-known results

For the latest “source of truth” numbers and commands, read:
- `docs/experiment.md`

It records:
- `E0001` (AVE-P0)
- `E0002` (anchor quality)
- `E0003` (official full-dataset validation)

## Notes / Caveats

- **Data licensing**: AVE and EPIC-SOUNDS require you to follow their respective dataset terms. This repo does not
  redistribute datasets.
- **Caches dominate runtime**: full runs are mostly bottlenecked by feature caching; head training is relatively cheap.
- **Git ignores artifacts**: `data/` and `runs/` are not tracked.

