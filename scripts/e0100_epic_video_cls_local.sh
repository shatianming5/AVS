#!/usr/bin/env bash
set -euo pipefail

# EPIC-SOUNDS downstream proxy (video-level multi-label recognition) on local EPIC mp4s.
#
# Requirements:
#   - EPIC-KITCHENS videos under data/EPIC_SOUNDS/raw/videos/<video_id>.mp4
#   - ffmpeg installed
#
# Example:
#   bash scripts/e0100_epic_video_cls_local.sh
#
# Notes:
#   - This script uses EPIC-SOUNDS annotations from `data/EPIC_SOUNDS/meta` (downloaded if missing).
#   - It runs on a limited number of train/val videos by default; increase LIMIT_* to scale.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VIDEOS_DIR="${VIDEOS_DIR:-data/EPIC_SOUNDS/raw/videos}"
META_DIR="${META_DIR:-data/EPIC_SOUNDS/meta}"

SELECTION="${SELECTION:-audio_anchored}" # uniform|random|audio_anchored
MAX_SECONDS="${MAX_SECONDS:-}"
MAX_STEPS="${MAX_STEPS:-120}"

EVENTNESS="${EVENTNESS:-energy}"
K="${K:-10}"
ANCHOR_RADIUS="${ANCHOR_RADIUS:-2}"
BACKGROUND_STRIDE="${BACKGROUND_STRIDE:-5}"
ANCHOR_SHIFT="${ANCHOR_SHIFT:-0}"
ANCHOR_STD_THRESHOLD="${ANCHOR_STD_THRESHOLD:-0.0}"

LOW_RES="${LOW_RES:-112}"
BASE_RES="${BASE_RES:-224}"
HIGH_RES="${HIGH_RES:-448}"
CACHE_RESOLUTIONS="${CACHE_RESOLUTIONS:-${LOW_RES},${BASE_RES},${HIGH_RES}}"

LIMIT_TRAIN_VIDEOS="${LIMIT_TRAIN_VIDEOS:-64}"
LIMIT_VAL_VIDEOS="${LIMIT_VAL_VIDEOS:-64}"
ALLOW_MISSING_VIDEOS="${ALLOW_MISSING_VIDEOS:-0}"
MIN_TRAIN_VIDEOS="${MIN_TRAIN_VIDEOS:-16}"
MIN_VAL_VIDEOS="${MIN_VAL_VIDEOS:-16}"

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-2e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
DROPOUT="${DROPOUT:-0.1}"

if [[ -z "${DEVICE:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && [[ "$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')" -gt 0 ]]; then
    DEVICE="cuda:0"
  else
    DEVICE="cpu"
  fi
fi
CLIP_DEVICE="${CLIP_DEVICE:-${DEVICE}}"
TRAIN_DEVICE="${TRAIN_DEVICE:-${DEVICE}}"

OUT_DIR="${OUT_DIR:-runs/E0100_epic_video_cls_local_${SELECTION}_$(date +%Y%m%d-%H%M%S)}"

cd "${REPO_ROOT}"

args=(
  --videos-dir "${VIDEOS_DIR}"
  --meta-dir "${META_DIR}"
  --out-dir "${OUT_DIR}"
  --selection "${SELECTION}"
  --max-steps "${MAX_STEPS}"
  --eventness-method "${EVENTNESS}"
  --k "${K}"
  --anchor-radius "${ANCHOR_RADIUS}"
  --background-stride "${BACKGROUND_STRIDE}"
  --anchor-shift "${ANCHOR_SHIFT}"
  --anchor-std-threshold "${ANCHOR_STD_THRESHOLD}"
  --low-res "${LOW_RES}"
  --base-res "${BASE_RES}"
  --high-res "${HIGH_RES}"
  --cache-resolutions "${CACHE_RESOLUTIONS}"
  --clip-device "${CLIP_DEVICE}"
  --train-device "${TRAIN_DEVICE}"
  --limit-train-videos "${LIMIT_TRAIN_VIDEOS}"
  --limit-val-videos "${LIMIT_VAL_VIDEOS}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --hidden-dim "${HIDDEN_DIM}"
  --dropout "${DROPOUT}"
)

if [[ "${ALLOW_MISSING_VIDEOS}" = "1" ]]; then
  args+=(--allow-missing-videos --min-train-videos "${MIN_TRAIN_VIDEOS}" --min-val-videos "${MIN_VAL_VIDEOS}")
fi

if [[ -n "${MAX_SECONDS}" ]]; then
  args+=(--max-seconds "${MAX_SECONDS}")
fi

python -m avs.experiments.epic_sounds_video_cls "${args[@]}"

echo "OK: ${OUT_DIR}/metrics.json"
