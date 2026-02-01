#!/usr/bin/env bash
set -euo pipefail

# Oracle ceiling sweep on official AVE: compares `uniform` vs `oracle_top2` under strict equal token budget.
#
# Requires:
#   - AVE metadata under data/AVE/meta/
#   - official raw videos under data/AVE/raw/videos/<video_id>.mp4 (symlink ok)
#
# This script:
#   1) Ensures caches exist (cache-only, skip-existing) for the union of required resolutions
#   2) Runs `avs.experiments.ave_p0_oracle_sweep` to produce `oracle_ceiling.json`

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

META_DIR="${META_DIR:-data/AVE/meta}"
RAW_VIDEOS_DIR="${RAW_VIDEOS_DIR:-data/AVE/raw/videos}"

SPLIT_TRAIN="${SPLIT_TRAIN:-train}"
SPLIT_EVAL="${SPLIT_EVAL:-val}" # val|test

DEFAULT_TRAIN_IDS_FILE="${META_DIR}/download_ok_train_official.txt"
if [[ ! -f "${DEFAULT_TRAIN_IDS_FILE}" ]]; then
  DEFAULT_TRAIN_IDS_FILE="${META_DIR}/download_ok_train_auto.txt"
fi
TRAIN_IDS_FILE="${TRAIN_IDS_FILE:-${DEFAULT_TRAIN_IDS_FILE}}"

DEFAULT_EVAL_IDS_FILE="${META_DIR}/download_ok_${SPLIT_EVAL}_official.txt"
if [[ ! -f "${DEFAULT_EVAL_IDS_FILE}" ]]; then
  DEFAULT_EVAL_IDS_FILE="${META_DIR}/download_ok_${SPLIT_EVAL}_auto.txt"
fi
EVAL_IDS_FILE="${EVAL_IDS_FILE:-${DEFAULT_EVAL_IDS_FILE}}"

LIMIT_TRAIN="${LIMIT_TRAIN:-3339}"
LIMIT_EVAL="${LIMIT_EVAL:-402}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-2e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"

# Use all visible GPUs by default (if present); otherwise fall back to CPU.
if command -v nvidia-smi >/dev/null 2>&1; then
  NUM_GPUS="$(nvidia-smi -L | wc -l | tr -d ' ')"
else
  NUM_GPUS="0"
fi

if [[ -z "${DEVICE:-}" ]]; then
  DEVICE="$([[ "${NUM_GPUS}" -gt 0 ]] && echo cuda:0 || echo cpu)"
fi
TRAIN_DEVICE="${TRAIN_DEVICE:-${DEVICE}}"

# CLIP vision backbone weights for cache build.
VISION_PRETRAINED="${VISION_PRETRAINED:-1}"
VISION_PRETRAINED_FLAG=()
if [[ "${VISION_PRETRAINED}" == "1" ]]; then
  VISION_PRETRAINED_FLAG+=(--vision-pretrained)
fi

# Multi-process cache build settings.
if [[ "${NUM_GPUS}" -gt 0 ]]; then
  CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-${NUM_GPUS}}"
  if [[ -z "${CACHE_DEVICES:-}" ]]; then
    devices=()
    for ((i = 0; i < NUM_GPUS; i++)); do
      devices+=("cuda:${i}")
    done
    CACHE_DEVICES="$(IFS=','; echo "${devices[*]}")"
  fi
else
  CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-1}"
  CACHE_DEVICES="${CACHE_DEVICES:-${DEVICE}}"
fi

# Union of resolutions used by default configs in `avs/experiments/ave_p0_oracle_sweep.py`.
CACHE_RESOLUTIONS="${CACHE_RESOLUTIONS:-112,160,224,352,448}"

# Reusable outputs (incremental).
PROCESSED_DIR="${PROCESSED_DIR:-runs/REAL_AVE_LOCAL/processed}"
CACHES_DIR="${CACHES_DIR:-runs/REAL_AVE_LOCAL/caches}"

PREPROCESS_JOBS="${PREPROCESS_JOBS:-32}"
ALLOW_MISSING="${ALLOW_MISSING:-1}"
ALLOW_MISSING_FLAG=()
if [[ "${ALLOW_MISSING}" == "1" ]]; then
  ALLOW_MISSING_FLAG+=(--allow-missing)
fi

OUT_DIR="${OUT_DIR:-runs/E0010_oracle_ceiling_official_${SPLIT_EVAL}_$(date +%Y%m%d-%H%M%S)}"

cd "${REPO_ROOT}"

python -m avs.pipeline.ave_p0_end2end \
  --mode none \
  --cache-only \
  --meta-dir "${META_DIR}" \
  --raw-videos-dir "${RAW_VIDEOS_DIR}" \
  --processed-dir "${PROCESSED_DIR}" \
  --preprocess-skip-existing \
  --preprocess-jobs "${PREPROCESS_JOBS}" \
  --caches-dir "${CACHES_DIR}" \
  --cache-skip-existing \
  --cache-num-workers "${CACHE_NUM_WORKERS}" \
  --cache-devices "${CACHE_DEVICES}" \
  --cache-resolutions "${CACHE_RESOLUTIONS}" \
  --out-dir "${OUT_DIR}/cache_only" \
  "${ALLOW_MISSING_FLAG[@]}" \
  --split-train "${SPLIT_TRAIN}" \
  --split-eval "${SPLIT_EVAL}" \
  --train-ids-file "${TRAIN_IDS_FILE}" \
  --eval-ids-file "${EVAL_IDS_FILE}" \
  --limit-train "${LIMIT_TRAIN}" \
  --limit-eval "${LIMIT_EVAL}" \
  "${VISION_PRETRAINED_FLAG[@]}" \
  --device "${DEVICE}"

python -m avs.experiments.ave_p0_oracle_sweep \
  --meta-dir "${META_DIR}" \
  --caches-dir "${CACHES_DIR}" \
  "${ALLOW_MISSING_FLAG[@]}" \
  --split-train "${SPLIT_TRAIN}" \
  --split-eval "${SPLIT_EVAL}" \
  --train-ids-file "${TRAIN_IDS_FILE}" \
  --eval-ids-file "${EVAL_IDS_FILE}" \
  --limit-train "${LIMIT_TRAIN}" \
  --limit-eval "${LIMIT_EVAL}" \
  --seeds "${SEEDS}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --train-device "${TRAIN_DEVICE}" \
  --out-dir "${OUT_DIR}"

echo "OK: ${OUT_DIR}/oracle_ceiling.json"
