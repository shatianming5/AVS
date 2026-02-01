#!/usr/bin/env bash
set -euo pipefail

# Wait for `scripts/ave_install_official.sh` to finish, then run real AVE experiments on the full official dataset.
#
# What it runs:
#   - E0002-style anchor eval on full val/test (LIMIT=402)
#   - E0001-style AVE-P0 on full train/val and train/test (defaults to full train; overridable)
#
# Usage:
#   bash scripts/ave_verify_official_after_install.sh
#
# Common overrides:
#   LIMIT_TRAIN=1024 SEEDS=0,1,2 bash scripts/ave_verify_official_after_install.sh
#   CACHE_NUM_WORKERS=4 CACHE_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 bash scripts/ave_verify_official_after_install.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

META_DIR="${META_DIR:-data/AVE/meta}"
RAW_VIDEOS_DIR="${RAW_VIDEOS_DIR:-data/AVE/raw/videos}"

TRAIN_IDS_FILE="${TRAIN_IDS_FILE:-${META_DIR}/download_ok_train_official.txt}"
VAL_IDS_FILE="${VAL_IDS_FILE:-${META_DIR}/download_ok_val_official.txt}"
TEST_IDS_FILE="${TEST_IDS_FILE:-${META_DIR}/download_ok_test_official.txt}"

EXPECTED_TRAIN="${EXPECTED_TRAIN:-3339}"
EXPECTED_VAL="${EXPECTED_VAL:-402}"
EXPECTED_TEST="${EXPECTED_TEST:-402}"

LIMIT_TRAIN="${LIMIT_TRAIN:-${EXPECTED_TRAIN}}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"

# Best config from docs/experiment.md (C0001).
EVENTNESS="${EVENTNESS:-energy}"
K="${K:-2}"
LOW_RES="${LOW_RES:-160}"
BASE_RES="${BASE_RES:-224}"
HIGH_RES="${HIGH_RES:-352}"
ANCHOR_SHIFT="${ANCHOR_SHIFT:-1}"
ANCHOR_STD_THRESHOLD="${ANCHOR_STD_THRESHOLD:-1.0}"
HEAD="${HEAD:-temporal_conv}"
TEMPORAL_KERNEL_SIZE="${TEMPORAL_KERNEL_SIZE:-3}"

VISION_PRETRAINED="${VISION_PRETRAINED:-1}"

# Use all visible GPUs by default (if present); otherwise fall back to CPU.
if command -v nvidia-smi >/dev/null 2>&1; then
  NUM_GPUS="$(nvidia-smi -L | wc -l | tr -d ' ')"
else
  NUM_GPUS="0"
fi

DEVICE="${DEVICE:-$([[ "${NUM_GPUS}" -gt 0 ]] && echo cuda:0 || echo cpu)}"
AUDIO_DEVICE="${AUDIO_DEVICE:-${DEVICE}}"

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

PREPROCESS_JOBS="${PREPROCESS_JOBS:-32}"

RUN_ROOT="${RUN_ROOT:-runs/REAL_AVE_OFFICIAL_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${RUN_ROOT}"

echo "[$(date)] Waiting for official AVE lists..."
while true; do
  if [[ -f "${TRAIN_IDS_FILE}" && -f "${VAL_IDS_FILE}" && -f "${TEST_IDS_FILE}" ]]; then
    N_TRAIN="$(wc -l < "${TRAIN_IDS_FILE}" | tr -d ' ')"
    N_VAL="$(wc -l < "${VAL_IDS_FILE}" | tr -d ' ')"
    N_TEST="$(wc -l < "${TEST_IDS_FILE}" | tr -d ' ')"
    if [[ "${N_TRAIN}" -ge "${EXPECTED_TRAIN}" && "${N_VAL}" -ge "${EXPECTED_VAL}" && "${N_TEST}" -ge "${EXPECTED_TEST}" ]]; then
      echo "[$(date)] Found lists: train=${N_TRAIN} val=${N_VAL} test=${N_TEST}"
      break
    fi
    echo "[$(date)] Lists not complete yet: train=${N_TRAIN}/${EXPECTED_TRAIN} val=${N_VAL}/${EXPECTED_VAL} test=${N_TEST}/${EXPECTED_TEST}"
  else
    echo "[$(date)] Lists missing; waiting..."
  fi
  sleep 300
done

echo "[$(date)] Running anchor eval (val/test)..."
MODE=local SRC_DIR="${RAW_VIDEOS_DIR}" SPLIT=val LIMIT="${EXPECTED_VAL}" RUN_DIR="${RUN_ROOT}/E0002_anchors_official_val" \
  bash scripts/e0002_anchor_eval_real.sh
MODE=local SRC_DIR="${RAW_VIDEOS_DIR}" SPLIT=test LIMIT="${EXPECTED_TEST}" RUN_DIR="${RUN_ROOT}/E0002_anchors_official_test" \
  bash scripts/e0002_anchor_eval_real.sh

echo "[$(date)] Running AVE-P0 (train→val)..."
PROCESSED_DIR="${RUN_ROOT}/processed" \
  CACHES_DIR="${RUN_ROOT}/caches_${LOW_RES}_${BASE_RES}_${HIGH_RES}" \
  PREPROCESS_JOBS="${PREPROCESS_JOBS}" \
  RAW_VIDEOS_DIR="${RAW_VIDEOS_DIR}" \
  ALLOW_MISSING=1 \
  TRAIN_IDS_FILE="${TRAIN_IDS_FILE}" \
  EVAL_IDS_FILE="${VAL_IDS_FILE}" \
  LIMIT_TRAIN="${LIMIT_TRAIN}" \
  LIMIT_EVAL="${EXPECTED_VAL}" \
  SEEDS="${SEEDS}" \
  EVENTNESS="${EVENTNESS}" \
  K="${K}" \
  LOW_RES="${LOW_RES}" \
  BASE_RES="${BASE_RES}" \
  HIGH_RES="${HIGH_RES}" \
  CACHE_RESOLUTIONS="${LOW_RES},${BASE_RES},${HIGH_RES}" \
  ANCHOR_SHIFT="${ANCHOR_SHIFT}" \
  ANCHOR_STD_THRESHOLD="${ANCHOR_STD_THRESHOLD}" \
  HEAD="${HEAD}" \
  TEMPORAL_KERNEL_SIZE="${TEMPORAL_KERNEL_SIZE}" \
  VISION_PRETRAINED="${VISION_PRETRAINED}" \
  DEVICE="${DEVICE}" \
  AUDIO_DEVICE="${AUDIO_DEVICE}" \
  CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS}" \
  CACHE_DEVICES="${CACHE_DEVICES}" \
  OUT_DIR="${RUN_ROOT}/p0_train${LIMIT_TRAIN}_val${EXPECTED_VAL}_${EVENTNESS}_${LOW_RES}_${BASE_RES}_${HIGH_RES}_k${K}_shift${ANCHOR_SHIFT}_std${ANCHOR_STD_THRESHOLD}_${HEAD}" \
  bash scripts/e0001_ave_p0_real_multigpu.sh

echo "[$(date)] Running AVE-P0 (train→test)..."
PROCESSED_DIR="${RUN_ROOT}/processed" \
  CACHES_DIR="${RUN_ROOT}/caches_${LOW_RES}_${BASE_RES}_${HIGH_RES}" \
  PREPROCESS_JOBS="${PREPROCESS_JOBS}" \
  RAW_VIDEOS_DIR="${RAW_VIDEOS_DIR}" \
  ALLOW_MISSING=1 \
  TRAIN_IDS_FILE="${TRAIN_IDS_FILE}" \
  EVAL_IDS_FILE="${TEST_IDS_FILE}" \
  LIMIT_TRAIN="${LIMIT_TRAIN}" \
  LIMIT_EVAL="${EXPECTED_TEST}" \
  SEEDS="${SEEDS}" \
  EVENTNESS="${EVENTNESS}" \
  K="${K}" \
  LOW_RES="${LOW_RES}" \
  BASE_RES="${BASE_RES}" \
  HIGH_RES="${HIGH_RES}" \
  CACHE_RESOLUTIONS="${LOW_RES},${BASE_RES},${HIGH_RES}" \
  ANCHOR_SHIFT="${ANCHOR_SHIFT}" \
  ANCHOR_STD_THRESHOLD="${ANCHOR_STD_THRESHOLD}" \
  HEAD="${HEAD}" \
  TEMPORAL_KERNEL_SIZE="${TEMPORAL_KERNEL_SIZE}" \
  VISION_PRETRAINED="${VISION_PRETRAINED}" \
  DEVICE="${DEVICE}" \
  AUDIO_DEVICE="${AUDIO_DEVICE}" \
  CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS}" \
  CACHE_DEVICES="${CACHE_DEVICES}" \
  OUT_DIR="${RUN_ROOT}/p0_train${LIMIT_TRAIN}_test${EXPECTED_TEST}_${EVENTNESS}_${LOW_RES}_${BASE_RES}_${HIGH_RES}_k${K}_shift${ANCHOR_SHIFT}_std${ANCHOR_STD_THRESHOLD}_${HEAD}" \
  bash scripts/e0001_ave_p0_real_multigpu.sh

echo "[$(date)] OK: ${RUN_ROOT}"
