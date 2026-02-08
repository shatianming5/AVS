#!/usr/bin/env bash
set -euo pipefail

# E0340: Build official AVE CLIP feature caches with extra mid resolutions for P0116 / E0341+.
#
# Why:
#   The next “拉大 C0003” attempt (candidate_set=ltl_top1med_band_midres_v1) needs mid resolutions
#   (192/208) and a cheaper high (320) so the band-budget DP can preserve more context under 2-high.
#
# This script reuses the official raw videos and (already extracted) processed frames/audio, then builds
# a new caches dir containing the required resolutions.
#
# Usage (recommended; multi-GPU):
#   CACHE_NUM_WORKERS=10 CACHE_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7,cuda:8,cuda:9 \\
#     bash scripts/e0340_ave_cache_official_midres.sh
#
# Outputs:
#   - ${CACHES_DIR}/<clip_id>.npz (+ .json meta)
#   - ${OUT_ROOT}/cache_val/* (logs)
#   - ${OUT_ROOT}/cache_test/* (logs)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

META_DIR="${META_DIR:-data/AVE/meta}"
RAW_VIDEOS_DIR="${RAW_VIDEOS_DIR:-data/AVE/raw/videos}"

DEFAULT_PROCESSED="data/AVE/processed"
if [[ -d "runs/REAL_AVE_OFFICIAL_20260201-124535/processed" ]]; then
  DEFAULT_PROCESSED="runs/REAL_AVE_OFFICIAL_20260201-124535/processed"
elif [[ -d "runs/REAL_AVE_LOCAL/processed" ]]; then
  DEFAULT_PROCESSED="runs/REAL_AVE_LOCAL/processed"
fi
PROCESSED_DIR="${PROCESSED_DIR:-${DEFAULT_PROCESSED}}"

DEFAULT_CACHES_DIR="runs/REAL_AVE_OFFICIAL_20260201-124535/caches_112_160_192_208_224_320_352"
CACHES_DIR="${CACHES_DIR:-${DEFAULT_CACHES_DIR}}"

LIMIT_TRAIN="${LIMIT_TRAIN:-3339}"
LIMIT_VAL="${LIMIT_VAL:-402}"
LIMIT_TEST="${LIMIT_TEST:-402}"

TRAIN_IDS_FILE="${TRAIN_IDS_FILE:-${META_DIR}/download_ok_train_official.txt}"
VAL_IDS_FILE="${VAL_IDS_FILE:-${META_DIR}/download_ok_val_official.txt}"
TEST_IDS_FILE="${TEST_IDS_FILE:-${META_DIR}/download_ok_test_official.txt}"

if [[ ! -f "${TRAIN_IDS_FILE}" ]]; then
  echo "ERROR: TRAIN_IDS_FILE not found: ${TRAIN_IDS_FILE}" >&2
  exit 2
fi
if [[ ! -f "${VAL_IDS_FILE}" ]]; then
  echo "ERROR: VAL_IDS_FILE not found: ${VAL_IDS_FILE}" >&2
  exit 2
fi
if [[ ! -f "${TEST_IDS_FILE}" ]]; then
  echo "ERROR: TEST_IDS_FILE not found: ${TEST_IDS_FILE}" >&2
  exit 2
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  DEVICE="${DEVICE:-cuda:0}"
else
  DEVICE="${DEVICE:-cpu}"
fi

if [[ -n "${CACHE_NUM_WORKERS:-}" || -n "${CACHE_DEVICES:-}" ]]; then
  CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-1}"
  CACHE_DEVICES="${CACHE_DEVICES:-${DEVICE}}"
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS="$(nvidia-smi -L | wc -l | tr -d ' ')"
    CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-${NUM_GPUS}}"
    devices=()
    for i in $(seq 0 $((NUM_GPUS - 1))); do
      devices+=("cuda:${i}")
    done
    CACHE_DEVICES="$(IFS=','; echo "${devices[*]}")"
  else
    CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-1}"
    CACHE_DEVICES="${CACHE_DEVICES:-${DEVICE}}"
  fi
fi

# Required for the new candidate set:
#   - 112: used by Stage-1 av_clipdiff_* methods as a fixed cheap visual res when available
#   - 160/224: canonical low/base
#   - 192/208: mid options for band-budget DP
#   - 320: cheaper high to preserve context in 2-high
#   - 352: baseline exact winner uses 352
CACHE_RESOLUTIONS="${CACHE_RESOLUTIONS:-112,160,192,208,224,320,352}"

OUT_ROOT="${OUT_ROOT:-runs/E0340_cache_official_midres_$(date +%Y%m%d-%H%M%S)}"

echo "[e0340] PROCESSED_DIR=${PROCESSED_DIR}"
echo "[e0340] CACHES_DIR=${CACHES_DIR}"
echo "[e0340] CACHE_RESOLUTIONS=${CACHE_RESOLUTIONS}"
echo "[e0340] CACHE_NUM_WORKERS=${CACHE_NUM_WORKERS}"
echo "[e0340] CACHE_DEVICES=${CACHE_DEVICES}"

mkdir -p "${OUT_ROOT}"
mkdir -p "${CACHES_DIR}"

# Pass low/base/high from the baseline exact triad so ave_p0_end2end validates required cache resolutions.
LOW_RES=160
BASE_RES=224
HIGH_RES=352

echo "[e0340] (1/2) build caches for train+val (union)"
python -m avs.pipeline.ave_p0_end2end \
  --mode none \
  --meta-dir "${META_DIR}" \
  --raw-videos-dir "${RAW_VIDEOS_DIR}" \
  --processed-dir "${PROCESSED_DIR}" \
  --preprocess-skip-existing \
  --allow-missing \
  --caches-dir "${CACHES_DIR}" \
  --cache-num-workers "${CACHE_NUM_WORKERS}" \
  --cache-devices "${CACHE_DEVICES}" \
  --cache-skip-existing \
  --cache-resolutions "${CACHE_RESOLUTIONS}" \
  --split-train train \
  --split-eval val \
  --train-ids-file "${TRAIN_IDS_FILE}" \
  --eval-ids-file "${VAL_IDS_FILE}" \
  --limit-train "${LIMIT_TRAIN}" \
  --limit-eval "${LIMIT_VAL}" \
  --low-res "${LOW_RES}" \
  --base-res "${BASE_RES}" \
  --high-res "${HIGH_RES}" \
  --vision-pretrained \
  --device "${DEVICE}" \
  --cache-only \
  --out-dir "${OUT_ROOT}/cache_val"

echo "[e0340] (2/2) build caches for train+test (fills missing test clips)"
python -m avs.pipeline.ave_p0_end2end \
  --mode none \
  --meta-dir "${META_DIR}" \
  --raw-videos-dir "${RAW_VIDEOS_DIR}" \
  --processed-dir "${PROCESSED_DIR}" \
  --preprocess-skip-existing \
  --allow-missing \
  --caches-dir "${CACHES_DIR}" \
  --cache-num-workers "${CACHE_NUM_WORKERS}" \
  --cache-devices "${CACHE_DEVICES}" \
  --cache-skip-existing \
  --cache-resolutions "${CACHE_RESOLUTIONS}" \
  --split-train train \
  --split-eval test \
  --train-ids-file "${TRAIN_IDS_FILE}" \
  --eval-ids-file "${TEST_IDS_FILE}" \
  --limit-train "${LIMIT_TRAIN}" \
  --limit-eval "${LIMIT_TEST}" \
  --low-res "${LOW_RES}" \
  --base-res "${BASE_RES}" \
  --high-res "${HIGH_RES}" \
  --vision-pretrained \
  --device "${DEVICE}" \
  --cache-only \
  --out-dir "${OUT_ROOT}/cache_test"

echo "OK: caches built under ${CACHES_DIR}"

