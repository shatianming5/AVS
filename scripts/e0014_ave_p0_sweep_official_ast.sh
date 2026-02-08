#!/usr/bin/env bash
set -euo pipefail

# AST-based fixed-space sweep on official AVE val402 to select a best config (for later test402 reproduction).
#
# Produces:
#   - runs/E0014_*/sweep_summary.json
#   - runs/E0014_*/best_config.json
#
# Notes:
# - This script assumes you already have processed audio/frames and feature caches.
#   If not, first run: bash scripts/ave_verify_official_after_install.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

META_DIR="${META_DIR:-data/AVE/meta}"

# Prefer an existing official processed+cache run directory if available.
if [[ -z "${OFFICIAL_RUN_DIR:-}" ]]; then
  OFFICIAL_RUN_DIR="$(ls -td runs/REAL_AVE_OFFICIAL_* 2>/dev/null | head -n 1 || true)"
fi

DEFAULT_PROCESSED_DIR="data/AVE/processed"
DEFAULT_CACHES_DIR=""
if [[ -n "${OFFICIAL_RUN_DIR}" && -d "${OFFICIAL_RUN_DIR}/processed" ]]; then
  DEFAULT_PROCESSED_DIR="${OFFICIAL_RUN_DIR}/processed"
  if [[ -d "${OFFICIAL_RUN_DIR}/caches_112_160_224_352_448" ]]; then
    DEFAULT_CACHES_DIR="${OFFICIAL_RUN_DIR}/caches_112_160_224_352_448"
  elif [[ -d "${OFFICIAL_RUN_DIR}/caches" ]]; then
    DEFAULT_CACHES_DIR="${OFFICIAL_RUN_DIR}/caches"
  fi
fi

PROCESSED_DIR="${PROCESSED_DIR:-${DEFAULT_PROCESSED_DIR}}"
CACHES_DIR="${CACHES_DIR:-${DEFAULT_CACHES_DIR}}"

if [[ ! -d "${PROCESSED_DIR}" ]]; then
  echo "ERROR: PROCESSED_DIR not found: ${PROCESSED_DIR}" >&2
  exit 2
fi
if [[ -z "${CACHES_DIR}" || ! -d "${CACHES_DIR}" ]]; then
  echo "ERROR: CACHES_DIR not found. Set CACHES_DIR to a dir containing <clip_id>.npz feature caches." >&2
  echo "Hint: run bash scripts/ave_verify_official_after_install.sh first." >&2
  exit 2
fi

SPLIT_TRAIN="${SPLIT_TRAIN:-train}"
SPLIT_EVAL="${SPLIT_EVAL:-val}"

DEFAULT_TRAIN_IDS_FILE="${META_DIR}/download_ok_train_official.txt"
if [[ ! -f "${DEFAULT_TRAIN_IDS_FILE}" ]]; then
  DEFAULT_TRAIN_IDS_FILE="${META_DIR}/download_ok_train_auto.txt"
fi
TRAIN_IDS_FILE="${TRAIN_IDS_FILE:-${DEFAULT_TRAIN_IDS_FILE}}"

DEFAULT_EVAL_IDS_FILE="${META_DIR}/download_ok_val_official.txt"
if [[ ! -f "${DEFAULT_EVAL_IDS_FILE}" ]]; then
  DEFAULT_EVAL_IDS_FILE="${META_DIR}/download_ok_val_auto.txt"
fi
EVAL_IDS_FILE="${EVAL_IDS_FILE:-${DEFAULT_EVAL_IDS_FILE}}"

LIMIT_TRAIN="${LIMIT_TRAIN:-3339}"
LIMIT_EVAL="${LIMIT_EVAL:-402}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-2e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
P_FILTER="${P_FILTER:-0.1}"

# Prefer GPU if available; fall back to CPU.
if command -v nvidia-smi >/dev/null 2>&1; then
  NUM_GPUS="$(nvidia-smi -L | wc -l | tr -d ' ')"
else
  NUM_GPUS="0"
fi
if [[ -z "${DEVICE:-}" ]]; then
  DEVICE="$([[ "${NUM_GPUS}" -gt 0 ]] && echo cuda:0 || echo cpu)"
fi
AUDIO_DEVICE="${AUDIO_DEVICE:-${DEVICE}}"
TRAIN_DEVICE="${TRAIN_DEVICE:-${DEVICE}}"

ALLOW_MISSING="${ALLOW_MISSING:-1}"
ALLOW_MISSING_FLAG=()
if [[ "${ALLOW_MISSING}" == "1" ]]; then
  ALLOW_MISSING_FLAG+=(--allow-missing)
fi

OUT_DIR="${OUT_DIR:-runs/E0014_ave_p0_sweep_official_val_ast_$(date +%Y%m%d-%H%M%S)}"

python -m avs.experiments.ave_p0_sweep sweep \
  --candidate-set ast_v1 \
  --meta-dir "${META_DIR}" \
  --processed-dir "${PROCESSED_DIR}" \
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
  --eventness-method ast \
  --ast-pretrained \
  --audio-device "${AUDIO_DEVICE}" \
  --train-device "${TRAIN_DEVICE}" \
  --p-filter "${P_FILTER}" \
  --out-dir "${OUT_DIR}"

echo "OK: ${OUT_DIR}/sweep_summary.json"
echo "BEST: ${OUT_DIR}/best_config.json"

