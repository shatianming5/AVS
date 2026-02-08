#!/usr/bin/env bash
set -euo pipefail

# E0203: Degradation suite (shift/noise/silence) on official AVE.
# Writes:
#   - runs/E0203_*/degradation_suite.json

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

META_DIR="${META_DIR:-data/AVE/meta}"

DEFAULT_PROCESSED="data/AVE/processed"
if [[ -d "runs/REAL_AVE_OFFICIAL_20260201-124535/processed" ]]; then
  DEFAULT_PROCESSED="runs/REAL_AVE_OFFICIAL_20260201-124535/processed"
elif [[ -d "runs/REAL_AVE_LOCAL/processed" ]]; then
  DEFAULT_PROCESSED="runs/REAL_AVE_LOCAL/processed"
fi
PROCESSED_DIR="${PROCESSED_DIR:-${DEFAULT_PROCESSED}}"

DEFAULT_CACHES="runs/REAL_AVE_LOCAL/caches"
if [[ -d "runs/REAL_AVE_OFFICIAL_20260201-124535/caches_112_160_224_352_448" ]]; then
  DEFAULT_CACHES="runs/REAL_AVE_OFFICIAL_20260201-124535/caches_112_160_224_352_448"
fi
CACHES_DIR="${CACHES_DIR:-${DEFAULT_CACHES}}"

SPLIT_TRAIN="${SPLIT_TRAIN:-train}"
SPLIT_EVAL="${SPLIT_EVAL:-test}"

DEFAULT_TRAIN_IDS_FILE="${META_DIR}/download_ok_train_official.txt"
if [[ ! -f "${DEFAULT_TRAIN_IDS_FILE}" ]]; then
  DEFAULT_TRAIN_IDS_FILE="${META_DIR}/download_ok_train_auto.txt"
fi
TRAIN_IDS_FILE="${TRAIN_IDS_FILE:-${DEFAULT_TRAIN_IDS_FILE}}"

DEFAULT_EVAL_IDS_FILE="${META_DIR}/download_ok_test_official.txt"
if [[ ! -f "${DEFAULT_EVAL_IDS_FILE}" ]]; then
  DEFAULT_EVAL_IDS_FILE="${META_DIR}/download_ok_test_auto.txt"
fi
EVAL_IDS_FILE="${EVAL_IDS_FILE:-${DEFAULT_EVAL_IDS_FILE}}"

LIMIT_TRAIN="${LIMIT_TRAIN:-3339}"
LIMIT_EVAL="${LIMIT_EVAL:-402}"

EVENTNESS="${EVENTNESS:-energy}"

# AST weights are opt-in elsewhere; for ast/ast_lr runs, default to pretrained unless explicitly disabled.
AST_PRETRAINED="${AST_PRETRAINED:-}"
if [[ -z "${AST_PRETRAINED}" ]]; then
  if [[ "${EVENTNESS}" == "ast" || "${EVENTNESS}" == "ast_lr" || "${EVENTNESS}" == "ast_emb_lr" || "${EVENTNESS}" == "ast_evt_mlp" || "${EVENTNESS}" == "ast_mlp_cls" || "${EVENTNESS}" == "ast_mlp_cls_target" ]]; then
    AST_PRETRAINED="1"
  else
    AST_PRETRAINED="0"
  fi
fi
AST_PRETRAINED_FLAG=()
if [[ "${AST_PRETRAINED}" == "1" ]]; then
  AST_PRETRAINED_FLAG+=(--ast-pretrained)
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  DEFAULT_AUDIO_DEVICE="cuda:0"
else
  DEFAULT_AUDIO_DEVICE="cpu"
fi
AUDIO_DEVICE="${AUDIO_DEVICE:-${DEFAULT_AUDIO_DEVICE}}"

ALLOW_MISSING="${ALLOW_MISSING:-1}"
ALLOW_MISSING_FLAG=()
if [[ "${ALLOW_MISSING}" == "1" ]]; then
  ALLOW_MISSING_FLAG+=(--allow-missing)
fi

SHIFT_GRID="${SHIFT_GRID:-}"
if [[ -z "${SHIFT_GRID}" ]]; then
  SHIFT_GRID="-0.5,0,0.5"
fi
SNR_GRID="${SNR_GRID:-20,10,0}"
SILENCE_GRID="${SILENCE_GRID:-0,0.5}"
DELTAS="${DELTAS:-0,1,2}"

OUT_DIR="${OUT_DIR:-runs/E0203_degradation_${EVENTNESS}_$(date +%Y%m%d-%H%M%S)}"

python -m avs.experiments.degradation_suite \
  --mode ave_official \
  --out-dir "${OUT_DIR}" \
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
  --eventness-method "${EVENTNESS}" \
  --audio-device "${AUDIO_DEVICE}" \
  "${AST_PRETRAINED_FLAG[@]}" \
  --shift-grid="${SHIFT_GRID}" \
  --snr-grid "${SNR_GRID}" \
  --silence-grid "${SILENCE_GRID}" \
  --deltas "${DELTAS}"

echo "OK: ${OUT_DIR}/degradation_suite.json"
