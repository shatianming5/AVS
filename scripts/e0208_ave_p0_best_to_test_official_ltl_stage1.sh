#!/usr/bin/env bash
set -euo pipefail

# E0208: Reproduce the best config selected by E0207 on official AVE test402.
#
# Usage:
#   BEST_CONFIG_JSON=runs/E0207_.../best_config.json EVENTNESS=av_clipdiff_lr bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh
#
# Produces:
#   - runs/E0208_*/metrics.json

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${BEST_CONFIG_JSON:-}" ]]; then
  latest_best="$(ls -t runs/E0207_*/best_config.json 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_best}" && -f "${latest_best}" ]]; then
    BEST_CONFIG_JSON="${latest_best}"
    echo "[e0208] BEST_CONFIG_JSON not set; using ${BEST_CONFIG_JSON}"
  else
    echo "ERROR: BEST_CONFIG_JSON is required (path to best_config.json from E0207)" >&2
    exit 2
  fi
fi
if [[ ! -f "${BEST_CONFIG_JSON}" ]]; then
  echo "ERROR: BEST_CONFIG_JSON not found: ${BEST_CONFIG_JSON}" >&2
  exit 2
fi

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
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"

EVENTNESS="${EVENTNESS:-av_clipdiff_lr}"

# AST backends require pretrained weights to be meaningful; default-enable when using an AST-based method.
AST_PRETRAINED="${AST_PRETRAINED:-}"
AST_FLAG=()
if [[ "${AST_PRETRAINED}" == "1" ]]; then
  AST_FLAG+=(--ast-pretrained)
elif [[ "${AST_PRETRAINED}" == "0" ]]; then
  AST_FLAG=()
else
  if [[ "${EVENTNESS}" == ast* || "${EVENTNESS}" == av_ast* ]]; then
    AST_FLAG+=(--ast-pretrained)
  fi
fi

EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-2e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"

if command -v nvidia-smi >/dev/null 2>&1; then
  DEVICE="${DEVICE:-cuda:0}"
else
  DEVICE="${DEVICE:-cpu}"
fi
TRAIN_DEVICE="${TRAIN_DEVICE:-${DEVICE}}"

# Stage-1 scoring is CPU-friendly for the default methods here.
AUDIO_DEVICE="${AUDIO_DEVICE:-cpu}"

ALLOW_MISSING="${ALLOW_MISSING:-1}"
ALLOW_MISSING_FLAG=()
if [[ "${ALLOW_MISSING}" == "1" ]]; then
  ALLOW_MISSING_FLAG+=(--allow-missing)
fi

# Reuse the sweep's score cache if present; auto-fills missing ids (e.g., test clips).
SCORES_JSON="${SCORES_JSON:-$(dirname "${BEST_CONFIG_JSON}")/eventness_scores.json}"
SCORES_FLAG=()
if [[ -f "${SCORES_JSON}" ]]; then
  SCORES_FLAG+=(--scores-json "${SCORES_JSON}")
fi

OUT_DIR="${OUT_DIR:-runs/E0208_ave_p0_best_to_test_official_${EVENTNESS}_$(date +%Y%m%d-%H%M%S)}"

python -m avs.experiments.ave_p0_sweep run \
  --config-json "${BEST_CONFIG_JSON}" \
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
  --eventness-method "${EVENTNESS}" \
  "${AST_FLAG[@]}" \
  --audio-device "${AUDIO_DEVICE}" \
  --train-device "${TRAIN_DEVICE}" \
  "${SCORES_FLAG[@]}" \
  --out-dir "${OUT_DIR}"

echo "OK: ${OUT_DIR}/metrics.json"
