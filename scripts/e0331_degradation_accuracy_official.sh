#!/usr/bin/env bash
set -euo pipefail

# E0331: Degradation suite with downstream accuracy + α lower bound (official AVE).
#
# Produces:
#   - runs/E0331_*/degradation_accuracy.json
#
# Usage (recommended):
#   bash scripts/e0331_degradation_accuracy_official.sh
#
# Optional overrides:
#   BEST_CONFIG_JSON=.../best_config.json
#   SCORES_JSON=.../eventness_scores.json
#   EVENTNESS_METHOD=av_clipdiff_mlp
#   SEEDS=0,1,2  LIMIT_TRAIN=3339 LIMIT_EVAL=402  TRAIN_DEVICE=cuda:0
#   SHIFT_GRID="-0.5,0,0.5"  SNR_GRID="20,10,0"  SILENCE_GRID="0,0.5"  ALPHA_GRID="0,0.5,1"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1

if [[ -z "${BEST_CONFIG_JSON:-}" ]]; then
  latest_best="$(ls -t runs/E0223_*/best_config.json 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_best}" && -f "${latest_best}" ]]; then
    BEST_CONFIG_JSON="${latest_best}"
    echo "[e0331] BEST_CONFIG_JSON not set; using ${BEST_CONFIG_JSON}"
  else
    echo "ERROR: BEST_CONFIG_JSON is required (expected runs/E0223_*/best_config.json)" >&2
    exit 2
  fi
fi
if [[ ! -f "${BEST_CONFIG_JSON}" ]]; then
  echo "ERROR: BEST_CONFIG_JSON not found: ${BEST_CONFIG_JSON}" >&2
  exit 2
fi

SCORES_JSON="${SCORES_JSON:-$(dirname "${BEST_CONFIG_JSON}")/eventness_scores.json}"
if [[ ! -f "${SCORES_JSON}" ]]; then
  echo "ERROR: SCORES_JSON not found: ${SCORES_JSON}" >&2
  echo "Hint: run E0223/E0224 first to generate eventness_scores.json for the chosen Stage-1 backend." >&2
  exit 2
fi

EVENTNESS_METHOD="${EVENTNESS_METHOD:-av_clipdiff_mlp}"

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
SEEDS="${SEEDS:-0,1,2}"

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

ALLOW_MISSING="${ALLOW_MISSING:-1}"
ALLOW_MISSING_FLAG=()
if [[ "${ALLOW_MISSING}" == "1" ]]; then
  ALLOW_MISSING_FLAG+=(--allow-missing)
fi

SHIFT_GRID="${SHIFT_GRID:--0.5,0.0,0.5}"
SNR_GRID="${SNR_GRID:-20,10,0}"
SILENCE_GRID="${SILENCE_GRID:-0.0,0.5}"
ALPHA_GRID="${ALPHA_GRID:-0.0,0.5,1.0}"

OUT_DIR="${OUT_DIR:-runs/E0331_degradation_accuracy_${EVENTNESS_METHOD}_$(date +%Y%m%d-%H%M%S)}"

python -m avs.experiments.degradation_accuracy \
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
  --eventness-method "${EVENTNESS_METHOD}" \
  --base-config-json "${BEST_CONFIG_JSON}" \
  --scores-json "${SCORES_JSON}" \
  --seeds "${SEEDS}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --train-device "${TRAIN_DEVICE}" \
  --shift-grid="${SHIFT_GRID}" \
  --snr-grid="${SNR_GRID}" \
  --silence-grid="${SILENCE_GRID}" \
  --alpha-grid="${ALPHA_GRID}"

echo "OK: ${OUT_DIR}/degradation_accuracy.json"
