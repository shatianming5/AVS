#!/usr/bin/env bash
set -euo pipefail

# E0214: Fixed-space sweep on official AVE val402 for learned anchors with always-maxHigh1.
#
# Goal: Find a stronger anchored_top2 configuration under the same downstream protocol by forcing
# `max_high_anchors=1` (candidate_set=ltl_maxhigh1_v1) to preserve more base-res context.
#
# Produces:
#   - runs/E0214_*/sweep_summary.json
#   - runs/E0214_*/best_config.json
#   - runs/E0214_*/eventness_scores.json (cached per-second scores; reused across candidates and later test reproduction)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

META_DIR="${META_DIR:-data/AVE/meta}"

# Prefer existing official processed/caches when present.
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

# For fast iteration, default to 3 seeds; override to 0..9 for full selection.
SEEDS="${SEEDS:-0,1,2}"

# Stage-1 anchor proposal method.
EVENTNESS="${EVENTNESS:-av_clipdiff_mlp}"

# Stage-2 candidate set.
CANDIDATE_SET="${CANDIDATE_SET:-ltl_maxhigh1_v1}"

EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-2e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
P_FILTER="${P_FILTER:-0.1}"

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

OUT_DIR="${OUT_DIR:-runs/E0214_ave_p0_sweep_official_val_${EVENTNESS}_${CANDIDATE_SET}_$(date +%Y%m%d-%H%M%S)}"

SCORES_JSON="${SCORES_JSON:-${OUT_DIR}/eventness_scores.json}"
SCORES_FLAG=()
if [[ -n "${SCORES_JSON}" ]]; then
  SCORES_FLAG+=(--scores-json "${SCORES_JSON}")
fi

python -m avs.experiments.ave_p0_sweep sweep \
  --candidate-set "${CANDIDATE_SET}" \
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
  --audio-device "${AUDIO_DEVICE}" \
  --train-device "${TRAIN_DEVICE}" \
  --p-filter "${P_FILTER}" \
  "${SCORES_FLAG[@]}" \
  --out-dir "${OUT_DIR}"

echo "OK: ${OUT_DIR}/sweep_summary.json"
echo "BEST: ${OUT_DIR}/best_config.json"
