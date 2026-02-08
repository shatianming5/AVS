#!/usr/bin/env bash
set -euo pipefail

# Confirm fusion adds on top of sampling on official AVE test402 using AST anchors:
# compares `audio_concat_uniform` vs `audio_concat_anchored_top2` under the best AST sampling config.
#
# Usage:
#   BEST_CONFIG_JSON=runs/E0014_.../best_config.json bash scripts/e0017_ave_fusion_confirm_official_ast.sh
#
# Produces:
#   - runs/E0017_*/metrics.json

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${BEST_CONFIG_JSON:-}" ]]; then
  latest_best="$(ls -t runs/E0014_*/best_config.json 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_best}" && -f "${latest_best}" ]]; then
    BEST_CONFIG_JSON="${latest_best}"
    echo "[e0017] BEST_CONFIG_JSON not set; using ${BEST_CONFIG_JSON}"
  else
    echo "ERROR: BEST_CONFIG_JSON is required (path to best_config.json from E0014)" >&2
    exit 2
  fi
fi
if [[ ! -f "${BEST_CONFIG_JSON}" ]]; then
  echo "ERROR: BEST_CONFIG_JSON not found: ${BEST_CONFIG_JSON}" >&2
  exit 2
fi

SCORES_JSON_FLAG=()
if [[ -z "${SCORES_JSON:-}" ]]; then
  cand="$(dirname "${BEST_CONFIG_JSON}")/eventness_scores.json"
  if [[ -f "${cand}" ]]; then
    SCORES_JSON="${cand}"
    echo "[e0017] SCORES_JSON not set; using ${SCORES_JSON}"
  fi
fi
if [[ -n "${SCORES_JSON:-}" ]]; then
  SCORES_JSON_FLAG+=(--scores-json "${SCORES_JSON}")
fi

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
  exit 2
fi

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

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-2e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"

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

OUT_DIR="${OUT_DIR:-runs/E0017_ave_fusion_confirm_official_ast_$(date +%Y%m%d-%H%M%S)}"

python -m avs.experiments.ave_p0_fusion_confirm \
  --config-json "${BEST_CONFIG_JSON}" \
  --meta-dir "${META_DIR}" \
  --processed-dir "${PROCESSED_DIR}" \
  --caches-dir "${CACHES_DIR}" \
  "${SCORES_JSON_FLAG[@]}" \
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
  --out-dir "${OUT_DIR}"

echo "OK: ${OUT_DIR}/metrics.json"
