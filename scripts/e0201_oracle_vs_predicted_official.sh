#!/usr/bin/env bash
set -euo pipefail

# E0201: Oracle vs Predicted gap report (Listen-then-Look MDE-2) on official AVE.
# Runs: uniform/random/anchored/oracle under a fixed token budget and writes:
#   - runs/E0201_*/oracle_vs_predicted.json
#   - runs/E0201_*/metrics_predicted.json (+ optional metrics_cheap_visual.json)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

META_DIR="${META_DIR:-data/AVE/meta}"

# Prefer existing full-cache artifacts when available (fallback to local layout).
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

EVENTNESS="${EVENTNESS:-energy}"

# Optional: fix all P0 knobs (including triad) via `ave_p0_sweep`'s best_config.json.
BASE_CONFIG_JSON="${BASE_CONFIG_JSON:-}"
BASE_CFG_FLAG=()
if [[ -n "${BASE_CONFIG_JSON}" ]]; then
  if [[ ! -f "${BASE_CONFIG_JSON}" ]]; then
    echo "ERROR: BASE_CONFIG_JSON not found: ${BASE_CONFIG_JSON}" >&2
    exit 2
  fi
  BASE_CFG_FLAG+=(--base-config-json "${BASE_CONFIG_JSON}")
fi

# Optional: Stage-1 scores cache (recommended for external teachers like PSP).
SCORES_JSON="${SCORES_JSON:-}"
SCORES_FLAG=()
if [[ -n "${SCORES_JSON}" ]]; then
  if [[ ! -f "${SCORES_JSON}" ]]; then
    echo "ERROR: SCORES_JSON not found: ${SCORES_JSON}" >&2
    exit 2
  fi
  SCORES_FLAG+=(--scores-json "${SCORES_JSON}")
fi

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

# Optional scale-invariant confidence gate (recommended for cross-method comparisons).
ANCHOR_CONF_METRIC="${ANCHOR_CONF_METRIC:-}"
ANCHOR_CONF_THRESHOLD="${ANCHOR_CONF_THRESHOLD:-}"
ANCHOR_CONF_FLAG=()
if [[ -n "${ANCHOR_CONF_METRIC}" || -n "${ANCHOR_CONF_THRESHOLD}" ]]; then
  if [[ -z "${ANCHOR_CONF_METRIC}" || -z "${ANCHOR_CONF_THRESHOLD}" ]]; then
    echo "ERROR: ANCHOR_CONF_METRIC and ANCHOR_CONF_THRESHOLD must be set together" >&2
    exit 2
  fi
  ANCHOR_CONF_FLAG+=(--anchor-conf-metric "${ANCHOR_CONF_METRIC}" --anchor-conf-threshold "${ANCHOR_CONF_THRESHOLD}")
fi

# Match the strongest known full-split protocol (head-only training).
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
AUDIO_DEVICE="${AUDIO_DEVICE:-${DEVICE}}"

ALLOW_MISSING="${ALLOW_MISSING:-1}"
ALLOW_MISSING_FLAG=()
if [[ "${ALLOW_MISSING}" == "1" ]]; then
  ALLOW_MISSING_FLAG+=(--allow-missing)
fi

INCLUDE_CHEAP_VISUAL="${INCLUDE_CHEAP_VISUAL:-1}"
CHEAP_VISUAL_FLAG=()
if [[ "${INCLUDE_CHEAP_VISUAL}" == "1" ]]; then
  CHEAP_VISUAL_FLAG+=(--include-cheap-visual)
fi

OUT_DIR="${OUT_DIR:-runs/E0201_oracle_vs_predicted_${EVENTNESS}_$(date +%Y%m%d-%H%M%S)}"

python -m avs.experiments.mde_ltl oracle_vs_predicted \
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
  --seeds "${SEEDS}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --train-device "${TRAIN_DEVICE}" \
  --audio-device "${AUDIO_DEVICE}" \
  --eventness-method "${EVENTNESS}" \
  "${BASE_CFG_FLAG[@]}" \
  "${SCORES_FLAG[@]}" \
  "${AST_PRETRAINED_FLAG[@]}" \
  "${ANCHOR_CONF_FLAG[@]}" \
  "${CHEAP_VISUAL_FLAG[@]}"

echo "OK: ${OUT_DIR}/oracle_vs_predicted.json"
