#!/usr/bin/env bash
set -euo pipefail

# E0330: Multi-budget Pareto grid on official AVE (Oracle→Predicted + controls).
#
# Produces:
#   - runs/E0330_*/pareto_report.json
#   - runs/E0330_*/pareto.png
#   - runs/E0330_*/metrics_*.json (per budget point)
#
# Usage (recommended):
#   EVENTNESS=av_clipdiff_mlp bash scripts/e0330_mde_pareto_grid_official.sh
#
# Optional overrides:
#   BASE_CONFIG_JSON=.../best_config.json
#   SCORES_JSON=.../eventness_scores.json
#   TRIADS="112,160,224;160,224,352;224,352,448"
#   BUDGET_MODE=band BUDGET_EPSILON_FRAC=0.05 BUDGET_EXTRA_RESOLUTIONS="112,160,224,352"
#   SEEDS=0,1,2  LIMIT_TRAIN=3339 LIMIT_EVAL=402  TRAIN_DEVICE=cuda:0  AUDIO_DEVICE=cpu
#   INCLUDE_CHEAP_VISUAL=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Ensure python logs are flushed to files/queues in real time.
export PYTHONUNBUFFERED=1

EVENTNESS="${EVENTNESS:-av_clipdiff_mlp}"

if [[ -z "${BASE_CONFIG_JSON:-}" ]]; then
  latest_best="$(ls -t runs/E0223_*/best_config.json 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_best}" && -f "${latest_best}" ]]; then
    BASE_CONFIG_JSON="${latest_best}"
    echo "[e0330] BASE_CONFIG_JSON not set; using ${BASE_CONFIG_JSON}"
  else
    echo "WARN: BASE_CONFIG_JSON not set and no runs/E0223_*/best_config.json found; using mde_ltl defaults." >&2
    BASE_CONFIG_JSON=""
  fi
fi
if [[ -n "${BASE_CONFIG_JSON}" && ! -f "${BASE_CONFIG_JSON}" ]]; then
  echo "ERROR: BASE_CONFIG_JSON not found: ${BASE_CONFIG_JSON}" >&2
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

# Stage-1 scoring can be GPU-heavy for AST; default to CPU otherwise.
AUDIO_DEVICE="${AUDIO_DEVICE:-cpu}"

ALLOW_MISSING="${ALLOW_MISSING:-1}"
ALLOW_MISSING_FLAG=()
if [[ "${ALLOW_MISSING}" == "1" ]]; then
  ALLOW_MISSING_FLAG+=(--allow-missing)
fi

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

TRIADS="${TRIADS:-112,160,224;160,224,352;224,352,448}"
# Default to `auto`: try exact budgets per-triad (so the mid-budget point matches the canonical
# test402 reproduction, e.g. E0224), and fall back to band budgets only when exact is impossible
# (e.g. 112/160/224 cannot exactly match tokens with only {low,base,high}).
BUDGET_MODE="${BUDGET_MODE:-auto}"
BUDGET_EPSILON_FRAC="${BUDGET_EPSILON_FRAC:-0.05}"
BUDGET_EXTRA_RESOLUTIONS="${BUDGET_EXTRA_RESOLUTIONS:-112,160,224,352,448}"

SCORES_JSON="${SCORES_JSON:-}"
if [[ -z "${SCORES_JSON}" && -n "${BASE_CONFIG_JSON}" ]]; then
  SCORES_JSON="$(dirname "${BASE_CONFIG_JSON}")/eventness_scores.json"
fi

INCLUDE_CHEAP_VISUAL="${INCLUDE_CHEAP_VISUAL:-1}"
CHEAP_VIS_FLAG=()
if [[ "${INCLUDE_CHEAP_VISUAL}" == "1" ]]; then
  CHEAP_VIS_FLAG+=(--include-cheap-visual)
fi

OUT_DIR="${OUT_DIR:-runs/E0330_mde_pareto_grid_official_${EVENTNESS}_$(date +%Y%m%d-%H%M%S)}"

BASE_CFG_FLAG=()
if [[ -n "${BASE_CONFIG_JSON}" ]]; then
  BASE_CFG_FLAG+=(--base-config-json "${BASE_CONFIG_JSON}")
fi

SCORES_FLAG=()
if [[ -n "${SCORES_JSON}" ]]; then
  SCORES_FLAG+=(--scores-json "${SCORES_JSON}")
fi

python -m avs.experiments.mde_ltl pareto_grid \
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
  --eventness-method "${EVENTNESS}" \
  "${CHEAP_VIS_FLAG[@]}" \
  --audio-device "${AUDIO_DEVICE}" \
  "${AST_FLAG[@]}" \
  --train-device "${TRAIN_DEVICE}" \
  "${BASE_CFG_FLAG[@]}" \
  --triads "${TRIADS}" \
  --budget-mode "${BUDGET_MODE}" \
  --budget-epsilon-frac "${BUDGET_EPSILON_FRAC}" \
  --budget-extra-resolutions "${BUDGET_EXTRA_RESOLUTIONS}" \
  "${SCORES_FLAG[@]}"

echo "OK: ${OUT_DIR}/pareto_report.json"
echo "OK: ${OUT_DIR}/pareto.png"
