#!/usr/bin/env bash
set -euo pipefail

# E0213: Diagnostic ablation for learned anchors.
#
# Take a known good learned-anchor config (best_config.json) and force:
#   - max_high_anchors=1
#   - anchor_high_policy=fixed
# to test whether the main failure mode is "2-high harms context".
#
# Produces:
#   - runs/E0213_*/metrics.json
#
# Notes:
# - This is a diagnostic; do not select hyperparams on test. Prefer running on val402 unless you have
#   a specific reason to evaluate on test402.

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

SEEDS="${SEEDS:-0,1,2}"
EVENTNESS="${EVENTNESS:-av_clipdiff_mlp}"

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

SOURCE_CONFIG_JSON="${SOURCE_CONFIG_JSON:-}"
if [[ -z "${SOURCE_CONFIG_JSON}" ]]; then
  # Prefer the known best adaptive_v1 selection for av_clipdiff_mlp if present; otherwise pick the newest E0207 best_config.json.
  candidate="runs/E0207_ave_p0_sweep_official_val_av_clipdiff_mlp_20260204-102403/best_config.json"
  if [[ -f "${candidate}" ]]; then
    SOURCE_CONFIG_JSON="${candidate}"
  else
    latest_best="$(ls -t runs/E0207_*/best_config.json 2>/dev/null | head -n 1 || true)"
    if [[ -n "${latest_best}" && -f "${latest_best}" ]]; then
      SOURCE_CONFIG_JSON="${latest_best}"
    else
      echo "ERROR: SOURCE_CONFIG_JSON is required (path to a best_config.json to ablate)" >&2
      exit 2
    fi
  fi
fi
if [[ ! -f "${SOURCE_CONFIG_JSON}" ]]; then
  echo "ERROR: SOURCE_CONFIG_JSON not found: ${SOURCE_CONFIG_JSON}" >&2
  exit 2
fi

OUT_DIR="${OUT_DIR:-runs/E0213_ave_p0_diagnostic_maxhigh1_${EVENTNESS}_$(date +%Y%m%d-%H%M%S)}"
CONFIG_JSON="${CONFIG_JSON:-${OUT_DIR}/config_maxhigh1.json}"

export SOURCE_CONFIG_JSON
export CONFIG_JSON

python - <<'PY'
import json
import os
from pathlib import Path

src = Path(os.environ["SOURCE_CONFIG_JSON"])
dst = Path(os.environ["CONFIG_JSON"])
dst.parent.mkdir(parents=True, exist_ok=True)

cfg = json.loads(src.read_text(encoding="utf-8"))
cfg["name"] = f"{cfg.get('name','cfg')}_maxHigh1"
cfg["max_high_anchors"] = 1
cfg["anchor_high_policy"] = "fixed"

dst.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(dst)
PY

# Reuse the source sweep's score cache if present (safe: only used to avoid recomputing Stage-1 scores).
SCORES_JSON="${SCORES_JSON:-$(dirname "${SOURCE_CONFIG_JSON}")/eventness_scores.json}"
SCORES_FLAG=()
if [[ -f "${SCORES_JSON}" ]]; then
  SCORES_FLAG+=(--scores-json "${SCORES_JSON}")
fi

python -m avs.experiments.ave_p0_sweep run \
  --config-json "${CONFIG_JSON}" \
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
  "${SCORES_FLAG[@]}" \
  --out-dir "${OUT_DIR}"

echo "OK: ${OUT_DIR}/metrics.json"
