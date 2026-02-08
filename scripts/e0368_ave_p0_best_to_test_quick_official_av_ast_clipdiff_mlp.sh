#!/usr/bin/env bash
set -euo pipefail

# E0368: Quick test402 reproduction (SEEDS=0..2) for the E0367 winner (av_ast_clipdiff_mlp).
#
# Usage:
#   EVENTNESS=av_ast_clipdiff_mlp bash scripts/e0368_ave_p0_best_to_test_quick_official_av_ast_clipdiff_mlp.sh
#
# If BEST_CONFIG_JSON is not set, uses the latest `runs/E0367_*/best_config.json`.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_ast_clipdiff_mlp}"

if [[ -z "${BEST_CONFIG_JSON:-}" ]]; then
  latest_best="$(ls -t runs/E0367_*/best_config.json 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_best}" && -f "${latest_best}" ]]; then
    BEST_CONFIG_JSON="${latest_best}"
    echo "[e0368] BEST_CONFIG_JSON not set; using ${BEST_CONFIG_JSON}"
  else
    echo "ERROR: BEST_CONFIG_JSON is required (expected runs/E0367_*/best_config.json)" >&2
    exit 2
  fi
fi
if [[ ! -f "${BEST_CONFIG_JSON}" ]]; then
  echo "ERROR: BEST_CONFIG_JSON not found: ${BEST_CONFIG_JSON}" >&2
  exit 2
fi

SEEDS="${SEEDS:-0,1,2}"

if command -v nvidia-smi >/dev/null 2>&1; then
  DEVICE="${DEVICE:-cuda:0}"
else
  DEVICE="${DEVICE:-cpu}"
fi
AUDIO_DEVICE="${AUDIO_DEVICE:-${DEVICE}}"
TRAIN_DEVICE="${TRAIN_DEVICE:-${DEVICE}}"
export AUDIO_DEVICE
export TRAIN_DEVICE

OUT_DIR="${OUT_DIR:-runs/E0368_quick_test402_${EVENTNESS}_$(date +%Y%m%d-%H%M%S)}"
export OUT_DIR
export BEST_CONFIG_JSON
export EVENTNESS
export SEEDS

bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh

