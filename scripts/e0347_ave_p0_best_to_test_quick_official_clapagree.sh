#!/usr/bin/env bash
set -euo pipefail

# E0347: Quick test402 reproduction (SEEDS=0..2) for the E0346 winner (av_clap_clip_agree).
#
# Usage:
#   BEST_CONFIG_JSON=runs/E0346_.../best_config.json bash scripts/e0347_ave_p0_best_to_test_quick_official_clapagree.sh
#
# Produces:
#   - runs/E0347_*/metrics.json

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clap_clip_agree}"

if [[ -z "${BEST_CONFIG_JSON:-}" ]]; then
  latest_best="$(ls -t runs/E0346_*/best_config.json 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_best}" && -f "${latest_best}" ]]; then
    BEST_CONFIG_JSON="${latest_best}"
    echo "[e0347] BEST_CONFIG_JSON not set; using ${BEST_CONFIG_JSON}"
  else
    echo "ERROR: BEST_CONFIG_JSON is required (E0346 best_config.json)" >&2
    exit 2
  fi
fi
if [[ ! -f "${BEST_CONFIG_JSON}" ]]; then
  echo "ERROR: BEST_CONFIG_JSON not found: ${BEST_CONFIG_JSON}" >&2
  exit 2
fi

SEEDS="${SEEDS:-0,1,2}"

# CLAP is expensive; default to GPU for Stage-1 scoring.
if command -v nvidia-smi >/dev/null 2>&1; then
  DEVICE="${DEVICE:-cuda:0}"
else
  DEVICE="${DEVICE:-cpu}"
fi
AUDIO_DEVICE="${AUDIO_DEVICE:-${DEVICE}}"
TRAIN_DEVICE="${TRAIN_DEVICE:-${DEVICE}}"
export AUDIO_DEVICE
export TRAIN_DEVICE

OUT_DIR="${OUT_DIR:-runs/E0347_quick_test402_${EVENTNESS}_$(date +%Y%m%d-%H%M%S)}"
export OUT_DIR
export BEST_CONFIG_JSON
export EVENTNESS
export SEEDS

bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh

