#!/usr/bin/env bash
set -euo pipefail

# E0397: Quick test402 reproduction (SEEDS=0..2) for the E0396 winner + diagnosis (E0344).
#
# If BEST_CONFIG_JSON is not set, uses the latest `runs/E0396_*/best_config.json`.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clipdiff_flow_mlp}"

if [[ -z "${BEST_CONFIG_JSON:-}" ]]; then
  latest_best="$(ls -t runs/E0396_*/best_config.json 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_best}" && -f "${latest_best}" ]]; then
    BEST_CONFIG_JSON="${latest_best}"
    echo "[e0397] BEST_CONFIG_JSON not set; using ${BEST_CONFIG_JSON}"
  else
    echo "ERROR: BEST_CONFIG_JSON is required (expected runs/E0396_*/best_config.json)" >&2
    exit 2
  fi
fi

SEEDS="${SEEDS:-0,1,2}"

if command -v nvidia-smi >/dev/null 2>&1; then
  AUDIO_DEVICE="${AUDIO_DEVICE:-cuda:0}"
  TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:0}"
fi

OUT_DIR="${OUT_DIR:-runs/E0397_quick_test402_${EVENTNESS}_k1ext_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export BEST_CONFIG_JSON
export SEEDS
export AUDIO_DEVICE
export TRAIN_DEVICE
export OUT_DIR

bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh

IN_METRICS="${OUT_DIR}/metrics.json" bash scripts/e0344_ave_p0_diagnose.sh

