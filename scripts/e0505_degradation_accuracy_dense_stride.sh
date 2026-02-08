#!/usr/bin/env bash
set -euo pipefail

# E0505: downstream degradation-accuracy rerun on dense-stride promoted config.
#
# Thin wrapper over E0331 with defaults pinned to the dense-stride config.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BEST_CONFIG_JSON="${BEST_CONFIG_JSON:-}"
if [[ -z "${BEST_CONFIG_JSON}" ]]; then
  BEST_CONFIG_JSON="$(ls -t runs/E0400c_*/ltltop1med_thr0p5_shift1/config.json 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${BEST_CONFIG_JSON}" ]]; then
  BEST_CONFIG_JSON="$(ls -t runs/E0400c_*/best_config.json 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${BEST_CONFIG_JSON}" ]]; then
  BEST_CONFIG_JSON="$(ls -t runs/E0400b_*/best_config.json 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${BEST_CONFIG_JSON}" || ! -f "${BEST_CONFIG_JSON}" ]]; then
  echo "ERROR: BEST_CONFIG_JSON not found; set BEST_CONFIG_JSON=<path/to/config.json>." >&2
  exit 2
fi

SCORES_JSON="${SCORES_JSON:-runs/E0405_shared_scores_av_clipdiff_flow_mlp_stride_test402.json}"
if [[ ! -f "${SCORES_JSON}" ]]; then
  echo "ERROR: SCORES_JSON not found: ${SCORES_JSON}" >&2
  exit 2
fi

EVENTNESS_METHOD="${EVENTNESS_METHOD:-av_clipdiff_flow_mlp_stride}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
SHIFT_GRID="${SHIFT_GRID:--0.5,0,0.5}"
SNR_GRID="${SNR_GRID:-20,10,0}"
SILENCE_GRID="${SILENCE_GRID:-0,0.5}"
ALPHA_GRID="${ALPHA_GRID:-0,0.5,1}"

OUT_DIR="${OUT_DIR:-runs/E0505_degradation_accuracy_dense_stride_$(date +%Y%m%d-%H%M%S)}"

export BEST_CONFIG_JSON
export SCORES_JSON
export EVENTNESS_METHOD
export SEEDS
export SHIFT_GRID
export SNR_GRID
export SILENCE_GRID
export ALPHA_GRID
export OUT_DIR

bash scripts/e0331_degradation_accuracy_official.sh
