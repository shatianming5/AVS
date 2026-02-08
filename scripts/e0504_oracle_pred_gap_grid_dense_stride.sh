#!/usr/bin/env bash
set -euo pipefail

# E0504: multi-budget Oracle→Predicted gap grid on dense-stride Stage-1.
#
# Thin wrapper over E0330 with defaults bound to `av_clipdiff_flow_mlp_stride`.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clipdiff_flow_mlp_stride}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
INCLUDE_CHEAP_VISUAL="${INCLUDE_CHEAP_VISUAL:-1}"
TRIADS="${TRIADS:-112,160,224;160,224,352;224,352,448}"
BUDGET_MODE="${BUDGET_MODE:-auto}"
BUDGET_EPSILON_FRAC="${BUDGET_EPSILON_FRAC:-0.05}"
BUDGET_EXTRA_RESOLUTIONS="${BUDGET_EXTRA_RESOLUTIONS:-112,160,224,352,448}"

BASE_CONFIG_JSON="${BASE_CONFIG_JSON:-}"
if [[ -z "${BASE_CONFIG_JSON}" ]]; then
  BASE_CONFIG_JSON="$(ls -t runs/E0400c_*/ltltop1med_thr0p5_shift1/config.json 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${BASE_CONFIG_JSON}" ]]; then
  BASE_CONFIG_JSON="$(ls -t runs/E0400c_*/best_config.json 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${BASE_CONFIG_JSON}" ]]; then
  BASE_CONFIG_JSON="$(ls -t runs/E0400b_*/best_config.json 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${BASE_CONFIG_JSON}" || ! -f "${BASE_CONFIG_JSON}" ]]; then
  echo "ERROR: BASE_CONFIG_JSON not found; set BASE_CONFIG_JSON=<path/to/config.json>." >&2
  exit 2
fi

SCORES_JSON="${SCORES_JSON:-runs/E0405_shared_scores_av_clipdiff_flow_mlp_stride_test402.json}"
OUT_DIR="${OUT_DIR:-runs/E0504_oracle_pred_gap_grid_dense_stride_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export BASE_CONFIG_JSON
export SCORES_JSON
export SEEDS
export INCLUDE_CHEAP_VISUAL
export TRIADS
export BUDGET_MODE
export BUDGET_EPSILON_FRAC
export BUDGET_EXTRA_RESOLUTIONS
export OUT_DIR

bash scripts/e0330_mde_pareto_grid_official.sh
