#!/usr/bin/env bash
set -euo pipefail

# E0302: Reproduce the E0301 selection on official AVE test402 (SEEDS=0..9).
#
# Usage:
#   BEST_CONFIG_JSON=runs/E0301_.../best_config.json bash scripts/e0302_ave_p0_best_to_test_official_ltl_top1med_k1_extreme_v1.sh
#
# This is a thin wrapper around `scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${BEST_CONFIG_JSON:-}" ]]; then
  latest_best="$(ls -t runs/E0301_*/best_config.json 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_best}" && -f "${latest_best}" ]]; then
    BEST_CONFIG_JSON="${latest_best}"
    echo "[e0302] BEST_CONFIG_JSON not set; using ${BEST_CONFIG_JSON}"
  else
    echo "ERROR: BEST_CONFIG_JSON is required (path to best_config.json from E0301)" >&2
    exit 2
  fi
fi

EVENTNESS="${EVENTNESS:-av_clipdiff_mlp}"

OUT_DIR="${OUT_DIR:-runs/E0302_ave_p0_best_to_test_official_${EVENTNESS}_k1extreme_$(date +%Y%m%d-%H%M%S)}"

export BEST_CONFIG_JSON
export EVENTNESS
export OUT_DIR

bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh

