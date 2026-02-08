#!/usr/bin/env bash
set -euo pipefail

# E0299: Reproduce the E0298 selection on official AVE test402 (SEEDS=0..9).
#
# Usage:
#   BEST_CONFIG_JSON=runs/E0298_.../best_config.json bash scripts/e0299_ave_p0_best_to_test_official_ltl_top1med_bridgealloc_v1.sh
#
# This is a thin wrapper around `scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh`.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${BEST_CONFIG_JSON:-}" ]]; then
  latest_best="$(ls -t runs/E0298_*/best_config.json 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_best}" && -f "${latest_best}" ]]; then
    BEST_CONFIG_JSON="${latest_best}"
    echo "[e0299] BEST_CONFIG_JSON not set; using ${BEST_CONFIG_JSON}"
  else
    echo "ERROR: BEST_CONFIG_JSON is required (path to best_config.json from E0298)" >&2
    exit 2
  fi
fi

EVENTNESS="${EVENTNESS:-av_clipdiff_mlp}"

OUT_DIR="${OUT_DIR:-runs/E0299_ave_p0_best_to_test_official_${EVENTNESS}_bridgealloc_$(date +%Y%m%d-%H%M%S)}"

export BEST_CONFIG_JSON
export EVENTNESS
export OUT_DIR

bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh

