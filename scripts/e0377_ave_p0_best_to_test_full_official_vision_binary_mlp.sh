#!/usr/bin/env bash
set -euo pipefail

# E0377: Full test402 reproduction (SEEDS=0..9) for the E0375 winner (gated by E0376 quick).
#
# Usage:
#   bash scripts/e0377_ave_p0_best_to_test_full_official_vision_binary_mlp.sh
#
# If BEST_CONFIG_JSON is not set, uses the latest `runs/E0375_*/best_config.json`.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-vision_binary_mlp}"

if [[ -z "${BEST_CONFIG_JSON:-}" ]]; then
  latest_best="$(ls -t runs/E0375_*/best_config.json 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_best}" && -f "${latest_best}" ]]; then
    BEST_CONFIG_JSON="${latest_best}"
    echo "[e0377] BEST_CONFIG_JSON not set; using ${BEST_CONFIG_JSON}"
  else
    echo "ERROR: BEST_CONFIG_JSON is required (expected runs/E0375_*/best_config.json)" >&2
    exit 2
  fi
fi

SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
OUT_DIR="${OUT_DIR:-runs/E0377_full_test402_${EVENTNESS}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export BEST_CONFIG_JSON
export SEEDS
export OUT_DIR

bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh

