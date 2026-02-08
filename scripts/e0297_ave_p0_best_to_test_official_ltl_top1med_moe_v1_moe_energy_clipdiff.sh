#!/usr/bin/env bash
set -euo pipefail

# E0297: Best-to-test reproduction on official AVE test402 for configs selected by E0296.
#
# Usage:
#   BEST_CONFIG_JSON=runs/E0296_.../best_config.json EVENTNESS=moe_energy_clipdiff bash scripts/e0297_ave_p0_best_to_test_official_ltl_top1med_moe_v1_moe_energy_clipdiff.sh
#
# If BEST_CONFIG_JSON is not set, this script uses the latest `runs/E0296_*/best_config.json`.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-moe_energy_clipdiff}"

if [[ -z "${BEST_CONFIG_JSON:-}" ]]; then
  latest_best="$(ls -t runs/E0296_*/best_config.json 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_best}" && -f "${latest_best}" ]]; then
    BEST_CONFIG_JSON="${latest_best}"
    echo "[e0297] BEST_CONFIG_JSON not set; using ${BEST_CONFIG_JSON}"
  else
    echo "ERROR: BEST_CONFIG_JSON is required (expected runs/E0296_*/best_config.json)" >&2
    exit 2
  fi
fi

OUT_DIR="${OUT_DIR:-runs/E0297_ave_p0_best_to_test_official_${EVENTNESS}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export BEST_CONFIG_JSON
export OUT_DIR

bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh

