#!/usr/bin/env bash
set -euo pipefail

# E0366: Full test402 reproduction (SEEDS=0..9) for configs selected by E0364 → attempt to prove C0003.
#
# Usage:
#   BEST_CONFIG_JSON=runs/E0364_.../best_config.json EVENTNESS=av_clipdiff_mlp bash scripts/e0366_ave_p0_best_to_test_full_official_ltl_top1med_keepadj_basealloc_highonly_v1.sh
#
# If BEST_CONFIG_JSON is not set, uses the latest `runs/E0364_*/best_config.json`.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clipdiff_mlp}"

if [[ -z "${BEST_CONFIG_JSON:-}" ]]; then
  latest_best="$(ls -t runs/E0364_*/best_config.json 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_best}" && -f "${latest_best}" ]]; then
    BEST_CONFIG_JSON="${latest_best}"
    echo "[e0366] BEST_CONFIG_JSON not set; using ${BEST_CONFIG_JSON}"
  else
    echo "ERROR: BEST_CONFIG_JSON is required (expected runs/E0364_*/best_config.json)" >&2
    exit 2
  fi
fi

OUT_DIR="${OUT_DIR:-runs/E0366_full_test402_${EVENTNESS}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export BEST_CONFIG_JSON
export OUT_DIR

bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh

