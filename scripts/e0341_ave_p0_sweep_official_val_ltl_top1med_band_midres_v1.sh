#!/usr/bin/env bash
set -euo pipefail

# E0341: Val402 sweep for the mid-res band-budget attempt (P0116).
#
# Usage:
#   EVENTNESS=av_clipdiff_mlp bash scripts/e0341_ave_p0_sweep_official_val_ltl_top1med_band_midres_v1.sh
#
# Produces:
#   - runs/E0341_*/sweep_summary.json
#   - runs/E0341_*/best_config.json
#   - runs/E0341_*/eventness_scores.json

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clipdiff_mlp}"
CANDIDATE_SET="${CANDIDATE_SET:-ltl_top1med_band_midres_v1}"

DEFAULT_CACHES_DIR="runs/REAL_AVE_OFFICIAL_20260201-124535/caches_112_160_192_208_224_320_352"
if [[ -z "${CACHES_DIR:-}" ]]; then
  if [[ -d "${DEFAULT_CACHES_DIR}" ]]; then
    CACHES_DIR="${DEFAULT_CACHES_DIR}"
  else
    echo "ERROR: CACHES_DIR not set and default mid-res caches not found: ${DEFAULT_CACHES_DIR}" >&2
    echo "Run: bash scripts/e0340_ave_cache_official_midres.sh" >&2
    exit 2
  fi
fi
export CACHES_DIR

OUT_DIR="${OUT_DIR:-runs/E0341_ave_p0_sweep_official_val_${EVENTNESS}_${CANDIDATE_SET}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export CANDIDATE_SET
export OUT_DIR

bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh

