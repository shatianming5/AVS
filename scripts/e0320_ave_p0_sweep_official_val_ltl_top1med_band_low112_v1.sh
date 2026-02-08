#!/usr/bin/env bash
set -euo pipefail

# E0320: Fixed-space sweep on official AVE val402 for a bold Stage-2 variant:
#   - band-budget planner (<=1% underbudget)
#   - low_res=112 + extra mid-res=160 to preserve context under the same token budget
#
# Thin wrapper around `scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` that:
#   - defaults EVENTNESS to `av_clipdiff_mlp` (current best Stage-1)
#   - defaults CANDIDATE_SET to `ltl_top1med_band_low112_v1`
#   - writes outputs under runs/E0320_*

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clipdiff_mlp}"
CANDIDATE_SET="${CANDIDATE_SET:-ltl_top1med_band_low112_v1}"

OUT_DIR="${OUT_DIR:-runs/E0320_ave_p0_sweep_official_val_${EVENTNESS}_${CANDIDATE_SET}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export CANDIDATE_SET
export OUT_DIR

bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh

