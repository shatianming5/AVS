#!/usr/bin/env bash
set -euo pipefail

# E0314: Fixed-space sweep on official AVE val402 for teacher-student Stage-1 "visual usefulness" anchors.
#
# Thin wrapper around `scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` that:
#   - defaults EVENTNESS to `av_clipdiff_visgain_mlp`
#   - defaults CANDIDATE_SET to `ltl_top1med_norm_v1` (scale-invariant gate)
#   - writes outputs under runs/E0314_*

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clipdiff_visgain_mlp}"
CANDIDATE_SET="${CANDIDATE_SET:-ltl_top1med_norm_v1}"

OUT_DIR="${OUT_DIR:-runs/E0314_ave_p0_sweep_official_val_${EVENTNESS}_${CANDIDATE_SET}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export CANDIDATE_SET
export OUT_DIR

bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh

