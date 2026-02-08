#!/usr/bin/env bash
set -euo pipefail

# E0286: Fixed-space sweep on official AVE val402 for learned anchors using:
#   EVENTNESS=av_clipdiff_mlp_r224 (Stage-1; clipdiff computed from 224px CLIP caches)
#   CANDIDATE_SET=ltl_top1med_v1 (Stage-2 + top1-med confidence gate)
#
# This is a thin wrapper around `scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clipdiff_mlp_r224}"
CANDIDATE_SET="${CANDIDATE_SET:-ltl_top1med_v1}"

OUT_DIR="${OUT_DIR:-runs/E0286_ave_p0_sweep_official_val_${EVENTNESS}_${CANDIDATE_SET}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export CANDIDATE_SET
export OUT_DIR

bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh

