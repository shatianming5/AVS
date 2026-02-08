#!/usr/bin/env bash
set -euo pipefail

# E0364: Fixed-space sweep on official AVE val402 for learned anchors using
# `candidate_set=ltl_top1med_keepadj_basealloc_highonly_v1`.
#
# Motivation: Under `adaptive_v3`, far-anchor clips are demoted to the 1-high regime, but the default planners
# still allocate base slots using distance-to-all anchors (including the demoted 2nd anchor), wasting budget.
# This sweep tests the new "*_high" base allocation modes that allocate base slots w.r.t. the high-set only.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clipdiff_mlp}"
CANDIDATE_SET="${CANDIDATE_SET:-ltl_top1med_keepadj_basealloc_highonly_v1}"

OUT_DIR="${OUT_DIR:-runs/E0364_ave_p0_sweep_official_val_${EVENTNESS}_${CANDIDATE_SET}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export CANDIDATE_SET
export OUT_DIR

bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh

