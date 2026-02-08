#!/usr/bin/env bash
set -euo pipefail

# E0284: Fixed-space sweep on official AVE val402 for learned anchors using
# `candidate_set=ltl_top1med_keepadj_basealloc_v1`.
#
# Motivation: Diagnostics show far-anchor 2-high cases are strongly harmful on test402. `adaptive_v3` demotes
# far anchors to the 1-high regime while keeping both anchors for base allocation. This sweep tests whether
# alternative base allocation strategies (mixed/context-preserving) improve transfer without changing Stage-1.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clipdiff_mlp}"
CANDIDATE_SET="${CANDIDATE_SET:-ltl_top1med_keepadj_basealloc_v1}"

OUT_DIR="${OUT_DIR:-runs/E0284_ave_p0_sweep_official_val_${EVENTNESS}_${CANDIDATE_SET}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export CANDIDATE_SET
export OUT_DIR

bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh

