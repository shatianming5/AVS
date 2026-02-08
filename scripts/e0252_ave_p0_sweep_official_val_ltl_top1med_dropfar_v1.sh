#!/usr/bin/env bash
set -euo pipefail

# E0252: Fixed-space sweep on official AVE val402 for learned anchors using `candidate_set=ltl_top1med_dropfar_v1`.
#
# Idea: keep adjacent top-2 anchors but drop the 2nd anchor when it is far from top1 to reduce harmful
# non-adjacent 2-anchor cases observed in E0224 diagnostics.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clipdiff_mlp}"
CANDIDATE_SET="${CANDIDATE_SET:-ltl_top1med_dropfar_v1}"

OUT_DIR="${OUT_DIR:-runs/E0252_ave_p0_sweep_official_val_${EVENTNESS}_${CANDIDATE_SET}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export CANDIDATE_SET
export OUT_DIR

bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh

