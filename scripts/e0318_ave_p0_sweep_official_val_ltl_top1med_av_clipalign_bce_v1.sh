#!/usr/bin/env bash
set -euo pipefail

# E0318: Fixed-space sweep on official AVE val402 for Stage-1 A/V correspondence anchors (BCE).
#
# Thin wrapper around `scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` that:
#   - defaults EVENTNESS to `av_ast_clipalign_bce`
#   - defaults CANDIDATE_SET to `ltl_top1med_norm_v1` (scale-invariant gate)
#   - writes outputs under runs/E0318_*

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_ast_clipalign_bce}"
CANDIDATE_SET="${CANDIDATE_SET:-ltl_top1med_norm_v1}"

OUT_DIR="${OUT_DIR:-runs/E0318_ave_p0_sweep_official_val_${EVENTNESS}_${CANDIDATE_SET}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export CANDIDATE_SET
export OUT_DIR

bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh

