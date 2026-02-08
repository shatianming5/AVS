#!/usr/bin/env bash
set -euo pipefail

# E0282: Fixed-space sweep on official AVE val402 for learned anchors using `candidate_set=ltl_top1med_farfb_v1`.
#
# Idea: if the top-2 anchors are far apart (dist > threshold), force a full fallback-to-uniform for that clip,
# targeting the strongly harmful far-anchor 2-high regime observed in E0224 diagnostics.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clipdiff_mlp}"
CANDIDATE_SET="${CANDIDATE_SET:-ltl_top1med_farfb_v1}"

OUT_DIR="${OUT_DIR:-runs/E0282_ave_p0_sweep_official_val_${EVENTNESS}_${CANDIDATE_SET}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export CANDIDATE_SET
export OUT_DIR

bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh

