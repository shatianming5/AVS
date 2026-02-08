#!/usr/bin/env bash
set -euo pipefail

# E0296: Fixed-space sweep on official AVE val402 for learned anchors using:
#   EVENTNESS=moe_energy_clipdiff (Stage-1; energy→CLIPdiff MOE)
#   CANDIDATE_SET=ltl_top1med_moe_v1 (Stage-2 + top1-med confidence gate, plus std_thr sweep for MOE)
#
# This is a thin wrapper around `scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-moe_energy_clipdiff}"
CANDIDATE_SET="${CANDIDATE_SET:-ltl_top1med_moe_v1}"

OUT_DIR="${OUT_DIR:-runs/E0296_ave_p0_sweep_official_val_${EVENTNESS}_${CANDIDATE_SET}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export CANDIDATE_SET
export OUT_DIR

bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh

