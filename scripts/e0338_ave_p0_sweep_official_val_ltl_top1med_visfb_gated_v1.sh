#!/usr/bin/env bash
set -euo pipefail

# E0338: Stage-1 sweep on official AVE val402 for the gated cheap-visual fallback candidate set
# (`candidate_set=ltl_top1med_visfb_gated_v1`).
#
# This is a thin wrapper around `scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` that:
#   - defaults EVENTNESS to the current learned-anchor backend (av_clipdiff_mlp)
#   - defaults CANDIDATE_SET to ltl_top1med_visfb_gated_v1
#   - writes outputs under runs/E0338_*

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clipdiff_mlp}"
CANDIDATE_SET="${CANDIDATE_SET:-ltl_top1med_visfb_gated_v1}"

OUT_DIR="${OUT_DIR:-runs/E0338_ave_p0_sweep_official_val_${EVENTNESS}_${CANDIDATE_SET}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export CANDIDATE_SET
export OUT_DIR

bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh

