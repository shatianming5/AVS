#!/usr/bin/env bash
set -euo pipefail

# E0245: Fixed-space sweep on official AVE val402 for learned anchors using per-clip autoshifted scores.
#
# Wrapper around `scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`:
#   - EVENTNESS defaults to av_clipdiff_mlp_autoshift
#   - CANDIDATE_SET defaults to ltl_top1med_autoshift_v1 (top1-med gate; anchor_shift fixed to 0)
#   - outputs under runs/E0245_*

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clipdiff_mlp_autoshift}"
CANDIDATE_SET="${CANDIDATE_SET:-ltl_top1med_autoshift_v1}"

OUT_DIR="${OUT_DIR:-runs/E0245_ave_p0_sweep_official_val_${EVENTNESS}_${CANDIDATE_SET}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export CANDIDATE_SET
export OUT_DIR

bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh

