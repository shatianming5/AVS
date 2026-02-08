#!/usr/bin/env bash
set -euo pipefail

# E0256: Fixed-space sweep on official AVE val402 for learned anchors using `candidate_set=ltl_top1med_headcap_v1`.
#
# This is a thin wrapper around `scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` that:
#   - defaults EVENTNESS to the current best learned anchor method (av_clipdiff_mlp)
#   - defaults CANDIDATE_SET to ltl_top1med_headcap_v1 (head_hidden_dim/head_dropout sweep under fixed top1-med gate)
#   - writes outputs under runs/E0256_*
#
# Usage:
#   SEEDS=0,1,2 EVENTNESS=av_clipdiff_mlp bash scripts/e0256_ave_p0_sweep_official_val_ltl_top1med_headcap_v1.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clipdiff_mlp}"
CANDIDATE_SET="${CANDIDATE_SET:-ltl_top1med_headcap_v1}"

OUT_DIR="${OUT_DIR:-runs/E0256_ave_p0_sweep_official_val_${EVENTNESS}_${CANDIDATE_SET}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export CANDIDATE_SET
export OUT_DIR

bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh

