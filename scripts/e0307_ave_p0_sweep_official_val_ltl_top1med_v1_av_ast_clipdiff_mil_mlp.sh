#!/usr/bin/env bash
set -euo pipefail

# E0307: Fixed-space sweep on official AVE val402 for AST-embedding + CLIPdiff MIL learned anchors
# (`EVENTNESS=av_ast_clipdiff_mil_mlp`) using the standard top1-med gate candidate set
# (`CANDIDATE_SET=ltl_top1med_v1`).
#
# This is a thin wrapper around `scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` that:
#   - defaults EVENTNESS to `av_ast_clipdiff_mil_mlp`
#   - defaults CANDIDATE_SET to `ltl_top1med_v1`
#   - defaults AUDIO_DEVICE to `cuda:0` when available (AST inference)
#   - writes outputs under runs/E0307_*

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_ast_clipdiff_mil_mlp}"
CANDIDATE_SET="${CANDIDATE_SET:-ltl_top1med_v1}"

if command -v nvidia-smi >/dev/null 2>&1; then
  AUDIO_DEVICE="${AUDIO_DEVICE:-cuda:0}"
else
  AUDIO_DEVICE="${AUDIO_DEVICE:-cpu}"
fi

OUT_DIR="${OUT_DIR:-runs/E0307_ave_p0_sweep_official_val_${EVENTNESS}_${CANDIDATE_SET}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export CANDIDATE_SET
export AUDIO_DEVICE
export OUT_DIR

bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh

