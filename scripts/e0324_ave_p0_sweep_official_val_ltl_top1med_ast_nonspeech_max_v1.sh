#!/usr/bin/env bash
set -euo pipefail

# E0324: Fixed-space sweep on official AVE val402 for an AST "non-speech max" anchor probe.
#
# Stage-1 backend:
#   EVENTNESS=ast_nonspeech_max
#   score[t] = max(sigmoid(AST_logits[t, non_speech_labels]))
#   where speech labels are vetoed by name match: {speech, conversation, narration}.
#
# Wrapper around `scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` that:
#   - defaults EVENTNESS to `ast_nonspeech_max`
#   - defaults CANDIDATE_SET to `ltl_top1med_norm_v1`
#   - writes outputs under runs/E0324_*

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-ast_nonspeech_max}"
CANDIDATE_SET="${CANDIDATE_SET:-ltl_top1med_norm_v1}"

OUT_DIR="${OUT_DIR:-runs/E0324_ave_p0_sweep_official_val_${EVENTNESS}_${CANDIDATE_SET}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export CANDIDATE_SET
export OUT_DIR

bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh

