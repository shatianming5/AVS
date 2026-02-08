#!/usr/bin/env bash
set -euo pipefail

# E0393: Stage-1 sweep on official AVE val402 for a new cheap visual signal:
# optical-flow magnitude (Farneback) on per-second frames.
#
# Method:
#   EVENTNESS=av_clipdiff_flow_mlp
#     = supervised per-second MLP trained on train split to predict event-vs-background using:
#       (audio basic features) + (CLIP feature diff scalar) + (optical-flow magnitude scalar)
#
# This is a thin wrapper around `scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh`.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clipdiff_flow_mlp}"
CANDIDATE_SET="${CANDIDATE_SET:-ltl_top1med_norm_v1}"

if command -v nvidia-smi >/dev/null 2>&1; then
  AUDIO_DEVICE="${AUDIO_DEVICE:-cuda:0}"
  TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:0}"
fi

OUT_DIR="${OUT_DIR:-runs/E0393_ave_p0_sweep_official_val_${EVENTNESS}_${CANDIDATE_SET}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export CANDIDATE_SET
export AUDIO_DEVICE
export TRAIN_DEVICE
export OUT_DIR

bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh

