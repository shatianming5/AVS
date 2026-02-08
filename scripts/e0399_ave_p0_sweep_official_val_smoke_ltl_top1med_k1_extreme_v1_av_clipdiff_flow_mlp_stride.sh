#!/usr/bin/env bash
set -euo pipefail

# E0399: Smoke sweep for new Stage-1 `av_clipdiff_flow_mlp_stride`.
#
# Purpose:
#   - Validate new method wiring end-to-end on tiny official-val subset.
#   - Produce a reusable score cache for fast iteration.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clipdiff_flow_mlp_stride}"
CANDIDATE_SET="${CANDIDATE_SET:-ltl_top1med_k1_extreme_v1}"

if command -v nvidia-smi >/dev/null 2>&1; then
  AUDIO_DEVICE="${AUDIO_DEVICE:-cuda:0}"
  TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:0}"
fi

LIMIT_TRAIN="${LIMIT_TRAIN:-64}"
LIMIT_EVAL="${LIMIT_EVAL:-32}"
SEEDS="${SEEDS:-0,1}"
EPOCHS="${EPOCHS:-1}"

OUT_DIR="${OUT_DIR:-runs/E0399_ave_p0_sweep_official_val_${EVENTNESS}_${CANDIDATE_SET}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export CANDIDATE_SET
export LIMIT_TRAIN
export LIMIT_EVAL
export SEEDS
export EPOCHS
export AUDIO_DEVICE
export TRAIN_DEVICE
export OUT_DIR

bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh

