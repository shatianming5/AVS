#!/usr/bin/env bash
set -euo pipefail

# E0403: Oracle vs Predicted (MDE-2) for `av_clipdiff_flow_mlp_stride`.
#
# Note:
#   This wrapper reuses E0201 harness and defaults to a quick validation budget.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clipdiff_flow_mlp_stride}"
SEEDS="${SEEDS:-0,1,2}"
LIMIT_TRAIN="${LIMIT_TRAIN:-3339}"
LIMIT_EVAL="${LIMIT_EVAL:-402}"

if command -v nvidia-smi >/dev/null 2>&1; then
  AUDIO_DEVICE="${AUDIO_DEVICE:-cuda:0}"
  TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:0}"
fi

OUT_DIR="${OUT_DIR:-runs/E0403_oracle_vs_predicted_${EVENTNESS}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export SEEDS
export LIMIT_TRAIN
export LIMIT_EVAL
export AUDIO_DEVICE
export TRAIN_DEVICE
export OUT_DIR

bash scripts/e0201_oracle_vs_predicted_official.sh

