#!/usr/bin/env bash
set -euo pipefail

# E0404: Degradation suite (shift/noise/silence) for `av_clipdiff_flow_mlp_stride`.
#
# Note:
#   This wrapper reuses E0203 harness and defaults to the standard official grid.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clipdiff_flow_mlp_stride}"
LIMIT_TRAIN="${LIMIT_TRAIN:-3339}"
LIMIT_EVAL="${LIMIT_EVAL:-402}"

if command -v nvidia-smi >/dev/null 2>&1; then
  AUDIO_DEVICE="${AUDIO_DEVICE:-cuda:0}"
fi

OUT_DIR="${OUT_DIR:-runs/E0404_degradation_${EVENTNESS}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export LIMIT_TRAIN
export LIMIT_EVAL
export AUDIO_DEVICE
export OUT_DIR

bash scripts/e0203_degradation_suite_official.sh

