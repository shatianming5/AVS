#!/usr/bin/env bash
set -euo pipefail

# E0400: Full official-val sweep (SEEDS=0..2) for new Stage-1 `av_clipdiff_flow_mlp_stride`.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clipdiff_flow_mlp_stride}"
CANDIDATE_SET="${CANDIDATE_SET:-ltl_top1med_k1_extreme_v1}"

if command -v nvidia-smi >/dev/null 2>&1; then
  AUDIO_DEVICE="${AUDIO_DEVICE:-cuda:0}"
  TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:0}"
fi

SEEDS="${SEEDS:-0,1,2}"
EPOCHS="${EPOCHS:-5}"
LIMIT_TRAIN="${LIMIT_TRAIN:-3339}"
LIMIT_EVAL="${LIMIT_EVAL:-402}"

OUT_DIR="${OUT_DIR:-runs/E0400_ave_p0_sweep_official_val_${EVENTNESS}_${CANDIDATE_SET}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export CANDIDATE_SET
export SEEDS
export EPOCHS
export LIMIT_TRAIN
export LIMIT_EVAL
export AUDIO_DEVICE
export TRAIN_DEVICE
export OUT_DIR

bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh

