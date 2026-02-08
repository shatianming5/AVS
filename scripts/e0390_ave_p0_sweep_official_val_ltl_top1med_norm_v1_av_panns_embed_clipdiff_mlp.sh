#!/usr/bin/env bash
set -euo pipefail

# E0390: Stage-1 sweep on official AVE val402 for a bold A+V anchor backend:
# pretrained PANNs embeddings + cheap visual motion proxy (CLIP feature diff), trained as a tiny MLP
# to predict event-vs-background per second (`EVENTNESS=av_panns_embed_clipdiff_mlp`) under the
# scale-invariant top1-med gate (`candidate_set=ltl_top1med_norm_v1`).
#
# This is a thin wrapper around `scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh` that:
#   - defaults EVENTNESS to av_panns_embed_clipdiff_mlp
#   - defaults CANDIDATE_SET to ltl_top1med_norm_v1
#   - defaults AUDIO_DEVICE/TRAIN_DEVICE to cuda:0 when available
#   - writes outputs under runs/E0390_*

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_panns_embed_clipdiff_mlp}"
CANDIDATE_SET="${CANDIDATE_SET:-ltl_top1med_norm_v1}"

if command -v nvidia-smi >/dev/null 2>&1; then
  AUDIO_DEVICE="${AUDIO_DEVICE:-cuda:0}"
  TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:0}"
fi

OUT_DIR="${OUT_DIR:-runs/E0390_ave_p0_sweep_official_val_${EVENTNESS}_${CANDIDATE_SET}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export CANDIDATE_SET
export AUDIO_DEVICE
export TRAIN_DEVICE
export OUT_DIR

bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh

