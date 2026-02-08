#!/usr/bin/env bash
set -euo pipefail

# E0371: Full test402 reproduction (SEEDS=0..9) for the E0318 winner (av_ast_clipalign_bce) → attempt to prove C0003.
#
# Usage:
#   SEEDS=0,1,2,3,4,5,6,7,8,9 AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0371_full_test402_av_ast_clipalign_bce.sh
#
# If BEST_CONFIG_JSON is not set, uses the latest `runs/E0318_*/best_config.json`.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_ast_clipalign_bce}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"

if command -v nvidia-smi >/dev/null 2>&1; then
  DEVICE="${DEVICE:-cuda:0}"
else
  DEVICE="${DEVICE:-cpu}"
fi
AUDIO_DEVICE="${AUDIO_DEVICE:-${DEVICE}}"
TRAIN_DEVICE="${TRAIN_DEVICE:-${DEVICE}}"
export AUDIO_DEVICE
export TRAIN_DEVICE

OUT_DIR="${OUT_DIR:-runs/E0371_full_test402_${EVENTNESS}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export SEEDS
export OUT_DIR

bash scripts/e0319_ave_p0_best_to_test_official_ltl_top1med_av_clipalign_bce_v1.sh

