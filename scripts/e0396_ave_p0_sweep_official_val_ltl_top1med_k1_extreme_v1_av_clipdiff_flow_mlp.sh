#!/usr/bin/env bash
set -euo pipefail

# E0396: Stage-2 "dynamic-K style" sweep on official AVE val402 for flow-augmented Stage-1 scores.
#
# Method:
#   EVENTNESS=av_clipdiff_flow_mlp (from E0393)
# Candidate set:
#   CANDIDATE_SET=ltl_top1med_k1_extreme_v1
#
# Default behavior:
#   - Reuse the latest E0393 eventness cache (SCORES_JSON) to avoid recomputing optical flow.
#   - Run through the standard E0207 sweep runner.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clipdiff_flow_mlp}"
CANDIDATE_SET="${CANDIDATE_SET:-ltl_top1med_k1_extreme_v1}"

if [[ -z "${SCORES_JSON:-}" ]]; then
  latest_scores="$(ls -t runs/E0393_*/eventness_scores.json 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_scores}" && -f "${latest_scores}" ]]; then
    SCORES_JSON="${latest_scores}"
    echo "[e0396] SCORES_JSON not set; reusing ${SCORES_JSON}"
  fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  AUDIO_DEVICE="${AUDIO_DEVICE:-cuda:0}"
  TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:0}"
fi

OUT_DIR="${OUT_DIR:-runs/E0396_ave_p0_sweep_official_val_${EVENTNESS}_${CANDIDATE_SET}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export CANDIDATE_SET
export SCORES_JSON
export AUDIO_DEVICE
export TRAIN_DEVICE
export OUT_DIR

bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh

