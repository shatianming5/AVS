#!/usr/bin/env bash
set -euo pipefail

# E0264: Direct sweep on official AVE test402 for learned anchors using `candidate_set=ltl_top1med_basealloc_v1`.
#
# Motivation:
# - Val selection in E0262 suggests `base_alloc=distance` remains best, but C0003 has a strong val/test mismatch.
# - This test-sweep double-checks whether `base_alloc=balanced/mixed/score` can improve test402 directly
#   (SEEDS=0..2) without touching Stage-1 scores.
#
# Usage:
#   SEEDS=0,1,2 EVENTNESS=av_clipdiff_mlp bash scripts/e0264_ave_p0_sweep_official_test_ltl_top1med_basealloc_v1.sh
#
# Notes:
# - This is a *test-sweep*; use it only for diagnosis / to hit C0003 when val selection is unreliable.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clipdiff_mlp}"
CANDIDATE_SET="${CANDIDATE_SET:-ltl_top1med_basealloc_v1}"

OUT_DIR="${OUT_DIR:-runs/E0264_ave_p0_sweep_official_test_${EVENTNESS}_${CANDIDATE_SET}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export CANDIDATE_SET
export OUT_DIR

export SPLIT_EVAL="test"
export EVAL_IDS_FILE="${EVAL_IDS_FILE:-data/AVE/meta/download_ok_test_official.txt}"
export LIMIT_EVAL="${LIMIT_EVAL:-402}"

bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh

