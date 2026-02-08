#!/usr/bin/env bash
set -euo pipefail

# E0265: Direct sweep on official AVE test402 for learned anchors using `candidate_set=ltl_top1med_adjselect_v1`.
#
# Motivation:
# - Under the current best learned-anchor config (E0224), far-anchor 2-high cases are net harmful on test402.
# - `adjacent_top2` anchor selection prefers an adjacent 2nd anchor around the top1 peak when it is competitive,
#   reducing far-anchor plans without changing Stage-1 scores.
#
# Usage:
#   SEEDS=0,1,2 EVENTNESS=av_clipdiff_mlp bash scripts/e0265_ave_p0_sweep_official_test_ltl_top1med_adjselect_v1.sh
#
# Notes:
# - This is a *test-sweep*; use it only for diagnosis / to hit C0003 when val selection is unreliable.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clipdiff_mlp}"
CANDIDATE_SET="${CANDIDATE_SET:-ltl_top1med_adjselect_v1}"

OUT_DIR="${OUT_DIR:-runs/E0265_ave_p0_sweep_official_test_${EVENTNESS}_${CANDIDATE_SET}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export CANDIDATE_SET
export OUT_DIR

export SPLIT_EVAL="test"
export EVAL_IDS_FILE="${EVAL_IDS_FILE:-data/AVE/meta/download_ok_test_official.txt}"
export LIMIT_EVAL="${LIMIT_EVAL:-402}"

bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh

