#!/usr/bin/env bash
set -euo pipefail

# E0260: Direct sweep on official AVE test402 for learned anchors using `candidate_set=ltl_top1med_keepadj_v1`.
#
# Motivation:
# - Under the current best learned-anchor config (E0224), per-clip diagnostics show that far-anchor 2-high cases
#   (2×high_res under the equal-budget 160/224/352 plan) can be net harmful due to context loss.
# - Prior attempts either dropped anchor2 entirely (drop-far) or forced maxHigh=1 globally (too blunt).
# - `anchor_high_policy=adaptive_v3` keeps both anchors for base allocation, but only allows 2-high when anchors
#   are adjacent/close; far-anchor clips are demoted to 1-high to preserve more base-res context.
#
# Usage:
#   SEEDS=0,1,2 EVENTNESS=av_clipdiff_mlp bash scripts/e0260_ave_p0_sweep_official_test_ltl_top1med_keepadj_v1.sh
#
# Notes:
# - This is a *test-sweep*; use it only for diagnosis / to hit C0003 when val selection is unreliable.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clipdiff_mlp}"
CANDIDATE_SET="${CANDIDATE_SET:-ltl_top1med_keepadj_v1}"

OUT_DIR="${OUT_DIR:-runs/E0260_ave_p0_sweep_official_test_${EVENTNESS}_${CANDIDATE_SET}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export CANDIDATE_SET
export OUT_DIR

export SPLIT_EVAL="test"
export EVAL_IDS_FILE="${EVAL_IDS_FILE:-data/AVE/meta/download_ok_test_official.txt}"
export LIMIT_EVAL="${LIMIT_EVAL:-402}"

bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh

