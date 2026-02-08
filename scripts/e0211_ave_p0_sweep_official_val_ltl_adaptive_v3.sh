#!/usr/bin/env bash
set -euo pipefail

# E0211: Fixed-space sweep on official AVE val402 for learned anchors using `candidate_set=ltl_adaptive_v3`.
#
# `ltl_adaptive_v3` uses `anchor_high_policy=adaptive_v2`:
#   - conf < std_thr: uniform fallback
#   - std_thr <= conf < high_conf_thr: anchored, but demote to 1 high-res anchor (more base-res context)
#   - conf >= high_conf_thr: anchored with up to 2 high-res anchors (plus adjacency demotion)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clipdiff_mlp}"
CANDIDATE_SET="${CANDIDATE_SET:-ltl_adaptive_v3}"

OUT_DIR="${OUT_DIR:-runs/E0211_ave_p0_sweep_official_val_${EVENTNESS}_${CANDIDATE_SET}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export CANDIDATE_SET
export OUT_DIR

bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh

