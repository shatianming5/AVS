#!/usr/bin/env bash
set -euo pipefail

# E0323: Reproduce the best config selected by E0322 on official AVE test402.
#
# Wrapper around `scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh` that:
#   - defaults EVENTNESS to `asr_vad`
#   - writes outputs under runs/E0323_*
#
# Usage:
#   BEST_CONFIG_JSON=runs/E0322_.../best_config.json bash scripts/e0323_ave_p0_best_to_test_official_ltl_top1med_asr_vad_v1.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-asr_vad}"
OUT_DIR="${OUT_DIR:-runs/E0323_ave_p0_best_to_test_official_${EVENTNESS}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export OUT_DIR

bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh

