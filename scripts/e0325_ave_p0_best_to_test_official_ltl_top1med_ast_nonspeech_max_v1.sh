#!/usr/bin/env bash
set -euo pipefail

# E0325: Reproduce the best config selected by E0324 on official AVE test402.
#
# Wrapper around `scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh` that:
#   - defaults EVENTNESS to `ast_nonspeech_max`
#   - writes outputs under runs/E0325_*
#
# Usage:
#   BEST_CONFIG_JSON=runs/E0324_.../best_config.json bash scripts/e0325_ave_p0_best_to_test_official_ltl_top1med_ast_nonspeech_max_v1.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-ast_nonspeech_max}"
OUT_DIR="${OUT_DIR:-runs/E0325_ave_p0_best_to_test_official_${EVENTNESS}_$(date +%Y%m%d-%H%M%S)}"

export EVENTNESS
export OUT_DIR

bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh

