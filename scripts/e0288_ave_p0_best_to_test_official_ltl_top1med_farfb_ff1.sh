#!/usr/bin/env bash
set -euo pipefail

# E0288: Targeted test402 run — far-anchor fallback-to-uniform (ff=1) on the fixed E0224 top1-med config.
#
# Motivation: diagnostics on the current best config (E0224) show dist>1 / 2-high cases are net harmful on test402.
# Enabling `anchor_fallback_far_dist=1` forces a full fallback (uniform) for those far-anchor clips, aiming to
# eliminate the harmful bucket without changing the Stage-1 scorer.
#
# Usage:
#   bash scripts/e0288_ave_p0_best_to_test_official_ltl_top1med_farfb_ff1.sh
#
# Smoke:
#   LIMIT_TRAIN=64 LIMIT_EVAL=32 SEEDS=0,1 EPOCHS=1 bash scripts/e0288_ave_p0_best_to_test_official_ltl_top1med_farfb_ff1.sh
#
# Full:
#   SEEDS=0,1,2,3,4,5,6,7,8,9 bash scripts/e0288_ave_p0_best_to_test_official_ltl_top1med_farfb_ff1.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clipdiff_mlp}"

# Base config: the E0223 val selection used by E0224.
BASE_BEST_CONFIG_JSON="${BASE_BEST_CONFIG_JSON:-runs/E0223_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_v1_20260204-135150/best_config.json}"
BASE_SCORES_JSON="${BASE_SCORES_JSON:-runs/E0223_ave_p0_sweep_official_val_av_clipdiff_mlp_ltl_top1med_v1_20260204-135150/eventness_scores.json}"

if [[ ! -f "${BASE_BEST_CONFIG_JSON}" ]]; then
  echo "ERROR: BASE_BEST_CONFIG_JSON not found: ${BASE_BEST_CONFIG_JSON}" >&2
  exit 2
fi
if [[ ! -f "${BASE_SCORES_JSON}" ]]; then
  echo "ERROR: BASE_SCORES_JSON not found: ${BASE_SCORES_JSON}" >&2
  exit 2
fi

OUT_DIR="${OUT_DIR:-runs/E0288_ave_p0_best_to_test_official_${EVENTNESS}_ltl_top1med_farfb_ff1_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${OUT_DIR}"

DERIVED_CONFIG_JSON="${DERIVED_CONFIG_JSON:-${OUT_DIR}/config_farfb_ff1.json}"

export BASE_BEST_CONFIG_JSON
export DERIVED_CONFIG_JSON
python - <<'PY'
import json
import os
from pathlib import Path

base = Path(os.environ["BASE_BEST_CONFIG_JSON"])
out = Path(os.environ["DERIVED_CONFIG_JSON"])

obj = json.loads(base.read_text())
obj["anchor_drop_far_dist"] = 0
obj["anchor_fallback_far_dist"] = 1

name = str(obj.get("name") or "cfg")
if not name.endswith("_ff1"):
    obj["name"] = f"{name}_ff1"

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
print(f"[e0288] wrote derived config: {out}")
PY

export EVENTNESS
export BEST_CONFIG_JSON="${DERIVED_CONFIG_JSON}"
export SCORES_JSON="${BASE_SCORES_JSON}"
export OUT_DIR

# Reuse the existing E0208 runner (official test402 reproduction).
bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh

