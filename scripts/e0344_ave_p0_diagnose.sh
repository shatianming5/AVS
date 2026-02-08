#!/usr/bin/env bash
set -euo pipefail

# E0344 (helper): Diagnose anchored gains / failure buckets for a given metrics.json.
#
# Usage:
#   IN_METRICS=runs/E0342_.../metrics.json bash scripts/e0344_ave_p0_diagnose.sh
#
# Produces:
#   - runs/E0344_*/diagnose.json

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${IN_METRICS:-}" ]]; then
  echo "ERROR: IN_METRICS is required (path to metrics.json)." >&2
  echo "Example: IN_METRICS=runs/E0224_.../metrics.json bash scripts/e0344_ave_p0_diagnose.sh" >&2
  exit 2
fi
if [[ ! -f "${IN_METRICS}" ]]; then
  echo "ERROR: IN_METRICS not found: ${IN_METRICS}" >&2
  exit 2
fi

META_DIR="${META_DIR:-data/AVE/meta}"
OUT_DIR="${OUT_DIR:-runs/E0344_ave_p0_diagnose_$(date +%Y%m%d-%H%M%S)}"

python -m avs.experiments.ave_p0_diagnose \
  --in-metrics "${IN_METRICS}" \
  --meta-dir "${META_DIR}" \
  --out-dir "${OUT_DIR}"

echo "OK: ${OUT_DIR}/diagnose.json"

