#!/usr/bin/env bash
set -euo pipefail

# Run `avs.experiments.ave_p0_diagnose` on a full official-split metrics.json (val/test402),
# producing a root-cause JSON report (fallback rates, anchor-distance buckets, recall↔delta correlation, etc).
#
# Usage:
#   IN_METRICS=runs/E0015_.../metrics.json bash scripts/e0016_ave_p0_diagnose_official.sh
#
# Produces:
#   - runs/E0016_*/diagnose.json

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${IN_METRICS:-}" ]]; then
  latest="$(ls -t runs/E0015_*/metrics.json 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest}" && -f "${latest}" ]]; then
    IN_METRICS="${latest}"
    echo "[e0016] IN_METRICS not set; using ${IN_METRICS}"
  else
    echo "ERROR: IN_METRICS is required (path to a metrics.json from E0015 or another full run)" >&2
    exit 2
  fi
fi
if [[ ! -f "${IN_METRICS}" ]]; then
  echo "ERROR: metrics file not found: ${IN_METRICS}" >&2
  exit 2
fi

META_DIR="${META_DIR:-data/AVE/meta}"
OUT_DIR="${OUT_DIR:-runs/E0016_ave_p0_diagnose_$(date +%Y%m%d-%H%M%S)}"

python -m avs.experiments.ave_p0_diagnose \
  --in-metrics "${IN_METRICS}" \
  --meta-dir "${META_DIR}" \
  --out-dir "${OUT_DIR}"

echo "OK: ${OUT_DIR}/diagnose.json"

