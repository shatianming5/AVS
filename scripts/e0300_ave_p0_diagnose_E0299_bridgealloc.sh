#!/usr/bin/env bash
set -euo pipefail

# E0300: Diagnose anchored-vs-uniform deltas for an E0299 metrics.json.
#
# Usage:
#   IN_METRICS=runs/E0299_*/metrics.json bash scripts/e0300_ave_p0_diagnose_E0299_bridgealloc.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${IN_METRICS:-}" ]]; then
  latest_metrics="$(ls -t runs/E0299_*/metrics.json 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_metrics}" && -f "${latest_metrics}" ]]; then
    IN_METRICS="${latest_metrics}"
    echo "[e0300] IN_METRICS not set; using ${IN_METRICS}"
  else
    echo "ERROR: IN_METRICS is required (path to metrics.json from E0299)" >&2
    exit 2
  fi
fi

OUT_DIR="${OUT_DIR:-runs/E0300_diagnose_E0299_bridgealloc_$(date +%Y%m%d-%H%M%S)}"
META_DIR="${META_DIR:-data/AVE/meta}"

python -m avs.experiments.ave_p0_diagnose \
  --in-metrics "${IN_METRICS}" \
  --meta-dir "${META_DIR}" \
  --out-dir "${OUT_DIR}"

echo "OK: ${OUT_DIR}/diagnose.json"

