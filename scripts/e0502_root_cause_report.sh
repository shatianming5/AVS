#!/usr/bin/env bash
set -euo pipefail

# E0502: aggregate root-cause report from latest AVE mechanism artifacts.
#
# Outputs:
#   runs/E0502_root_cause_report_<ts>/root_cause_report.json

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

latest_file() {
  local pattern="$1"
  ls -t ${pattern} 2>/dev/null | head -n 1 || true
}

METRICS_JSON="${METRICS_JSON:-$(latest_file 'runs/E0402_full_test402_av_clipdiff_flow_mlp_stride_alt_top1med_thr0p5_*/metrics.json')}"
if [[ -z "${METRICS_JSON}" || ! -f "${METRICS_JSON}" ]]; then
  echo "ERROR: METRICS_JSON not found. Set METRICS_JSON=<path/to/metrics.json>." >&2
  exit 2
fi

ORACLE_JSON="${ORACLE_JSON:-$(latest_file 'runs/E0407_oracle_vs_predicted_av_clipdiff_flow_mlp_stride_top1med_thr0p5_s0-9_*/oracle_vs_predicted.json')}"
EVIDENCE_JSON="${EVIDENCE_JSON:-$(latest_file 'runs/E0411_evidence_alignment_av_clipdiff_flow_mlp_stride_top1med_thr0p5_*/evidence_alignment.json')}"
DEGRADATION_JSON="${DEGRADATION_JSON:-$(latest_file 'runs/E0412_degradation_accuracy_av_clipdiff_flow_mlp_stride_top1med_thr0p5_s0-9_*/degradation_accuracy.json')}"
EPIC_COMPARE_JSON="${EPIC_COMPARE_JSON:-$(latest_file 'runs/E0413_auto_latest_compare.json')}"

TARGET_DELTA="${TARGET_DELTA:-0.02}"
TARGET_P="${TARGET_P:-0.05}"
ORACLE_GAP_THRESHOLD="${ORACLE_GAP_THRESHOLD:-0.015}"
FALLBACK_THRESHOLD="${FALLBACK_THRESHOLD:-0.60}"
COVERAGE_THRESHOLD="${COVERAGE_THRESHOLD:-0.20}"
CORR_THRESHOLD="${CORR_THRESHOLD:-0.15}"

OUT_DIR="${OUT_DIR:-runs/E0502_root_cause_report_$(date +%Y%m%d-%H%M%S)}"

args=(
  --metrics-json "${METRICS_JSON}"
  --target-delta "${TARGET_DELTA}"
  --target-p "${TARGET_P}"
  --oracle-gap-threshold "${ORACLE_GAP_THRESHOLD}"
  --fallback-threshold "${FALLBACK_THRESHOLD}"
  --coverage-threshold "${COVERAGE_THRESHOLD}"
  --corr-threshold "${CORR_THRESHOLD}"
  --out-dir "${OUT_DIR}"
)

if [[ -n "${ORACLE_JSON}" && -f "${ORACLE_JSON}" ]]; then
  args+=(--oracle-vs-predicted-json "${ORACLE_JSON}")
fi
if [[ -n "${EVIDENCE_JSON}" && -f "${EVIDENCE_JSON}" ]]; then
  args+=(--evidence-alignment-json "${EVIDENCE_JSON}")
fi
if [[ -n "${DEGRADATION_JSON}" && -f "${DEGRADATION_JSON}" ]]; then
  args+=(--degradation-accuracy-json "${DEGRADATION_JSON}")
fi
if [[ -n "${EPIC_COMPARE_JSON}" && -f "${EPIC_COMPARE_JSON}" ]]; then
  args+=(--epic-compare-json "${EPIC_COMPARE_JSON}")
fi

out_json="$(python -m avs.experiments.root_cause_report "${args[@]}")"
echo "OK: ${out_json}"
