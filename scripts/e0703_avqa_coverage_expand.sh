#!/usr/bin/env bash
set -euo pipefail

# E0703: Expand AVQA coverage (best effort) and run coverage sensitivity check.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TS="${TS:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-runs/E0703_avqa_coverage_expand_${TS}}"
mkdir -p "$RUN_ROOT"

JOBS="${JOBS:-8}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-60}"
LISTS_TAG="${LISTS_TAG:-val_full_${TS}}"
EVAL_LIMIT="${EVAL_LIMIT:-1024}"   # evaluate on a large subset after expansion
EVAL_MIN_ITEMS="${EVAL_MIN_ITEMS:-200}"
BASELINE_METRICS="${BASELINE_METRICS:-runs/E0616_avqa_vlm_eval_val_b4_20260211-051556/metrics.json}"
DOWNLOAD_LIMIT="${DOWNLOAD_LIMIT:-0}"   # 0 means full split
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"     # 1 means reuse current local AVQA cache

DEVICE="${DEVICE:-cuda:1}"
DTYPE="${DTYPE:-bfloat16}"
QL2L_CLAP_DEVICE="${QL2L_CLAP_DEVICE:-cuda:2}"
QL2L_ASR_DEVICE="${QL2L_ASR_DEVICE:-cpu}"
QL2L_CLIP_DEVICE="${QL2L_CLIP_DEVICE:-cpu}"
STRATEGY="${STRATEGY:-ppl}"

echo "[E0703] counting current AVQA raw videos ..."
BEFORE_COUNT="$(find data/AVQA/raw/videos -maxdepth 1 -type f -name '*.mp4' | wc -l | xargs)"
echo "[E0703] mp4_count_before=${BEFORE_COUNT}"

DOWNLOAD_JSON="$RUN_ROOT/avqa_download_val_full.json"
if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
  echo "[E0703] SKIP_DOWNLOAD=1, reuse existing local AVQA cache"
  cat > "$DOWNLOAD_JSON" <<JSON
{
  "ok": true,
  "mode": "skip_download",
  "num_ok": 0,
  "num_fail": 0,
  "meta": {
    "skip_download": true,
    "download_limit": ${DOWNLOAD_LIMIT},
    "jobs": ${JOBS},
    "timeout_seconds": ${TIMEOUT_SECONDS}
  }
}
JSON
else
  DL_ARGS=(
    --split val
    --jobs "$JOBS"
    --timeout-seconds "$TIMEOUT_SECONDS"
    --write-meta-lists
    --lists-tag "$LISTS_TAG"
    --out-json "$DOWNLOAD_JSON"
  )
  if [[ "${DOWNLOAD_LIMIT}" != "0" ]]; then
    DL_ARGS+=(--limit "$DOWNLOAD_LIMIT")
  fi
  python -m avs.datasets.avqa_download "${DL_ARGS[@]}"
fi

AFTER_COUNT="$(find data/AVQA/raw/videos -maxdepth 1 -type f -name '*.mp4' | wc -l | xargs)"
echo "[E0703] mp4_count_after=${AFTER_COUNT}"

# Re-evaluate key methods on expanded coverage.
EXPANDED_OUT_DIR="runs/E0703_avqa_vlm_eval_val_b4_expand_${TS}"
SPLIT=val \
LIMIT="$EVAL_LIMIT" \
METHODS="uniform,cheap_visual,ql2l_asr_bm25,text_only" \
B_FRAMES=4 \
MAX_SECONDS=120 \
SEED=0 \
STRATEGY="$STRATEGY" \
DEVICE="$DEVICE" \
DTYPE="$DTYPE" \
QL2L_CLAP_DEVICE="$QL2L_CLAP_DEVICE" \
QL2L_ASR_DEVICE="$QL2L_ASR_DEVICE" \
QL2L_CLIP_DEVICE="$QL2L_CLIP_DEVICE" \
ALLOW_MISSING_VIDEOS=1 \
MIN_ITEMS="$EVAL_MIN_ITEMS" \
OUT_DIR="$EXPANDED_OUT_DIR" \
bash scripts/e0615_avqa_vlm_eval.sh

python -m avs.experiments.qa_coverage_sensitivity \
  --baseline-metrics "$BASELINE_METRICS" \
  --expanded-metrics "${EXPANDED_OUT_DIR}/metrics.json" \
  --download-json "$DOWNLOAD_JSON" \
  --out-dir "$RUN_ROOT/sensitivity"

echo "$RUN_ROOT"
