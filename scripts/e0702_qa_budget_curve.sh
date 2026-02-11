#!/usr/bin/env bash
set -euo pipefail

# E0702: QA budget curve (B_FRAMES in {2,4,8,16}) with fixed methods.
#
# Default: run missing budgets and reuse existing B4/B16 artifacts where available.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TS="${TS:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-runs/E0702_qa_budget_curve_${TS}}"
mkdir -p "$RUN_ROOT"

METHODS="${METHODS:-uniform,random,ql2l_clap,ql2l_asr_bm25,text_only}"
BUDGETS="${BUDGETS:-2,4,8,16}"
SEED="${SEED:-0}"

DEVICE="${DEVICE:-cuda:1}"
DTYPE="${DTYPE:-bfloat16}"
QL2L_CLAP_DEVICE="${QL2L_CLAP_DEVICE:-cuda:2}"
QL2L_ASR_DEVICE="${QL2L_ASR_DEVICE:-cpu}"
QL2L_CLIP_DEVICE="${QL2L_CLIP_DEVICE:-cpu}"
STRATEGY="${STRATEGY:-ppl}"

REUSE_EXISTING="${REUSE_EXISTING:-1}"

INTENTQA_METRICS=()
AVQA_METRICS=()

IFS=',' read -r -a B_ARR <<< "$BUDGETS"

for B in "${B_ARR[@]}"; do
  B="$(echo "$B" | xargs)"

  # ----- IntentQA -----
  if [[ "$REUSE_EXISTING" == "1" && "$B" == "16" ]]; then
    INTENTQA_METRICS+=("runs/E0604_intentqa_vlm_eval_val_s0_20260210-125048/metrics.json")
    INTENTQA_METRICS+=("runs/E0617_intentqa_vlm_eval_val_text_only_20260211-053301/metrics.json")
  else
    OUT_DIR="runs/E0702_intentqa_val_b${B}_s${SEED}_${TS}"
    SPLIT=val \
    LIMIT=256 \
    METHODS="$METHODS" \
    B_FRAMES="$B" \
    MAX_SECONDS=120 \
    SEED="$SEED" \
    STRATEGY="$STRATEGY" \
    DEVICE="$DEVICE" \
    DTYPE="$DTYPE" \
    QL2L_CLAP_DEVICE="$QL2L_CLAP_DEVICE" \
    QL2L_ASR_DEVICE="$QL2L_ASR_DEVICE" \
    QL2L_CLIP_DEVICE="$QL2L_CLIP_DEVICE" \
    ALLOW_MISSING_VIDEOS=1 \
    MIN_ITEMS=250 \
    OUT_DIR="$OUT_DIR" \
    bash scripts/e0600_intentqa_vlm_eval.sh
    INTENTQA_METRICS+=("${OUT_DIR}/metrics.json")
  fi

  # ----- AVQA -----
  if [[ "$REUSE_EXISTING" == "1" && "$B" == "4" ]]; then
    AVQA_METRICS+=("runs/E0616_avqa_vlm_eval_val_b4_20260211-051556/metrics.json")
  elif [[ "$REUSE_EXISTING" == "1" && "$B" == "16" ]]; then
    AVQA_METRICS+=("runs/E0615_avqa_vlm_eval_val_20260211-043508/metrics.json")
  else
    OUT_DIR="runs/E0702_avqa_val_b${B}_s${SEED}_${TS}"
    SPLIT=val \
    LIMIT=256 \
    METHODS="$METHODS" \
    B_FRAMES="$B" \
    MAX_SECONDS=120 \
    SEED="$SEED" \
    STRATEGY="$STRATEGY" \
    DEVICE="$DEVICE" \
    DTYPE="$DTYPE" \
    QL2L_CLAP_DEVICE="$QL2L_CLAP_DEVICE" \
    QL2L_ASR_DEVICE="$QL2L_ASR_DEVICE" \
    QL2L_CLIP_DEVICE="$QL2L_CLIP_DEVICE" \
    ALLOW_MISSING_VIDEOS=1 \
    MIN_ITEMS=200 \
    OUT_DIR="$OUT_DIR" \
    bash scripts/e0615_avqa_vlm_eval.sh
    AVQA_METRICS+=("${OUT_DIR}/metrics.json")
  fi
done

# ----- Build budget curves -----
INT_ARGS=()
for m in "${INTENTQA_METRICS[@]}"; do
  INT_ARGS+=(--run-metrics "$m")
done
python -m avs.experiments.qa_budget_curve \
  --task IntentQA_val \
  --methods "$METHODS" \
  "${INT_ARGS[@]}" \
  --out-dir "$RUN_ROOT/intentqa_curve"

AVQA_ARGS=()
for m in "${AVQA_METRICS[@]}"; do
  AVQA_ARGS+=(--run-metrics "$m")
done
python -m avs.experiments.qa_budget_curve \
  --task AVQA_val \
  --methods "$METHODS" \
  "${AVQA_ARGS[@]}" \
  --out-dir "$RUN_ROOT/avqa_curve"

echo "$RUN_ROOT"

