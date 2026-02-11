#!/usr/bin/env bash
set -euo pipefail

# E0701: QA multi-seed robustness on long-video QA add-on.
#
# IntentQA val: seeds 1,2 (seed0 reused from prior runs).
# AVQA val (B=4): seeds 1,2 (seed0 reused from prior run).
# Then aggregate into per-task multi-seed summaries.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TS="${TS:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-runs/E0701_qa_multiseed_${TS}}"
mkdir -p "$RUN_ROOT"

INTENTQA_METHODS="${INTENTQA_METHODS:-uniform,random,ql2l_clap,ql2l_asr_bm25,text_only}"
AVQA_METHODS="${AVQA_METHODS:-uniform,random,ql2l_clap,ql2l_asr_bm25,text_only}"

DEVICE="${DEVICE:-cuda:1}"
DTYPE="${DTYPE:-bfloat16}"
QL2L_CLAP_DEVICE="${QL2L_CLAP_DEVICE:-cuda:2}"
QL2L_ASR_DEVICE="${QL2L_ASR_DEVICE:-cpu}"
QL2L_CLIP_DEVICE="${QL2L_CLIP_DEVICE:-cpu}"
STRATEGY="${STRATEGY:-ppl}"

# --- IntentQA seeds 1,2 ---
for SEED in 1 2; do
  OUT_DIR="runs/E0701_intentqa_val_b16_s${SEED}_${TS}"
  SPLIT=val \
  LIMIT=256 \
  METHODS="$INTENTQA_METHODS" \
  B_FRAMES=16 \
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
done

# --- AVQA seeds 1,2 (tight budget B=4) ---
for SEED in 1 2; do
  OUT_DIR="runs/E0701_avqa_val_b4_s${SEED}_${TS}"
  SPLIT=val \
  LIMIT=256 \
  METHODS="$AVQA_METHODS" \
  B_FRAMES=4 \
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
done

# --- Aggregation ---
python -m avs.experiments.qa_multiseed_summary \
  --task IntentQA_val_b16 \
  --methods "$INTENTQA_METHODS" \
  --run-metrics runs/E0604_intentqa_vlm_eval_val_s0_20260210-125048/metrics.json \
  --run-metrics runs/E0617_intentqa_vlm_eval_val_text_only_20260211-053301/metrics.json \
  --run-metrics "runs/E0701_intentqa_val_b16_s1_${TS}/metrics.json" \
  --run-metrics "runs/E0701_intentqa_val_b16_s2_${TS}/metrics.json" \
  --out-dir "$RUN_ROOT/intentqa_multiseed"

python -m avs.experiments.qa_multiseed_summary \
  --task AVQA_val_b4 \
  --methods "$AVQA_METHODS" \
  --run-metrics runs/E0616_avqa_vlm_eval_val_b4_20260211-051556/metrics.json \
  --run-metrics "runs/E0701_avqa_val_b4_s1_${TS}/metrics.json" \
  --run-metrics "runs/E0701_avqa_val_b4_s2_${TS}/metrics.json" \
  --out-dir "$RUN_ROOT/avqa_multiseed"

echo "$RUN_ROOT"

