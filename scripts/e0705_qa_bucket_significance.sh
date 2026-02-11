#!/usr/bin/env bash
set -euo pipefail

# E0705: bucket-level significance report (bootstrap CI + p_boot) for long-video QA.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TS="${TS:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-runs/E0705_qa_bucket_significance_${TS}}"
mkdir -p "$OUT_DIR"

N_BOOT="${N_BOOT:-5000}"
MIN_N="${MIN_N:-20}"
SEED="${SEED:-0}"

python -m avs.experiments.qa_bucket_significance \
  --predictions runs/E0604_intentqa_vlm_eval_val_s0_20260210-125048/predictions.jsonl \
  --meta data/IntentQA/IntentQA/val.csv \
  --meta-id-key video_id,qid \
  --meta-bucket-field type \
  --uniform-method uniform \
  --primary-method ql2l_clap \
  --qbar-method ql2l_clap \
  --n-boot "$N_BOOT" \
  --bootstrap-seed "$SEED" \
  --min-n "$MIN_N" \
  --out-dir "$OUT_DIR/intentqa"

python -m avs.experiments.qa_bucket_significance \
  --predictions runs/E0616_avqa_vlm_eval_val_b4_20260211-051556/predictions.jsonl \
  --meta data/AVQA/meta/val_qa.json \
  --meta-id-key id \
  --meta-bucket-field question_type \
  --uniform-method uniform \
  --primary-method ql2l_asr_bm25 \
  --qbar-method ql2l_asr_bm25 \
  --n-boot "$N_BOOT" \
  --bootstrap-seed "$SEED" \
  --min-n "$MIN_N" \
  --out-dir "$OUT_DIR/avqa"

python -m avs.experiments.qa_bucket_significance \
  --predictions runs/E0605_egoschema_eval_subset500_s0_20260210-125048/predictions.jsonl \
  --uniform-method uniform \
  --primary-method ql2l_clap \
  --qbar-method ql2l_clap \
  --n-boot "$N_BOOT" \
  --bootstrap-seed "$SEED" \
  --min-n "$MIN_N" \
  --out-dir "$OUT_DIR/egoschema"

echo "$OUT_DIR"

