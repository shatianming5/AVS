#!/usr/bin/env bash
set -euo pipefail

# E0704: answer-prior bias baselines for IntentQA / EgoSchema / AVQA.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TS="${TS:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-runs/E0704_qa_bias_baselines_${TS}}"
mkdir -p "$OUT_DIR"

python -m avs.experiments.qa_answer_prior \
  --task intentqa \
  --predictions runs/E0617_intentqa_vlm_eval_val_text_only_20260211-053301/predictions.jsonl \
  --metrics runs/E0617_intentqa_vlm_eval_val_text_only_20260211-053301/metrics.json \
  --out-dir "$OUT_DIR/intentqa"

python -m avs.experiments.qa_answer_prior \
  --task egoschema \
  --predictions runs/E0618_egoschema_eval_subset500_text_only_20260211-055131/predictions.jsonl \
  --metrics runs/E0618_egoschema_eval_subset500_text_only_20260211-055131/metrics.json \
  --out-dir "$OUT_DIR/egoschema"

python -m avs.experiments.qa_answer_prior \
  --task avqa \
  --predictions runs/E0616_avqa_vlm_eval_val_b4_20260211-051556/predictions.jsonl \
  --metrics runs/E0616_avqa_vlm_eval_val_b4_20260211-051556/metrics.json \
  --avqa-meta-dir data/AVQA/meta \
  --out-dir "$OUT_DIR/avqa"

echo "$OUT_DIR"

