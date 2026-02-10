#!/usr/bin/env bash
set -euo pipefail

# E0619: Bucketed "when does audio help?" analysis for Long-Video QA add-on.
#
# This is a *post-hoc* analysis over existing predictions.jsonl artifacts; it does
# not run any VLM.
#
# Outputs:
#   $OUT_DIR/intentqa/bucket_report.{json,md}
#   $OUT_DIR/avqa/bucket_report.{json,md}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-runs/E0619_qa_bucket_report_$(date +%Y%m%d-%H%M%S)}"

# Defaults: use the already-completed runs recorded in docs/experiment.md.
INTENTQA_PRED="${INTENTQA_PRED:-runs/E0604_intentqa_vlm_eval_val_s0_20260210-125048/predictions.jsonl}"
INTENTQA_META="${INTENTQA_META:-data/IntentQA/IntentQA/val.csv}"

AVQA_PRED="${AVQA_PRED:-runs/E0616_avqa_vlm_eval_val_b4_20260211-051556/predictions.jsonl}"
AVQA_META="${AVQA_META:-data/AVQA/meta/val_qa.json}"

EGOSCHEMA_PRED="${EGOSCHEMA_PRED:-runs/E0605_egoschema_eval_subset500_s0_20260210-125048/predictions.jsonl}"

mkdir -p "$OUT_DIR/intentqa" "$OUT_DIR/avqa" "$OUT_DIR/egoschema"

python -m avs.experiments.qa_bucket_report \
  --predictions "$INTENTQA_PRED" \
  --meta "$INTENTQA_META" \
  --meta-id-key video_id,qid \
  --meta-bucket-field type \
  --primary-method ql2l_clap \
  --qbar-method ql2l_clap \
  --out-dir "$OUT_DIR/intentqa"

python -m avs.experiments.qa_bucket_report \
  --predictions "$AVQA_PRED" \
  --meta "$AVQA_META" \
  --meta-id-key id \
  --meta-bucket-field question_type \
  --primary-method ql2l_asr_bm25 \
  --qbar-method ql2l_asr_bm25 \
  --out-dir "$OUT_DIR/avqa"

python -m avs.experiments.qa_bucket_report \
  --predictions "$EGOSCHEMA_PRED" \
  --primary-method ql2l_clap \
  --qbar-method ql2l_clap \
  --out-dir "$OUT_DIR/egoschema"

echo "$OUT_DIR"
