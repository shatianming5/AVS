#!/usr/bin/env bash
set -euo pipefail

# E0503: val-only gate sweep for dense-stride Stage-1 (`av_clipdiff_flow_mlp_stride`).
#
# Outputs:
#   runs/E0503_gate_sweep_dense_stride_<ts>/gate_sweep.json
#   runs/E0503_gate_sweep_dense_stride_<ts>/best_gate.json

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

META_DIR="${META_DIR:-data/AVE/meta}"
PROCESSED_DIR="${PROCESSED_DIR:-runs/REAL_AVE_OFFICIAL_20260201-124535/processed}"
CACHES_DIR="${CACHES_DIR:-runs/REAL_AVE_OFFICIAL_20260201-124535/caches_112_160_224_352_448}"
TRAIN_IDS_FILE="${TRAIN_IDS_FILE:-${META_DIR}/download_ok_train_official.txt}"
EVAL_IDS_FILE="${EVAL_IDS_FILE:-${META_DIR}/download_ok_val_official.txt}"

EVENTNESS="${EVENTNESS:-av_clipdiff_flow_mlp_stride}"
SEEDS="${SEEDS:-0,1,2}"
GATE_METRIC="${GATE_METRIC:-top1_med}"
GATE_THRESHOLDS="${GATE_THRESHOLDS:-0.4,0.5,0.6,0.7}"
LIMIT_TRAIN="${LIMIT_TRAIN:-3339}"
LIMIT_EVAL="${LIMIT_EVAL:-402}"
ALLOW_MISSING="${ALLOW_MISSING:-1}"

if command -v nvidia-smi >/dev/null 2>&1; then
  TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:0}"
  AUDIO_DEVICE="${AUDIO_DEVICE:-cuda:0}"
else
  TRAIN_DEVICE="${TRAIN_DEVICE:-cpu}"
  AUDIO_DEVICE="${AUDIO_DEVICE:-cpu}"
fi

OUT_DIR="${OUT_DIR:-runs/E0503_gate_sweep_dense_stride_$(date +%Y%m%d-%H%M%S)}"

allow_missing_flag=()
if [[ "${ALLOW_MISSING}" == "1" ]]; then
  allow_missing_flag+=(--allow-missing)
fi

python -m avs.experiments.mde_ltl gate_sweep \
  --mode ave_official \
  --out-dir "${OUT_DIR}" \
  --meta-dir "${META_DIR}" \
  --processed-dir "${PROCESSED_DIR}" \
  --caches-dir "${CACHES_DIR}" \
  "${allow_missing_flag[@]}" \
  --split-train train \
  --split-eval val \
  --train-ids-file "${TRAIN_IDS_FILE}" \
  --eval-ids-file "${EVAL_IDS_FILE}" \
  --limit-train "${LIMIT_TRAIN}" \
  --limit-eval "${LIMIT_EVAL}" \
  --eventness-method "${EVENTNESS}" \
  --seeds "${SEEDS}" \
  --epochs 5 \
  --batch-size 16 \
  --lr 0.002 \
  --weight-decay 0.0 \
  --train-device "${TRAIN_DEVICE}" \
  --audio-device "${AUDIO_DEVICE}" \
  --k 2 \
  --low-res 160 \
  --base-res 224 \
  --high-res 352 \
  --anchor-shift 1 \
  --anchor-std-threshold 1.0 \
  --anchor-select topk \
  --anchor-nms-radius 2 \
  --anchor-nms-strong-gap 0.6 \
  --anchor-window 3 \
  --anchor-base-alloc distance \
  --anchor-high-policy adaptive_v1 \
  --anchor-high-adjacent-dist 1 \
  --anchor-high-gap-threshold 0 \
  --temporal-kernel-size 3 \
  --gate-metric "${GATE_METRIC}" \
  --gate-thresholds "${GATE_THRESHOLDS}"

echo "OK: ${OUT_DIR}/gate_sweep.json"
echo "OK: ${OUT_DIR}/best_gate.json"
