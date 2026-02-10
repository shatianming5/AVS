#!/usr/bin/env bash
set -euo pipefail

# E0600: IntentQA VLM evaluation under budgeted frame selection.
#
# Outputs:
#   runs/E0600_intentqa_vlm_eval_<ts>/metrics.json
#   runs/E0600_intentqa_vlm_eval_<ts>/predictions.jsonl

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="${OUT_DIR:-runs/E0600_intentqa_vlm_eval_$(date +%Y%m%d-%H%M%S)}"
SPLIT="${SPLIT:-val}"
LIMIT="${LIMIT:-64}"
METHODS="${METHODS:-uniform,random,audio,cheap_visual,fused,ql2l_clap,ql2l_asr_bm25}"
B_FRAMES="${B_FRAMES:-16}"
MAX_SECONDS="${MAX_SECONDS:-120}"
SEED="${SEED:-0}"
STRATEGY="${STRATEGY:-ppl}" # ppl|generate
ALLOW_MISSING_VIDEOS="${ALLOW_MISSING_VIDEOS:-0}"
MIN_ITEMS="${MIN_ITEMS:-16}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2-VL-2B-Instruct}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-}"

QL2L_CLAP_DEVICE="${QL2L_CLAP_DEVICE:-cpu}"
QL2L_ASR_DEVICE="${QL2L_ASR_DEVICE:-cpu}"
QL2L_CLIP_DEVICE="${QL2L_CLIP_DEVICE:-cpu}"

mkdir -p "${OUT_DIR}"

args=(
  --split "${SPLIT}"
  --limit "${LIMIT}"
  --methods "${METHODS}"
  --budget-frames "${B_FRAMES}"
  --max-seconds "${MAX_SECONDS}"
  --seed "${SEED}"
  --strategy "${STRATEGY}"
  --out-dir "${OUT_DIR}"
  --model-name "${MODEL_NAME}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --ql2l-clap-device "${QL2L_CLAP_DEVICE}"
  --ql2l-asr-device "${QL2L_ASR_DEVICE}"
  --ql2l-clip-device "${QL2L_CLIP_DEVICE}"
)
if [[ -n "${ATTN_IMPL}" ]]; then
  args+=(--attn-implementation "${ATTN_IMPL}")
fi
if [[ "${ALLOW_MISSING_VIDEOS}" = "1" ]]; then
  args+=(--allow-missing-videos --min-items "${MIN_ITEMS}")
fi

python -m avs.experiments.intentqa_vlm_eval "${args[@]}"
