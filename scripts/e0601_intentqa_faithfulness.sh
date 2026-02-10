#!/usr/bin/env bash
set -euo pipefail

# E0601: IntentQA faithfulness proxy (delete-and-predict) under fixed frame budget.
#
# Outputs:
#   runs/E0601_intentqa_faithfulness_<ts>/faithfulness.json
#   runs/E0601_intentqa_faithfulness_<ts>/rows.jsonl

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="${OUT_DIR:-runs/E0601_intentqa_faithfulness_$(date +%Y%m%d-%H%M%S)}"
SPLIT="${SPLIT:-val}"
LIMIT="${LIMIT:-64}"
METHOD="${METHOD:-ql2l_clap}"
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

mkdir -p "${OUT_DIR}"

args=(
  --split "${SPLIT}"
  --limit "${LIMIT}"
  --method "${METHOD}"
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
)
if [[ -n "${ATTN_IMPL}" ]]; then
  args+=(--attn-implementation "${ATTN_IMPL}")
fi
if [[ "${ALLOW_MISSING_VIDEOS}" = "1" ]]; then
  args+=(--allow-missing-videos --min-items "${MIN_ITEMS}")
fi

python -m avs.experiments.intentqa_faithfulness "${args[@]}"
