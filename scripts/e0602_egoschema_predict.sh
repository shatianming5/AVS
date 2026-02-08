#!/usr/bin/env bash
set -euo pipefail

# E0602: EgoSchema VLM prediction generation under budgeted frame selection.
#
# Prereq:
#   - Extract videos into data/EgoSchema/videos/*.mp4 (see scripts/datasets/egoschema_extract_videos.sh)
#
# Outputs:
#   runs/E0602_egoschema_predict_<ts>/metrics.json
#   runs/E0602_egoschema_predict_<ts>/predictions.jsonl

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="${OUT_DIR:-runs/E0602_egoschema_predict_$(date +%Y%m%d-%H%M%S)}"
CONFIG="${CONFIG:-MC}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-64}"
METHODS="${METHODS:-uniform,ql2l_clap,ql2l_asr_bm25}"
B_FRAMES="${B_FRAMES:-16}"
MAX_SECONDS="${MAX_SECONDS:-120}"
SEED="${SEED:-0}"
STRATEGY="${STRATEGY:-ppl}" # ppl|generate

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2-VL-2B-Instruct}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-}"

QL2L_CLAP_DEVICE="${QL2L_CLAP_DEVICE:-cpu}"
QL2L_ASR_DEVICE="${QL2L_ASR_DEVICE:-cpu}"

mkdir -p "${OUT_DIR}"

args=(
  --config "${CONFIG}"
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
)
if [[ -n "${ATTN_IMPL}" ]]; then
  args+=(--attn-implementation "${ATTN_IMPL}")
fi

python -m avs.experiments.egoschema_vlm_eval "${args[@]}"

