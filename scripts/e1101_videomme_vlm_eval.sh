#!/usr/bin/env bash
set -euo pipefail

# E1101: Video-MME controlled transfer (YouTube-backed) under fixed frame budget.
#
# Outputs:
#   runs/E1101_videomme_vlm_eval_<ts>/metrics.json
#   runs/E1101_videomme_vlm_eval_<ts>/predictions.jsonl
#
# Notes:
# - Uses `MAX_SECONDS` (default=180) to cap compute and to match yt-dlp clipping during install.
# - Includes priors controls: `text_only` and `random_frame1`.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="${OUT_DIR:-runs/E1101_videomme_vlm_eval_$(date +%Y%m%d-%H%M%S)}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-256}"
SEED="${SEED:-0}"
ORDER="${ORDER:-hash}" # hash|original
METHODS="${METHODS:-uniform,random,random_frame1,text_only,audio,cheap_visual,fused,ql2l_clip,qframe_gumbel_clip,maxinfo_maxvol_clip,mdp3_dpp_clip}"
B_FRAMES="${B_FRAMES:-16}"
MAX_SECONDS="${MAX_SECONDS:-180}"
STRATEGY="${STRATEGY:-ppl}" # ppl|generate
ALLOW_MISSING_VIDEOS="${ALLOW_MISSING_VIDEOS:-1}"
MIN_ITEMS="${MIN_ITEMS:-128}"

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
  --seed "${SEED}"
  --order "${ORDER}"
  --methods "${METHODS}"
  --budget-frames "${B_FRAMES}"
  --max-seconds "${MAX_SECONDS}"
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

python -m avs.experiments.videomme_vlm_eval "${args[@]}"

