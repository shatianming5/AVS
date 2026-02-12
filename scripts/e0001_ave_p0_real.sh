#!/usr/bin/env bash
set -euo pipefail

# Real-AVE AVE-P0 experiment (equal token budget baselines).
#
# Default: yt-dlp download a small train+val subset into runs/, preprocess, build caches, and run P0.
#
# Usage examples:
#   bash scripts/e0001_ave_p0_real.sh
#   MODE=local SRC_DIR=/path/to/ave/mp4s bash scripts/e0001_ave_p0_real.sh
#   LIMIT_TRAIN=32 LIMIT_EVAL=16 SEEDS=0,1,2 bash scripts/e0001_ave_p0_real.sh
#   EVENTNESS=ast AST_PRETRAINED=1 bash scripts/e0001_ave_p0_real.sh

MODE="${MODE:-yt-dlp}"          # yt-dlp|local|none
SRC_DIR="${SRC_DIR:-}"          # required when MODE=local
META_DIR="${META_DIR:-data/AVE/meta}"

LIMIT_TRAIN="${LIMIT_TRAIN:-8}"
LIMIT_EVAL="${LIMIT_EVAL:-4}"
SEEDS="${SEEDS:-0,1,2}"

EVENTNESS="${EVENTNESS:-energy}"     # energy|energy_delta|ast|panns|audiomae|audio_basic_lr|audio_basic_mlp|audio_fbank_mlp|audio_basic_mlp_cls|audio_basic_mlp_cls_target
AST_PRETRAINED="${AST_PRETRAINED:-0}"
ANCHOR_SHIFT="${ANCHOR_SHIFT:-0}"
ANCHOR_STD_THRESHOLD="${ANCHOR_STD_THRESHOLD:-0.0}"
ANCHOR_SELECT="${ANCHOR_SELECT:-}"
ANCHOR_NMS_RADIUS="${ANCHOR_NMS_RADIUS:-}"
ANCHOR_NMS_STRONG_GAP="${ANCHOR_NMS_STRONG_GAP:-}"
ANCHOR_BASE_ALLOC="${ANCHOR_BASE_ALLOC:-}"
ANCHOR_HIGH_POLICY="${ANCHOR_HIGH_POLICY:-}"
ANCHOR_HIGH_ADJACENT_DIST="${ANCHOR_HIGH_ADJACENT_DIST:-}"
ANCHOR_HIGH_GAP_THRESHOLD="${ANCHOR_HIGH_GAP_THRESHOLD:-}"

VISION_PRETRAINED="${VISION_PRETRAINED:-1}"   # 1 downloads CLIP weights from HF
VISION_MODEL_NAME="${VISION_MODEL_NAME:-}"
if [[ -z "${DEVICE:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && [[ "$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')" -gt 0 ]]; then
    DEVICE="cuda:0"
  else
    DEVICE="cpu"
  fi
fi

TRAIN_DEVICE="${TRAIN_DEVICE:-${DEVICE}}"

RUN_DIR="${RUN_DIR:-runs/E0001_ave_p0_real_$(date +%Y%m%d-%H%M%S)}"
RAW_DIR="${RUN_DIR}/raw/videos"
PROC_DIR="${RUN_DIR}/processed"
CACHES_DIR="${RUN_DIR}/caches"
OUT_DIR="${RUN_DIR}/ave_p0_end2end"

mkdir -p "${RUN_DIR}"

args=(
  --mode "${MODE}"
  --allow-missing
  --meta-dir "${META_DIR}"
  --raw-videos-dir "${RAW_DIR}"
  --processed-dir "${PROC_DIR}"
  --caches-dir "${CACHES_DIR}"
  --out-dir "${OUT_DIR}"
  --split-train train
  --split-eval val
  --limit-train "${LIMIT_TRAIN}"
  --limit-eval "${LIMIT_EVAL}"
  --seeds "${SEEDS}"
  --eventness-method "${EVENTNESS}"
  --anchor-shift "${ANCHOR_SHIFT}"
  --anchor-std-threshold "${ANCHOR_STD_THRESHOLD}"
  --device "${DEVICE}"
  --train-device "${TRAIN_DEVICE}"
)

if [[ "${MODE}" == "local" ]]; then
  if [[ -z "${SRC_DIR}" ]]; then
    echo "MODE=local requires SRC_DIR=/path/to/<video_id>.mp4 directory" 1>&2
    exit 2
  fi
  args+=(--src-dir "${SRC_DIR}")
fi

if [[ "${AST_PRETRAINED}" == "1" ]]; then
  args+=(--ast-pretrained)
fi

if [[ -n "${ANCHOR_HIGH_POLICY}" ]]; then
  args+=(--anchor-high-policy "${ANCHOR_HIGH_POLICY}")
fi
if [[ -n "${ANCHOR_SELECT}" ]]; then
  args+=(--anchor-select "${ANCHOR_SELECT}")
fi
if [[ -n "${ANCHOR_NMS_RADIUS}" ]]; then
  args+=(--anchor-nms-radius "${ANCHOR_NMS_RADIUS}")
fi
if [[ -n "${ANCHOR_NMS_STRONG_GAP}" ]]; then
  args+=(--anchor-nms-strong-gap "${ANCHOR_NMS_STRONG_GAP}")
fi
if [[ -n "${ANCHOR_BASE_ALLOC}" ]]; then
  args+=(--anchor-base-alloc "${ANCHOR_BASE_ALLOC}")
fi
if [[ -n "${ANCHOR_HIGH_ADJACENT_DIST}" ]]; then
  args+=(--anchor-high-adjacent-dist "${ANCHOR_HIGH_ADJACENT_DIST}")
fi
if [[ -n "${ANCHOR_HIGH_GAP_THRESHOLD}" ]]; then
  args+=(--anchor-high-gap-threshold "${ANCHOR_HIGH_GAP_THRESHOLD}")
fi

if [[ "${VISION_PRETRAINED}" == "1" ]]; then
  args+=(--vision-pretrained)
fi
if [[ -n "${VISION_MODEL_NAME}" ]]; then
  args+=(--vision-model-name "${VISION_MODEL_NAME}")
fi

python -m avs.pipeline.ave_p0_end2end "${args[@]}"

echo "OK: ${OUT_DIR}/metrics.json"
