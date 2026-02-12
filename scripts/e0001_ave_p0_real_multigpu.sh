#!/usr/bin/env bash
set -euo pipefail

# Real-AVE AVE-P0 experiment (equal token budget baselines) using an existing local AVE raw-video cache.
#
# Requires:
#   - AVE metadata under data/AVE/meta/
#   - raw videos under data/AVE/raw/videos/<video_id>.mp4
#
# This script does NOT download videos; it runs in MODE=none and reuses processed/caches across runs.
#
# Usage examples:
#   bash scripts/e0001_ave_p0_real_multigpu.sh
#   LIMIT_EVAL=113 EVAL_IDS_FILE=data/AVE/meta/download_ok_test_auto.txt bash scripts/e0001_ave_p0_real_multigpu.sh
#   EVENTNESS=ast AUDIO_DEVICE=cuda:0 bash scripts/e0001_ave_p0_real_multigpu.sh
#   LOW_RES=160 BASE_RES=224 HIGH_RES=352 CACHE_RESOLUTIONS=160,224,352 bash scripts/e0001_ave_p0_real_multigpu.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

META_DIR="${META_DIR:-data/AVE/meta}"
RAW_VIDEOS_DIR="${RAW_VIDEOS_DIR:-data/AVE/raw/videos}"

DEFAULT_TRAIN_IDS_FILE="${META_DIR}/download_ok_train_official.txt"
if [[ ! -f "${DEFAULT_TRAIN_IDS_FILE}" ]]; then
  DEFAULT_TRAIN_IDS_FILE="${META_DIR}/download_ok_train_auto.txt"
fi
if [[ ! -f "${DEFAULT_TRAIN_IDS_FILE}" ]]; then
  DEFAULT_TRAIN_IDS_FILE="${META_DIR}/download_ok_train200.txt"
fi
TRAIN_IDS_FILE="${TRAIN_IDS_FILE:-${DEFAULT_TRAIN_IDS_FILE}}"

DEFAULT_EVAL_IDS_FILE="${META_DIR}/download_ok_val_official.txt"
if [[ ! -f "${DEFAULT_EVAL_IDS_FILE}" ]]; then
  DEFAULT_EVAL_IDS_FILE="${META_DIR}/download_ok_val_auto.txt"
fi
EVAL_IDS_FILE="${EVAL_IDS_FILE:-${DEFAULT_EVAL_IDS_FILE}}"

LIMIT_TRAIN="${LIMIT_TRAIN:-180}"
LIMIT_EVAL="${LIMIT_EVAL:-165}"
SEEDS="${SEEDS:-0,1,2}"

EVENTNESS="${EVENTNESS:-energy}"   # energy|energy_delta|ast|panns|audiomae|audio_basic_lr|audio_basic_mlp|audio_fbank_mlp|audio_basic_mlp_cls|audio_basic_mlp_cls_target
AST_PRETRAINED="${AST_PRETRAINED:-0}"
K="${K:-2}"

LOW_RES="${LOW_RES:-112}"
BASE_RES="${BASE_RES:-224}"
HIGH_RES="${HIGH_RES:-448}"
PATCH_SIZE="${PATCH_SIZE:-16}"
MAX_HIGH_ANCHORS="${MAX_HIGH_ANCHORS:-}"

ANCHOR_SHIFT="${ANCHOR_SHIFT:-0}"
ANCHOR_STD_THRESHOLD="${ANCHOR_STD_THRESHOLD:-0.0}"
ANCHOR_SELECT="${ANCHOR_SELECT:-}"
ANCHOR_NMS_RADIUS="${ANCHOR_NMS_RADIUS:-}"
ANCHOR_NMS_STRONG_GAP="${ANCHOR_NMS_STRONG_GAP:-}"
ANCHOR_BASE_ALLOC="${ANCHOR_BASE_ALLOC:-}"
ANCHOR_HIGH_POLICY="${ANCHOR_HIGH_POLICY:-}"
ANCHOR_HIGH_ADJACENT_DIST="${ANCHOR_HIGH_ADJACENT_DIST:-}"
ANCHOR_HIGH_GAP_THRESHOLD="${ANCHOR_HIGH_GAP_THRESHOLD:-}"

HEAD="${HEAD:-mlp}"                 # mlp|temporal_conv
HEAD_HIDDEN_DIM="${HEAD_HIDDEN_DIM:-128}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.0}"
TEMPORAL_KERNEL_SIZE="${TEMPORAL_KERNEL_SIZE:-3}"

VISION_PRETRAINED="${VISION_PRETRAINED:-1}"
VISION_MODEL_NAME="${VISION_MODEL_NAME:-}"

if [[ -z "${DEVICE:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && [[ "$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')" -gt 0 ]]; then
    DEVICE="cuda:0"
  else
    DEVICE="cpu"
  fi
fi
AUDIO_DEVICE="${AUDIO_DEVICE:-${DEVICE}}"
TRAIN_DEVICE="${TRAIN_DEVICE:-${DEVICE}}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"

# Multi-process cache build settings.
CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-4}"
CACHE_DEVICES="${CACHE_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"
CACHE_RESOLUTIONS="${CACHE_RESOLUTIONS:-${LOW_RES},${BASE_RES},${HIGH_RES}}"

# Reusable outputs (incremental).
PROCESSED_DIR="${PROCESSED_DIR:-runs/REAL_AVE_LOCAL/processed}"
CACHES_DIR="${CACHES_DIR:-runs/REAL_AVE_LOCAL/caches}"
PREPROCESS_JOBS="${PREPROCESS_JOBS:-1}"

OUT_DIR="${OUT_DIR:-runs/E0001_ave_p0_real_multigpu_$(date +%Y%m%d-%H%M%S)}"

args=(
  --mode none
  # Allow best-effort large-scale runs to skip broken/missing clips (preprocess/cache).
  --meta-dir "${META_DIR}"
  --raw-videos-dir "${RAW_VIDEOS_DIR}"
  --processed-dir "${PROCESSED_DIR}"
  --preprocess-skip-existing
  --preprocess-jobs "${PREPROCESS_JOBS}"
  --caches-dir "${CACHES_DIR}"
  --cache-skip-existing
  --cache-num-workers "${CACHE_NUM_WORKERS}"
  --cache-devices "${CACHE_DEVICES}"
  --out-dir "${OUT_DIR}"
  --train-ids-file "${TRAIN_IDS_FILE}"
  --eval-ids-file "${EVAL_IDS_FILE}"
  --limit-train "${LIMIT_TRAIN}"
  --limit-eval "${LIMIT_EVAL}"
  --seeds "${SEEDS}"
  --eventness-method "${EVENTNESS}"
  --k "${K}"
  --anchor-shift "${ANCHOR_SHIFT}"
  --anchor-std-threshold "${ANCHOR_STD_THRESHOLD}"
  --low-res "${LOW_RES}"
  --base-res "${BASE_RES}"
  --high-res "${HIGH_RES}"
  --patch-size "${PATCH_SIZE}"
  --head "${HEAD}"
  --head-hidden-dim "${HEAD_HIDDEN_DIM}"
  --head-dropout "${HEAD_DROPOUT}"
  --temporal-kernel-size "${TEMPORAL_KERNEL_SIZE}"
  --device "${DEVICE}"
  --audio-device "${AUDIO_DEVICE}"
  --train-device "${TRAIN_DEVICE}"
  --cache-resolutions "${CACHE_RESOLUTIONS}"
)

if [[ -n "${MAX_HIGH_ANCHORS}" ]]; then
  args+=(--max-high-anchors "${MAX_HIGH_ANCHORS}")
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

if [[ -n "${ANCHOR_HIGH_POLICY}" ]]; then
  args+=(--anchor-high-policy "${ANCHOR_HIGH_POLICY}")
fi
if [[ -n "${ANCHOR_HIGH_ADJACENT_DIST}" ]]; then
  args+=(--anchor-high-adjacent-dist "${ANCHOR_HIGH_ADJACENT_DIST}")
fi
if [[ -n "${ANCHOR_HIGH_GAP_THRESHOLD}" ]]; then
  args+=(--anchor-high-gap-threshold "${ANCHOR_HIGH_GAP_THRESHOLD}")
fi

if [[ "${ALLOW_MISSING}" == "1" ]]; then
  args+=(--allow-missing)
fi

if [[ "${AST_PRETRAINED}" == "1" ]]; then
  args+=(--ast-pretrained)
fi

if [[ "${VISION_PRETRAINED}" == "1" ]]; then
  args+=(--vision-pretrained)
fi
if [[ -n "${VISION_MODEL_NAME}" ]]; then
  args+=(--vision-model-name "${VISION_MODEL_NAME}")
fi

cd "${REPO_ROOT}"
python -m avs.pipeline.ave_p0_end2end "${args[@]}"

echo "OK: ${OUT_DIR}/metrics.json"
