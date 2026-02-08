#!/usr/bin/env bash
set -euo pipefail

# E0349: Val402 sweep for CLAP+CLIP semantic agreement Stage-1 anchors (av_clap_clip_agree) with k=1 candidate set.
#
# Usage:
#   AUDIO_DEVICE=cuda:0 TRAIN_DEVICE=cuda:0 bash scripts/e0349_ave_p0_sweep_official_val_ltl_top1med_k1_v1_clapagree.sh
#
# Optional (recommended): reuse the E0346 score cache to skip expensive CLAP scoring.
#   BASE_SCORES_JSON=runs/E0346_.../eventness_scores.json bash scripts/e0349_...
#
# Produces:
#   - runs/E0349_*/sweep_summary.json
#   - runs/E0349_*/best_config.json
#   - runs/E0349_*/eventness_scores.json

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clap_clip_agree}"
CANDIDATE_SET="${CANDIDATE_SET:-ltl_top1med_k1_v1}"

DEFAULT_CACHES_DIR="runs/REAL_AVE_OFFICIAL_20260201-124535/caches_112_160_224_352_448"
if [[ -z "${CACHES_DIR:-}" ]]; then
  if [[ -d "${DEFAULT_CACHES_DIR}" ]]; then
    CACHES_DIR="${DEFAULT_CACHES_DIR}"
  else
    echo "ERROR: CACHES_DIR not set and default caches not found: ${DEFAULT_CACHES_DIR}" >&2
    exit 2
  fi
fi
export CACHES_DIR

# CLAP is expensive; default to GPU for Stage-1 scoring.
if command -v nvidia-smi >/dev/null 2>&1; then
  DEVICE="${DEVICE:-cuda:0}"
else
  DEVICE="${DEVICE:-cpu}"
fi
AUDIO_DEVICE="${AUDIO_DEVICE:-${DEVICE}}"
TRAIN_DEVICE="${TRAIN_DEVICE:-${DEVICE}}"
export AUDIO_DEVICE
export TRAIN_DEVICE

OUT_DIR="${OUT_DIR:-runs/E0349_ave_p0_sweep_official_val_${EVENTNESS}_${CANDIDATE_SET}_$(date +%Y%m%d-%H%M%S)}"
export OUT_DIR
export EVENTNESS
export CANDIDATE_SET

# Reuse existing score cache if available; copy into OUT_DIR for auditable artifacts.
if [[ -z "${BASE_SCORES_JSON:-}" ]]; then
  latest_scores="$(ls -t runs/E0346_*/eventness_scores.json 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_scores}" && -f "${latest_scores}" ]]; then
    BASE_SCORES_JSON="${latest_scores}"
    echo "[e0349] BASE_SCORES_JSON not set; using ${BASE_SCORES_JSON}"
  fi
fi
if [[ -n "${BASE_SCORES_JSON:-}" ]]; then
  if [[ ! -f "${BASE_SCORES_JSON}" ]]; then
    echo "ERROR: BASE_SCORES_JSON not found: ${BASE_SCORES_JSON}" >&2
    exit 2
  fi
  mkdir -p "${OUT_DIR}"
  cp "${BASE_SCORES_JSON}" "${OUT_DIR}/eventness_scores.json"
  export SCORES_JSON="${OUT_DIR}/eventness_scores.json"
fi

bash scripts/e0207_ave_p0_sweep_official_val_ltl_stage1.sh

