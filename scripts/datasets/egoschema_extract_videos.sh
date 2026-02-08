#!/usr/bin/env bash
set -euo pipefail

# Extract EgoSchema zip shards into a flat videos directory:
#   data/EgoSchema/videos/<video_idx>.mp4
#
# Prereq:
#   - zip shards present under data/hf_repos/egoschema/videos_chunked_*.zip
#
# Usage:
#   bash scripts/datasets/egoschema_extract_videos.sh
#
# Optional:
#   REPO_DIR=data/hf_repos/egoschema OUT_DIR=data/EgoSchema/videos bash scripts/datasets/egoschema_extract_videos.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

REPO_DIR="${REPO_DIR:-data/hf_repos/egoschema}"
OUT_DIR="${OUT_DIR:-data/EgoSchema/videos}"

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "ERROR: missing REPO_DIR=${REPO_DIR}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

shards=( "${REPO_DIR}"/videos_chunked_*.zip )
if [[ "${#shards[@]}" -eq 1 && "${shards[0]}" == "${REPO_DIR}/videos_chunked_*.zip" ]]; then
  echo "ERROR: no zip shards found under ${REPO_DIR} (expected videos_chunked_*.zip)" >&2
  exit 2
fi

total_before="$(find "${OUT_DIR}" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ')"
echo "[egoschema_extract] out_dir=${OUT_DIR} mp4_before=${total_before}"

for z in "${shards[@]}"; do
  bn="$(basename "${z}")"
  echo "[egoschema_extract] extracting ${bn} ..."
  # -n: never overwrite existing files (resumable)
  # -j: junk paths (zip contains videos/<uuid>.mp4)
  unzip -n -j "${z}" 'videos/*.mp4' -d "${OUT_DIR}" >/dev/null
  cur="$(find "${OUT_DIR}" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ')"
  echo "[egoschema_extract] shard_done=${bn} mp4_now=${cur}"
done

total_after="$(find "${OUT_DIR}" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ')"
echo "[egoschema_extract] ALL_DONE mp4_after=${total_after}"
