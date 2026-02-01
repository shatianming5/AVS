#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <local_src_dir> <video_id> [video_id...]" >&2
  echo "  Copies <video_id>.mp4 from <local_src_dir> into data/AVE/raw/videos/" >&2
  exit 2
fi

SRC_DIR="$1"
shift

VIDEO_ARGS=()
for VID in "$@"; do
  VIDEO_ARGS+=(--video-id "${VID}")
done

PYTHONPATH="${REPO_ROOT}" python -m avs.datasets.ave_download --mode local --src-dir "${SRC_DIR}" "${VIDEO_ARGS[@]}" --out-dir "${REPO_ROOT}/data/AVE/raw/videos"
