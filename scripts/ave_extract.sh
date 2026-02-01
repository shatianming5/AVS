#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input_video.mp4> <output_dir> [clip_id]" >&2
  exit 2
fi

IN_VIDEO="$1"
OUT_DIR="$2"
CLIP_ID="${3:-$(basename "${IN_VIDEO%.*}")}"

PYTHONPATH="${REPO_ROOT}" python - <<PY
from pathlib import Path
from avs.preprocess.ave_extract import preprocess_one

paths = preprocess_one(Path("${IN_VIDEO}"), Path("${OUT_DIR}"), clip_id="${CLIP_ID}")
print("audio:", paths["audio"])
print("frames:", len(paths["frames"]))
PY

