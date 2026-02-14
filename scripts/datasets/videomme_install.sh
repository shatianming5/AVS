#!/usr/bin/env bash
set -euo pipefail

# Install helper for Video-MME (YouTube-backed).
#
# What it does:
#  1) Snapshot HF metadata into `data/VideoMME/meta/test.jsonl`
#  2) Download a deterministic subset of raw videos (first MAX_SECONDS seconds) into `data/VideoMME/raw/videos/`
#
# Outputs:
#  - `data/VideoMME/meta/test.jsonl`
#  - `runs/videomme_download_*/download_report.json`

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-256}"          # number of QA items; unique videos are derived from them
SEED="${SEED:-0}"
ORDER="${ORDER:-hash}"         # hash|original
JOBS="${JOBS:-4}"
MAX_SECONDS="${MAX_SECONDS:-180}"
MIN_VIDEOS="${MIN_VIDEOS:-64}" # fail if too many videos are missing/unavailable
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"

python - <<'PY'
from avs.datasets.videomme import ensure_videomme_meta
from pathlib import Path
print(ensure_videomme_meta(Path("data/VideoMME/meta"), split="test"))
PY

python -m avs.datasets.videomme_download \
  --split "${SPLIT}" \
  --limit "${LIMIT}" \
  --seed "${SEED}" \
  --order "${ORDER}" \
  --jobs "${JOBS}" \
  --max-seconds "${MAX_SECONDS}" \
  --timeout-seconds "${TIMEOUT_SECONDS}" \
  --min-videos "${MIN_VIDEOS}"

