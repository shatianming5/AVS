#!/usr/bin/env bash
set -euo pipefail

# Download the official AVE dataset zip (all videos) from the authors' Google Drive link.
#
# This is the recommended way to get the "complete dataset" (4143 clips), compared to best-effort yt-dlp.
#
# Notes:
# - The download is resumable (`--continue-at -`).
# - The URL bypasses the Drive "virus scan warning" page via `confirm=t`.
#
# Usage:
#   bash scripts/ave_download_official.sh
#   OUT_ZIP=data/AVE/raw/AVE_Dataset.zip bash scripts/ave_download_official.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUT_ZIP="${OUT_ZIP:-${REPO_ROOT}/data/AVE/raw/AVE_Dataset.zip}"
URL="https://drive.usercontent.google.com/download?id=1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK&export=download&confirm=t"

mkdir -p "$(dirname "${OUT_ZIP}")"
attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "[ave_download_official] attempt=${attempt} out=${OUT_ZIP}"
  if curl -L --fail --retry 5 --retry-delay 5 --continue-at - -o "${OUT_ZIP}" "${URL}"; then
    break
  fi
  echo "[ave_download_official] download failed; retrying in 30s" 1>&2
  sleep 30
done
echo "OK: ${OUT_ZIP}"
