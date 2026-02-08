#!/usr/bin/env bash
set -euo pipefail

# One-shot installer for the official AVE dataset zip:
#   - downloads the zip (resumable)
#   - unpacks into data/AVE/raw/AVE_Dataset/
#   - installs it as the default raw videos dir (data/AVE/raw/videos -> AVE_Dataset)
#   - writes split availability lists under data/AVE/meta/
#
# Usage:
#   bash scripts/ave_install_official.sh
#
# Useful env overrides:
#   OUT_ZIP=...              # passed to ave_download_official.sh
#   ZIP_PATH=...             # passed to ave_unpack_official.sh
#   EXTRACT_ROOT=...         # passed to ave_unpack_official.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${REPO_ROOT}"

bash scripts/ave_download_official.sh

INSTALL_AS_DEFAULT=1 bash scripts/ave_unpack_official.sh

OFFICIAL_META_SRC="data/AVE/raw/AVE_Dataset"
if [[ -f "${OFFICIAL_META_SRC}/Annotations.txt" && -f "${OFFICIAL_META_SRC}/trainSet.txt" && -f "${OFFICIAL_META_SRC}/valSet.txt" && -f "${OFFICIAL_META_SRC}/testSet.txt" ]]; then
  mkdir -p data/AVE/meta
  cp -f "${OFFICIAL_META_SRC}/Annotations.txt" data/AVE/meta/Annotations.txt
  cp -f "${OFFICIAL_META_SRC}/trainSet.txt" data/AVE/meta/trainSet.txt
  cp -f "${OFFICIAL_META_SRC}/valSet.txt" data/AVE/meta/valSet.txt
  cp -f "${OFFICIAL_META_SRC}/testSet.txt" data/AVE/meta/testSet.txt
  echo "OK: synced official AVE meta into data/AVE/meta (Annotations.txt + {train,val,test}Set.txt)"
else
  echo "WARN: official meta files not found under ${OFFICIAL_META_SRC}; falling back to GitHub meta" 1>&2
fi

python scripts/ave_build_official_lists.py \
  --meta-dir data/AVE/meta \
  --raw-videos-dir data/AVE/raw/videos \
  --tag official

echo "OK: official AVE dataset installed under data/AVE/raw/videos"
