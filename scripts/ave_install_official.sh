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

python scripts/ave_build_official_lists.py \
  --meta-dir data/AVE/meta \
  --raw-videos-dir data/AVE/raw/videos \
  --tag official

echo "OK: official AVE dataset installed under data/AVE/raw/videos"

