#!/usr/bin/env bash
set -euo pipefail

# Unpack the official AVE dataset zip, and (optionally) expose it under a convenient videos directory.
#
# By default, this script:
#   1) Unzips into `data/AVE/raw/` (creating `data/AVE/raw/AVE_Dataset/`)
#   2) Creates a symlink `data/AVE/raw/videos_official -> data/AVE/raw/AVE_Dataset`
#
# You can optionally install it as the default raw videos directory used by the codebase:
#   INSTALL_AS_DEFAULT=1 bash scripts/ave_unpack_official.sh
#
# Usage:
#   bash scripts/ave_unpack_official.sh
#   ZIP_PATH=data/AVE/raw/AVE_Dataset.zip bash scripts/ave_unpack_official.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ZIP_PATH="${ZIP_PATH:-${REPO_ROOT}/data/AVE/raw/AVE_Dataset.zip}"
EXTRACT_ROOT="${EXTRACT_ROOT:-${REPO_ROOT}/data/AVE/raw}"

INSTALL_AS_DEFAULT="${INSTALL_AS_DEFAULT:-0}"

if [[ ! -f "${ZIP_PATH}" ]]; then
  echo "Missing ZIP_PATH: ${ZIP_PATH}" 1>&2
  exit 2
fi

mkdir -p "${EXTRACT_ROOT}"
unzip -n "${ZIP_PATH}" -d "${EXTRACT_ROOT}"

EXTRACTED_DIR="${EXTRACT_ROOT}/AVE_Dataset"
if [[ ! -d "${EXTRACTED_DIR}" ]]; then
  echo "Expected extracted dir not found: ${EXTRACTED_DIR}" 1>&2
  exit 2
fi

# In the official zip, videos typically live under `AVE_Dataset/AVE/*.mp4`.
VIDEO_DIR="${EXTRACTED_DIR}/AVE"
if [[ ! -d "${VIDEO_DIR}" ]]; then
  # Fallback for unexpected layouts: treat `AVE_Dataset/` itself as the videos dir.
  VIDEO_DIR="${EXTRACTED_DIR}"
fi

OFFICIAL_LINK="${EXTRACT_ROOT}/videos_official"
if [[ -L "${OFFICIAL_LINK}" ]]; then
  cur="$(readlink "${OFFICIAL_LINK}" || true)"
  if [[ "$(readlink -f "${OFFICIAL_LINK}")" != "$(readlink -f "${VIDEO_DIR}")" ]]; then
    rm -f "${OFFICIAL_LINK}"
    ln -s "${VIDEO_DIR}" "${OFFICIAL_LINK}"
    echo "Relinked: ${OFFICIAL_LINK} -> ${VIDEO_DIR} (was: ${cur})"
  else
    echo "Exists (ok): ${OFFICIAL_LINK} -> ${cur}"
  fi
elif [[ -e "${OFFICIAL_LINK}" ]]; then
  echo "Exists (skip; not a symlink): ${OFFICIAL_LINK}"
else
  ln -s "${VIDEO_DIR}" "${OFFICIAL_LINK}"
  echo "Linked: ${OFFICIAL_LINK} -> ${VIDEO_DIR}"
fi

if [[ "${INSTALL_AS_DEFAULT}" == "1" ]]; then
  DEFAULT_RAW="${EXTRACT_ROOT}/videos"
  if [[ -e "${DEFAULT_RAW}" && ! -L "${DEFAULT_RAW}" ]]; then
    BACKUP="${EXTRACT_ROOT}/videos_ytdlp_$(date +%Y%m%d-%H%M%S)"
    mv "${DEFAULT_RAW}" "${BACKUP}"
    echo "Moved existing raw videos dir: ${DEFAULT_RAW} -> ${BACKUP}"
  fi
  if [[ -L "${DEFAULT_RAW}" ]]; then
    cur="$(readlink "${DEFAULT_RAW}" || true)"
    if [[ "$(readlink -f "${DEFAULT_RAW}")" != "$(readlink -f "${VIDEO_DIR}")" ]]; then
      rm -f "${DEFAULT_RAW}"
      ln -s "${VIDEO_DIR}" "${DEFAULT_RAW}"
      echo "Reinstalled: ${DEFAULT_RAW} -> ${VIDEO_DIR} (was: ${cur})"
    else
      echo "Exists (ok): ${DEFAULT_RAW} -> ${cur}"
    fi
  elif [[ -e "${DEFAULT_RAW}" ]]; then
    echo "Exists (skip; not a symlink): ${DEFAULT_RAW}"
  else
    ln -s "${VIDEO_DIR}" "${DEFAULT_RAW}"
    echo "Installed: ${DEFAULT_RAW} -> ${VIDEO_DIR}"
  fi
fi

echo "OK: extracted=${EXTRACTED_DIR} videos=${VIDEO_DIR}"
