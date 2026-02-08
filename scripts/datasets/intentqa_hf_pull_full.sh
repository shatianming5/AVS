#!/usr/bin/env bash
set -euo pipefail

# Pull missing IntentQA LFS videos using `hf-mirror.com` + curl-resume.
#
# Why:
# - `git lfs pull` can be unstable on some networks (connection resets on batch endpoint).
# - The mirror's `resolve/main/...` redirects to signed CAS URLs that work well with `curl --continue-at -`.
#
# Expected layout:
#   data/hf_repos/IntentQA/IntentQA/videos/<video_id>.mp4
#
# Usage:
#   bash scripts/datasets/intentqa_hf_pull_full.sh
#
# Optional:
#   REPO_DIR=data/hf_repos/IntentQA JOBS=4 HF_MIRROR_BASE=https://hf-mirror.com bash scripts/datasets/intentqa_hf_pull_full.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

REPO_DIR="${REPO_DIR:-data/hf_repos/IntentQA}"
HF_MIRROR_BASE="${HF_MIRROR_BASE:-https://hf-mirror.com}"
JOBS="${JOBS:-2}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-6}"

VIDEOS_DIR="${REPO_DIR}/IntentQA/videos"
if [[ ! -d "${VIDEOS_DIR}" ]]; then
  echo "ERROR: missing videos dir: ${VIDEOS_DIR}" >&2
  echo "Hint: clone the HF dataset repo into ${REPO_DIR} first." >&2
  exit 2
fi

FFPROBE="${FFPROBE:-${REPO_ROOT}/data/tools/ffmpeg/bin/ffprobe}"
if [[ ! -x "${FFPROBE}" ]]; then
  FFPROBE="ffprobe"
fi

is_lfs_pointer() {
  local path="$1"
  # Avoid `head -n 1` on binary blobs (could scan until a newline); gate on small size.
  local sz
  sz="$(stat -c%s "${path}" 2>/dev/null || echo 0)"
  if [[ "${sz}" -le 0 || "${sz}" -gt 2048 ]]; then
    return 1
  fi
  local hdr=""
  hdr="$(head -c 200 "${path}" 2>/dev/null || true)"
  [[ "${hdr}" == *"version https://git-lfs.github.com/spec/v1"* ]]
}
export -f is_lfs_pointer

has_ftyp_header() {
  local path="$1"
  python - "${path}" <<'PY'
import sys
from pathlib import Path

p = Path(sys.argv[1])
try:
    with p.open("rb") as f:
        head = f.read(4096)
except Exception:
    sys.exit(1)
sys.exit(0 if b"ftyp" in head else 1)
PY
}
export -f has_ftyp_header

is_valid_mp4() {
  local path="$1"
  "${FFPROBE}" -hide_banner -v error -show_entries format=duration -of default=nw=1:nk=1 "${path}" >/dev/null 2>&1
}
export -f is_valid_mp4

download_one() {
  local path="$1"
  local bn
  bn="$(basename "${path}")"

  local url
  url="${HF_MIRROR_BASE}/datasets/hamedrahimi/IntentQA/resolve/main/IntentQA/videos/${bn}"

  # Fast path: already valid.
  if [[ -f "${path}" ]] && is_valid_mp4 "${path}"; then
    return 0
  fi

  # If still an LFS pointer (or clearly not an MP4), delete it first; resume would corrupt the file.
  if [[ -f "${path}" ]] && is_lfs_pointer "${path}"; then
    rm -f "${path}"
  elif [[ -f "${path}" ]] && ! has_ftyp_header "${path}"; then
    rm -f "${path}"
  fi

  local attempt=1
  while [[ "${attempt}" -le "${MAX_ATTEMPTS}" ]]; do
    echo "[intentqa_hf] pull file=${bn} attempt=${attempt}/${MAX_ATTEMPTS} ts=$(date -Is)"

    # Some environments inject a broken localhost proxy; explicitly disable proxy env vars.
    set +e
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
      curl -4 -L --fail --retry 8 --retry-delay 5 --retry-all-errors \
        --continue-at - --silent --show-error --no-progress-meter \
        -o "${path}" "${url}"
    rc=$?
    set -e

    if [[ -f "${path}" ]] && ! is_lfs_pointer "${path}" && is_valid_mp4 "${path}"; then
      return 0
    fi

    # If curl succeeded but we got an HTML/error body, don't keep it around for resume.
    if [[ -f "${path}" ]] && ! has_ftyp_header "${path}"; then
      rm -f "${path}"
    fi

    echo "[intentqa_hf] retry file=${bn} rc=${rc} size=$(stat -c%s \"${path}\" 2>/dev/null || echo 0)" >&2
    sleep $((attempt * 2))
    attempt=$((attempt + 1))
  done

  echo "[intentqa_hf] ERROR: failed to download a valid mp4 after ${MAX_ATTEMPTS} attempts: ${bn}" >&2
  return 1
}
export -f download_one
export HF_MIRROR_BASE
export FFPROBE
export MAX_ATTEMPTS

mapfile -d '' -t mp4s < <(find "${VIDEOS_DIR}" -maxdepth 1 -type f -name '*.mp4' -print0)
echo "[intentqa_hf] videos_dir=${VIDEOS_DIR} mp4s=${#mp4s[@]} jobs=${JOBS} max_attempts=${MAX_ATTEMPTS} ts=$(date -Is)"
if [[ "${#mp4s[@]}" -eq 0 ]]; then
  echo "[intentqa_hf] ERROR: no mp4s found under: ${VIDEOS_DIR}" >&2
  exit 2
fi

set +e
printf '%s\0' "${mp4s[@]}" | xargs -0 -n 1 -P "${JOBS}" bash -c 'download_one "$@"' _ 
rc=$?
set -e

# Re-check pointer count and print a summary.
left="$(find "${VIDEOS_DIR}" -maxdepth 1 -type f -name '*.mp4' -size -1024c | wc -l | tr -d ' ')"
echo "[intentqa_hf] DONE rc=${rc} pointers_left=${left} ts=$(date -Is)"
if [[ "${left}" -ne 0 || "${rc}" -ne 0 ]]; then
  echo "[intentqa_hf] WARNING: some mp4s are still missing/invalid. Re-run this script (it is resumable)." >&2
  exit 1
fi
