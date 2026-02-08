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
JOBS="${JOBS:-4}"

VIDEOS_DIR="${REPO_DIR}/IntentQA/videos"
if [[ ! -d "${VIDEOS_DIR}" ]]; then
  echo "ERROR: missing videos dir: ${VIDEOS_DIR}" >&2
  echo "Hint: clone the HF dataset repo into ${REPO_DIR} first." >&2
  exit 2
fi

download_one() {
  local path="$1"
  local bn
  bn="$(basename "${path}")"

  # If still an LFS pointer, delete it first; resume would corrupt the file.
  local sz
  sz="$(stat -c%s "${path}" 2>/dev/null || echo 0)"
  if [[ "${sz}" -gt 0 && "${sz}" -le 1024 ]]; then
    rm -f "${path}"
  fi

  local url
  url="${HF_MIRROR_BASE}/datasets/hamedrahimi/IntentQA/resolve/main/IntentQA/videos/${bn}"

  # Some environments inject a broken localhost proxy; explicitly disable proxy env vars.
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    curl -4 -L --fail --retry 8 --retry-delay 5 --retry-all-errors \
      --continue-at - -o "${path}" "${url}"
}
export -f download_one
export HF_MIRROR_BASE

mapfile -t ptrs < <(find "${VIDEOS_DIR}" -maxdepth 1 -type f -name '*.mp4' -size -1024c | sort)
echo "[intentqa_hf] videos_dir=${VIDEOS_DIR} pointers=${#ptrs[@]} jobs=${JOBS} ts=$(date -Is)"
if [[ "${#ptrs[@]}" -eq 0 ]]; then
  echo "[intentqa_hf] ALL_DONE (no pointer mp4s)" >&2
  exit 0
fi

printf '%s\0' "${ptrs[@]}" | xargs -0 -n 1 -P "${JOBS}" bash -c 'download_one "$@"' _ || true

# Re-check pointer count and print a summary.
left="$(find "${VIDEOS_DIR}" -maxdepth 1 -type f -name '*.mp4' -size -1024c | wc -l | tr -d ' ')"
echo "[intentqa_hf] DONE pointers_left=${left} ts=$(date -Is)"
if [[ "${left}" -ne 0 ]]; then
  echo "[intentqa_hf] WARNING: some pointer mp4s remain. Re-run this script (it is resumable)." >&2
  exit 1
fi
