#!/usr/bin/env bash
set -euo pipefail

# Pull EPIC-KITCHENS videos packaged as zip shards from the public HF dataset:
#   Zzitang/EpicKitchens
#
# This is used as a potential untrimmed video source for EPIC-SOUNDS.
#
# Notes:
# - Files are large (~52GB total). This script is resumable and will retry.
# - Some environments inject a broken localhost proxy. We explicitly disable proxy env vars.
# - We do not rely on git-lfs; we fetch via hf-mirror resolve URLs + signed CAS redirects.
#
# Usage:
#   bash scripts/datasets/epic_kitchens_zztitang_hf_pull_full.sh
#
# Optional:
#   REPO_DIR=data/hf_repos/epic_kitchens_zztitang bash scripts/datasets/epic_kitchens_zztitang_hf_pull_full.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

REPO_DIR="${REPO_DIR:-data/hf_repos/epic_kitchens_zztitang}"
mkdir -p "${REPO_DIR}"
cd "${REPO_DIR}"

HF_MIRROR_BASE="${HF_MIRROR_BASE:-https://hf-mirror.com}"

files=(
  EpicKitchens_videos_chunked_01.zip
  EpicKitchens_videos_chunked_02.zip
  EpicKitchens_videos_chunked_03.zip
  EpicKitchens_videos_chunked_04.zip
  EpicKitchens_videos_chunked_05.zip
  EpicKitchens_videos_chunked_06.zip
  EpicKitchens_videos_chunked_07.zip
  EpicKitchens_videos_chunked_08.zip
  EpicKitchens_videos_chunked_09.zip
  EpicKitchens_videos_chunked_10.zip
  EpicKitchens_videos_chunked_11.zip
  EpicKitchens_videos_chunked_12.zip
)

_content_length() {
  local url="$1"
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    curl -sS -I "${url}" \
    | awk 'BEGIN{IGNORECASE=1} $1=="Content-Length:" {gsub("\r","",$2); print $2; exit}'
}

attempt=0
for f in "${files[@]}"; do
  while true; do
    attempt=$((attempt + 1))

    resolve_url="${HF_MIRROR_BASE}/datasets/Zzitang/EpicKitchens/resolve/main/${f}"
    effective_url="$(
      env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
        curl -sS -I -L -o /dev/null -w '%{url_effective}' "${resolve_url}"
    )"
    expect="$(_content_length "${effective_url}" || echo "")"
    cur="$(stat -c%s "${f}" 2>/dev/null || echo 0)"

    echo "[epic_kitchens_hf] file=${f} attempt=${attempt} size=${cur} expect=${expect:-unknown} ts=$(date -Is)"

    if [[ -n "${expect}" && "${cur}" -eq "${expect}" ]]; then
      echo "[epic_kitchens_hf] file_done=${f}"
      break
    fi

    # If the file is still a tiny pointer/bad partial, restart from 0.
    if [[ "${cur}" -gt 0 && "${cur}" -le 1024 ]]; then
      rm -f "${f}"
      cur=0
    fi

    set +e
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
      curl -L --fail --retry 10 --retry-all-errors --retry-delay 5 --continue-at - \
        --silent --show-error --no-progress-meter -o "${f}" "${effective_url}"
    rc=$?
    set -e

    cur2="$(stat -c%s "${f}" 2>/dev/null || echo 0)"
    echo "[epic_kitchens_hf] pull_rc=${rc} file=${f} size_after=${cur2}"
    sleep 2
  done
done

echo "[epic_kitchens_hf] ALL_DONE ts=$(date -Is)"
