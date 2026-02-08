#!/usr/bin/env bash
set -euo pipefail

# Pull the full EgoSchema video shards from the HuggingFace dataset repo.
#
# This repo expects EgoSchema to be present under:
#   data/EgoSchema/  (symlink is fine)
#
# Notes:
# - These files are very large (~100GB total). This script is resumable and will retry.
# - Some environments inject a broken localhost proxy. We explicitly disable proxy env vars.
# - If git-lfs batch requests are unstable, set EGO_METHOD=curl to fetch via `hf-mirror.com` resolve URLs.
#
# Usage:
#   bash scripts/datasets/egoschema_hf_pull_full.sh
#
# Optional:
#   REPO_DIR=data/hf_repos/egoschema bash scripts/datasets/egoschema_hf_pull_full.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

REPO_DIR="${REPO_DIR:-data/hf_repos/egoschema}"
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  echo "ERROR: missing HF repo clone at ${REPO_DIR}. Expected a git repo (with .git/)." >&2
  echo "Hint: git clone https://huggingface.co/datasets/lmms-lab/egoschema ${REPO_DIR}" >&2
  exit 2
fi

cd "${REPO_DIR}"

# auto|gitlfs|curl
EGO_METHOD="${EGO_METHOD:-auto}"
HF_MIRROR_BASE="${HF_MIRROR_BASE:-https://hf-mirror.com}"

# As of 2026-02-08, the public repo publishes 5 zip shards.
files=(videos_chunked_01.zip videos_chunked_02.zip videos_chunked_03.zip videos_chunked_04.zip videos_chunked_05.zip)
expects=(21424643499 21405519046 21430009244 21382497595 20084367404)

attempt=0
for i in "${!files[@]}"; do
  f="${files[$i]}"
  e="${expects[$i]}"
  while true; do
    attempt=$((attempt + 1))
    cur="$(stat -c%s "${f}" 2>/dev/null || echo 0)"
    echo "[egoschema_hf] file=${f} attempt=${attempt} size=${cur} expect=${e} ts=$(date +%Y-%m-%dT%H:%M:%S)"
    if [[ "${cur}" -eq "${e}" ]]; then
      echo "[egoschema_hf] file_done=${f}"
      break
    fi

    set +e
    rc=0
    if [[ "${EGO_METHOD}" == "gitlfs" || "${EGO_METHOD}" == "auto" ]]; then
      env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
        git lfs pull --include "${f}" --exclude ""
      rc=$?
    fi
    # Fallback: use hf-mirror resolve URL to bypass git-lfs batch endpoint instability.
    if [[ "${EGO_METHOD}" == "curl" || ( "${EGO_METHOD}" == "auto" && "${rc}" != "0" ) ]]; then
      # If the file is still an LFS pointer (tiny), restart the download from 0.
      cur_ptr="$(stat -c%s "${f}" 2>/dev/null || echo 0)"
      if [[ "${cur_ptr}" -gt 0 && "${cur_ptr}" -le 1024 ]]; then
        rm -f "${f}"
      fi
      resolve_url="${HF_MIRROR_BASE}/datasets/lmms-lab/egoschema/resolve/main/${f}"
      # Follow redirects to get a fresh signed CAS URL, then resume against that final URL.
      effective_url="$(
        env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
          curl -sS -I -L -o /dev/null -w '%{url_effective}' "${resolve_url}"
      )"
      env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
        curl -L --fail --retry 5 --retry-delay 5 --continue-at - -o "${f}" "${effective_url}"
      rc=$?
    fi
    set -e
    cur2="$(stat -c%s "${f}" 2>/dev/null || echo 0)"
    echo "[egoschema_hf] pull_rc=${rc} file=${f} size_after=${cur2}"
    sleep 3
  done
done

echo "[egoschema_hf] ALL_DONE ts=$(date +%Y-%m-%dT%H:%M:%S)"
