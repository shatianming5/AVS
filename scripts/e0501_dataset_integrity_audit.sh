#!/usr/bin/env bash
set -euo pipefail

# E0501: dataset integrity audit (decode/probe) for AVE + EPIC video roots.
#
# Outputs:
#   runs/E0501_dataset_integrity_<ts>/index.json
#   runs/E0501_dataset_integrity_<ts>/ave/dataset_integrity_audit.json
#   runs/E0501_dataset_integrity_<ts>/epic/dataset_integrity_audit.json

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="${OUT_DIR:-runs/E0501_dataset_integrity_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${OUT_DIR}"

AVE_VIDEOS_DIR="${AVE_VIDEOS_DIR:-data/AVE/raw/videos}"
EPIC_VIDEOS_DIR="${EPIC_VIDEOS_DIR:-data/EPIC_SOUNDS/raw/videos}"
PATTERN="${PATTERN:-*.mp4}"
DECODE_CHECK="${DECODE_CHECK:-sampled}"  # none|sampled|full
DECODE_LIMIT="${DECODE_LIMIT:-64}"
LIMIT="${LIMIT:-}"

index_file="${OUT_DIR}/index.json"
echo "{\"ok\": true, \"runs\": []}" > "${index_file}"

run_one() {
  local name="$1"
  local videos_dir="$2"
  local subdir="${OUT_DIR}/${name}"

  if [[ ! -d "${videos_dir}" ]]; then
    echo "WARN: skip ${name}; missing directory: ${videos_dir}" >&2
    return 0
  fi

  local args=(
    --videos-dir "${videos_dir}"
    --pattern "${PATTERN}"
    --decode-check "${DECODE_CHECK}"
    --decode-limit "${DECODE_LIMIT}"
    --out-dir "${subdir}"
  )
  if [[ -n "${LIMIT}" ]]; then
    args+=(--limit "${LIMIT}")
  fi

  local out_json
  out_json="$(python -m avs.experiments.dataset_integrity_audit "${args[@]}")"
  echo "OK: ${name} -> ${out_json}"

  python - "${index_file}" "${name}" "${out_json}" <<'PY'
import json
import sys
from pathlib import Path

index_path = Path(sys.argv[1])
name = sys.argv[2]
out_json = sys.argv[3]
obj = json.loads(index_path.read_text(encoding="utf-8"))
obj.setdefault("runs", []).append({"name": name, "artifact": out_json})
index_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_one "ave" "${AVE_VIDEOS_DIR}"
run_one "epic" "${EPIC_VIDEOS_DIR}"

echo "OK: ${index_file}"
