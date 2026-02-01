#!/usr/bin/env bash
set -euo pipefail

# Quick status for the official AVE dataset install + verification jobs.
#
# Usage:
#   bash scripts/ave_status.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "== AVE status ($(date)) =="

ZIP_PATH="data/AVE/raw/AVE_Dataset.zip"
if [[ -f "${ZIP_PATH}" ]]; then
  python - <<'PY'
import os
import subprocess
from pathlib import Path

p = Path("data/AVE/raw/AVE_Dataset.zip")
size = p.stat().st_size

# Official AVE zip URL (Google Drive direct link). We use a HEAD request to get Content-Length.
url = "https://drive.usercontent.google.com/download?id=1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK&export=download&confirm=t"

total = None
try:
    out = subprocess.check_output(
        ["curl", "-sI", "-L", "--connect-timeout", "5", "--max-time", "10", url],
        stderr=subprocess.DEVNULL,
        text=True,
    )
    for line in out.splitlines():
        if line.lower().startswith("content-length:"):
            total = int(line.split(":", 1)[1].strip())
except Exception:
    total = None

if total is None:
    # Fallback: known size observed from HEAD (bytes). If the upstream changes, percent may be off.
    total = 5_665_119_358

pct = 100.0 * size / total if total > 0 else 0.0
print(f"zip: {p} size={size/1024/1024:.1f} MiB / {total/1024/1024:.1f} MiB (~{pct:.1f}%)")
PY
else
  echo "zip: missing (${ZIP_PATH})"
fi

echo
echo "tmux:"
tmux ls 2>/dev/null | rg "ave-official-(install|verify)" || echo "  (no ave-official tmux sessions)"

echo
echo "logs (tail):"
test -f runs/ave_official_install.log && { echo "-- runs/ave_official_install.log"; tail -n 3 runs/ave_official_install.log; } || true
test -f runs/ave_official_verify.log && { echo "-- runs/ave_official_verify.log"; tail -n 5 runs/ave_official_verify.log; } || true

echo
df -h "${REPO_ROOT}" | tail -n 1
