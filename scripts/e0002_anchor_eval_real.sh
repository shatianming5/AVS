#!/usr/bin/env bash
set -euo pipefail

# Real-AVE anchor quality evaluation (Recall@K / Recall@K,Δ).
#
# Default: yt-dlp download a small val subset into runs/, extract audio, and run anchor eval.
#
# Usage examples:
#   bash scripts/e0002_anchor_eval_real.sh
#   MODE=local SRC_DIR=/path/to/ave/mp4s bash scripts/e0002_anchor_eval_real.sh
#   LIMIT=64 bash scripts/e0002_anchor_eval_real.sh

MODE="${MODE:-yt-dlp}"          # yt-dlp|local
SRC_DIR="${SRC_DIR:-}"          # required when MODE=local (dir containing <video_id>.mp4)
SPLIT="${SPLIT:-val}"           # train|val|test (only used to pick ids from AVE meta)
LIMIT="${LIMIT:-32}"            # number of ids to attempt
METHOD="${METHOD:-energy}"      # energy|energy_delta|ast|panns|audiomae (depending on what you implement)
K="${K:-2}"
DELTAS="${DELTAS:-0,1,2}"
SEED="${SEED:-0}"
META_DIR="${META_DIR:-data/AVE/meta}"
RUN_DIR="${RUN_DIR:-runs/E0002_anchors_real_$(date +%Y%m%d-%H%M%S)}"

RAW_DIR="${RUN_DIR}/raw/videos"
PROC_DIR="${RUN_DIR}/processed"
DOWNLOAD_JSON="${RUN_DIR}/download.json"
CLIPS_JSONL="${RUN_DIR}/clips.jsonl"
OUT_DIR="${RUN_DIR}/anchor_eval"

mkdir -p "${RUN_DIR}"

download_args=(--mode "${MODE}" --split "${SPLIT}" --limit "${LIMIT}" --meta-dir "${META_DIR}" --out-dir "${RAW_DIR}" --out-json "${DOWNLOAD_JSON}")
if [[ "${MODE}" == "local" ]]; then
  if [[ -z "${SRC_DIR}" ]]; then
    echo "MODE=local requires SRC_DIR=/path/to/<video_id>.mp4 directory" 1>&2
    exit 2
  fi
  download_args+=(--src-dir "${SRC_DIR}")
fi

python -m avs.datasets.ave_download "${download_args[@]}" || true

OK_IDS="$(python - "${DOWNLOAD_JSON}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("")
    raise SystemExit(0)
obj = json.loads(path.read_text(encoding="utf-8"))
ids = [r["video_id"] for r in obj.get("results", []) if r.get("ok")]
print(" ".join(ids))
PY
)"

if [[ -z "${OK_IDS}" ]]; then
  echo "No successful downloads; see ${DOWNLOAD_JSON}" 1>&2
  exit 2
fi

AUDIO_IDS="$(python - "${RAW_DIR}" "${PROC_DIR}" ${OK_IDS} <<'PY'
import sys
from pathlib import Path

from avs.preprocess.ave_extract import extract_wav

raw_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
video_ids = sys.argv[3:]

ok = []
for vid in video_ids:
    try:
        extract_wav(raw_dir / f"{vid}.mp4", out_dir / vid / "audio.wav")
        ok.append(vid)
    except Exception as e:
        print(f"skip {vid}: {e}", file=sys.stderr)

print(" ".join(ok))
PY
)"

if [[ -z "${AUDIO_IDS}" ]]; then
  echo "No audio extracted successfully; see ${DOWNLOAD_JSON}" 1>&2
  exit 2
fi

python - "${META_DIR}" "${PROC_DIR}" "${CLIPS_JSONL}" ${AUDIO_IDS} <<'PY'
import json
import sys
from pathlib import Path

from avs.datasets.ave import AVEIndex, ensure_ave_meta

meta_dir = Path(sys.argv[1])
processed_dir = Path(sys.argv[2])
out_jsonl = Path(sys.argv[3])
video_ids = sys.argv[4:]

ensure_ave_meta(meta_dir)
index = AVEIndex.from_meta_dir(meta_dir)
clip_by_id = {c.video_id: c for c in index.clips}

rows = []
for vid in video_ids:
    wav_path = processed_dir / vid / "audio.wav"
    if not wav_path.exists():
        continue
    clip = clip_by_id.get(vid)
    if clip is None:
        continue
    gt = [i for i, lab in enumerate(index.segment_labels(clip)) if int(lab) != 0]
    rows.append({"clip_id": vid, "wav_path": str(wav_path), "gt_segments": gt})

out_jsonl.parent.mkdir(parents=True, exist_ok=True)
out_jsonl.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")
print(f"Wrote {len(rows)} clips -> {out_jsonl}")
PY

python -m avs.experiments.ave_anchor_eval \
  --clips-jsonl "${CLIPS_JSONL}" \
  --method "${METHOD}" \
  --k "${K}" \
  --deltas "${DELTAS}" \
  --seed "${SEED}" \
  --out-dir "${OUT_DIR}"

echo "OK: ${OUT_DIR}/anchors_metrics.json"
