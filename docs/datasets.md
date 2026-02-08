# Datasets

This repo aims to keep dataset installation reproducible and auditable.

The canonical “what to run next” gate is:

```bash
bash scripts/datasets/verify_all.sh
```

It writes a JSON report under `runs/` and prints next steps for missing datasets.

---

## 1) AVE (Audio-Visual Event)

**Required for:** AVE-P0 / Listen-then-Look MDE on AVE (`E0001`/`E0002`/`E0003`/`E0010+`).

**Auto-download supported:** Yes (official zip).

Install:

```bash
bash scripts/ave_install_official.sh
```

Notes:
- `data/AVE/raw/videos` is installed as a symlink to the extracted `AVE_Dataset/AVE/` folder.

---

## 2) EPIC-SOUNDS (EPIC-KITCHENS audio events)

**Required for:** Long-video proxy / downstream task (`E0100`).

**Auto-download supported:** No (requires EPIC-KITCHENS access / credentials).

Expected local layout:
- Metadata CSVs: `data/EPIC_SOUNDS/meta/*.csv` (already tracked / small)
- Raw videos: `data/EPIC_SOUNDS/raw/videos/<video_id>.mp4`

Materialize videos (recommended):
- If you already have EPIC-KITCHENS videos locally, import/symlink them into the expected layout:
  - `python scripts/datasets/epic_sounds_fill_missing.py --meta-dir data/EPIC_SOUNDS/meta --raw-videos-dir data/EPIC_SOUNDS/raw/videos --src-root /path/to/EPIC-KITCHENS --src-mode symlink --require-meta-duration`
- If you have official EPIC-KITCHENS downloader access, you can also drive `epic_downloader.py` one video at a time:
  - `python scripts/datasets/epic_sounds_fill_missing.py --meta-dir data/EPIC_SOUNDS/meta --raw-videos-dir data/EPIC_SOUNDS/raw/videos --downloader-dir /path/to/epic_downloader --staging-dir /tmp/epic_stage --require-meta-duration`

Preprocessing helpers:
- Extract untrimmed audio: `python -m avs.preprocess.epic_sounds_audio --help`
- Extract per-second frames: `python -m avs.preprocess.epic_sounds_frames --help`
- End-to-end pack (synthetic smoke available): `python -m avs.experiments.epic_sounds_long_pack --help`

---

## 3) Long-video QA (EgoSchema / IntentQA)

**Required for:** Final “long video QA” validation (`E0600+`).

**Auto-download supported:** No (dataset terms vary).

### IntentQA

Expected local layout (canonical):
- CSVs: `data/IntentQA/IntentQA/{train,val,test}.csv`
- Videos: `data/IntentQA/IntentQA/videos/<video_id>.mp4`

This repo also supports a common HF-clone layout:
- `data/hf_repos/IntentQA/IntentQA/...`

If you cloned from HF into `data/hf_repos/IntentQA`, symlink it:

```bash
ln -sfn "$(pwd)/data/hf_repos/IntentQA" "$(pwd)/data/IntentQA"
```

### EgoSchema

Expected local layout:
- HF repo clone: `data/hf_repos/egoschema/` (parquet metadata + zip shards)
- Extracted videos: `data/EgoSchema/videos/<video_idx>.mp4`

If you already pulled the zip shards, extract them with:

```bash
bash scripts/datasets/egoschema_extract_videos.sh
```

---

## 4) AVQA (Audio-Visual Question Answering)

**Status:** Metadata IO + prompt templates are implemented (`P0020`), but it is not currently the primary “final QA” benchmark in `docs/plan.md`.
