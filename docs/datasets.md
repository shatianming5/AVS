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

Preprocessing helpers:
- Extract untrimmed audio: `python -m avs.preprocess.epic_sounds_audio --help`
- Extract per-second frames: `python -m avs.preprocess.epic_sounds_frames --help`
- End-to-end pack (synthetic smoke available): `python -m avs.experiments.epic_sounds_long_pack --help`

---

## 3) Long-video QA (EgoSchema / IntentQA)

**Required for:** Final “long video QA” validation (planned; see `C0007+` and `E0201+`).

**Auto-download supported:** No (dataset terms vary).

Expected local layout (planned defaults):
- `data/EgoSchema/` or `data/IntentQA/` (exact IO spec will be added when the dataset is selected).

---

## 4) AVQA (Audio-Visual Question Answering)

**Status:** Metadata IO + prompt templates are implemented (`P0020`), but it is not currently the primary “final QA” benchmark in `docs/plan.md`.

