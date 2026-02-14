from __future__ import annotations

import csv
import json
import os
import statistics as stats
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetStatus:
    name: str
    required: bool
    ok: bool
    details: dict
    next_steps: list[str]

    def to_jsonable(self) -> dict:
        return {
            "name": str(self.name),
            "required": bool(self.required),
            "ok": bool(self.ok),
            "details": dict(self.details),
            "next_steps": [str(x) for x in self.next_steps],
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _count_mp4_dir(path: Path) -> int:
    # Follow symlinks.
    try:
        return sum(1 for _ in path.glob("*.mp4"))
    except Exception:
        return 0


def _count_small_files(path: Path, pattern: str, *, max_bytes: int) -> int:
    try:
        return sum(1 for p in path.glob(pattern) if p.is_file() and p.stat().st_size <= int(max_bytes))
    except Exception:
        return 0


def _size_summary_bytes(path: Path, pattern: str) -> dict:
    try:
        sizes = [int(p.stat().st_size) for p in path.glob(pattern) if p.is_file()]
    except Exception:
        sizes = []
    if not sizes:
        return {}
    sizes_sorted = sorted(sizes)
    out = {
        "count": int(len(sizes_sorted)),
        "min": int(sizes_sorted[0]),
        "p10": int(stats.quantiles(sizes_sorted, n=10)[0]) if len(sizes_sorted) >= 10 else int(sizes_sorted[0]),
        "median": int(stats.median(sizes_sorted)),
        "p90": int(stats.quantiles(sizes_sorted, n=10)[8]) if len(sizes_sorted) >= 10 else int(sizes_sorted[-1]),
        "max": int(sizes_sorted[-1]),
        "total": int(sum(sizes_sorted)),
    }
    return out


def _read_epic_required_video_ids(meta_dir: Path, *, include_test: bool) -> list[str]:
    files = [
        meta_dir / "EPIC_Sounds_train.csv",
        meta_dir / "EPIC_Sounds_validation.csv",
    ]
    if include_test:
        files.append(meta_dir / "EPIC_Sounds_recognition_test_timestamps.csv")

    required: set[str] = set()
    for csv_path in files:
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = str(row.get("video_id", "")).strip()
                if vid:
                    required.add(vid)
    return sorted(required)


def _parse_epic_timestamp_s(ts: str) -> float | None:
    """
    Parse EPIC timestamp strings like "HH:MM:SS.mmm" into seconds.
    Returns None for empty/invalid inputs.
    """
    s = str(ts).strip()
    if not s:
        return None
    parts = s.split(":")
    if len(parts) != 3:
        return None
    try:
        h = int(parts[0])
        m = int(parts[1])
        sec = float(parts[2])
    except Exception:
        return None
    return float(h * 3600 + m * 60) + float(sec)


def _epic_required_stop_by_video(meta_dir: Path, *, include_test: bool) -> dict[str, float]:
    """
    For each video_id, compute the required minimum duration as max(stop_timestamp).
    """
    files = [
        meta_dir / "EPIC_Sounds_train.csv",
        meta_dir / "EPIC_Sounds_validation.csv",
    ]
    if include_test:
        files.append(meta_dir / "EPIC_Sounds_recognition_test_timestamps.csv")

    req: dict[str, float] = {}
    for csv_path in files:
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = str(row.get("video_id", "")).strip()
                stop_ts = row.get("stop_timestamp")
                stop = _parse_epic_timestamp_s(str(stop_ts) if stop_ts is not None else "")
                if not vid or stop is None:
                    continue
                req[vid] = max(float(req.get(vid, 0.0)), float(stop))
    return req


def _ffprobe_bin(root: Path) -> str:
    # Prefer repo-local static ffprobe (installed via scripts/setup/install_ffmpeg_static.py).
    cand = root / "data" / "tools" / "ffmpeg" / "bin" / "ffprobe"
    if cand.exists():
        return str(cand)
    return "ffprobe"


def _ffprobe_duration_s(*, ffprobe: str, path: Path) -> float | None:
    """
    Return duration in seconds via ffprobe, or None if probing fails.
    """
    r = subprocess.run(
        [str(ffprobe), "-hide_banner", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return None
    try:
        return float(r.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def _epic_duration_audit(
    *,
    root: Path,
    meta_dir: Path,
    raw_videos_dir: Path,
    include_test: bool,
    margin_s: float = 0.1,
    limit_videos: int | None = None,
) -> dict:
    """
    Audit whether each `video_id.mp4` is long enough to cover the max stop timestamp in the EPIC-SOUNDS metadata.
    """
    req = _epic_required_stop_by_video(meta_dir, include_test=bool(include_test))
    ffprobe = _ffprobe_bin(root)

    video_ids = sorted(req.keys())
    if limit_videos is not None:
        video_ids = video_ids[: int(limit_videos)]

    missing: list[str] = []
    too_short: list[dict] = []
    ok_count = 0
    for vid in video_ids:
        need = float(req[vid])
        p = raw_videos_dir / f"{vid}.mp4"
        if not p.exists():
            missing.append(vid)
            continue
        dur = _ffprobe_duration_s(ffprobe=ffprobe, path=p)
        if dur is None:
            too_short.append({"video_id": vid, "need_s": need, "dur_s": None, "shortfall_s": None, "size_bytes": int(p.stat().st_size)})
            continue
        shortfall = float(need) - float(dur) - float(margin_s)
        if shortfall > 0.0:
            too_short.append(
                {"video_id": vid, "need_s": need, "dur_s": float(dur), "shortfall_s": float(shortfall), "size_bytes": int(p.stat().st_size)}
            )
        else:
            ok_count += 1

    too_short_sorted = sorted(
        [r for r in too_short if r.get("shortfall_s") is not None],
        key=lambda r: float(r["shortfall_s"]),  # type: ignore[index]
        reverse=True,
    )
    return {
        "include_test": bool(include_test),
        "margin_s": float(margin_s),
        "checked_videos": int(len(video_ids)),
        "ok_videos": int(ok_count),
        "missing_files": int(len(missing)),
        "too_short_files": int(len(too_short)),
        "missing_examples": missing[:20],
        "too_short_examples": too_short_sorted[:20],
    }


def check_ave(*, root: Path) -> DatasetStatus:
    meta_dir = root / "data" / "AVE" / "meta"
    raw_videos_dir = root / "data" / "AVE" / "raw" / "videos"
    ann = meta_dir / "Annotations.txt"
    mp4_count = _count_mp4_dir(raw_videos_dir) if raw_videos_dir.exists() else 0

    ok = ann.exists() and raw_videos_dir.exists() and mp4_count > 0
    next_steps = []
    if not ok:
        next_steps.append("bash scripts/ave_install_official.sh")

    return DatasetStatus(
        name="AVE",
        required=True,
        ok=ok,
        details={"meta_dir": str(meta_dir), "raw_videos_dir": str(raw_videos_dir), "annotations": str(ann), "mp4_count": int(mp4_count)},
        next_steps=next_steps,
    )


def check_epic_sounds(*, root: Path) -> DatasetStatus:
    meta_dir = root / "data" / "EPIC_SOUNDS" / "meta"
    raw_videos_dir = root / "data" / "EPIC_SOUNDS" / "raw" / "videos"
    has_meta = meta_dir.exists()
    mp4_count = _count_mp4_dir(raw_videos_dir) if raw_videos_dir.exists() else 0

    include_test = bool(os.environ.get("EPIC_INCLUDE_TEST", "").strip() not in ("", "0", "false", "False"))
    required = _read_epic_required_video_ids(meta_dir, include_test=include_test) if has_meta else []
    have = {p.stem for p in raw_videos_dir.glob("*.mp4")} if raw_videos_dir.exists() else set()
    missing = [vid for vid in required if vid not in have]

    size_summary = _size_summary_bytes(raw_videos_dir, "*.mp4") if raw_videos_dir.exists() else {}
    small_lt_5mb = _count_small_files(raw_videos_dir, "*.mp4", max_bytes=5_000_000) if raw_videos_dir.exists() else 0

    # "ok" here is about coverage, not about whether videos are truly untrimmed.
    ok = has_meta and (len(required) > 0) and (len(missing) == 0)

    next_steps = []
    if has_meta and (len(required) > 0) and (len(missing) > 0):
        next_steps.append("Missing EPIC videos. If you have EPIC-KITCHENS access, use the official downloader or import from a local EPIC-KITCHENS videos folder.")
        next_steps.append("Suggested: python scripts/datasets/epic_sounds_fill_missing.py --help")
    if has_meta and mp4_count > 0 and small_lt_5mb > 0:
        next_steps.append("Warning: many EPIC videos are very small (<5MB). If these are trimmed/partial, timestamps in EPIC_Sounds_*.csv will not align. Consider re-importing full untrimmed videos.")

    duration_audit = None
    do_duration_audit = bool(os.environ.get("EPIC_DURATION_AUDIT", "").strip() not in ("", "0", "false", "False"))
    if do_duration_audit and has_meta and raw_videos_dir.exists() and required:
        duration_audit = _epic_duration_audit(
            root=root,
            meta_dir=meta_dir,
            raw_videos_dir=raw_videos_dir,
            include_test=include_test,
            margin_s=float(os.environ.get("EPIC_DURATION_MARGIN_S", "0.1")),
            limit_videos=(int(os.environ["EPIC_DURATION_LIMIT"]) if os.environ.get("EPIC_DURATION_LIMIT") else None),
        )
        if int(duration_audit.get("too_short_files", 0)) > 0:
            next_steps.append("EPIC duration audit: some mp4s are shorter than required stop timestamps. Import full untrimmed EPIC-KITCHENS videos to make timestamp-based eval meaningful.")

    return DatasetStatus(
        name="EPIC_SOUNDS",
        required=False,
        ok=ok,
        details={
            "meta_dir": str(meta_dir),
            "raw_videos_dir": str(raw_videos_dir),
            "mp4_count": int(mp4_count),
            "required_video_ids": int(len(required)),
            "have_video_ids": int(len(have)),
            "missing_video_ids": int(len(missing)),
            "small_lt_5mb": int(small_lt_5mb),
            "size_summary_bytes": size_summary,
            "duration_audit": duration_audit,
        },
        next_steps=next_steps,
    )


def check_egoschema(*, root: Path) -> DatasetStatus:
    extracted_root = root / "data" / "EgoSchema"
    videos_dir = extracted_root / "videos"
    mp4_count = _count_mp4_dir(videos_dir) if videos_dir.exists() else 0

    hf_repo = root / "data" / "hf_repos" / "egoschema"
    zips = sorted(hf_repo.glob("videos_chunked_*.zip")) if hf_repo.exists() else []
    zip_count = int(len(zips))
    zip_pointer = int(sum(1 for p in zips if p.is_file() and p.stat().st_size <= 1024))
    zip_downloaded = zip_count - zip_pointer
    zip_total_bytes = int(sum(int(p.stat().st_size) for p in zips if p.is_file() and p.stat().st_size > 1024))

    # For VLM eval, extracted mp4s are required.
    ok = mp4_count > 0
    next_steps: list[str] = []
    if not hf_repo.exists():
        next_steps.append("Clone the HF dataset repo into data/hf_repos/egoschema (see scripts/datasets/egoschema_hf_pull_full.sh).")
    if zip_count == 0 and hf_repo.exists():
        next_steps.append("Pull the EgoSchema zip shards (git-lfs): bash scripts/datasets/egoschema_hf_pull_full.sh")
    if zip_count > 0 and zip_pointer > 0:
        next_steps.append("Some EgoSchema zip shards are still LFS pointers. Re-run: bash scripts/datasets/egoschema_hf_pull_full.sh")
    if zip_count > 0 and zip_pointer == 0 and mp4_count == 0:
        next_steps.append("Extract EgoSchema videos: bash scripts/datasets/egoschema_extract_videos.sh")
    return DatasetStatus(
        name="EgoSchema",
        required=False,
        ok=ok,
        details={
            "extracted_root": str(extracted_root),
            "videos_dir": str(videos_dir),
            "mp4_count": int(mp4_count),
            "hf_repo": str(hf_repo),
            "zip_parts": int(zip_count),
            "zip_pointer_parts": int(zip_pointer),
            "zip_downloaded_parts": int(zip_downloaded),
            "zip_total_bytes": int(zip_total_bytes),
        },
        next_steps=next_steps,
    )


def check_intentqa(*, root: Path) -> DatasetStatus:
    d = root / "data" / "IntentQA"
    hf = root / "data" / "hf_repos" / "IntentQA"

    # Canonical layout: data/IntentQA/IntentQA/videos/*.mp4
    videos_dir = d / "IntentQA" / "videos"
    if not videos_dir.exists():
        # Common HF clone layout:
        #   data/hf_repos/IntentQA/IntentQA/videos/*.mp4
        videos_dir = hf / "IntentQA" / "videos"
    mp4_count = _count_mp4_dir(videos_dir) if videos_dir.exists() else 0
    mp4_pointer = _count_small_files(videos_dir, "*.mp4", max_bytes=1024) if videos_dir.exists() else 0
    ok = mp4_count > 0 and mp4_pointer == 0
    next_steps: list[str] = []
    if not ok and hf.exists():
        next_steps.append("If this is a HF clone under data/hf_repos/IntentQA, symlink it to the canonical path: ln -sfn $(pwd)/data/hf_repos/IntentQA $(pwd)/data/IntentQA")
    if not ok:
        next_steps.append("Download/pull IntentQA videos into data/IntentQA (HF LFS).")
    return DatasetStatus(
        name="IntentQA",
        required=False,
        ok=ok,
        details={
            "root": str(d),
            "hf_repo": str(hf),
            "videos_dir": str(videos_dir),
            "mp4_count": int(mp4_count),
            "mp4_pointer_files": int(mp4_pointer),
        },
        next_steps=next_steps,
    )


def check_avqa(*, root: Path) -> DatasetStatus:
    meta_dir = root / "data" / "AVQA" / "meta"
    raw_videos_dir = root / "data" / "AVQA" / "raw" / "videos"
    train = meta_dir / "train_qa.json"
    val = meta_dir / "val_qa.json"

    has_meta = train.exists() and val.exists()
    mp4_count = _count_mp4_dir(raw_videos_dir) if raw_videos_dir.exists() else 0
    mp4_pointer = _count_small_files(raw_videos_dir, "*.mp4", max_bytes=1024) if raw_videos_dir.exists() else 0

    # For VLM eval, both meta + raw clips are required.
    ok = has_meta and mp4_count > 0 and mp4_pointer == 0

    next_steps: list[str] = []
    if not has_meta:
        next_steps.append(
            "Download AVQA metadata: python -c \"from pathlib import Path; from avs.datasets.avqa import ensure_avqa_meta; ensure_avqa_meta(Path('data/AVQA/meta'))\""
        )
    if has_meta and (mp4_count == 0 or mp4_pointer > 0):
        next_steps.append("Download AVQA clips (yt-dlp, VGGSound slicing): python -m avs.datasets.avqa_download --split val --limit 256")

    return DatasetStatus(
        name="AVQA",
        required=False,
        ok=ok,
        details={
            "meta_dir": str(meta_dir),
            "raw_videos_dir": str(raw_videos_dir),
            "has_meta": bool(has_meta),
            "mp4_count": int(mp4_count),
            "mp4_pointer_files": int(mp4_pointer),
        },
        next_steps=next_steps,
    )


def check_videomme(*, root: Path) -> DatasetStatus:
    meta_dir = root / "data" / "VideoMME" / "meta"
    raw_videos_dir = root / "data" / "VideoMME" / "raw" / "videos"
    meta = meta_dir / "test.jsonl"

    has_meta = meta.exists() and meta.is_file() and meta.stat().st_size > 0
    mp4_count = _count_mp4_dir(raw_videos_dir) if raw_videos_dir.exists() else 0
    mp4_pointer = _count_small_files(raw_videos_dir, "*.mp4", max_bytes=1024) if raw_videos_dir.exists() else 0

    ok = has_meta and mp4_count > 0 and mp4_pointer == 0
    next_steps: list[str] = []
    if not has_meta:
        next_steps.append("Generate metadata snapshot: python -c \"from pathlib import Path; from avs.datasets.videomme import ensure_videomme_meta; ensure_videomme_meta(Path('data/VideoMME/meta'))\"")
    if has_meta and (mp4_count == 0 or mp4_pointer > 0):
        next_steps.append("Download raw videos (deterministic subset): bash scripts/datasets/videomme_install.sh")

    return DatasetStatus(
        name="VideoMME",
        required=False,
        ok=ok,
        details={
            "meta_dir": str(meta_dir),
            "meta_jsonl": str(meta),
            "raw_videos_dir": str(raw_videos_dir),
            "has_meta": bool(has_meta),
            "mp4_count": int(mp4_count),
            "mp4_pointer_files": int(mp4_pointer),
        },
        next_steps=next_steps,
    )


def main() -> int:
    root = _repo_root()
    out_dir = root / "runs" / f"datasets_verify_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    statuses = [
        check_ave(root=root),
        check_epic_sounds(root=root),
        check_egoschema(root=root),
        check_intentqa(root=root),
        check_avqa(root=root),
        check_videomme(root=root),
    ]

    ok = all((not s.required) or s.ok for s in statuses)
    payload = {"ok": ok, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "datasets": [s.to_jsonable() for s in statuses]}

    out_json = out_dir / "datasets_verify.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_json)

    # Do not hard-fail on optional datasets; keep this as a guide + log generator.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
