#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
import zipfile
from pathlib import Path


def _read_required_video_ids(meta_dir: Path, *, include_test: bool) -> list[str]:
    files = [
        meta_dir / "EPIC_Sounds_train.csv",
        meta_dir / "EPIC_Sounds_validation.csv",
    ]
    if include_test:
        files.append(meta_dir / "EPIC_Sounds_recognition_test_timestamps.csv")

    required: set[str] = set()
    for csv_path in files:
        if not csv_path.exists():
            raise FileNotFoundError(f"missing meta csv: {csv_path}")
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = str(row.get("video_id", "")).strip()
                if vid:
                    required.add(vid)
    return sorted(required)


def _iter_zip_shards(zips_dir: Path) -> list[Path]:
    return sorted(zips_dir.glob("EpicKitchens_videos_chunked_*.zip"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Import EPIC-SOUNDS untrimmed videos from EPIC-KITCHENS zip shards.")
    p.add_argument("--meta-dir", type=Path, required=True)
    p.add_argument("--raw-videos-dir", type=Path, required=True)
    p.add_argument("--zips-dir", type=Path, required=True, help="Folder containing EpicKitchens_videos_chunked_*.zip")
    p.add_argument("--include-test", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log-path", type=Path, default=Path("runs") / f"epic_import_from_zips_{time.strftime('%Y%m%d-%H%M%S')}.json")
    p.add_argument("--dry-run", action="store_true")
    return p


def _build_index(shards: list[Path], required: set[str]) -> dict[str, tuple[Path, str]]:
    """
    Map video_id -> (zip_path, member_name) for required ids only.
    """
    index: dict[str, tuple[Path, str]] = {}
    for zp in shards:
        with zipfile.ZipFile(zp, "r") as z:
            for name in z.namelist():
                low = name.lower()
                if not low.endswith(".mp4"):
                    continue
                vid = Path(name).stem
                if vid in required and vid not in index:
                    index[vid] = (zp, name)
    return index


def _extract_member(*, zip_path: Path, member: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(member, "r") as src, tmp.open("wb") as out:
            shutil.copyfileobj(src, out, length=1024 * 1024)
    tmp.replace(dst)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.meta_dir = args.meta_dir.resolve()
    args.raw_videos_dir = args.raw_videos_dir.resolve()
    args.zips_dir = args.zips_dir.resolve()
    args.log_path = args.log_path.resolve()

    required_ids = _read_required_video_ids(args.meta_dir, include_test=bool(args.include_test))
    required_set = set(required_ids)
    shards = _iter_zip_shards(args.zips_dir)
    if not shards:
        raise SystemExit(f"no zip shards found under: {args.zips_dir}")

    index = _build_index(shards, required_set)
    missing_in_zips = [vid for vid in required_ids if vid not in index]

    start = max(0, int(args.start_index))
    queue = required_ids[start:]
    if args.limit is not None:
        queue = queue[: max(0, int(args.limit))]

    summary: dict[str, object] = {
        "meta_dir": str(args.meta_dir),
        "raw_videos_dir": str(args.raw_videos_dir),
        "zips_dir": str(args.zips_dir),
        "zip_shards": [str(p) for p in shards],
        "required_count": int(len(required_ids)),
        "index_found": int(len(index)),
        "missing_in_zips": int(len(missing_in_zips)),
        "missing_in_zips_examples": missing_in_zips[:50],
        "start_index": int(start),
        "limit": None if args.limit is None else int(args.limit),
        "queued_count": int(len(queue)),
        "overwrite": bool(args.overwrite),
        "dry_run": bool(args.dry_run),
    }

    extracted = 0
    skipped = 0
    missing = 0
    errors: list[dict[str, object]] = []
    for vid in queue:
        dst = args.raw_videos_dir / f"{vid}.mp4"
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        if vid not in index:
            missing += 1
            continue
        zip_path, member = index[vid]
        try:
            if not args.dry_run:
                _extract_member(zip_path=zip_path, member=member, dst=dst)
            extracted += 1
        except Exception as exc:  # noqa: BLE001
            errors.append({"video_id": vid, "zip": str(zip_path), "member": str(member), "error": str(exc)})

    summary.update(
        {
            "extracted": int(extracted),
            "skipped_existing": int(skipped),
            "missing_from_zips_in_queue": int(missing),
            "errors": errors,
        }
    )

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.log_path)
    return 0 if (missing == 0 and not errors) else 2


if __name__ == "__main__":
    raise SystemExit(main())

