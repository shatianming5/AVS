#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
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


def _current_have(raw_videos_dir: Path) -> set[str]:
    if not raw_videos_dir.exists():
        return set()
    return {p.stem for p in raw_videos_dir.glob("*.mp4")}


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd is not None else None)  # noqa: S603


def _find_staged_video(staging_dir: Path, video_id: str) -> Path:
    candidates = list(staging_dir.glob(f"EPIC-KITCHENS/**/{video_id}.MP4"))
    if not candidates:
        candidates = list(staging_dir.glob(f"EPIC-KITCHENS/**/{video_id}.mp4"))
    if not candidates:
        raise FileNotFoundError(f"download finished but staged file not found for {video_id}")
    return candidates[0]


def _clean_staging(staging_dir: Path) -> None:
    if not staging_dir.exists():
        return
    # Keep root folder, clear generated payload only.
    payload = staging_dir / "EPIC-KITCHENS"
    if payload.exists():
        shutil.rmtree(payload)


def _transcode(
    *,
    src: Path,
    dst: Path,
    max_seconds: int | None,
    reencode: bool,
    crf: int,
    height: int | None,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not reencode and max_seconds is None:
        shutil.move(str(src), str(dst))
        return

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src)]
    if max_seconds is not None:
        cmd += ["-t", str(int(max_seconds))]

    if reencode:
        if height is not None and int(height) > 0:
            cmd += ["-vf", f"scale=-2:{int(height)}"]
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(int(crf)),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-movflags",
            "+faststart",
            str(dst),
        ]
    else:
        cmd += ["-c", "copy", str(dst)]
    _run(cmd)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Fill missing EPIC-SOUNDS videos by driving official epic_downloader.py one video at a time, "
            "with resume support and optional size control."
        )
    )
    p.add_argument("--meta-dir", type=Path, required=True)
    p.add_argument("--raw-videos-dir", type=Path, required=True)
    p.add_argument("--downloader-dir", type=Path, required=True, help="Folder containing epic_downloader.py")
    p.add_argument("--staging-dir", type=Path, required=True, help="Temporary output path passed to epic_downloader.py")
    p.add_argument("--include-test", action="store_true", help="Also require test split video_ids")
    p.add_argument("--max-seconds", type=int, default=None, help="Trim each downloaded video to this duration")
    p.add_argument("--reencode", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--crf", type=int, default=28)
    p.add_argument("--height", type=int, default=360, help="Target height when re-encoding; <=0 disables scaling")
    p.add_argument("--start-index", type=int, default=0, help="Start index inside current missing list")
    p.add_argument("--limit", type=int, default=None, help="Max number of videos to process this run")
    p.add_argument("--num-workers", type=int, default=1, help="Split queue across workers by modulo")
    p.add_argument("--worker-index", type=int, default=0, help="Worker slot in [0, num_workers)")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--sleep-seconds", type=float, default=1.0)
    p.add_argument("--log-path", type=Path, default=Path("runs") / f"epic_fill_missing_{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.meta_dir = args.meta_dir.resolve()
    args.raw_videos_dir = args.raw_videos_dir.resolve()
    args.downloader_dir = args.downloader_dir.resolve()
    args.staging_dir = args.staging_dir.resolve()
    args.log_path = args.log_path.resolve()

    args.raw_videos_dir.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)

    required = _read_required_video_ids(args.meta_dir, include_test=bool(args.include_test))
    have = _current_have(args.raw_videos_dir)
    missing = [vid for vid in required if vid not in have]

    start = max(0, int(args.start_index))
    queue = missing[start:]
    num_workers = max(1, int(args.num_workers))
    worker_index = int(args.worker_index)
    if worker_index < 0 or worker_index >= num_workers:
        raise SystemExit(f"--worker-index must be in [0, {num_workers})")
    if num_workers > 1:
        queue = [vid for i, vid in enumerate(queue) if i % num_workers == worker_index]
    if args.limit is not None:
        queue = queue[: max(0, int(args.limit))]

    summary = {
        "required_count": len(required),
        "have_count": len(have),
        "missing_count": len(missing),
        "queued_count": len(queue),
        "start_index": int(start),
        "limit": None if args.limit is None else int(args.limit),
        "num_workers": int(num_workers),
        "worker_index": int(worker_index),
        "max_seconds": None if args.max_seconds is None else int(args.max_seconds),
        "reencode": bool(args.reencode),
        "crf": int(args.crf),
        "height": int(args.height),
    }
    print(json.dumps(summary, ensure_ascii=True))
    if not queue:
        return 0

    processed = 0
    failed = 0
    for idx, vid in enumerate(queue, start=1):
        out_path = args.raw_videos_dir / f"{vid}.mp4"
        if out_path.exists():
            continue

        record: dict[str, object] = {
            "video_id": vid,
            "queue_index": idx,
            "ts_start": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ok": False,
        }
        t0 = time.time()

        try:
            if not args.dry_run:
                success = False
                last_err: str | None = None
                for attempt in range(1, int(args.retries) + 2):
                    try:
                        _run(
                            [
                                "python",
                                "epic_downloader.py",
                                "--videos",
                                "--specific-videos",
                                vid,
                                "--output-path",
                                str(args.staging_dir),
                            ],
                            cwd=args.downloader_dir,
                        )
                        staged = _find_staged_video(args.staging_dir, vid)
                        _transcode(
                            src=staged,
                            dst=out_path,
                            max_seconds=args.max_seconds,
                            reencode=bool(args.reencode),
                            crf=int(args.crf),
                            height=(None if int(args.height) <= 0 else int(args.height)),
                        )
                        success = out_path.exists() and out_path.stat().st_size > 0
                        _clean_staging(args.staging_dir)
                        if success:
                            break
                    except Exception as exc:  # noqa: BLE001
                        last_err = str(exc)
                        _clean_staging(args.staging_dir)
                        time.sleep(float(args.sleep_seconds))
                if not success:
                    raise RuntimeError(last_err or f"download failed for {vid}")

            record["ok"] = True
            record["size_bytes"] = int(out_path.stat().st_size) if out_path.exists() else 0
            processed += 1
        except Exception as exc:  # noqa: BLE001
            record["error"] = str(exc)
            failed += 1
        finally:
            record["elapsed_sec"] = round(float(time.time() - t0), 3)
            with args.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
            print(json.dumps(record, ensure_ascii=True))

    final = {
        "processed_ok": int(processed),
        "failed": int(failed),
        "remaining_missing_after_run": len([v for v in required if not (args.raw_videos_dir / f"{v}.mp4").exists()]),
        "log_path": str(args.log_path),
    }
    print(json.dumps(final, ensure_ascii=True))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
