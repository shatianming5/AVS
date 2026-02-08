#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
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


def _ffmpeg_bin() -> str:
    """
    Return an ffmpeg binary path.

    Prefer the repo-local static build (installed via scripts/setup/install_ffmpeg_static.py)
    to avoid relying on system packages / sudo.
    """
    try:
        from avs.utils.ffmpeg import ffmpeg_bin  # noqa: PLC0415

        return str(ffmpeg_bin())
    except Exception:  # noqa: BLE001
        return "ffmpeg"


def _ffprobe_bin() -> str:
    try:
        from avs.utils.ffmpeg import ffprobe_bin  # noqa: PLC0415

        return str(ffprobe_bin())
    except Exception:  # noqa: BLE001
        return "ffprobe"


def _parse_timestamp_s(ts: str) -> float | None:
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


def _required_stop_by_video(meta_dir: Path, *, include_test: bool) -> dict[str, float]:
    """
    Compute per-video required minimum duration from EPIC-SOUNDS timestamps: max(stop_timestamp).
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
                stop = _parse_timestamp_s(str(row.get("stop_timestamp", "")).strip())
                if not vid or stop is None:
                    continue
                req[vid] = max(float(req.get(vid, 0.0)), float(stop))
    return req


def _probe_duration_s(path: Path) -> float | None:
    r = subprocess.run(
        [
            _ffprobe_bin(),
            "-hide_banner",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return None
    try:
        return float(r.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def _index_video_tree(root: Path) -> dict[str, Path]:
    """
    Build a mapping from video_id -> best candidate path by scanning a local EPIC video tree.

    If multiple files share the same stem, keep the largest file to avoid picking tiny derived clips.
    """
    out: dict[str, Path] = {}
    if not root.exists():
        return out

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() != ".mp4":
            continue
        vid = p.stem
        try:
            sz = int(p.stat().st_size)
        except OSError:
            continue
        prev = out.get(vid)
        if prev is None:
            out[vid] = p
            continue
        try:
            prev_sz = int(prev.stat().st_size)
        except OSError:
            prev_sz = -1
        if sz > prev_sz:
            out[vid] = p
    return out


def _materialize_from_local(
    *,
    src: Path,
    dst: Path,
    mode: str,
    max_seconds: int | None,
    reencode: bool,
    crf: int,
    height: int | None,
) -> None:
    """
    Materialize `dst` from a local `src` video.

    - If no trimming/re-encoding is requested: symlink/hardlink/copy.
    - Otherwise: use ffmpeg to (optionally) trim and/or re-encode into `dst` (src is preserved).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if (not reencode) and (max_seconds is None):
        if mode == "symlink":
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)
            return
        if mode == "hardlink":
            if dst.exists():
                dst.unlink()
            os.link(src, dst)
            return
        if mode == "copy":
            shutil.copy2(src, dst)
            return
        if mode != "transcode":
            raise ValueError(f"unknown --src-mode: {mode!r}")

    cmd = [_ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src)]
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

    cmd = [_ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src)]
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
            "with resume support and optional size control. Optionally import from a local EPIC-KITCHENS video tree first."
        )
    )
    p.add_argument("--meta-dir", type=Path, required=True)
    p.add_argument("--raw-videos-dir", type=Path, required=True)
    p.add_argument(
        "--src-root",
        type=Path,
        default=None,
        help=(
            "Optional local EPIC-KITCHENS video root. If provided, the script will try to materialize "
            "<video_id>.mp4 from this tree before invoking epic_downloader.py."
        ),
    )
    p.add_argument("--src-mode", type=str, default="symlink", choices=["symlink", "hardlink", "copy", "transcode"])
    p.add_argument("--min-size-mb", type=float, default=5.0, help="Treat existing videos smaller than this as missing.")
    p.add_argument("--downloader-dir", type=Path, default=None, help="Folder containing epic_downloader.py (optional).")
    p.add_argument("--staging-dir", type=Path, default=None, help="Temporary output path passed to epic_downloader.py (optional).")
    p.add_argument("--include-test", action="store_true", help="Also require test split video_ids")
    p.add_argument(
        "--require-meta-duration",
        action="store_true",
        help="Treat existing videos as missing if their duration is shorter than the max stop timestamp in the metadata.",
    )
    p.add_argument("--duration-margin-s", type=float, default=0.1)
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
    args.src_root = args.src_root.resolve() if args.src_root is not None else None
    args.downloader_dir = args.downloader_dir.resolve() if args.downloader_dir is not None else None
    args.staging_dir = args.staging_dir.resolve() if args.staging_dir is not None else None
    args.log_path = args.log_path.resolve()

    args.raw_videos_dir.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)

    required = _read_required_video_ids(args.meta_dir, include_test=bool(args.include_test))
    min_size_bytes = int(max(0.0, float(args.min_size_mb)) * 1024.0 * 1024.0)
    req_stop = _required_stop_by_video(args.meta_dir, include_test=bool(args.include_test)) if args.require_meta_duration else {}
    margin_s = float(args.duration_margin_s)

    have: set[str] = set()
    too_short_existing = 0
    for p in args.raw_videos_dir.glob("*.mp4"):
        try:
            if int(p.stat().st_size) < min_size_bytes:
                continue
        except OSError:
            continue
        vid = p.stem
        if args.require_meta_duration:
            need = req_stop.get(vid)
            if need is not None:
                dur = _probe_duration_s(p)
                if dur is None or (float(dur) + float(margin_s)) < float(need):
                    too_short_existing += 1
                    continue
        have.add(vid)
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
        "src_root": None if args.src_root is None else str(args.src_root),
        "src_mode": str(args.src_mode),
        "min_size_mb": float(args.min_size_mb),
        "require_meta_duration": bool(args.require_meta_duration),
        "duration_margin_s": float(args.duration_margin_s),
        "too_short_existing": int(too_short_existing),
        "downloader_dir": None if args.downloader_dir is None else str(args.downloader_dir),
        "staging_dir": None if args.staging_dir is None else str(args.staging_dir),
        "max_seconds": None if args.max_seconds is None else int(args.max_seconds),
        "reencode": bool(args.reencode),
        "crf": int(args.crf),
        "height": int(args.height),
    }
    print(json.dumps(summary, ensure_ascii=True))
    if not queue:
        return 0

    src_index: dict[str, Path] = {}
    if args.src_root is not None and args.src_root.exists():
        src_index = _index_video_tree(args.src_root)

    if (not args.dry_run) and (not src_index) and (args.downloader_dir is None or args.staging_dir is None):
        raise SystemExit("No usable source provided. Pass --src-root (local EPIC videos) or --downloader-dir + --staging-dir.")

    processed = 0
    failed = 0
    for idx, vid in enumerate(queue, start=1):
        out_path = args.raw_videos_dir / f"{vid}.mp4"
        if out_path.exists():
            try:
                sz_ok = int(out_path.stat().st_size) >= min_size_bytes
            except OSError:
                sz_ok = False
            if sz_ok:
                if not args.require_meta_duration:
                    continue
                need = req_stop.get(vid)
                if need is None:
                    continue
                dur = _probe_duration_s(out_path)
                if dur is not None and (float(dur) + float(margin_s)) >= float(need):
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
                # 1) Import from a local EPIC video tree (preferred when available).
                if vid in src_index:
                    record["method"] = "import"
                    record["src_path"] = str(src_index[vid])
                    _materialize_from_local(
                        src=src_index[vid],
                        dst=out_path,
                        mode=str(args.src_mode),
                        max_seconds=args.max_seconds,
                        reencode=bool(args.reencode),
                        crf=int(args.crf),
                        height=(None if int(args.height) <= 0 else int(args.height)),
                    )
                    success = out_path.exists() and int(out_path.stat().st_size) >= min_size_bytes
                    if not success:
                        raise RuntimeError(f"import produced an invalid file for {vid}: {out_path}")
                    if args.require_meta_duration:
                        need = req_stop.get(vid)
                        if need is not None:
                            dur = _probe_duration_s(out_path)
                            if dur is None or (float(dur) + float(margin_s)) < float(need):
                                raise RuntimeError(f"imported video is too short for timestamps: video_id={vid} need_s={need} dur_s={dur}")
                else:
                    # 2) Fallback: drive epic_downloader.py (requires user credentials/access).
                    if args.downloader_dir is None or args.staging_dir is None:
                        raise RuntimeError(
                            "missing --downloader-dir/--staging-dir and --src-root did not contain this video_id"
                        )

                    record["method"] = "downloader"
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
                            success = out_path.exists() and int(out_path.stat().st_size) >= min_size_bytes
                            if success and args.require_meta_duration:
                                need = req_stop.get(vid)
                                if need is not None:
                                    dur = _probe_duration_s(out_path)
                                    if dur is None or (float(dur) + float(margin_s)) < float(need):
                                        raise RuntimeError(
                                            f"downloaded video is too short for timestamps: video_id={vid} need_s={need} dur_s={dur}"
                                        )
                            _clean_staging(args.staging_dir)
                            if success:
                                break
                        except Exception as exc:  # noqa: BLE001
                            last_err = str(exc)
                            try:
                                _clean_staging(args.staging_dir)
                            except Exception:  # noqa: BLE001
                                pass
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

    remaining = 0
    for v in required:
        p = args.raw_videos_dir / f"{v}.mp4"
        if (not p.exists()) or int(p.stat().st_size) < min_size_bytes:
            remaining += 1

    final = {
        "processed_ok": int(processed),
        "failed": int(failed),
        "remaining_missing_after_run": int(remaining),
        "log_path": str(args.log_path),
    }
    print(json.dumps(final, ensure_ascii=True))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
