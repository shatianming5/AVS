from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from avs.datasets.ave import AVEIndex, ensure_ave_meta
from avs.datasets.layout import ave_paths


@dataclass(frozen=True)
class DownloadResult:
    video_id: str
    ok: bool
    mode: str
    out_path: str
    error: str | None = None


def _copy_local(video_id: str, *, src_dir: Path, out_dir: Path, overwrite: bool) -> DownloadResult:
    src = src_dir / f"{video_id}.mp4"
    dst = out_dir / f"{video_id}.mp4"
    out_dir.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return DownloadResult(video_id=video_id, ok=True, mode="local", out_path=str(dst))
    if not src.exists():
        return DownloadResult(video_id=video_id, ok=False, mode="local", out_path=str(dst), error=f"missing source: {src}")
    shutil.copyfile(src, dst)
    return DownloadResult(video_id=video_id, ok=True, mode="local", out_path=str(dst))


def _cleanup_partial_files(dst: Path) -> None:
    for p in (dst, Path(str(dst) + ".part"), Path(str(dst) + ".ytdl")):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            # Best-effort cleanup: ignore permission races or transient FS errors.
            pass


def _download_ytdlp(
    video_id: str,
    *,
    out_dir: Path,
    ytdlp: str,
    overwrite: bool,
    timeout_seconds: float | None,
    keep_part: bool,
) -> DownloadResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{video_id}.mp4"
    if dst.exists() and not overwrite:
        return DownloadResult(video_id=video_id, ok=True, mode="yt-dlp", out_path=str(dst))

    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        ytdlp,
        "--no-progress",
        "--quiet",
        "--no-warnings",
        "-f",
        "mp4/bestvideo+bestaudio/best",
        "-o",
        str(dst),
        url,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=timeout_seconds)  # noqa: S603,S607 - controlled args
        ok = dst.exists() and dst.stat().st_size > 0
        if not ok and not keep_part:
            _cleanup_partial_files(dst)
        return DownloadResult(video_id=video_id, ok=ok, mode="yt-dlp", out_path=str(dst), error=None if ok else "download produced empty file")
    except FileNotFoundError:
        if not keep_part:
            _cleanup_partial_files(dst)
        return DownloadResult(video_id=video_id, ok=False, mode="yt-dlp", out_path=str(dst), error=f"missing downloader binary: {ytdlp}")
    except subprocess.TimeoutExpired:
        if not keep_part:
            _cleanup_partial_files(dst)
        return DownloadResult(video_id=video_id, ok=False, mode="yt-dlp", out_path=str(dst), error="yt-dlp timed out")
    except subprocess.CalledProcessError as e:
        if not keep_part:
            _cleanup_partial_files(dst)
        return DownloadResult(video_id=video_id, ok=False, mode="yt-dlp", out_path=str(dst), error=f"yt-dlp failed: {e}")


def materialize_ave_videos(
    *,
    video_ids: list[str],
    out_dir: Path,
    mode: str,
    src_dir: Path | None = None,
    ytdlp: str = "yt-dlp",
    overwrite: bool = False,
    timeout_seconds: float | None = 600.0,
    jobs: int = 1,
    keep_part: bool = False,
) -> list[DownloadResult]:
    if mode not in {"local", "yt-dlp"}:
        raise ValueError(f"unknown mode: {mode}")

    if mode == "local" and src_dir is None:
        raise ValueError("src_dir is required for local mode")

    timeout = None if (timeout_seconds is None or float(timeout_seconds) <= 0.0) else float(timeout_seconds)
    max_workers = max(1, int(jobs))

    def _run_one(vid: str) -> DownloadResult:
        if mode == "local":
            assert src_dir is not None
            return _copy_local(vid, src_dir=src_dir, out_dir=out_dir, overwrite=overwrite)
        return _download_ytdlp(vid, out_dir=out_dir, ytdlp=ytdlp, overwrite=overwrite, timeout_seconds=timeout, keep_part=keep_part)

    if max_workers == 1 or len(video_ids) <= 1:
        return [_run_one(v) for v in video_ids]

    results_by_id: dict[str, DownloadResult] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_one, vid): vid for vid in video_ids}
        for fut in concurrent.futures.as_completed(futures):
            vid = futures[fut]
            try:
                results_by_id[vid] = fut.result()
            except Exception as e:  # noqa: BLE001 - best-effort batch runner
                dst = out_dir / f"{vid}.mp4"
                results_by_id[vid] = DownloadResult(video_id=vid, ok=False, mode=str(mode), out_path=str(dst), error=repr(e))

    return [results_by_id[vid] for vid in video_ids]


def _select_video_ids(meta_dir: Path, split: str, limit: int | None) -> list[str]:
    ensure_ave_meta(meta_dir)
    index = AVEIndex.from_meta_dir(meta_dir)
    ids = index.splits[split]
    if limit is not None:
        ids = ids[:limit]
    return [index.clips[int(i)].video_id for i in ids]

def _read_ids_file(path: Path, limit: int | None) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = str(line).strip()
        if not s:
            continue
        ids.append(s)
        if limit is not None and len(ids) >= int(limit):
            break
    return ids


def _stable_unique(ids: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in ids:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Materialize AVE raw videos into data/AVE/raw/videos/")
    p.add_argument("--mode", type=str, default="local", choices=["local", "yt-dlp"])
    p.add_argument("--video-id", action="append", default=[], help="YouTube video id (repeatable)")
    p.add_argument("--ids-file", type=Path, default=None, help="Optional file with one YouTube video id per line.")
    p.add_argument("--split", type=str, choices=["train", "val", "test"], help="Select ids from AVE split (requires meta)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--meta-dir", type=Path, default=ave_paths().meta_dir)
    p.add_argument("--out-dir", type=Path, default=ave_paths().raw_videos_dir)
    p.add_argument("--src-dir", type=Path, default=None, help="Local dir containing <video_id>.mp4 (for mode=local)")
    p.add_argument("--ytdlp", type=str, default="yt-dlp", help="yt-dlp binary (for mode=yt-dlp)")
    p.add_argument("--jobs", type=int, default=4, help="Number of parallel download workers.")
    p.add_argument(
        "--timeout-seconds",
        type=float,
        default=600.0,
        help="Per-video timeout for yt-dlp downloads (0 disables).",
    )
    p.add_argument("--keep-part", action="store_true", help="Keep yt-dlp partial artifacts (.part/.ytdl) on failures.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--write-meta-lists", action="store_true", help="Write download_ok/download_fail lists under meta-dir.")
    p.add_argument("--lists-tag", type=str, default=None, help="Tag for ok/fail list filenames (e.g., train/val/test).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video_ids: list[str] = list(args.video_id)
    if args.ids_file is not None:
        video_ids.extend(_read_ids_file(args.ids_file, args.limit))
    if args.split:
        video_ids.extend(_select_video_ids(args.meta_dir, args.split, args.limit))
    video_ids = _stable_unique([v for v in video_ids if v])
    if not video_ids:
        raise SystemExit("no video ids provided; use --video-id, --ids-file, or --split")

    results = materialize_ave_videos(
        video_ids=video_ids,
        out_dir=args.out_dir,
        mode=args.mode,
        src_dir=args.src_dir,
        ytdlp=args.ytdlp,
        overwrite=bool(args.overwrite),
        timeout_seconds=float(args.timeout_seconds),
        jobs=int(args.jobs),
        keep_part=bool(args.keep_part),
    )

    payload = {
        "ts": time.strftime("%Y%m%d-%H%M%S"),
        "mode": args.mode,
        "out_dir": str(args.out_dir),
        "results": [r.__dict__ for r in results],
        "ok": all(r.ok for r in results),
        "num_ok": int(sum(1 for r in results if r.ok)),
        "num_fail": int(sum(1 for r in results if not r.ok)),
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    if bool(args.write_meta_lists):
        tag = str(args.lists_tag or "").strip()
        if not tag:
            raise SystemExit("--write-meta-lists requires --lists-tag (e.g., train/val/test)")
        ok_ids = [r.video_id for r in results if r.ok]
        fail_ids = [r.video_id for r in results if not r.ok]
        ok_path = Path(args.meta_dir) / f"download_ok_{tag}_auto.txt"
        fail_path = Path(args.meta_dir) / f"download_fail_{tag}_auto.txt"
        ok_path.parent.mkdir(parents=True, exist_ok=True)
        ok_path.write_text("\n".join(ok_ids) + ("\n" if ok_ids else ""), encoding="utf-8")
        fail_path.write_text("\n".join(fail_ids) + ("\n" if fail_ids else ""), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
