from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from avs.datasets.layout import videomme_paths
from avs.datasets.videomme import load_videomme_split
from avs.utils.ffmpeg import ensure_ffmpeg_in_path


@dataclass(frozen=True)
class DownloadResult:
    youtube_id: str
    ok: bool
    out_path: str
    url: str
    error: str | None = None

    def to_jsonable(self) -> dict:
        return {
            "youtube_id": str(self.youtube_id),
            "ok": bool(self.ok),
            "out_path": str(self.out_path),
            "url": str(self.url),
            "error": None if self.error is None else str(self.error),
        }


def _hhmmss(t: int) -> str:
    s = max(0, int(t))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _cleanup_partial_files(dst: Path) -> None:
    for p in (dst, Path(str(dst) + ".part"), Path(str(dst) + ".ytdl")):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass


def _download_ytdlp_video(
    youtube_id: str,
    *,
    url: str,
    out_dir: Path,
    ytdlp: str,
    overwrite: bool,
    timeout_seconds: float | None,
    keep_part: bool,
    max_seconds: int | None,
) -> DownloadResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{str(youtube_id)}.mp4"
    if dst.exists() and not overwrite:
        return DownloadResult(youtube_id=str(youtube_id), ok=True, out_path=str(dst), url=str(url))

    ensure_ffmpeg_in_path()

    cmd = [
        ytdlp,
        "--no-progress",
        "--quiet",
        "--no-warnings",
        "-f",
        "mp4/bestvideo+bestaudio/best",
    ]
    if max_seconds is not None:
        end = _hhmmss(int(max_seconds))
        cmd += ["--download-sections", f"*00:00:00-{end}"]
    cmd += ["-o", str(dst), str(url)]

    try:
        subprocess.run(  # noqa: S603,S607 - controlled args
            cmd,
            check=True,
            timeout=timeout_seconds,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if not dst.exists() or dst.stat().st_size <= 1024:
            raise RuntimeError(f"downloaded file missing or too small: {dst}")
        return DownloadResult(youtube_id=str(youtube_id), ok=True, out_path=str(dst), url=str(url))
    except Exception as e:  # noqa: BLE001
        if not keep_part:
            _cleanup_partial_files(dst)
        return DownloadResult(youtube_id=str(youtube_id), ok=False, out_path=str(dst), url=str(url), error=repr(e))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download (a subset of) Video-MME YouTube videos into data/VideoMME/raw/videos/")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--limit", type=int, default=256, help="Number of QA items (questions) to sample; unique videos are derived from them.")
    p.add_argument("--seed", type=int, default=0, help="Seed used for deterministic item ordering (hash order).")
    p.add_argument("--order", type=str, default="hash", choices=["hash", "original"])
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--ytdlp", type=str, default="yt-dlp")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--timeout-seconds", type=float, default=600.0)
    p.add_argument("--keep-part", action="store_true", help="Keep partial .part files on failure for debugging/resume.")
    p.add_argument("--max-seconds", type=int, default=180, help="Download only the first N seconds (controlled transfer).")
    p.add_argument("--min-videos", type=int, default=64, help="Fail if fewer than this many videos are downloaded successfully.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    p = videomme_paths()
    out_videos_dir = p.raw_videos_dir

    items = load_videomme_split(split=str(args.split), limit=int(args.limit), seed=int(args.seed), order=str(args.order))
    by_vid: dict[str, str] = {}
    for it in items:
        vid = str(it.youtube_id).strip()
        if not vid:
            continue
        url = str(it.url).strip() or f"https://www.youtube.com/watch?v={vid}"
        by_vid[vid] = url

    youtube_ids = sorted(by_vid.keys())
    run_dir = Path("runs") / f"videomme_download_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results: list[DownloadResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.jobs)) as ex:
        futs = []
        for vid in youtube_ids:
            futs.append(
                ex.submit(
                    _download_ytdlp_video,
                    vid,
                    url=by_vid[vid],
                    out_dir=out_videos_dir,
                    ytdlp=str(args.ytdlp),
                    overwrite=bool(args.overwrite),
                    timeout_seconds=float(args.timeout_seconds) if args.timeout_seconds else None,
                    keep_part=bool(args.keep_part),
                    max_seconds=int(args.max_seconds) if args.max_seconds else None,
                )
            )
        for f in concurrent.futures.as_completed(futs):
            results.append(f.result())

    dt = float(time.time() - t0)
    ok = [r for r in results if r.ok]
    fail = [r for r in results if not r.ok]
    payload = {
        "ok": len(ok) >= int(args.min_videos),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split": str(args.split),
        "limit_items": int(args.limit),
        "unique_videos": int(len(youtube_ids)),
        "downloaded_ok": int(len(ok)),
        "downloaded_fail": int(len(fail)),
        "max_seconds": int(args.max_seconds) if args.max_seconds else None,
        "elapsed_s": float(dt),
        "raw_videos_dir": str(out_videos_dir),
        "examples_fail": [r.to_jsonable() for r in fail[:20]],
    }
    out = run_dir / "download_report.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

