from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from avs.datasets.avqa import AVQAIndex, ensure_avqa_meta
from avs.datasets.layout import avqa_paths
from avs.utils.ffmpeg import ensure_ffmpeg_in_path


@dataclass(frozen=True)
class DownloadResult:
    video_name: str
    ok: bool
    mode: str
    out_path: str
    error: str | None = None


def parse_avqa_video_name(video_name: str) -> tuple[str, int]:
    """
    AVQA video_name encodes VGGSound clip ids like:
      <youtube_id>_<start_sec:06d>

    Note: youtube_id itself may contain underscores, so we parse by taking
    the last "_" segment as start_sec and joining the rest.
    """
    parts = str(video_name).strip().split("_")
    if len(parts) < 2:
        raise ValueError(f"invalid video_name (expected <youtube_id>_<start_sec>): {video_name!r}")
    start_s = parts[-1].strip()
    if not start_s.isdigit():
        raise ValueError(f"invalid start_sec suffix in video_name={video_name!r} (suffix={start_s!r})")
    youtube_id = "_".join(parts[:-1]).strip()
    if not youtube_id:
        raise ValueError(f"empty youtube_id parsed from video_name={video_name!r}")
    return youtube_id, int(start_s)


def _hhmmss(t: int) -> str:
    s = max(0, int(t))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _copy_local(video_name: str, *, src_dir: Path, out_dir: Path, overwrite: bool) -> DownloadResult:
    src = src_dir / f"{video_name}.mp4"
    dst = out_dir / f"{video_name}.mp4"
    out_dir.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return DownloadResult(video_name=video_name, ok=True, mode="local", out_path=str(dst))
    if not src.exists():
        return DownloadResult(video_name=video_name, ok=False, mode="local", out_path=str(dst), error=f"missing source: {src}")
    shutil.copyfile(src, dst)
    return DownloadResult(video_name=video_name, ok=True, mode="local", out_path=str(dst))


def _cleanup_partial_files(dst: Path) -> None:
    for p in (dst, Path(str(dst) + ".part"), Path(str(dst) + ".ytdl")):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass


def _download_ytdlp_clip(
    video_name: str,
    *,
    out_dir: Path,
    ytdlp: str,
    overwrite: bool,
    timeout_seconds: float | None,
    keep_part: bool,
    clip_duration_seconds: float,
) -> DownloadResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{video_name}.mp4"
    if dst.exists() and not overwrite:
        return DownloadResult(video_name=video_name, ok=True, mode="yt-dlp", out_path=str(dst))

    try:
        youtube_id, start_sec = parse_avqa_video_name(video_name)
    except Exception as e:  # noqa: BLE001
        return DownloadResult(video_name=video_name, ok=False, mode="yt-dlp", out_path=str(dst), error=f"parse failed: {e!r}")

    # yt-dlp needs ffmpeg for section cutting and/or A+V merge.
    ensure_ffmpeg_in_path()

    start = int(start_sec)
    end = int(start_sec) + max(1, int(round(float(clip_duration_seconds))))
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    cmd = [
        ytdlp,
        "--no-progress",
        "--quiet",
        "--no-warnings",
        "-f",
        "mp4/bestvideo+bestaudio/best",
        "--download-sections",
        f"*{_hhmmss(start)}-{_hhmmss(end)}",
        "-o",
        str(dst),
        url,
    ]
    try:
        subprocess.run(  # noqa: S603,S607 - controlled args
            cmd,
            check=True,
            timeout=timeout_seconds,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        ok = dst.exists() and dst.stat().st_size > 0
        if not ok and not keep_part:
            _cleanup_partial_files(dst)
        return DownloadResult(video_name=video_name, ok=ok, mode="yt-dlp", out_path=str(dst), error=None if ok else "download produced empty file")
    except FileNotFoundError:
        if not keep_part:
            _cleanup_partial_files(dst)
        return DownloadResult(video_name=video_name, ok=False, mode="yt-dlp", out_path=str(dst), error=f"missing downloader binary: {ytdlp}")
    except subprocess.TimeoutExpired:
        if not keep_part:
            _cleanup_partial_files(dst)
        return DownloadResult(video_name=video_name, ok=False, mode="yt-dlp", out_path=str(dst), error="yt-dlp timed out")
    except subprocess.CalledProcessError as e:
        if not keep_part:
            _cleanup_partial_files(dst)
        tail = ""
        try:
            lines = (e.stderr or "").splitlines()
            tail = lines[-1].strip() if lines else ""
        except Exception:
            tail = ""
        msg = f"yt-dlp failed: {e}"
        if tail:
            msg += f" (stderr_tail={tail!r})"
        return DownloadResult(video_name=video_name, ok=False, mode="yt-dlp", out_path=str(dst), error=msg)


def _stable_unique(xs: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in xs:
        s = str(x).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _select_video_names(meta_dir: Path, split: str, limit: int | None) -> list[str]:
    ensure_avqa_meta(meta_dir)
    idx = AVQAIndex.from_meta_dir(meta_dir)
    items = idx.train if str(split) == "train" else idx.val
    names = _stable_unique([it.video_name for it in items])
    if limit is not None:
        names = names[: int(limit)]
    return names


def materialize_avqa_videos(
    *,
    video_names: list[str],
    out_dir: Path,
    mode: str,
    src_dir: Path | None = None,
    ytdlp: str = "yt-dlp",
    overwrite: bool = False,
    timeout_seconds: float | None = 600.0,
    jobs: int = 1,
    keep_part: bool = False,
    clip_duration_seconds: float = 10.0,
) -> list[DownloadResult]:
    if mode not in {"local", "yt-dlp"}:
        raise ValueError(f"unknown mode: {mode}")
    if mode == "local" and src_dir is None:
        raise ValueError("src_dir is required for local mode")

    timeout = None if (timeout_seconds is None or float(timeout_seconds) <= 0.0) else float(timeout_seconds)
    max_workers = max(1, int(jobs))
    names = _stable_unique([n for n in video_names if n])
    if not names:
        return []

    def _run_one(name: str) -> DownloadResult:
        if mode == "local":
            assert src_dir is not None
            return _copy_local(name, src_dir=src_dir, out_dir=out_dir, overwrite=overwrite)
        return _download_ytdlp_clip(
            name,
            out_dir=out_dir,
            ytdlp=ytdlp,
            overwrite=overwrite,
            timeout_seconds=timeout,
            keep_part=keep_part,
            clip_duration_seconds=float(clip_duration_seconds),
        )

    t0 = time.time()
    results: list[DownloadResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_run_one, n) for n in names]
        for fut in concurrent.futures.as_completed(futs):
            results.append(fut.result())
    dt = float(time.time() - t0)

    # Stable order for logs.
    results = sorted(results, key=lambda r: str(r.video_name))
    ok = int(sum(1 for r in results if r.ok))
    fail = int(len(results) - ok)
    print(f"[avqa_download] done: ok={ok} fail={fail} total={len(results)} dt_s={dt:.1f}", flush=True)
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Materialize AVQA VGGSound clips into data/AVQA/raw/videos/")
    p.add_argument("--mode", type=str, default="yt-dlp", choices=["local", "yt-dlp"])
    p.add_argument("--video-name", action="append", default=[], help="AVQA video_name (repeatable)")
    p.add_argument("--names-file", type=Path, default=None, help="Optional file with one video_name per line.")
    p.add_argument("--split", type=str, choices=["train", "val"], help="Select video_names from AVQA split (downloads meta)")
    p.add_argument("--limit", type=int, default=None)

    p.add_argument("--meta-dir", type=Path, default=avqa_paths().meta_dir)
    p.add_argument("--out-dir", type=Path, default=avqa_paths().raw_videos_dir)
    p.add_argument("--src-dir", type=Path, default=None, help="Local dir containing <video_name>.mp4 (for mode=local)")
    p.add_argument("--ytdlp", type=str, default="yt-dlp", help="yt-dlp binary (for mode=yt-dlp)")

    p.add_argument("--clip-duration-seconds", type=float, default=10.0, help="Clip duration to download (seconds).")
    p.add_argument("--jobs", type=int, default=4, help="Number of parallel download workers.")
    p.add_argument("--timeout-seconds", type=float, default=600.0, help="Per-video timeout for yt-dlp downloads (0 disables).")
    p.add_argument("--keep-part", action="store_true", help="Keep yt-dlp partial artifacts (.part/.ytdl) on failures.")
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--write-meta-lists", action="store_true", help="Write download_ok/download_fail lists under meta-dir.")
    p.add_argument("--lists-tag", type=str, default=None, help="Tag for ok/fail list filenames (e.g., train/val).")
    return p


def _read_names_file(path: Path, limit: int | None) -> list[str]:
    names = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if limit is not None:
        names = names[: int(limit)]
    return names


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    names: list[str] = list(args.video_name)
    if args.names_file is not None:
        names.extend(_read_names_file(args.names_file, args.limit))
    if args.split:
        names.extend(_select_video_names(Path(args.meta_dir), str(args.split), args.limit))
    names = _stable_unique([n for n in names if n])
    if not names:
        raise SystemExit("no video_names provided; use --video-name, --names-file, or --split")

    results = materialize_avqa_videos(
        video_names=names,
        out_dir=Path(args.out_dir),
        mode=str(args.mode),
        src_dir=Path(args.src_dir) if args.src_dir is not None else None,
        ytdlp=str(args.ytdlp),
        overwrite=bool(args.overwrite),
        timeout_seconds=float(args.timeout_seconds),
        jobs=int(args.jobs),
        keep_part=bool(args.keep_part),
        clip_duration_seconds=float(args.clip_duration_seconds),
    )

    payload = {
        "ts": time.strftime("%Y%m%d-%H%M%S"),
        "mode": str(args.mode),
        "out_dir": str(args.out_dir),
        "meta_dir": str(args.meta_dir),
        "clip_duration_seconds": float(args.clip_duration_seconds),
        "results": [r.__dict__ for r in results],
        "ok": all(r.ok for r in results),
        "num_ok": int(sum(1 for r in results if r.ok)),
        "num_fail": int(sum(1 for r in results if not r.ok)),
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if bool(args.write_meta_lists):
        tag = str(args.lists_tag or "").strip()
        if not tag:
            raise SystemExit("--write-meta-lists requires --lists-tag (e.g., train/val)")
        ok_names = [r.video_name for r in results if r.ok]
        fail_names = [r.video_name for r in results if not r.ok]
        meta_dir = Path(args.meta_dir)
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / f"download_ok_{tag}_auto.txt").write_text("\n".join(ok_names) + ("\n" if ok_names else ""), encoding="utf-8")
        (meta_dir / f"download_fail_{tag}_auto.txt").write_text("\n".join(fail_names) + ("\n" if fail_names else ""), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
