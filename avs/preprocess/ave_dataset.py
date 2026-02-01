from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path

from avs.preprocess.ave_extract import preprocess_one


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


def _is_preprocessed(clip_root: Path, *, num_frames: int = 10) -> bool:
    audio_path = clip_root / "audio.wav"
    frames_dir = clip_root / "frames"
    if not audio_path.exists():
        return False
    for t in range(int(num_frames)):
        if not (frames_dir / f"{int(t)}.jpg").exists():
            return False
    return True


def preprocess_ave_videos(
    *,
    raw_videos_dir: Path,
    out_dir: Path,
    video_ids: list[str],
    skip_existing: bool = False,
    allow_missing: bool = False,
    jobs: int = 1,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    max_workers = max(1, int(jobs))

    def _run_one(vid: str) -> str | None:
        clip_root = out_dir / vid
        if bool(skip_existing) and _is_preprocessed(clip_root, num_frames=10):
            return vid
        video_path = raw_videos_dir / f"{vid}.mp4"
        if not video_path.exists():
            if allow_missing:
                return None
            raise FileNotFoundError(f"missing video: {video_path}")
        try:
            preprocess_one(video_path, out_dir, clip_id=vid)
            return vid
        except Exception:
            if allow_missing:
                return None
            raise

    if max_workers == 1 or len(video_ids) <= 1:
        return [v for vid in video_ids if (v := _run_one(vid)) is not None]

    done: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_one, vid): vid for vid in video_ids}
        for fut in concurrent.futures.as_completed(futures):
            v = fut.result()
            if v is not None:
                done.append(v)
    return done


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preprocess AVE raw videos into per-clip audio+frames.")
    p.add_argument("--raw-videos-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--video-id", action="append", default=[], help="Video id (repeatable)")
    p.add_argument("--ids-file", type=Path, default=None, help="Optional file with one video id per line.")
    p.add_argument("--limit", type=int, default=None, help="Optional limit for --ids-file.")
    p.add_argument("--skip-existing", action="store_true", help="Skip preprocessing for clips that already exist in out-dir.")
    p.add_argument("--allow-missing", action="store_true", help="Skip missing raw videos instead of failing.")
    p.add_argument("--jobs", type=int, default=1, help="Number of parallel preprocessing workers.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video_ids: list[str] = list(args.video_id)
    if args.ids_file is not None:
        video_ids.extend(_read_ids_file(args.ids_file, args.limit))
    video_ids = _stable_unique([v for v in video_ids if v])
    if not video_ids:
        raise SystemExit("at least one --video-id or --ids-file is required")
    preprocess_ave_videos(
        raw_videos_dir=args.raw_videos_dir,
        out_dir=args.out_dir,
        video_ids=video_ids,
        skip_existing=bool(args.skip_existing),
        allow_missing=bool(args.allow_missing),
        jobs=int(args.jobs),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
