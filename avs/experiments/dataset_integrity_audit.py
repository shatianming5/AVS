from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def _run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603 - controlled args
    return int(proc.returncode), str(proc.stdout), str(proc.stderr)


def _probe_video(path: Path) -> tuple[bool, dict, str | None]:
    code, out, err = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_name,width,height,avg_frame_rate,duration,nb_frames",
            "-of",
            "json",
            str(path),
        ]
    )
    if code != 0:
        return False, {}, (err.strip() or out.strip() or "ffprobe failed")

    try:
        payload = json.loads(out or "{}")
    except json.JSONDecodeError:
        return False, {}, "ffprobe returned non-json payload"

    streams = payload.get("streams") or []
    video_stream = None
    audio_stream = None
    for stream in streams:
        if not isinstance(stream, dict):
            continue
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
        if stream.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = stream

    if video_stream is None:
        return False, {"streams": streams}, "missing video stream"

    meta = {
        "video_codec": video_stream.get("codec_name"),
        "audio_codec": audio_stream.get("codec_name") if isinstance(audio_stream, dict) else None,
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "nb_frames": video_stream.get("nb_frames"),
        "duration": video_stream.get("duration"),
        "avg_frame_rate": video_stream.get("avg_frame_rate"),
    }
    return True, meta, None


def _decode_video(path: Path, *, max_error_lines: int = 20) -> tuple[bool, int, list[str]]:
    code, _out, err = _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ]
    )
    lines = [ln.strip() for ln in str(err).splitlines() if str(ln).strip()]
    error_lines = [ln for ln in lines if ("error" in ln.lower()) or ("invalid" in ln.lower())]
    decode_ok = (code == 0) and (len(error_lines) == 0)
    return decode_ok, len(error_lines), error_lines[: int(max_error_lines)]


def run_integrity_audit(
    *,
    videos_dir: Path,
    pattern: str,
    out_dir: Path,
    decode_check: str,
    decode_limit: int | None,
    limit: int | None,
) -> dict:
    videos_dir = Path(videos_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(videos_dir.glob(pattern))
    if limit is not None:
        files = files[: int(limit)]

    decode_budget = None if decode_limit is None else int(decode_limit)
    decode_mode = str(decode_check)

    rows: list[dict] = []
    probe_ok_count = 0
    decode_ok_count = 0
    decode_ran_count = 0
    decode_error_total = 0
    corrupted: list[str] = []

    for idx, path in enumerate(files):
        probe_ok, probe_meta, probe_error = _probe_video(path)
        if probe_ok:
            probe_ok_count += 1

        should_decode = False
        if decode_mode == "full":
            should_decode = True
        elif decode_mode == "sampled":
            should_decode = decode_budget is None or idx < decode_budget

        decode_ok = None
        decode_error_count = 0
        decode_errors: list[str] = []
        if should_decode:
            decode_ran_count += 1
            decode_ok, decode_error_count, decode_errors = _decode_video(path)
            decode_error_total += int(decode_error_count)
            if decode_ok:
                decode_ok_count += 1

        bad = (not probe_ok) or (decode_ok is False)
        if bad:
            corrupted.append(str(path))

        rows.append(
            {
                "path": str(path),
                "size_bytes": int(path.stat().st_size) if path.exists() else 0,
                "probe_ok": bool(probe_ok),
                "probe_meta": probe_meta,
                "probe_error": probe_error,
                "decode_checked": bool(should_decode),
                "decode_ok": decode_ok,
                "decode_error_count": int(decode_error_count),
                "decode_errors": decode_errors,
            }
        )

    summary = {
        "videos_dir": str(videos_dir),
        "pattern": str(pattern),
        "decode_check": decode_mode,
        "decode_limit": decode_budget,
        "total_files": int(len(files)),
        "probe_ok": int(probe_ok_count),
        "probe_failed": int(len(files) - probe_ok_count),
        "decode_checked": int(decode_ran_count),
        "decode_ok": int(decode_ok_count),
        "decode_failed": int(decode_ran_count - decode_ok_count),
        "decode_error_lines_total": int(decode_error_total),
        "corrupted_files": int(len(corrupted)),
        "corrupted_paths": corrupted,
    }
    payload = {"ok": True, "summary": summary, "files": rows}
    out_json = out_dir / "dataset_integrity_audit.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"out_json": str(out_json), "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit video decoding integrity (ffprobe + optional ffmpeg decode pass).")
    p.add_argument("--videos-dir", type=Path, required=True, help="Directory containing source videos.")
    p.add_argument("--pattern", type=str, default="*.mp4", help="Glob pattern under --videos-dir.")
    p.add_argument("--out-dir", type=Path, default=Path("runs") / f"E0501_dataset_integrity_audit_{time.strftime('%Y%m%d-%H%M%S')}")
    p.add_argument("--decode-check", type=str, default="sampled", choices=["none", "sampled", "full"])
    p.add_argument("--decode-limit", type=int, default=64, help="When decode-check=sampled, decode at most this many files.")
    p.add_argument("--limit", type=int, default=None, help="Optional hard limit on scanned files.")
    p.add_argument("--fail-on-corrupt", action="store_true", help="Return non-zero when corrupted files are detected.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    decode_limit = int(args.decode_limit) if args.decode_limit is not None else None
    if str(args.decode_check) != "sampled":
        decode_limit = None

    rep = run_integrity_audit(
        videos_dir=Path(args.videos_dir),
        pattern=str(args.pattern),
        out_dir=Path(args.out_dir),
        decode_check=str(args.decode_check),
        decode_limit=decode_limit,
        limit=int(args.limit) if args.limit is not None else None,
    )
    print(rep["out_json"])
    if bool(args.fail_on_corrupt) and int(rep["summary"].get("corrupted_files", 0)) > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
