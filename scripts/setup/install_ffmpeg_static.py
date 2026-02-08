#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import stat
import tarfile
import time
from pathlib import Path


DEFAULT_URL = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _download_url(url: str, dest: Path) -> None:
    import urllib.request

    # Some sandboxes inject a localhost proxy env that is not reachable from this process.
    # If the proxy points to localhost, disable proxies for this download.
    proxy_env = " ".join(
        [
            os.environ.get("http_proxy", ""),
            os.environ.get("https_proxy", ""),
            os.environ.get("HTTP_PROXY", ""),
            os.environ.get("HTTPS_PROXY", ""),
            os.environ.get("ALL_PROXY", ""),
        ]
    ).lower()
    if "127.0.0.1" in proxy_env or "localhost" in proxy_env:
        urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - controlled URL
    os.replace(tmp, dest)


def _chmod_x(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _find_extracted_bin(root: Path, name: str) -> Path:
    candidates = list(root.glob(f"ffmpeg-*-amd64-static/{name}"))
    if not candidates:
        candidates = list(root.glob(f"ffmpeg-*-amd64-static/bin/{name}"))
    if not candidates:
        raise FileNotFoundError(f"extracted archive does not contain {name}")
    return candidates[0]


def install_ffmpeg(*, url: str, out_dir: Path, force: bool) -> dict[str, str]:
    out_dir = out_dir.resolve()
    bin_dir = out_dir / "bin"
    ffmpeg_dst = bin_dir / "ffmpeg"
    ffprobe_dst = bin_dir / "ffprobe"

    if not force and ffmpeg_dst.exists() and ffprobe_dst.exists():
        return {"ok": "true", "ffmpeg": str(ffmpeg_dst), "ffprobe": str(ffprobe_dst), "skipped": "true"}

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    tmp_root = out_dir / "_tmp_extract"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    archive = out_dir / "ffmpeg-static.tar.xz"
    _download_url(url, archive)

    with tarfile.open(archive, mode="r:xz") as tf:
        # Safe-ish extraction: trust upstream layout but prevent path traversal.
        for m in tf.getmembers():
            if not m.name or m.name.startswith("/") or ".." in Path(m.name).parts:
                raise ValueError(f"unsafe tar member: {m.name}")
        tf.extractall(tmp_root)  # noqa: S202 - validated member paths above

    ffmpeg_src = _find_extracted_bin(tmp_root, "ffmpeg")
    ffprobe_src = _find_extracted_bin(tmp_root, "ffprobe")

    shutil.copy2(ffmpeg_src, ffmpeg_dst)
    shutil.copy2(ffprobe_src, ffprobe_dst)
    _chmod_x(ffmpeg_dst)
    _chmod_x(ffprobe_dst)

    # Leave the archive to avoid re-downloading; clear extraction temp.
    shutil.rmtree(tmp_root)

    stamp = out_dir / "INSTALL_STAMP.txt"
    stamp.write_text(
        f"ts={time.strftime('%Y-%m-%d %H:%M:%S')}\nurl={url}\nffmpeg={ffmpeg_dst}\nffprobe={ffprobe_dst}\n",
        encoding="utf-8",
    )

    return {"ok": "true", "ffmpeg": str(ffmpeg_dst), "ffprobe": str(ffprobe_dst), "skipped": "false"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Install static ffmpeg/ffprobe into data/tools/ffmpeg/ (no sudo).")
    p.add_argument("--url", type=str, default=os.environ.get("AVS_FFMPEG_URL", DEFAULT_URL))
    p.add_argument("--out-dir", type=Path, default=_repo_root() / "data" / "tools" / "ffmpeg")
    p.add_argument("--force", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rep = install_ffmpeg(url=str(args.url), out_dir=Path(args.out_dir), force=bool(args.force))
    print(rep["ffmpeg"])
    print(rep["ffprobe"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
