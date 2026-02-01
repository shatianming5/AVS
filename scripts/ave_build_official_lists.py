#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from avs.datasets.ave import AVEIndex, ensure_ave_meta
from avs.datasets.layout import ave_paths


@dataclass(frozen=True)
class SplitLists:
    ok: list[str]
    fail: list[str]


def _write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def _split_ids(index: AVEIndex, split: str) -> list[str]:
    return [index.clips[int(i)].video_id for i in index.splits[split]]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build AVE download_ok/download_fail lists by checking which <video_id>.mp4 exist in a raw videos dir."
    )
    p.add_argument("--meta-dir", type=Path, default=ave_paths().meta_dir)
    p.add_argument("--raw-videos-dir", type=Path, required=True)
    p.add_argument("--tag", type=str, default="official", help="Filename tag (e.g., 'official' or 'auto').")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    meta_dir: Path = args.meta_dir
    raw_videos_dir: Path = args.raw_videos_dir
    tag = str(args.tag).strip()
    if not tag:
        raise SystemExit("--tag must be non-empty")

    ensure_ave_meta(meta_dir)
    index = AVEIndex.from_meta_dir(meta_dir)

    for split in ("train", "val", "test"):
        ids = _split_ids(index, split)
        ok: list[str] = []
        fail: list[str] = []
        for vid in ids:
            if (raw_videos_dir / f"{vid}.mp4").exists():
                ok.append(vid)
            else:
                fail.append(vid)

        ok_path = meta_dir / f"download_ok_{split}_{tag}.txt"
        fail_path = meta_dir / f"download_fail_{split}_{tag}.txt"
        _write_ids(ok_path, ok)
        _write_ids(fail_path, fail)
        print(f"{split}: ok={len(ok):4d} fail={len(fail):4d} -> {ok_path} {fail_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

