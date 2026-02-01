#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from avs.datasets.ave import AVEIndex, ensure_ave_meta
from avs.datasets.layout import ave_paths


def _write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(x) for x in ids) + "\n", encoding="utf-8")


def _split_ids(index: AVEIndex, split: str) -> list[str]:
    return [index.clips[int(i)].video_id for i in index.splits[split]]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export AVE train/val/test YouTube ids to text files.")
    p.add_argument("--meta-dir", type=Path, default=ave_paths().meta_dir)
    p.add_argument("--out-dir", type=Path, default=ave_paths().meta_dir)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_ave_meta(args.meta_dir)
    index = AVEIndex.from_meta_dir(args.meta_dir)

    for split in ("train", "val", "test"):
        ids = _split_ids(index, split)
        _write_ids(args.out_dir / f"ave_{split}_all_ids.txt", ids)

    print(str(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

