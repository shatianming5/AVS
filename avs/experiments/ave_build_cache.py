from __future__ import annotations

import argparse
import time
from pathlib import Path

from avs.vision.clip_vit import ClipVisionEncoder, ClipVisionEncoderConfig
from avs.vision.feature_cache import build_clip_feature_cache


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build multi-resolution CLIP feature caches for processed clips.")
    p.add_argument("--processed-dir", type=Path, required=True, help="Dir containing <clip_id>/frames/*.jpg")
    p.add_argument("--clip-id", action="append", default=[], help="Clip id to cache (repeatable)")
    p.add_argument("--resolutions", type=str, default="112,224,448")
    p.add_argument("--out-dir", type=Path, default=Path("runs") / f"features_cache_{time.strftime('%Y%m%d-%H%M%S')}")
    p.add_argument("--pretrained", action="store_true", help="Use pretrained CLIP weights (downloads from HF)")
    p.add_argument("--device", type=str, default="cpu")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.clip_id:
        raise SystemExit("at least one --clip-id is required")

    resolutions = [int(x) for x in str(args.resolutions).split(",") if str(x).strip()]
    encoder = ClipVisionEncoder(ClipVisionEncoderConfig(pretrained=bool(args.pretrained), device=args.device))

    for cid in args.clip_id:
        frames_dir = args.processed_dir / cid / "frames"
        cache = build_clip_feature_cache(frames_dir=frames_dir, resolutions=resolutions, encoder=encoder)
        cache.save_npz(args.out_dir / f"{cid}.npz")

    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

