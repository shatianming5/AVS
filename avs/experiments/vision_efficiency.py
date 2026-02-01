from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from avs.sampling.token_budget import TokenBudget
from avs.vision.clip_vit import ClipVisionEncoder, ClipVisionEncoderConfig
from avs.vision.vit_flops import VitFlopsConfig, estimate_vit_forward_flops


@dataclass(frozen=True)
class VisionEfficiencyConfig:
    resolutions: list[int]
    batch_size: int = 8
    warmup: int = 2
    iters: int = 10
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"
    pretrained: bool = False


def bench_clip_vision_encoder(cfg: VisionEfficiencyConfig) -> dict:
    budget = TokenBudget(patch_size=16)
    encoder = ClipVisionEncoder(ClipVisionEncoderConfig(pretrained=bool(cfg.pretrained), device=str(cfg.device), dtype=str(cfg.dtype)))

    vision_cfg = encoder.model.vision_model.config
    flops_cfg = VitFlopsConfig(
        num_layers=int(vision_cfg.num_hidden_layers),
        hidden_size=int(vision_cfg.hidden_size),
        intermediate_size=int(vision_cfg.intermediate_size),
        patch_size=int(vision_cfg.patch_size),
        num_channels=3,
        include_class_token=True,
    )

    rng = np.random.default_rng(int(cfg.seed))
    base_res = int(max(cfg.resolutions) if cfg.resolutions else 224)
    images = [
        Image.fromarray(rng.integers(0, 256, size=(base_res, base_res, 3), dtype=np.uint8), mode="RGB")
        for _ in range(int(cfg.batch_size))
    ]

    results_by_resolution: list[dict] = []
    for r in cfg.resolutions:
        r = int(r)
        # Warmup.
        for _ in range(int(cfg.warmup)):
            _ = encoder.encode(images, resolution=r)

        if cfg.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(int(cfg.iters)):
            _ = encoder.encode(images, resolution=r)
        if cfg.device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        sec = float(t1 - t0) / max(1, int(cfg.iters))
        ms_per_batch = 1000.0 * sec
        ms_per_image = ms_per_batch / max(1, int(cfg.batch_size))
        tokens_per_image = int(budget.tokens_for_resolution(r))
        flops_per_image = int(estimate_vit_forward_flops(flops_cfg, resolution=r))

        results_by_resolution.append(
            {
                "resolution": r,
                "batch_size": int(cfg.batch_size),
                "iters": int(cfg.iters),
                "ms_per_batch": float(ms_per_batch),
                "ms_per_image": float(ms_per_image),
                "tokens_per_image": tokens_per_image,
                "tokens_per_batch": int(tokens_per_image) * int(cfg.batch_size),
                "approx_flops_per_image": flops_per_image,
                "approx_flops_per_batch": int(flops_per_image) * int(cfg.batch_size),
            }
        )

    return {
        "config": {
            "resolutions": [int(r) for r in cfg.resolutions],
            "batch_size": int(cfg.batch_size),
            "warmup": int(cfg.warmup),
            "iters": int(cfg.iters),
            "seed": int(cfg.seed),
            "device": str(cfg.device),
            "dtype": str(cfg.dtype),
            "pretrained": bool(cfg.pretrained),
            "vit": {
                "num_layers": int(flops_cfg.num_layers),
                "hidden_size": int(flops_cfg.hidden_size),
                "intermediate_size": int(flops_cfg.intermediate_size),
                "patch_size": int(flops_cfg.patch_size),
            },
        },
        "results_by_resolution": results_by_resolution,
    }


def _parse_int_list(text: str) -> list[int]:
    return [int(x) for x in str(text).split(",") if str(x).strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark ClipVisionEncoder wall-clock by resolution (ms/image).")
    p.add_argument("--resolutions", type=str, default="112,224,448", help="Comma-separated list, e.g. 112,224,448")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32", "bfloat16"])
    p.add_argument("--vision-pretrained", action="store_true", help="Use pretrained CLIP weights (downloads from HF)")
    p.add_argument("--out-dir", type=Path, default=Path("runs") / f"vision_efficiency_{time.strftime('%Y%m%d-%H%M%S')}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = VisionEfficiencyConfig(
        resolutions=_parse_int_list(args.resolutions),
        batch_size=int(args.batch_size),
        warmup=int(args.warmup),
        iters=int(args.iters),
        seed=int(args.seed),
        device=str(args.device),
        dtype=str(args.dtype),
        pretrained=bool(args.vision_pretrained),
    )
    payload = bench_clip_vision_encoder(cfg)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "vision_efficiency.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
