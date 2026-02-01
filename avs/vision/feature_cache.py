from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from avs.vision.clip_vit import ClipVisionEncoder


@dataclass(frozen=True)
class FeatureCache:
    resolutions: list[int]
    features_by_resolution: dict[int, np.ndarray]  # [T, D]

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"res_{int(r)}": self.features_by_resolution[int(r)] for r in self.resolutions}
        np.savez_compressed(path, **arrays)
        meta = {"resolutions": [int(r) for r in self.resolutions]}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load_npz(cls, path: Path) -> "FeatureCache":
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            resolutions = [int(r) for r in meta["resolutions"]]
        else:
            # Fallback: infer from keys
            with np.load(path) as z:
                resolutions = sorted(int(k.split("_", 1)[1]) for k in z.files if k.startswith("res_"))

        feats: dict[int, np.ndarray] = {}
        with np.load(path) as z:
            for r in resolutions:
                feats[r] = z[f"res_{r}"]
        return cls(resolutions=resolutions, features_by_resolution=feats)


def build_clip_feature_cache(
    *,
    frames_dir: Path,
    resolutions: list[int],
    encoder: ClipVisionEncoder,
    num_segments: int = 10,
) -> FeatureCache:
    features_by_resolution: dict[int, np.ndarray] = {}

    for r in resolutions:
        images: list[Image.Image] = []
        for t in range(num_segments):
            frame_path = frames_dir / f"{t}.jpg"
            if not frame_path.exists():
                raise FileNotFoundError(f"missing frame: {frame_path}")
            with Image.open(frame_path) as im:
                # Force-load into memory so the file handle can be closed before encoding.
                images.append(im.convert("RGB").copy())
        embs = encoder.encode(images, resolution=int(r)).numpy()
        features_by_resolution[int(r)] = embs.astype(np.float32)

    return FeatureCache(resolutions=[int(r) for r in resolutions], features_by_resolution=features_by_resolution)


def build_clip_feature_cache_from_seconds(
    *,
    frames_dir: Path,
    seconds: list[int],
    resolutions: list[int],
    encoder: ClipVisionEncoder,
) -> FeatureCache:
    features_by_resolution: dict[int, np.ndarray] = {}

    secs = [int(s) for s in seconds]
    if not secs:
        raise ValueError("seconds must be non-empty")

    for r in resolutions:
        images: list[Image.Image] = []
        for sec in secs:
            frame_path = frames_dir / f"{int(sec)}.jpg"
            if not frame_path.exists():
                raise FileNotFoundError(f"missing frame: {frame_path}")
            with Image.open(frame_path) as im:
                images.append(im.convert("RGB").copy())
        embs = encoder.encode(images, resolution=int(r)).numpy()
        features_by_resolution[int(r)] = embs.astype(np.float32)

    return FeatureCache(resolutions=[int(r) for r in resolutions], features_by_resolution=features_by_resolution)
