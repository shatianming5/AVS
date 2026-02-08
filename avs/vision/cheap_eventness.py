from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def list_frames(frames_dir: Path) -> list[Path]:
    """
    List `{t}.jpg` frames in a directory, sorted by integer `t`.
    """
    paths = [p for p in Path(frames_dir).glob("*.jpg") if p.is_file()]
    out: list[tuple[int, Path]] = []
    for p in paths:
        try:
            t = int(p.stem)
        except Exception:
            continue
        out.append((t, p))
    out.sort(key=lambda x: x[0])
    return [p for _, p in out]


def _load_gray(path: Path, *, size: int = 32) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize((int(size), int(size)))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def frame_diff_eventness(frames: list[Path], *, size: int = 32) -> list[float]:
    """
    Cheap visual eventness based on frame-to-frame absolute difference (motion/scene change proxy).

    Returns one score per frame (score[0]=0).
    """
    if not frames:
        return []
    prev = _load_gray(frames[0], size=int(size))
    scores: list[float] = [0.0]
    for p in frames[1:]:
        cur = _load_gray(p, size=int(size))
        diff = float(np.mean(np.abs(cur - prev)))
        scores.append(float(diff))
        prev = cur
    return scores


def clip_feature_diff_eventness(features: np.ndarray, *, metric: str = "cosine") -> list[float]:
    """
    Cheap visual eventness from cached CLIP features (semantic motion proxy).

    Args:
      features: [T, D] float array (e.g., CLIP embeddings per second)
      metric:
        - "cosine": score[t] = 1 - cos(features[t-1], features[t])
        - "l2":     score[t] = ||features[t] - features[t-1]||_2

    Returns:
      length-T list with score[0]=0.0.
    """
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or int(x.shape[0]) <= 0:
        return []

    t = int(x.shape[0])
    scores: list[float] = [0.0]
    mode = str(metric)
    if mode not in ("cosine", "l2"):
        raise ValueError(f"unknown metric={mode!r}; expected 'cosine' or 'l2'")

    prev = x[0]
    prev_norm = float(np.linalg.norm(prev)) + 1e-12
    for i in range(1, t):
        cur = x[i]
        if mode == "l2":
            s = float(np.linalg.norm(cur - prev))
        else:
            cur_norm = float(np.linalg.norm(cur)) + 1e-12
            cos = float(np.dot(prev, cur) / (prev_norm * cur_norm))
            # Clamp for numerical stability then convert to a distance-like score.
            cos = float(max(-1.0, min(1.0, cos)))
            s = float(1.0 - cos)
            prev_norm = cur_norm
        scores.append(float(s))
        prev = cur
    return scores


def optical_flow_mag_eventness(frames: list[Path], *, size: int = 64) -> list[float]:
    """
    Cheap visual eventness based on optical-flow magnitude (motion proxy).

    Uses Farneback optical flow on resized grayscale frames and returns one score per frame (score[0]=0).

    Notes:
      - Deterministic given the frames and OpenCV version.
      - Intended as a stronger alternative to `frame_diff_eventness` for motion-heavy events.
    """
    if not frames:
        return []

    try:
        import cv2  # type: ignore
    except Exception as e:  # noqa: BLE001 - optional dependency
        raise ImportError("optical_flow_mag_eventness requires opencv-python (cv2)") from e

    prev = _load_gray(frames[0], size=int(size)).astype(np.float32, copy=False)
    scores: list[float] = [0.0]
    for p in frames[1:]:
        cur = _load_gray(p, size=int(size)).astype(np.float32, copy=False)
        flow = cv2.calcOpticalFlowFarneback(
            prev,
            cur,
            None,
            0.5,  # pyr_scale
            3,  # levels
            15,  # winsize
            3,  # iterations
            5,  # poly_n
            1.2,  # poly_sigma
            0,  # flags
        )
        mag = np.sqrt(np.square(flow[..., 0]) + np.square(flow[..., 1]))
        scores.append(float(np.mean(mag)))
        prev = cur
    return scores
