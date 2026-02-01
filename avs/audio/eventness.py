from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AudioEventness:
    scores: list[float]
    sample_rate: int


def load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        pcm = wf.readframes(num_frames)

    if sample_width != 2:
        raise ValueError(f"expected 16-bit PCM wav, got sampwidth={sample_width} for {path}")

    audio = np.frombuffer(pcm, dtype=np.int16)
    if num_channels > 1:
        audio = audio.reshape(-1, num_channels).mean(axis=1)
    audio = audio.astype(np.float32) / 32768.0
    return audio, int(sample_rate)


def eventness_energy_per_second(audio: np.ndarray, sample_rate: int, *, num_segments: int = 10) -> list[float]:
    seg_len = int(sample_rate)
    target_len = seg_len * int(num_segments)

    # Real-world WAVs extracted via ffmpeg can be off by a few samples due to rounding.
    # For AVE's fixed 10s protocol, we pad/trim to exactly `num_segments` seconds.
    n = int(audio.shape[0])
    if n < target_len:
        pad = target_len - n
        audio = np.pad(audio, (0, pad), mode="constant")
    elif n > target_len:
        audio = audio[:target_len]

    scores: list[float] = []
    for t in range(num_segments):
        seg = audio[t * seg_len : (t + 1) * seg_len]
        ms = float(np.mean(seg * seg))
        scores.append(math.log(ms + 1e-12))
    return scores


def compute_eventness_wav_energy(path: Path, *, num_segments: int = 10) -> AudioEventness:
    audio, sr = load_wav_mono(path)
    scores = eventness_energy_per_second(audio, sr, num_segments=num_segments)
    return AudioEventness(scores=scores, sample_rate=sr)


def eventness_energy_delta_per_second(audio: np.ndarray, sample_rate: int, *, num_segments: int = 10) -> list[float]:
    energy = eventness_energy_per_second(audio, sample_rate, num_segments=num_segments)
    if not energy:
        return []
    scores: list[float] = [0.0]
    for t in range(1, len(energy)):
        scores.append(float(abs(float(energy[t]) - float(energy[t - 1]))))
    return scores


def compute_eventness_wav_energy_delta(path: Path, *, num_segments: int = 10) -> AudioEventness:
    audio, sr = load_wav_mono(path)
    scores = eventness_energy_delta_per_second(audio, sr, num_segments=num_segments)
    return AudioEventness(scores=scores, sample_rate=sr)


def topk_anchors(scores: list[float], k: int) -> list[int]:
    if k <= 0:
        return []
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return order[: min(k, len(order))]


def topk_anchors_nms(scores: list[float], k: int, *, radius: int = 1) -> list[int]:
    """
    NMS-style Top-K selection on a 1D score sequence.

    Picks the highest score index, then suppresses any candidate within ±`radius` seconds,
    repeating until K anchors are selected or candidates are exhausted.
    """
    if k <= 0:
        return []
    radius = int(radius)
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")

    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    out: list[int] = []
    for i in order:
        if any(abs(int(i) - int(j)) <= radius for j in out):
            continue
        out.append(int(i))
        if len(out) >= int(k):
            break
    return out


def topk_anchors_nms_strong(
    scores: list[float],
    k: int,
    *,
    radius: int = 1,
    max_gap: float = 0.0,
) -> list[int]:
    """
    "Strong-peak" Top-K selection that prefers diverse anchors only when they are competitive.

    For K>=2:
      1) Pick the global best anchor `a1`.
      2) Consider the best candidate `a_far` with distance > `radius` from `a1`.
         If `score[a1] - score[a_far] <= max_gap`, pick `a_far` as the next anchor.
         Otherwise, fall back to the 2nd-best overall (even if adjacent to `a1`).

    This is a practical compromise between:
      - plain Top-K (often picks adjacent seconds when a single event spans multiple seconds),
      - hard NMS (can force very weak far-away anchors, hurting downstream accuracy).
    """
    if k <= 0:
        return []
    radius = int(radius)
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    max_gap = float(max_gap)
    if max_gap < 0.0:
        raise ValueError(f"max_gap must be >= 0, got {max_gap}")

    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    if not order:
        return []
    if len(order) == 1 or int(k) == 1:
        return [int(order[0])]

    a1 = int(order[0])
    a2_default = int(order[1])

    a_far = None
    for i in order[1:]:
        if abs(int(i) - a1) > radius:
            a_far = int(i)
            break

    out: list[int] = [a1]
    if a_far is not None and float(scores[a1] - scores[a_far]) <= float(max_gap):
        out.append(a_far)
    else:
        out.append(a2_default)

    # If K>2, fill remaining anchors with standard NMS starting from the selected set.
    if len(out) < int(k):
        for i in order[2:]:
            if any(abs(int(i) - int(j)) <= radius for j in out):
                continue
            out.append(int(i))
            if len(out) >= int(k):
                break

    # Ensure stable ordering by descending score (then index), matching other selectors.
    out = sorted(set(out), key=lambda i: (-scores[int(i)], int(i)))
    return out[: min(int(k), len(out))]


def anchors_from_scores(
    scores: list[float],
    *,
    k: int,
    num_segments: int | None = None,
    shift: int = 0,
    std_threshold: float = 0.0,
    select: str = "topk",
    nms_radius: int = 1,
    nms_strong_gap: float = 0.0,
) -> list[int]:
    """
    Select Top-K anchors from scores with optional robustness knobs.

    - `shift`: integer offset applied to each selected anchor (models A/V misalignment).
    - `std_threshold`: if `std(scores) < std_threshold`, return [] (fallback to uniform sampling).
    - `select`: "topk" (default), "nms" (hard temporal NMS), or "nms_strong" (prefers far anchors only if strong).
    - `nms_radius`: only for `select="nms"`, suppression radius in segments.
    - `nms_strong_gap`: only for `select="nms_strong"`, max allowed (top1 - far_candidate) score gap to accept a far anchor.
    """
    if num_segments is None:
        num_segments = len(scores)
    if num_segments <= 0:
        return []

    if std_threshold > 0.0:
        s = np.asarray(scores, dtype=np.float32)
        if float(s.std()) < float(std_threshold):
            return []

    select = str(select)
    if select == "topk":
        anchors = topk_anchors(scores, k=k)
    elif select == "nms":
        anchors = topk_anchors_nms(scores, k=k, radius=int(nms_radius))
    elif select == "nms_strong":
        anchors = topk_anchors_nms_strong(scores, k=k, radius=int(nms_radius), max_gap=float(nms_strong_gap))
    else:
        raise ValueError(f"unknown select={select!r}; expected 'topk', 'nms', or 'nms_strong'")

    if shift:
        out: list[int] = []
        for a in anchors:
            b = int(a) + int(shift)
            if 0 <= b < int(num_segments) and b not in out:
                out.append(b)
        anchors = out
    return anchors
