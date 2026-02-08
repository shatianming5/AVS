from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from avs.metrics.time_windows import TimeWindow
from avs.utils.scores import minmax_01


@dataclass(frozen=True)
class AudioEventness:
    scores: list[float]
    sample_rate: int


@dataclass(frozen=True)
class AnchorSelectionResult:
    anchors: list[int]
    conf_metric: str
    conf_value: float
    conf_threshold: float
    fallback_used: bool
    fallback_reason: str | None


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


def eventness_energy_stride(
    audio: np.ndarray,
    sample_rate: int,
    *,
    stride_s: float,
    win_s: float,
    pad: bool = True,
) -> list[float]:
    """
    Sliding-window log-energy at a given stride/window size.

    Each score corresponds to the center time:
      t_i = i * stride_s + win_s / 2
    """
    sr = int(sample_rate)
    if sr <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sample_rate}")

    stride = int(round(float(stride_s) * float(sr)))
    win = int(round(float(win_s) * float(sr)))
    if stride <= 0:
        raise ValueError(f"stride_s too small: {stride_s}")
    if win <= 0:
        raise ValueError(f"win_s too small: {win_s}")

    x = audio.astype(np.float32, copy=False)
    if x.size < win and pad:
        x = np.pad(x, (0, win - int(x.size)), mode="constant")

    scores: list[float] = []
    start = 0
    while start < int(x.size):
        end = start + win
        seg = x[start:end]
        if seg.size < win:
            if not pad:
                break
            seg = np.pad(seg, (0, win - int(seg.size)), mode="constant")
        ms = float(np.mean(seg * seg))
        scores.append(float(math.log(ms + 1e-12)))
        start += stride
        if start + 1 > int(x.size) and not pad:
            break
    return scores


def compute_eventness_wav_energy_stride(
    path: Path,
    *,
    stride_s: float,
    win_s: float,
) -> AudioEventness:
    audio, sr = load_wav_mono(path)
    scores = eventness_energy_stride(audio, sr, stride_s=float(stride_s), win_s=float(win_s), pad=True)
    return AudioEventness(scores=scores, sample_rate=sr)


def eventness_energy_stride_max_per_second(
    audio: np.ndarray,
    sample_rate: int,
    *,
    num_segments: int = 10,
    stride_s: float = 0.2,
    win_s: float = 0.4,
) -> list[float]:
    """
    Dense stride-based log-energy aggregated into `num_segments` per-second scores by max pooling.

    This is intended as a higher-recall anchor proposal than the vanilla 10×1s energy:
      - compute sliding-window log-energy at sub-second stride
      - assign each window score to a second index by its center time
      - take max within each second
    """
    seg_len = int(sample_rate)
    target_len = seg_len * int(num_segments)

    # Match `eventness_energy_per_second` behavior: pad/trim to fixed-length protocol.
    n = int(audio.shape[0])
    if n < target_len:
        audio = np.pad(audio, (0, target_len - n), mode="constant")
    elif n > target_len:
        audio = audio[:target_len]

    dense = eventness_energy_stride(audio, int(sample_rate), stride_s=float(stride_s), win_s=float(win_s), pad=True)
    out = [float("-inf")] * int(num_segments)
    for i, s in enumerate(dense):
        center_s = float(int(i)) * float(stride_s) + float(win_s) / 2.0
        if not (0.0 <= float(center_s) < float(num_segments)):
            continue
        t = int(center_s)
        out[t] = float(max(float(out[t]), float(s)))

    valid = [x for x in out if float(x) != float("-inf")]
    fill = float(min(valid)) if valid else 0.0
    return [float(fill if float(x) == float("-inf") else x) for x in out]


def compute_eventness_wav_energy_stride_max(
    path: Path,
    *,
    num_segments: int = 10,
    stride_s: float = 0.2,
    win_s: float = 0.4,
) -> AudioEventness:
    audio, sr = load_wav_mono(path)
    scores = eventness_energy_stride_max_per_second(
        audio,
        int(sr),
        num_segments=int(num_segments),
        stride_s=float(stride_s),
        win_s=float(win_s),
    )
    return AudioEventness(scores=scores, sample_rate=sr)


def local_maxima_indices(scores: list[float]) -> list[int]:
    if not scores:
        return []
    out: list[int] = []
    n = len(scores)
    for i in range(n):
        left = scores[i - 1] if i - 1 >= 0 else None
        right = scores[i + 1] if i + 1 < n else None
        if left is None and right is None:
            out.append(int(i))
            continue
        if left is None:
            if float(scores[i]) > float(right):
                out.append(int(i))
            continue
        if right is None:
            if float(scores[i]) > float(left):
                out.append(int(i))
            continue
        if float(scores[i]) > float(left) and float(scores[i]) >= float(right):
            out.append(int(i))
    return out


def topk_anchors_localmax_nms(scores: list[float], k: int, *, radius: int = 1) -> list[int]:
    """
    Local-maxima Top-K with NMS.
    """
    if k <= 0:
        return []
    cand = local_maxima_indices(scores)
    if not cand:
        return topk_anchors_nms(scores, k=int(k), radius=int(radius))

    order = sorted(cand, key=lambda i: (-float(scores[i]), int(i)))
    out: list[int] = []
    for i in order:
        if any(abs(int(i) - int(j)) <= int(radius) for j in out):
            continue
        out.append(int(i))
        if len(out) >= int(k):
            break
    return out


@dataclass(frozen=True)
class AnchorWindows:
    stride_s: float
    win_s: float
    delta_s: float
    scores: list[float]
    anchors_idx: list[int]
    anchors_s: list[float]
    windows: list[TimeWindow]

    def to_jsonable(self) -> dict:
        return {
            "stride_s": float(self.stride_s),
            "win_s": float(self.win_s),
            "delta_s": float(self.delta_s),
            "scores": [float(x) for x in self.scores],
            "anchors_idx": [int(x) for x in self.anchors_idx],
            "anchors_s": [float(x) for x in self.anchors_s],
            "windows": [{"start_s": float(w.start_s), "end_s": float(w.end_s)} for w in self.windows],
        }


def anchor_windows_from_scores(
    scores: list[float],
    *,
    stride_s: float,
    win_s: float,
    delta_s: float,
    k: int,
    nms_radius: int = 1,
) -> AnchorWindows:
    """
    Convert stride-based score sequence into explicit time windows.
    """
    anchors_idx = topk_anchors_localmax_nms(scores, k=int(k), radius=int(nms_radius))
    anchors_s = [float(int(i) * float(stride_s) + float(win_s) / 2.0) for i in anchors_idx]
    windows: list[TimeWindow] = []
    for t in anchors_s:
        start = max(0.0, float(t) - float(delta_s))
        end = float(t) + float(delta_s)
        windows.append(TimeWindow(start_s=float(start), end_s=float(end)))
    return AnchorWindows(
        stride_s=float(stride_s),
        win_s=float(win_s),
        delta_s=float(delta_s),
        scores=[float(x) for x in scores],
        anchors_idx=[int(x) for x in anchors_idx],
        anchors_s=[float(x) for x in anchors_s],
        windows=windows,
    )


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


def topk_anchors_adjacent_top2(
    scores: list[float],
    k: int,
    *,
    radius: int = 1,
    max_gap: float = 0.0,
) -> list[int]:
    """
    Top-K selection that prefers a near-by 2nd anchor when it is competitive with the top1 anchor.

    This is designed for K>=2:
      1) Pick the global best anchor `a1`.
      2) Find the best candidate within ±`radius` of `a1` (excluding `a1`), call it `a_adj`.
         If `score[a1] - score[a_adj] <= max_gap`, select `a_adj` as the 2nd anchor.
         Otherwise, select the 2nd-best overall anchor.

    For K>2, remaining anchors are filled with NMS (radius) from the global order.
    """
    if k <= 0:
        return []
    radius = int(radius)
    if radius < 1:
        radius = 1
    max_gap = float(max_gap)
    if max_gap < 0.0:
        raise ValueError(f"max_gap must be >= 0, got {max_gap}")

    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), int(i)))
    if not order:
        return []
    if len(order) == 1 or int(k) == 1:
        return [int(order[0])]

    a1 = int(order[0])
    a2_default = int(order[1])

    best_adj = None
    best_adj_score = float("-inf")
    lo = max(0, int(a1) - int(radius))
    hi = min(len(scores), int(a1) + int(radius) + 1)
    for i in range(lo, hi):
        if int(i) == int(a1):
            continue
        s = float(scores[int(i)])
        if s > best_adj_score + 1e-12 or (abs(s - best_adj_score) <= 1e-12 and (best_adj is None or int(i) < int(best_adj))):
            best_adj = int(i)
            best_adj_score = float(s)

    out: list[int] = [a1]
    if best_adj is not None and float(scores[a1] - float(best_adj_score)) <= float(max_gap):
        out.append(int(best_adj))
    else:
        out.append(int(a2_default))

    # If K>2, fill remaining anchors with standard NMS starting from the selected set.
    if len(out) < int(k):
        for i in order[2:]:
            if any(abs(int(i) - int(j)) <= int(radius) for j in out):
                continue
            out.append(int(i))
            if len(out) >= int(k):
                break

    # Ensure stable ordering by descending score (then index), matching other selectors.
    out = sorted(set(out), key=lambda i: (-scores[int(i)], int(i)))
    return out[: min(int(k), len(out))]


def smooth_scores(scores: list[float], *, window: int, mode: str = "mean") -> list[float]:
    """
    Smooth a 1D score sequence with a centered sliding window.

    - window=0/1: no-op (returns a copy).
    - window must be odd for symmetric behavior.
    - mode: "mean" or "sum".
    """
    w = int(window)
    if w <= 1:
        return [float(x) for x in scores]
    if w % 2 == 0:
        raise ValueError(f"window must be odd, got {w}")

    mode = str(mode)
    if mode not in ("mean", "sum"):
        raise ValueError(f"unknown mode={mode!r}; expected 'mean' or 'sum'")

    half = w // 2
    out: list[float] = []
    for i in range(len(scores)):
        lo = max(0, i - half)
        hi = min(len(scores), i + half + 1)
        vals = scores[lo:hi]
        if not vals:
            out.append(float(scores[i]))
            continue
        s = float(sum(float(x) for x in vals))
        if mode == "mean":
            s /= float(len(vals))
        out.append(float(s))
    return out


def confidence_std(scores: list[float]) -> float:
    if not scores:
        return 0.0
    s = np.asarray(scores, dtype=np.float32)
    return float(s.std())


def confidence_std_norm(scores: list[float]) -> float:
    # Scale-invariant variant: compute std after per-clip min-max normalization to [0,1].
    return confidence_std(minmax_01(scores))


def confidence_top1_minus_median(scores: list[float]) -> float:
    if not scores:
        return 0.0
    s = np.asarray(scores, dtype=np.float32)
    top1 = float(np.max(s))
    med = float(np.median(s))
    return float(top1 - med)


def confidence_top1_minus_median_norm(scores: list[float]) -> float:
    # Scale-invariant variant: compute (top1 - median) after per-clip min-max normalization to [0,1].
    return confidence_top1_minus_median(minmax_01(scores))


def confidence_top12_gap(scores: list[float]) -> float:
    if len(scores) < 2:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    return float(float(scores[order[0]]) - float(scores[order[1]]))


def confidence_top12_gap_norm(scores: list[float]) -> float:
    # Scale-invariant variant: compute (top1 - top2) after per-clip min-max normalization to [0,1].
    return confidence_top12_gap(minmax_01(scores))


def confidence_top3_bottom3_gap_norm(scores: list[float]) -> float:
    """
    Scale-invariant "separation" confidence:

      conf = mean(top-3(scores_norm)) - mean(bottom-3(scores_norm))

    Unlike peakiness measures (e.g. top1-med), this stays high for broad multi-second events as long as
    there is a clear separation between high-score and low-score segments.
    """
    s01 = minmax_01(scores)
    if not s01:
        return 0.0
    n = len(s01)
    k = min(3, n)
    xs = sorted(float(x) for x in s01)
    bot = float(sum(xs[:k]) / float(k))
    top = float(sum(xs[-k:]) / float(k))
    return float(top - bot)


def confidence_gini(scores: list[float]) -> float:
    """
    Gini coefficient as a simple "peakiness" measure.

    Since scores can be negative (e.g., log-energy), we shift them to be non-negative.
    """
    if not scores:
        return 0.0
    x = np.asarray(scores, dtype=np.float64)
    x = x - float(np.min(x))
    if float(np.sum(x)) <= 0.0:
        return 0.0
    x = np.sort(x)
    n = int(x.size)
    idx = np.arange(1, n + 1, dtype=np.float64)
    g = (2.0 * float(np.sum(idx * x)) / (float(n) * float(np.sum(x)))) - (float(n) + 1.0) / float(n)
    return float(max(0.0, min(1.0, g)))


def _confidence(scores: list[float], metric: str) -> float:
    metric = str(metric)
    if metric == "std":
        return confidence_std(scores)
    if metric == "std_norm":
        return confidence_std_norm(scores)
    if metric == "top1_med":
        return confidence_top1_minus_median(scores)
    if metric == "top1_med_norm":
        return confidence_top1_minus_median_norm(scores)
    if metric == "top12_gap":
        return confidence_top12_gap(scores)
    if metric == "top12_gap_norm":
        return confidence_top12_gap_norm(scores)
    if metric == "top3_bottom3_gap_norm":
        return confidence_top3_bottom3_gap_norm(scores)
    if metric == "gini":
        return confidence_gini(scores)
    raise ValueError(
        "unknown conf_metric="
        f"{metric!r}; expected 'std', 'std_norm', 'top1_med', 'top1_med_norm', 'top12_gap', 'top12_gap_norm', "
        "'top3_bottom3_gap_norm', or 'gini'"
    )


def window_topk_anchors(scores: list[float], k: int, *, window: int = 3, nms_radius: int = 1) -> list[int]:
    """
    Select anchors by applying window aggregation then Top-K with NMS.

    This helps avoid selecting adjacent seconds when a single event spans multiple seconds.
    """
    agg = smooth_scores(scores, window=int(window), mode="mean")
    return topk_anchors_nms(agg, k=int(k), radius=int(nms_radius))


def anchors_from_scores_with_debug(
    scores: list[float],
    *,
    k: int,
    num_segments: int | None = None,
    shift: int = 0,
    std_threshold: float = 0.0,
    select: str = "topk",
    nms_radius: int = 1,
    nms_strong_gap: float = 0.0,
    anchor_window: int = 3,
    smooth_window: int = 0,
    smooth_mode: str = "mean",
    conf_metric: str | None = None,
    conf_threshold: float | None = None,
) -> AnchorSelectionResult:
    if num_segments is None:
        num_segments = len(scores)
    num_segments = int(num_segments)
    if num_segments <= 0:
        return AnchorSelectionResult(
            anchors=[],
            conf_metric=str(conf_metric or "std"),
            conf_value=0.0,
            conf_threshold=float(conf_threshold if conf_threshold is not None else std_threshold),
            fallback_used=True,
            fallback_reason="empty_scores",
        )

    # Clip scores to evaluated range for stability.
    scores = [float(x) for x in scores[:num_segments]]

    # Backward-compatible defaults: std_threshold == conf_threshold when conf_* is unset.
    if conf_metric is None:
        conf_metric = "std"
    if conf_threshold is None:
        conf_threshold = float(std_threshold)

    conf_value = _confidence(scores, str(conf_metric))
    if float(conf_threshold) > 0.0 and float(conf_value) < float(conf_threshold):
        return AnchorSelectionResult(
            anchors=[],
            conf_metric=str(conf_metric),
            conf_value=float(conf_value),
            conf_threshold=float(conf_threshold),
            fallback_used=True,
            fallback_reason="conf_below_threshold",
        )

    select = str(select)
    scores_sel = scores
    if int(smooth_window) > 1:
        scores_sel = smooth_scores(scores_sel, window=int(smooth_window), mode=str(smooth_mode))

    if select == "topk":
        anchors = topk_anchors(scores_sel, k=k)
    elif select == "nms":
        anchors = topk_anchors_nms(scores_sel, k=k, radius=int(nms_radius))
    elif select == "nms_strong":
        anchors = topk_anchors_nms_strong(scores_sel, k=k, radius=int(nms_radius), max_gap=float(nms_strong_gap))
    elif select == "adjacent_top2":
        anchors = topk_anchors_adjacent_top2(scores_sel, k=k, radius=int(nms_radius), max_gap=float(nms_strong_gap))
    elif select == "window_topk":
        anchors = window_topk_anchors(scores_sel, k=k, window=int(anchor_window), nms_radius=int(nms_radius))
    else:
        raise ValueError(
            f"unknown select={select!r}; expected 'topk', 'nms', 'nms_strong', 'adjacent_top2', or 'window_topk'"
        )

    if shift:
        out: list[int] = []
        for a in anchors:
            b = int(a) + int(shift)
            if 0 <= b < int(num_segments) and b not in out:
                out.append(b)
        anchors = out

    return AnchorSelectionResult(
        anchors=[int(x) for x in anchors],
        conf_metric=str(conf_metric),
        conf_value=float(conf_value),
        conf_threshold=float(conf_threshold),
        fallback_used=False,
        fallback_reason=None,
    )


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
    anchor_window: int = 3,
    smooth_window: int = 0,
    smooth_mode: str = "mean",
    conf_metric: str | None = None,
    conf_threshold: float | None = None,
) -> list[int]:
    """
    Select Top-K anchors from scores with optional robustness knobs.

    - `shift`: integer offset applied to each selected anchor (models A/V misalignment).
    - `std_threshold`: legacy alias for `conf_threshold` when `conf_*` is not set.
    - `select`: "topk" (default), "nms" (hard temporal NMS), "nms_strong" (prefers far anchors only if strong),
      or "window_topk" (window aggregation + NMS).
    - `nms_radius`: only for `select="nms"`, suppression radius in segments.
    - `nms_strong_gap`: only for `select="nms_strong"`, max allowed (top1 - far_candidate) score gap to accept a far anchor.
    """
    res = anchors_from_scores_with_debug(
        scores,
        k=int(k),
        num_segments=num_segments,
        shift=int(shift),
        std_threshold=float(std_threshold),
        select=str(select),
        nms_radius=int(nms_radius),
        nms_strong_gap=float(nms_strong_gap),
        anchor_window=int(anchor_window),
        smooth_window=int(smooth_window),
        smooth_mode=str(smooth_mode),
        conf_metric=str(conf_metric) if conf_metric is not None else None,
        conf_threshold=float(conf_threshold) if conf_threshold is not None else None,
    )
    return list(res.anchors)
