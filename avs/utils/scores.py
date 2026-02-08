from __future__ import annotations


AV_FUSED_SCORE_SCALE = 3.5


def pad_or_truncate(scores: list[float], *, num_segments: int, pad_value: float = 0.0) -> list[float]:
    n = int(num_segments)
    if n <= 0:
        return []
    xs = [float(x) for x in scores[:n]]
    if len(xs) < n:
        xs.extend([float(pad_value)] * (n - len(xs)))
    return xs


def minmax_01(scores: list[float], *, eps: float = 1e-12) -> list[float]:
    """
    Per-sequence min-max normalization to [0,1].

    If the sequence is constant (max-min <= eps), returns all zeros to avoid NaNs.
    """
    if not scores:
        return []
    xs = [float(x) for x in scores]
    lo = float(min(xs))
    hi = float(max(xs))
    rng = float(hi - lo)
    if rng <= float(eps):
        return [0.0 for _ in xs]
    return [(float(x) - lo) / rng for x in xs]


def fuse_max(a: list[float], b: list[float], *, num_segments: int) -> list[float]:
    aa = pad_or_truncate(a, num_segments=int(num_segments), pad_value=0.0)
    bb = pad_or_truncate(b, num_segments=int(num_segments), pad_value=0.0)
    return [float(max(float(x), float(y))) for x, y in zip(aa, bb, strict=True)]


def fuse_prod(a: list[float], b: list[float], *, num_segments: int) -> list[float]:
    aa = pad_or_truncate(a, num_segments=int(num_segments), pad_value=0.0)
    bb = pad_or_truncate(b, num_segments=int(num_segments), pad_value=0.0)
    return [float(float(x) * float(y)) for x, y in zip(aa, bb, strict=True)]


def scale(scores: list[float], factor: float) -> list[float]:
    f = float(factor)
    return [float(x) * f for x in scores]


def stride_max_pool_per_second(
    scores: list[float],
    *,
    num_segments: int,
    stride_s: float = 0.2,
    win_s: float = 0.6,
) -> list[float]:
    """
    Convert per-second scores into a denser stride timeline, apply local max pooling,
    then aggregate back to one score per second.

    This is useful when anchor proposals are derived from sub-second windows rather
    than only second-level local maxima.
    """
    n = int(num_segments)
    if n <= 0:
        return []
    xs = pad_or_truncate(scores, num_segments=n, pad_value=0.0)
    if float(stride_s) <= 0.0 or float(win_s) <= 0.0:
        return xs

    import numpy as np

    step = float(stride_s)
    dense_n = max(1, int(round(float(max(0, n - 1)) / step)) + 1)
    dense_t = np.arange(dense_n, dtype=np.float32) * np.float32(step)
    base_t = np.arange(n, dtype=np.float32)
    base = np.asarray(xs, dtype=np.float32)
    dense = np.interp(dense_t, base_t, base).astype(np.float32, copy=False)

    win_bins = max(1, int(round(float(win_s) / step)))
    half = int(win_bins // 2)
    pooled = np.empty_like(dense)
    for i in range(dense_n):
        lo = max(0, int(i - half))
        hi = min(int(dense_n), int(i + half + 1))
        pooled[i] = float(np.max(dense[lo:hi]))

    out: list[float] = []
    for sec in range(n):
        lo = int(np.floor(float(sec) / step))
        hi = int(np.floor((float(sec) + 1.0 - 1e-6) / step)) + 1
        lo = max(0, min(lo, dense_n - 1))
        hi = max(lo + 1, min(hi, dense_n))
        out.append(float(np.max(pooled[lo:hi])))
    return out


def shift_scores(scores: list[float], shift: int, *, pad_value: float | None = None) -> list[float]:
    """
    Shift a score sequence by `shift` steps with padding.

    Convention: positive `shift` moves scores to the right (later indices).
    """
    if not scores:
        return []
    s = int(shift)
    xs = [float(x) for x in scores]
    if s == 0:
        return xs

    pad = float(min(xs)) if pad_value is None else float(pad_value)
    n = len(xs)
    out = [pad] * n
    if s > 0:
        for i in range(0, n - s):
            out[i + s] = xs[i]
    else:
        k = -s
        for i in range(k, n):
            out[i - k] = xs[i]
    return out


def best_shift_by_corr(
    a: list[float], b: list[float], *, shifts: list[int] = (-2, -1, 0, 1, 2), eps: float = 1e-6
) -> int:
    """
    Pick the integer shift that maximizes Pearson correlation between `a` and `b`.

    Tie-break: prefer smaller |shift|, then smaller shift (deterministic).
    """
    import numpy as np

    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    n = int(min(int(aa.size), int(bb.size)))
    if n < 2:
        return 0
    aa = aa[:n]
    bb = bb[:n]

    def _corr(x: np.ndarray, y: np.ndarray) -> float:
        x = x - float(x.mean())
        y = y - float(y.mean())
        denom = float(np.sqrt(float((x * x).sum())) * np.sqrt(float((y * y).sum())))
        if denom <= float(eps):
            return 0.0
        return float((x * y).sum() / denom)

    best_s = 0
    best_v = float("-inf")
    for s in [int(x) for x in shifts]:
        if s >= 0:
            if n - s < 2:
                v = float("-inf")
            else:
                v = _corr(aa[: n - s], bb[s:n])
        else:
            k = -s
            if n - k < 2:
                v = float("-inf")
            else:
                v = _corr(aa[k:n], bb[: n - k])

        if v > best_v + 1e-12:
            best_s, best_v = int(s), float(v)
            continue
        if abs(float(v) - float(best_v)) <= 1e-12:
            if abs(int(s)) < abs(int(best_s)) or (abs(int(s)) == abs(int(best_s)) and int(s) < int(best_s)):
                best_s, best_v = int(s), float(v)
    return int(best_s)
