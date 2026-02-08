from __future__ import annotations

import math

import numpy as np


def shift_audio(*, audio: np.ndarray, sample_rate: int, shift_s: float) -> np.ndarray:
    """
    Time-shift audio with zero padding.

    - shift_s > 0: delay (prepend zeros)
    - shift_s < 0: advance (drop from the start)
    """
    sr = int(sample_rate)
    if sr <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
    n = int(audio.shape[0])
    if n == 0:
        return audio

    shift = int(round(float(shift_s) * float(sr)))
    if shift == 0:
        return audio
    if shift > 0:
        pad = np.zeros((shift,), dtype=audio.dtype)
        out = np.concatenate([pad, audio], axis=0)
        return out[:n]

    # shift < 0
    shift = int(-shift)
    if shift >= n:
        return np.zeros_like(audio)
    out = audio[shift:]
    pad = np.zeros((shift,), dtype=audio.dtype)
    return np.concatenate([out, pad], axis=0)


def add_noise_snr_db(*, audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """
    Add zero-mean Gaussian noise at a target SNR (dB) measured by RMS.
    """
    x = audio.astype(np.float32, copy=False)
    sig_rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    if sig_rms <= 0.0:
        return x

    snr = float(snr_db)
    if not math.isfinite(snr):
        raise ValueError(f"snr_db must be finite, got {snr_db}")

    noise_rms = sig_rms / float(10.0 ** (snr / 20.0))
    noise = rng.standard_normal(size=x.shape[0]).astype(np.float32) * float(noise_rms)
    return x + noise


def apply_silence_ratio(*, audio: np.ndarray, silence_ratio: float, rng: np.random.Generator) -> np.ndarray:
    """
    Zero out a contiguous random chunk of the signal.
    """
    r = float(silence_ratio)
    if r <= 0.0:
        return audio
    if r >= 1.0:
        return np.zeros_like(audio)

    n = int(audio.shape[0])
    k = max(1, int(round(r * n)))
    if k >= n:
        return np.zeros_like(audio)

    start = int(rng.integers(low=0, high=n - k + 1))
    out = audio.copy()
    out[start : start + k] = 0.0
    return out

