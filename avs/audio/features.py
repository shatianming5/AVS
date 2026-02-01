from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from avs.audio.eventness import load_wav_mono


def _pad_or_trim_audio(audio: np.ndarray, *, target_len: int) -> np.ndarray:
    n = int(audio.shape[0])
    if n < int(target_len):
        return np.pad(audio, (0, int(target_len) - n), mode="constant")
    if n > int(target_len):
        return audio[: int(target_len)]
    return audio


def audio_features_per_second(
    wav_path: Path,
    *,
    num_segments: int = 10,
    feature_set: str = "basic",
) -> np.ndarray:
    """
    Compute per-second audio features for a fixed-length protocol (AVE: 10s → 10 segments).

    Output shape: [num_segments, F] float32.

    Feature sets:
      - "basic": log-energy + energy-delta + waveform stats + simple spectral stats + 4 band energies (F=10)
      - "fbank_stats": kaldi-style log-fbank mean+std per second (F=80)
    """
    feature_set = str(feature_set)
    if feature_set not in ("basic", "fbank_stats"):
        raise ValueError(f"unsupported feature_set: {feature_set}")

    audio, sr = load_wav_mono(wav_path)
    seg_len = int(sr)
    target_len = seg_len * int(num_segments)
    audio = _pad_or_trim_audio(audio, target_len=target_len)

    if feature_set == "fbank_stats":
        import torch
        import torchaudio

        wav = torch.from_numpy(audio).to(dtype=torch.float32).unsqueeze(0)  # [1, T_total]
        fb = torchaudio.compliance.kaldi.fbank(
            wav,
            num_mel_bins=40,
            sample_frequency=float(sr),
            frame_length=25.0,
            frame_shift=10.0,
            dither=0.0,
            snip_edges=False,
        )
        # [num_frames, 40], with snip_edges=False this is typically divisible by num_segments (AVE: 1000 frames / 10s).
        num_frames = int(fb.shape[0])
        if num_frames <= 0:
            raise ValueError(f"fbank produced empty features for {wav_path}")
        if num_frames % int(num_segments) != 0:
            # Guardrail: pad (or truncate as a last resort) to be safely reshaped.
            frames_per_seg = int(math.ceil(num_frames / float(num_segments)))
            target = int(num_segments) * int(frames_per_seg)
            if num_frames < target:
                pad = fb[-1:, :].repeat(target - num_frames, 1)
                fb = torch.cat([fb, pad], dim=0)
                num_frames = int(fb.shape[0])
            elif num_frames > target:
                fb = fb[:target]
                num_frames = int(fb.shape[0])

        frames_per_seg = num_frames // int(num_segments)
        fb = fb.reshape(int(num_segments), int(frames_per_seg), 40)
        mean = fb.mean(dim=1)
        std = fb.std(dim=1, unbiased=False)
        out = torch.cat([mean, std], dim=1).detach().cpu().numpy().astype(np.float32, copy=False)
        if out.shape != (int(num_segments), 80):
            raise AssertionError(f"bug: expected fbank_stats shape ({num_segments}, 80), got {tuple(out.shape)}")
        return out

    freqs = np.fft.rfftfreq(seg_len, d=1.0 / float(sr))
    nyq = 0.5 * float(sr)

    # Coarse log-band energies (Hz).
    band_edges = [0.0, 250.0, 1000.0, 4000.0, nyq + 1.0]
    band_masks = [(freqs >= lo) & (freqs < hi) for lo, hi in zip(band_edges[:-1], band_edges[1:], strict=True)]

    rows: list[list[float]] = []
    prev_log_energy: float | None = None
    for t in range(int(num_segments)):
        seg = audio[t * seg_len : (t + 1) * seg_len]

        ms = float(np.mean(seg * seg))
        log_energy = float(math.log(ms + 1e-12))
        energy_delta = 0.0 if prev_log_energy is None else float(abs(log_energy - float(prev_log_energy)))
        prev_log_energy = log_energy

        wave_std = float(np.std(seg))
        zcr = float(np.mean((seg[:-1] * seg[1:]) < 0.0)) if seg_len > 1 else 0.0

        # Spectral stats from power spectrum.
        spec = np.fft.rfft(seg.astype(np.float64, copy=False), n=seg_len)
        power = (np.abs(spec) ** 2).astype(np.float64, copy=False)
        if power.shape[0] > 0:
            power[0] = 0.0  # ignore DC
        total = float(power.sum())

        if total <= 0.0:
            centroid = 0.0
            rolloff = 0.0
        else:
            centroid = float((freqs * power).sum() / total) / nyq  # normalize to [0,1]
            cumsum = np.cumsum(power)
            idx = int(np.searchsorted(cumsum, 0.85 * total))
            idx = min(max(idx, 0), int(freqs.shape[0] - 1))
            rolloff = float(freqs[idx]) / nyq

        band_logs: list[float] = []
        for m in band_masks:
            e = float(power[m].sum())
            band_logs.append(float(math.log(e + 1e-12)))

        rows.append(
            [
                log_energy,
                energy_delta,
                wave_std,
                zcr,
                centroid,
                rolloff,
                *band_logs,
            ]
        )

    out = np.asarray(rows, dtype=np.float32)
    if out.shape != (int(num_segments), 10):
        raise AssertionError(f"bug: expected features shape ({num_segments}, 10), got {tuple(out.shape)}")
    return out
