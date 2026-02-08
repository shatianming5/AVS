from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from avs.audio.eventness import load_wav_mono


@dataclass(frozen=True)
class WebRtcVadConfig:
    aggressiveness: int = 2  # 0..3 (higher => more aggressive filtering)
    frame_ms: int = 30  # 10|20|30
    target_sample_rate: int = 16000


def webrtcvad_speech_ratio_from_array(
    audio: np.ndarray,
    sample_rate: int,
    *,
    num_segments: int = 10,
    cfg: WebRtcVadConfig = WebRtcVadConfig(),
) -> list[float]:
    try:
        import webrtcvad  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "webrtcvad is required for WebRTC-VAD speech ratio. Install it via `pip install webrtcvad==2.0.10`."
        ) from e

    num_segments = int(num_segments)
    if num_segments <= 0:
        return []

    sr = int(sample_rate)
    tgt_sr = int(cfg.target_sample_rate)
    if sr <= 0 or tgt_sr <= 0:
        raise ValueError(f"invalid sample_rate: src={sr}, tgt={tgt_sr}")

    x = np.asarray(audio, dtype=np.float32)
    if sr != tgt_sr:
        from scipy.signal import resample_poly

        g = int(math.gcd(sr, tgt_sr))
        up = int(tgt_sr // g)
        down = int(sr // g)
        x = resample_poly(x, up, down).astype(np.float32, copy=False)
        sr = int(tgt_sr)

    # Pad/trim to fixed-length protocol (like other AVE audio utilities).
    target_len = int(sr) * int(num_segments)
    n = int(x.shape[0])
    if n < target_len:
        x = np.pad(x, (0, target_len - n), mode="constant")
    elif n > target_len:
        x = x[:target_len]

    frame_ms = int(cfg.frame_ms)
    if frame_ms not in (10, 20, 30):
        raise ValueError(f"frame_ms must be 10|20|30, got {frame_ms}")
    if (int(sr) * int(frame_ms)) % 1000 != 0:
        raise ValueError(f"sample_rate={sr} is incompatible with frame_ms={frame_ms} for WebRTC-VAD")
    frame_len = (int(sr) * int(frame_ms)) // 1000

    # Ensure length is a multiple of frame_len.
    pad = (-int(x.shape[0])) % int(frame_len)
    if pad:
        x = np.pad(x, (0, pad), mode="constant")

    pcm = np.clip(x, -1.0, 1.0)
    pcm = (pcm * 32768.0).round().clip(-32768.0, 32767.0).astype(np.int16, copy=False)

    vad = webrtcvad.Vad(int(cfg.aggressiveness))

    speech = np.zeros((int(num_segments),), dtype=np.int64)
    total = np.zeros((int(num_segments),), dtype=np.int64)

    num_frames = int(pcm.shape[0]) // int(frame_len)
    for i in range(int(num_frames)):
        start = int(i) * int(frame_len)
        end = start + int(frame_len)
        frame = pcm[start:end]
        if int(frame.shape[0]) != int(frame_len):
            continue
        center_s = (float(start) + float(frame_len) / 2.0) / float(sr)
        t = int(center_s)
        if t < 0:
            continue
        if t >= int(num_segments):
            break
        is_speech = bool(vad.is_speech(frame.tobytes(), int(sr)))
        speech[int(t)] += int(is_speech)
        total[int(t)] += 1

    out: list[float] = []
    for t in range(int(num_segments)):
        denom = int(total[t])
        out.append(float(speech[t]) / float(denom) if denom > 0 else 0.0)
    return out


def webrtcvad_speech_ratio_per_second(
    wav_path: Path,
    *,
    num_segments: int = 10,
    cfg: WebRtcVadConfig = WebRtcVadConfig(),
) -> list[float]:
    audio, sr = load_wav_mono(Path(wav_path))
    return webrtcvad_speech_ratio_from_array(audio, int(sr), num_segments=int(num_segments), cfg=cfg)

