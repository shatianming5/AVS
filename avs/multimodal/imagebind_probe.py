from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torchvision import transforms

from avs.utils.scores import minmax_01


def _load_wav_10s_mono(*, wav_path: Path, sample_rate: int, num_segments: int) -> torch.Tensor:
    """
    Load WAV and normalize to the AVE fixed-length protocol (10 seconds by default).

    Returns:
      waveform: torch.Tensor of shape [1, N] at `sample_rate`.
    """
    waveform, sr = torchaudio.load(str(wav_path))  # [C, N]
    if int(sr) != int(sample_rate):
        waveform = torchaudio.functional.resample(waveform, orig_freq=int(sr), new_freq=int(sample_rate))
    if int(waveform.ndim) != 2:
        raise ValueError(f"unexpected waveform ndim={waveform.ndim} for {wav_path}")
    if int(waveform.shape[0]) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    target_len = int(sample_rate) * int(num_segments)
    n = int(waveform.shape[1])
    if n < target_len:
        waveform = F.pad(waveform, (0, target_len - n), mode="constant", value=0.0)
    elif n > target_len:
        waveform = waveform[:, :target_len]
    return waveform


def _imagebind_audio_mels_centered_2s(
    *,
    wav_path: Path,
    device: torch.device,
    num_segments: int,
    sample_rate: int = 16000,
    num_mel_bins: int = 128,
    target_length: int = 204,
    mean: float = -4.268,
    std: float = 9.138,
) -> torch.Tensor:
    """
    ImageBind audio transform, but made deterministic for AVE:

    For each second t in [0..num_segments-1], use a 2-second window centered at (t+0.5)s:
      window = [t-0.5, t+1.5], with zero-padding at boundaries.

    Returns:
      mels: torch.Tensor of shape [T, 1, num_mel_bins, target_length] on `device`.
    """
    # Import locally so ImageBind is an optional dependency for most of the repo.
    from imagebind.data import waveform2melspec

    wf = _load_wav_10s_mono(wav_path=wav_path, sample_rate=int(sample_rate), num_segments=int(num_segments))

    pad = int(sample_rate // 2)  # 0.5s
    wf = F.pad(wf, (pad, pad), mode="constant", value=0.0)  # [1, (T+1)s]

    normalize = transforms.Normalize(mean=float(mean), std=float(std))
    out: list[torch.Tensor] = []
    for t in range(int(num_segments)):
        start = int(t) * int(sample_rate)
        seg = wf[:, start : start + 2 * int(sample_rate)]  # 2s window
        mel = waveform2melspec(seg, int(sample_rate), int(num_mel_bins), int(target_length))  # [1, mel, frames]
        out.append(normalize(mel).to(device))
    return torch.stack(out, dim=0)


def compute_imagebind_av_sim_scores_by_clip(
    *,
    clip_ids: list[str],
    processed_dir: Path,
    num_segments: int = 10,
    device: str = "cpu",
    pretrained: bool = True,
    batch_size: int | None = None,
) -> dict[str, list[float]]:
    """
    Compute per-second ImageBind AV-consistency scores:
      s(t) = cosine(emb_audio(t), emb_image(t))

    Notes:
    - This is intended as a Stage-1 eventness signal (anchor proposal). We min-max normalize per-clip
      to make downstream confidence thresholds comparable to other Stage-1 methods.
    - This function is compute-heavy; use `batch_size` and a CUDA device when available.
    """
    if not clip_ids:
        return {}

    dev = torch.device(str(device))
    if batch_size is None:
        batch_size = 2 if dev.type == "cpu" else 8

    # Local imports so ImageBind is optional unless this path is used.
    from imagebind.data import load_and_transform_vision_data
    from imagebind.models.imagebind_model import ModalityType, imagebind_huge

    # Build model once (downloads weights on first use if pretrained=True).
    model = imagebind_huge(pretrained=bool(pretrained)).eval().to(dev)

    out: dict[str, list[float]] = {}
    # Avoid OOM by keeping batches small; the model is large.
    n = int(len(clip_ids))
    steps = int(math.ceil(n / int(batch_size)))
    for step in range(steps):
        batch = clip_ids[step * int(batch_size) : (step + 1) * int(batch_size)]

        # Vision inputs: flatten to (B*T) image paths.
        img_paths: list[str] = []
        for cid in batch:
            frames_dir = processed_dir / cid / "frames"
            for t in range(int(num_segments)):
                p = frames_dir / f"{t}.jpg"
                if not p.exists():
                    raise FileNotFoundError(f"missing frame: {p}")
                img_paths.append(str(p))

        vision = load_and_transform_vision_data(img_paths, dev)  # [B*T, 3, 224, 224]

        # Audio inputs: build deterministic 2s-centered mel specs per clip, then flatten to (B*T).
        audio_mels: list[torch.Tensor] = []
        for cid in batch:
            wav_path = processed_dir / cid / "audio.wav"
            if not wav_path.exists():
                raise FileNotFoundError(f"missing wav: {wav_path}")
            mel = _imagebind_audio_mels_centered_2s(
                wav_path=wav_path,
                device=dev,
                num_segments=int(num_segments),
            )  # [T, 1, 128, 204]
            audio_mels.append(mel)
        audio = torch.cat(audio_mels, dim=0)  # [B*T, 1, 128, 204]

        with torch.no_grad():
            emb_v = model({ModalityType.VISION: vision})[ModalityType.VISION]  # [B*T, D]
            emb_a = model({ModalityType.AUDIO: audio})[ModalityType.AUDIO]  # [B*T, D]

        emb_v = F.normalize(emb_v, dim=-1)
        emb_a = F.normalize(emb_a, dim=-1)
        sim = (emb_v * emb_a).sum(dim=-1).view(len(batch), int(num_segments))  # [B, T]

        for i, cid in enumerate(batch):
            s = sim[i].detach().cpu().numpy().astype(np.float32, copy=False)
            out[cid] = minmax_01([float(x) for x in s.tolist()])

        if (step + 1) % 10 == 0 or (step + 1) == steps:
            print(f"[imagebind_av_sim] {min((step+1)*int(batch_size), n)}/{n} clips", flush=True)

    return out

