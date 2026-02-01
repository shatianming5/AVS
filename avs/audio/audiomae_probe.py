from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from avs.audio.eventness import load_wav_mono


def _pad_or_trim_frames(mel_bt128: torch.Tensor, *, target_frames: int) -> torch.Tensor:
    """
    Pad or trim a tensor shaped [B, frames, 128] along the `frames` dim to `target_frames`.
    """
    if mel_bt128.ndim != 3:
        raise ValueError(f"expected mel [B, frames, 128], got shape={tuple(mel_bt128.shape)}")
    frames = int(mel_bt128.shape[1])
    if frames == int(target_frames):
        return mel_bt128
    if frames > int(target_frames):
        return mel_bt128[:, : int(target_frames), :]
    return torch.nn.functional.pad(mel_bt128, (0, 0, 0, int(target_frames) - frames), mode="constant", value=0.0)


def _log_mel_1024x128(audio_1d: np.ndarray, sr: int, *, device: torch.device) -> torch.Tensor:
    """
    Compute a log-mel spectrogram shaped as (1, 1, 1024, 128) suitable for an AudioMAE-style ViT.

    Note: this is a lightweight approximation intended for probing/eventness, not a faithful AudioMAE reproduction.
    """
    import torchaudio

    wav = torch.from_numpy(audio_1d.astype(np.float32, copy=False)).to(device=device)
    if wav.ndim != 1:
        raise ValueError(f"expected mono waveform 1D, got shape={tuple(wav.shape)}")
    wav = wav.unsqueeze(0)  # [1, T]

    if sr != 16000:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
        sr = 16000

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=1024,
        win_length=1024,
        hop_length=160,
        center=True,
        power=2.0,
        n_mels=128,
    ).to(device=device)(wav)  # [1, 128, frames]

    mel = torch.clamp(mel, min=1e-10).log()
    mel = mel.transpose(1, 2)  # [1, frames, 128]
    mel = _pad_or_trim_frames(mel, target_frames=1024)
    return mel.unsqueeze(1)  # [1, 1, 1024, 128]


@dataclass(frozen=True)
class AudioMAEProbeConfig:
    pretrained: bool = False
    checkpoint_path: Path | None = None
    device: str = "cpu"
    dtype: str = "float32"

    # Input shape used by AudioMAE-style ViT (T x F).
    time_frames: int = 1024
    mel_bins: int = 128
    patch_size: int = 16

    # Lightweight default architecture (fast on CPU).
    embed_dim: int = 256
    depth: int = 4
    num_heads: int = 4


class AudioMAEEventnessProbe:
    """
    AudioMAE-style ViT probe producing per-segment eventness scores s(t) in [0,1].

    - For smoke runs: use random weights (cfg.pretrained=False).
    - For real runs: provide `checkpoint_path` and set cfg.pretrained=True (no implicit downloads).
    """

    def __init__(self, cfg: AudioMAEProbeConfig = AudioMAEProbeConfig()):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = getattr(torch, cfg.dtype)

        try:
            import timm
        except Exception as e:  # noqa: BLE001 - optional dependency
            raise ImportError("AudioMAE probe requires `timm`. Install it or use --eventness-method energy/ast/panns.") from e

        # Treat log-mel as a single-channel image of size (time_frames, mel_bins).
        self.vit = timm.models.vision_transformer.VisionTransformer(
            img_size=(cfg.time_frames, cfg.mel_bins),
            patch_size=cfg.patch_size,
            in_chans=1,
            num_classes=0,  # no classifier head
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
        ).to(self.device, dtype=self.dtype)

        self.head = torch.nn.Linear(cfg.embed_dim, 1).to(self.device, dtype=self.dtype)

        if cfg.pretrained:
            if cfg.checkpoint_path is None:
                raise ValueError("AudioMAE pretrained requested but checkpoint_path is None.")
            if not cfg.checkpoint_path.exists():
                raise FileNotFoundError(f"AudioMAE checkpoint not found: {cfg.checkpoint_path}")
            state = torch.load(cfg.checkpoint_path, map_location=self.device)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            missing, unexpected = self.vit.load_state_dict(state, strict=False)
            # Allow missing/unexpected keys since checkpoints vary.
            _ = (missing, unexpected)

        self.vit.eval()
        self.head.eval()

    @torch.no_grad()
    def eventness_per_second(self, wav_path: Path, *, num_segments: int = 10) -> list[float]:
        audio, sr = load_wav_mono(wav_path)
        seg_len = int(sr)
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
            x = _log_mel_1024x128(seg, sr, device=self.device)  # [1,1,1024,128]
            emb = self.vit(x)  # [1, D]
            logit = self.head(emb).squeeze(-1)
            prob = torch.sigmoid(logit).detach().cpu().numpy().astype(np.float32)
            scores.append(float(prob.reshape(-1)[0]))
        return scores


def audiomae_eventness(
    wav_path: Path,
    *,
    pretrained: bool,
    checkpoint_path: Path | None = None,
    num_segments: int = 10,
) -> list[float]:
    probe = AudioMAEEventnessProbe(AudioMAEProbeConfig(pretrained=pretrained, checkpoint_path=checkpoint_path))
    scores = probe.eventness_per_second(wav_path, num_segments=num_segments)
    return [float(x) for x in np.asarray(scores, dtype=np.float32)]
