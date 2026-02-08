from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from avs.audio.eventness import load_wav_mono


def _resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(f"invalid sample rates: orig_sr={orig_sr}, target_sr={target_sr}")
    if audio.ndim != 1:
        raise ValueError(f"expected mono audio 1D array, got shape={audio.shape}")

    target_len = int(round(audio.shape[0] * float(target_sr) / float(orig_sr)))
    if target_len < 2:
        raise ValueError(f"audio too short to resample: n={audio.shape[0]} samples")

    x_old = np.arange(audio.shape[0], dtype=np.float64)
    x_new = np.linspace(0.0, float(audio.shape[0] - 1), num=target_len, dtype=np.float64)
    out = np.interp(x_new, x_old, audio.astype(np.float64)).astype(np.float32)
    return out


@dataclass(frozen=True)
class PANNsProbeConfig:
    pretrained: bool = True
    checkpoint_path: Path | None = None
    device: str = "cpu"
    dtype: str = "float32"

    # Matches `panns_inference.inference.AudioTagging` defaults.
    sample_rate: int = 32000
    window_size: int = 1024
    hop_size: int = 320
    mel_bins: int = 64
    fmin: int = 50
    fmax: int = 14000

    # Guardrail: `panns_inference` triggers auto-download when checkpoint < 3e8 bytes.
    min_checkpoint_bytes: int = 300_000_000


class PANNsEventnessProbe:
    """
    Eventness probe using PANNs Cnn14 (AudioSet).

    For smoke runs, set `pretrained=False` (random weights, no downloads).
    For real runs, pass a real checkpoint via `checkpoint_path` (no implicit download).
    """

    def __init__(self, cfg: PANNsProbeConfig = PANNsProbeConfig()):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = getattr(torch, cfg.dtype)

        try:
            from panns_inference.config import classes_num
            from panns_inference.models import Cnn14
        except Exception as e:  # noqa: BLE001 - optional dependency
            raise ImportError(
                "PANNs backend requires `panns-inference` (and its deps). "
                "Install it and re-run, or use --eventness-method energy/ast."
            ) from e

        self.model = Cnn14(
            sample_rate=cfg.sample_rate,
            window_size=cfg.window_size,
            hop_size=cfg.hop_size,
            mel_bins=cfg.mel_bins,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
            classes_num=classes_num,
        ).to(self.device)

        if cfg.pretrained:
            ckpt_path = cfg.checkpoint_path or (Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth")
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"PANNs checkpoint not found: {ckpt_path}. "
                    "Download it from Zenodo (Cnn14_mAP=0.431.pth) and pass --panns-checkpoint, "
                    "or use --panns-random for a no-download smoke run."
                )
            if ckpt_path.stat().st_size < cfg.min_checkpoint_bytes:
                raise ValueError(
                    f"PANNs checkpoint looks incomplete (size={ckpt_path.stat().st_size} bytes): {ckpt_path}. "
                    "Re-download and retry."
                )

            checkpoint = torch.load(ckpt_path, map_location=self.device)
            state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            self.model.load_state_dict(state)

        self.model.eval()

    @torch.no_grad()
    def clipwise_output_from_array(self, audio: np.ndarray, sample_rate: int, *, num_segments: int = 10) -> np.ndarray:
        """
        Compute per-second PANNs clipwise outputs (AudioSet class probabilities).

        Returns:
          np.ndarray[float32] with shape [T=num_segments, C=527].

        This is useful for supervised calibration (`panns_lr`) where we want a train-split-only mapping
        from PANNs outputs → a single scalar eventness logit.
        """
        audio = _resample_linear(audio, int(sample_rate), self.cfg.sample_rate)

        seg_len = int(self.cfg.sample_rate)
        target_len = seg_len * int(num_segments)

        # Real-world WAVs extracted via ffmpeg can be off by a few samples due to rounding.
        # For AVE's fixed 10s protocol, we pad/trim to exactly `num_segments` seconds.
        n = int(audio.shape[0])
        if n < target_len:
            pad = target_len - n
            audio = np.pad(audio, (0, pad), mode="constant")
        elif n > target_len:
            audio = audio[:target_len]

        segments = np.stack([audio[t * seg_len : (t + 1) * seg_len] for t in range(num_segments)], axis=0)
        x = torch.from_numpy(segments).to(device=self.device, dtype=self.dtype)
        out = self.model(x, None)
        probs = out["clipwise_output"]  # [T, classes]
        return probs.detach().cpu().numpy().astype(np.float32, copy=False)

    @torch.no_grad()
    def embeddings_from_array(self, audio: np.ndarray, sample_rate: int, *, num_segments: int = 10) -> np.ndarray:
        """
        Compute per-second PANNs embeddings.

        Returns:
          np.ndarray[float32] with shape [T=num_segments, D=2048].
        """
        audio = _resample_linear(audio, int(sample_rate), self.cfg.sample_rate)

        seg_len = int(self.cfg.sample_rate)
        target_len = seg_len * int(num_segments)

        n = int(audio.shape[0])
        if n < target_len:
            pad = target_len - n
            audio = np.pad(audio, (0, pad), mode="constant")
        elif n > target_len:
            audio = audio[:target_len]

        segments = np.stack([audio[t * seg_len : (t + 1) * seg_len] for t in range(num_segments)], axis=0)
        x = torch.from_numpy(segments).to(device=self.device, dtype=self.dtype)
        out = self.model(x, None)
        emb = out["embedding"]  # [T, 2048]
        return emb.detach().cpu().numpy().astype(np.float32, copy=False)

    @torch.no_grad()
    def eventness_from_array(self, audio: np.ndarray, sample_rate: int, *, num_segments: int = 10) -> list[float]:
        """
        Compute per-second eventness from an in-memory waveform (useful for augmentations).
        """
        probs = self.clipwise_output_from_array(audio, int(sample_rate), num_segments=int(num_segments))
        scores = probs.max(axis=-1).astype(np.float32, copy=False)
        return [float(x) for x in scores.tolist()]

    @torch.no_grad()
    def eventness_per_second(self, wav_path: Path, *, num_segments: int = 10) -> list[float]:
        audio, sr = load_wav_mono(wav_path)
        return self.eventness_from_array(audio, int(sr), num_segments=num_segments)

    @torch.no_grad()
    def clipwise_output_per_second(self, wav_path: Path, *, num_segments: int = 10) -> np.ndarray:
        audio, sr = load_wav_mono(wav_path)
        return self.clipwise_output_from_array(audio, int(sr), num_segments=int(num_segments))

    @torch.no_grad()
    def embeddings_per_second(self, wav_path: Path, *, num_segments: int = 10) -> np.ndarray:
        audio, sr = load_wav_mono(wav_path)
        return self.embeddings_from_array(audio, int(sr), num_segments=int(num_segments))


def panns_eventness(
    wav_path: Path,
    *,
    pretrained: bool,
    checkpoint_path: Path | None = None,
    num_segments: int = 10,
) -> list[float]:
    probe = PANNsEventnessProbe(PANNsProbeConfig(pretrained=pretrained, checkpoint_path=checkpoint_path))
    scores = probe.eventness_per_second(wav_path, num_segments=num_segments)
    return [float(x) for x in np.asarray(scores, dtype=np.float32)]
