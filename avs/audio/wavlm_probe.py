from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor, WavLMConfig, WavLMModel


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


@dataclass(frozen=True)
class WavLMProbeConfig:
    model_name: str = "microsoft/wavlm-base-plus"
    pretrained: bool = True
    device: str = "cpu"
    dtype: str = "float32"


class WavLMEmbeddingProbe:
    """
    Per-second audio embedding probe using WavLM.

    This is used as a feature backbone for supervised Stage-1 eventness heads
    (see `avs.experiments.ave_p0_sweep`).
    """

    def __init__(self, cfg: WavLMProbeConfig = WavLMProbeConfig()):
        self.cfg = cfg
        self.device = torch.device(str(cfg.device))
        self.dtype = getattr(torch, str(cfg.dtype))

        if bool(cfg.pretrained):
            # Prefer local cache first: repeated sweeps/reruns should not depend on HF hub availability.
            # If cache is missing/incomplete, fall back to the online code path.
            try:
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(str(cfg.model_name), local_files_only=True)
                self.model = WavLMModel.from_pretrained(str(cfg.model_name), local_files_only=True).to(
                    self.device, dtype=self.dtype
                )
            except Exception:
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(str(cfg.model_name))
                self.model = WavLMModel.from_pretrained(str(cfg.model_name)).to(self.device, dtype=self.dtype)
        else:
            # Smoke-friendly tiny random model (no downloads).
            self.feature_extractor = Wav2Vec2FeatureExtractor()
            config = WavLMConfig(
                hidden_size=96,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=192,
                conv_dim=(32, 32, 32),
                conv_stride=(5, 2, 2),
                conv_kernel=(10, 3, 3),
            )
            self.model = WavLMModel(config).to(self.device, dtype=self.dtype)

        self.model.eval()

        sr = int(getattr(self.feature_extractor, "sampling_rate", 16000))
        if sr <= 0:
            raise ValueError(f"invalid feature_extractor sampling_rate={sr}")
        self.sample_rate = sr

    @torch.no_grad()
    def embeddings_per_second_by_clip_ids(
        self,
        *,
        clip_ids: list[str],
        processed_dir: Path,
        num_segments: int = 10,
        batch_size: int | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Compute per-second embeddings for each clip_id.

        Returns:
          dict[clip_id] -> np.ndarray of shape [T, D] (T=num_segments).
        """
        if not clip_ids:
            return {}

        dev = self.device
        if batch_size is None:
            batch_size = 4 if dev.type == "cpu" else 16
        batch_size = max(1, int(batch_size))

        out: dict[str, np.ndarray] = {}

        n = int(len(clip_ids))
        steps = int(math.ceil(n / float(batch_size)))
        for step in range(steps):
            batch = clip_ids[step * batch_size : (step + 1) * batch_size]

            segs: list[np.ndarray] = []
            for cid in batch:
                wav_path = processed_dir / cid / "audio.wav"
                if not wav_path.exists():
                    raise FileNotFoundError(f"missing wav: {wav_path}")
                wf = _load_wav_10s_mono(wav_path=wav_path, sample_rate=int(self.sample_rate), num_segments=int(num_segments))
                # Flatten into T segments of 1 second each.
                sr = int(self.sample_rate)
                for t in range(int(num_segments)):
                    seg = wf[:, t * sr : (t + 1) * sr]  # [1, sr]
                    segs.append(seg.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False))

            inputs = self.feature_extractor(
                segs,
                sampling_rate=int(self.sample_rate),
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs["input_values"].to(device=dev, dtype=self.dtype)

            out_model = self.model(input_values, return_dict=True)
            hs = out_model.last_hidden_state  # [B*T, S, D]
            emb = hs.mean(dim=1)  # [B*T, D]
            emb_np = emb.detach().cpu().numpy().astype(np.float32, copy=False)
            emb_np = emb_np.reshape(len(batch), int(num_segments), -1)

            for i, cid in enumerate(batch):
                out[str(cid)] = emb_np[i]

            if (step + 1) % 10 == 0 or (step + 1) == steps:
                print(f"[wavlm] {min((step + 1) * batch_size, n)}/{n} clips", flush=True)

        return out
