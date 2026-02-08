from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import ClapConfig, ClapModel, ClapProcessor

from avs.audio.eventness import load_wav_mono


@dataclass(frozen=True)
class ClapProbeConfig:
    model_name: str = "laion/clap-htsat-fused"
    pretrained: bool = True
    device: str = "cpu"
    dtype: str = "float32"


def _l2_normalize(x: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + float(eps))


class ClapProbe:
    """
    Minimal CLAP (audio-text) probe for Stage-1 semantic scoring.

    Notes:
      - CLAP feature extractor expects 48kHz input for the common LAION checkpoints.
      - This probe is used for per-second embeddings/similarity, not for training.
    """

    def __init__(self, cfg: ClapProbeConfig = ClapProbeConfig()):
        self.cfg = cfg
        self.device = torch.device(str(cfg.device))
        self.dtype = getattr(torch, str(cfg.dtype))

        if bool(cfg.pretrained):
            self.processor = ClapProcessor.from_pretrained(str(cfg.model_name))
            self.model = ClapModel.from_pretrained(str(cfg.model_name)).to(self.device)
        else:
            # Random weights for smoke/debug only. Note that this is not expected to be useful.
            self.processor = ClapProcessor.from_pretrained(str(cfg.model_name))
            self.model = ClapModel(ClapConfig()).to(self.device)

        self.model.eval()
        self.target_sample_rate = int(self.processor.feature_extractor.sampling_rate)

    def _pad_trim_seconds(self, audio: np.ndarray, sample_rate: int, *, num_segments: int) -> np.ndarray:
        sr = int(sample_rate)
        if sr <= 0:
            raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
        seg_len = int(sr)
        target_len = seg_len * int(num_segments)
        x = np.asarray(audio, dtype=np.float32)
        n = int(x.shape[0])
        if n < target_len:
            x = np.pad(x, (0, target_len - n), mode="constant")
        elif n > target_len:
            x = x[:target_len]
        return x

    def _resample(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        sr = int(sample_rate)
        if sr == int(self.target_sample_rate):
            return np.asarray(audio, dtype=np.float32)
        x = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)  # [1, N]
        y = torchaudio.functional.resample(x, orig_freq=int(sr), new_freq=int(self.target_sample_rate))
        return y.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

    @torch.no_grad()
    def audio_embeddings_per_second(self, wav_path: Path, *, num_segments: int = 10, batch_size: int | None = None) -> np.ndarray:
        """
        Return per-second CLAP audio embeddings (shape: [T, D]) in the CLAP joint embedding space.
        """
        audio, sr = load_wav_mono(wav_path)
        audio = self._pad_trim_seconds(audio, int(sr), num_segments=int(num_segments))
        audio = self._resample(audio, int(sr))

        seg_len = int(self.target_sample_rate)
        target_len = seg_len * int(num_segments)
        n = int(audio.shape[0])
        if n < target_len:
            audio = np.pad(audio, (0, target_len - n), mode="constant")
        elif n > target_len:
            audio = audio[:target_len]

        segments = [audio[t * seg_len : (t + 1) * seg_len] for t in range(int(num_segments))]
        bs = None if batch_size is None else max(1, int(batch_size))
        if bs is None or bs >= int(num_segments):
            # transformers >=4.44 deprecated `audios=` in favor of `audio=`.
            inputs = self.processor(audio=segments, sampling_rate=int(self.target_sample_rate), return_tensors="pt")
            inputs = {k: (v.to(device=self.device, dtype=self.dtype) if v.is_floating_point() else v.to(device=self.device)) for k, v in inputs.items()}
            out = self.model.get_audio_features(**inputs)
            # transformers returns BaseModelOutputWithPooling; pooler_output is projected + normalized.
            emb = getattr(out, "pooler_output", out)
            emb = _l2_normalize(emb)
            return emb.detach().cpu().numpy().astype(np.float32, copy=False)

        outs: list[np.ndarray] = []
        for start in range(0, int(num_segments), int(bs)):
            segs = segments[start : start + int(bs)]
            inputs = self.processor(audio=segs, sampling_rate=int(self.target_sample_rate), return_tensors="pt")
            inputs = {k: (v.to(device=self.device, dtype=self.dtype) if v.is_floating_point() else v.to(device=self.device)) for k, v in inputs.items()}
            out = self.model.get_audio_features(**inputs)
            emb = getattr(out, "pooler_output", out)
            emb = _l2_normalize(emb).detach().cpu().numpy().astype(np.float32, copy=False)
            outs.append(emb)
        return np.concatenate(outs, axis=0).astype(np.float32, copy=False)

    @torch.no_grad()
    def text_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Return CLAP text embeddings (shape: [N, D]) in the CLAP joint embedding space.
        """
        inputs = self.processor(text=[str(x) for x in texts], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device=self.device) for k, v in inputs.items()}
        out = self.model.get_text_features(**inputs)
        emb = getattr(out, "pooler_output", out)
        emb = _l2_normalize(emb)
        return emb.detach().cpu().numpy().astype(np.float32, copy=False)
