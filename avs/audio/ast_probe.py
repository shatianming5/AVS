from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification

from avs.audio.eventness import load_wav_mono


@dataclass(frozen=True)
class ASTProbeConfig:
    model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    pretrained: bool = True
    device: str = "cpu"
    dtype: str = "float32"


class ASTEventnessProbe:
    """
    Eventness probe using Audio Spectrogram Transformer (AST).

    This is a drop-in stand-in for the “PANNs/AudioMAE probe” described in `plan.md`:
    we only need per-second eventness scores `s(t)` to generate anchors.
    """

    def __init__(self, cfg: ASTProbeConfig = ASTProbeConfig()):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = getattr(torch, cfg.dtype)

        if cfg.pretrained:
            self.feature_extractor = ASTFeatureExtractor.from_pretrained(cfg.model_name)
            self.model = ASTForAudioClassification.from_pretrained(cfg.model_name).to(self.device)
        else:
            self.feature_extractor = ASTFeatureExtractor()
            config = ASTConfig(
                hidden_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=256,
                num_labels=10,
            )
            self.model = ASTForAudioClassification(config).to(self.device)

        self.model.eval()

    @torch.no_grad()
    def eventness_per_second(self, wav_path: Path, *, num_segments: int = 10) -> list[float]:
        audio, sr = load_wav_mono(wav_path)
        if sr != self.feature_extractor.sampling_rate:
            raise ValueError(
                f"expected sampling_rate={self.feature_extractor.sampling_rate}, got {sr} for {wav_path}. "
                "Re-extract audio as 16kHz mono first."
            )

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

        # Batch all 1-second segments into one forward for efficiency.
        segments = [audio[t * seg_len : (t + 1) * seg_len] for t in range(num_segments)]
        inputs = self.feature_extractor(segments, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device=self.device, dtype=self.dtype) for k, v in inputs.items()}
        logits = self.model(**inputs).logits  # [T, num_labels]
        probs = torch.softmax(logits, dim=-1)
        scores = probs.max(dim=-1).values.detach().cpu().numpy().astype(np.float32)

        # Stabilize: ensure python floats (JSON-friendly)
        return [float(x) for x in scores.tolist()]


def ast_eventness(wav_path: Path, *, pretrained: bool, num_segments: int = 10) -> list[float]:
    probe = ASTEventnessProbe(ASTProbeConfig(pretrained=pretrained))
    scores = probe.eventness_per_second(wav_path, num_segments=num_segments)
    return [float(x) for x in np.asarray(scores, dtype=np.float32)]
