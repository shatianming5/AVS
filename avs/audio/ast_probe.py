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
    # AudioSet is multi-label; prefer sigmoid-based "peakiness" instead of softmax.
    # Keep this configurable for ablations.
    score_mode: str = "sigmoid_max"  # sigmoid_max|softmax_max|sigmoid_nonsilence


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

    def _segments_from_array(self, audio: np.ndarray, sample_rate: int, *, num_segments: int) -> list[np.ndarray]:
        sr = int(sample_rate)
        if sr != self.feature_extractor.sampling_rate:
            raise ValueError(
                f"expected sampling_rate={self.feature_extractor.sampling_rate}, got {sr}. "
                "Re-extract audio as 16kHz mono first."
            )

        seg_len = int(sr)
        target_len = seg_len * int(num_segments)

        # Real-world WAVs extracted via ffmpeg can be off by a few samples due to rounding.
        # For AVE's fixed 10s protocol, we pad/trim to exactly `num_segments` seconds.
        x = np.asarray(audio, dtype=np.float32)
        n = int(x.shape[0])
        if n < target_len:
            pad = target_len - n
            x = np.pad(x, (0, pad), mode="constant")
        elif n > target_len:
            x = x[:target_len]

        return [x[t * seg_len : (t + 1) * seg_len] for t in range(int(num_segments))]

    @torch.no_grad()
    def eventness_per_second(self, wav_path: Path, *, num_segments: int = 10) -> list[float]:
        audio, sr = load_wav_mono(wav_path)
        return self.eventness_from_array(audio, int(sr), num_segments=int(num_segments))

    @torch.no_grad()
    def eventness_from_array(self, audio: np.ndarray, sample_rate: int, *, num_segments: int = 10) -> list[float]:
        segments = self._segments_from_array(audio, int(sample_rate), num_segments=int(num_segments))
        inputs = self.feature_extractor(segments, sampling_rate=int(sample_rate), return_tensors="pt")
        inputs = {k: v.to(device=self.device, dtype=self.dtype) for k, v in inputs.items()}
        logits = self.model(**inputs).logits  # [T, num_labels]

        mode = str(self.cfg.score_mode)
        if mode == "sigmoid_max":
            probs = torch.sigmoid(logits)
            scores_t = probs.max(dim=-1).values
        elif mode == "softmax_max":
            probs = torch.softmax(logits, dim=-1)
            scores_t = probs.max(dim=-1).values
        elif mode == "sigmoid_nonsilence":
            probs = torch.sigmoid(logits)
            sil_idx = None
            for k, v in self.model.config.id2label.items():
                if str(v).strip().lower() == "silence":
                    sil_idx = int(k)
                    break
            if sil_idx is None:
                raise ValueError("AST model config has no 'Silence' label; cannot use score_mode='sigmoid_nonsilence'")
            scores_t = 1.0 - probs[:, sil_idx]
        else:
            raise ValueError(f"unknown score_mode={mode!r}; expected 'sigmoid_max', 'softmax_max', or 'sigmoid_nonsilence'")

        scores = scores_t.detach().cpu().numpy().astype(np.float32)

        # Stabilize: ensure python floats (JSON-friendly)
        return [float(x) for x in scores.tolist()]

    @torch.no_grad()
    def logits_per_second(self, wav_path: Path, *, num_segments: int = 10) -> np.ndarray:
        """
        Return raw per-second logits (shape: [T, num_labels]) for downstream probing / linear calibration.
        """
        audio, sr = load_wav_mono(wav_path)
        return self.logits_from_array(audio, int(sr), num_segments=int(num_segments))

    @torch.no_grad()
    def logits_from_array(self, audio: np.ndarray, sample_rate: int, *, num_segments: int = 10) -> np.ndarray:
        segments = self._segments_from_array(audio, int(sample_rate), num_segments=int(num_segments))
        inputs = self.feature_extractor(segments, sampling_rate=int(sample_rate), return_tensors="pt")
        inputs = {k: v.to(device=self.device, dtype=self.dtype) for k, v in inputs.items()}
        logits = self.model(**inputs).logits  # [T, num_labels]
        return logits.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def embeddings_per_second(self, wav_path: Path, *, num_segments: int = 10) -> np.ndarray:
        """
        Return per-second AST embeddings (shape: [T, hidden_size]) for downstream probing.

        Uses the last-layer CLS token as the embedding.
        """
        audio, sr = load_wav_mono(wav_path)
        return self.embeddings_from_array(audio, int(sr), num_segments=int(num_segments))

    @torch.no_grad()
    def embeddings_from_array(self, audio: np.ndarray, sample_rate: int, *, num_segments: int = 10) -> np.ndarray:
        segments = self._segments_from_array(audio, int(sample_rate), num_segments=int(num_segments))
        inputs = self.feature_extractor(segments, sampling_rate=int(sample_rate), return_tensors="pt")
        inputs = {k: v.to(device=self.device, dtype=self.dtype) for k, v in inputs.items()}
        out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states
        if hs is None or len(hs) == 0:
            raise ValueError("AST model did not return hidden_states; cannot compute embeddings")
        last = hs[-1]  # [T, S, H]
        if int(last.ndim) != 3 or int(last.shape[0]) != int(num_segments):
            raise ValueError(f"unexpected AST hidden state shape: {tuple(int(x) for x in last.shape)}")
        emb = last[:, 0, :]  # CLS token
        return emb.detach().cpu().numpy().astype(np.float32)


def ast_eventness(
    wav_path: Path,
    *,
    pretrained: bool,
    num_segments: int = 10,
    score_mode: str = "sigmoid_max",
) -> list[float]:
    probe = ASTEventnessProbe(ASTProbeConfig(pretrained=pretrained, score_mode=str(score_mode)))
    scores = probe.eventness_per_second(wav_path, num_segments=num_segments)
    return [float(x) for x in np.asarray(scores, dtype=np.float32)]
