from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPTokenizer, CLIPVisionConfig


@dataclass(frozen=True)
class ClipTextProbeConfig:
    model_name: str = "openai/clip-vit-base-patch16"
    pretrained: bool = True
    device: str = "cpu"
    dtype: str = "float32"


def _l2_normalize(x: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + float(eps))


class ClipTextProbe:
    """
    CLIP text encoder + visual projection helper for zero-shot class scoring.

    This is intentionally lightweight: Stage-2 caches store CLIPVisionModel pooled outputs (dim=768),
    while CLIP text similarity uses projected image features (dim=512). We reuse CLIPModel's
    `visual_projection` to map cached features into the joint space.
    """

    def __init__(self, cfg: ClipTextProbeConfig = ClipTextProbeConfig()):
        self.cfg = cfg
        self.device = torch.device(str(cfg.device))
        self.dtype = getattr(torch, str(cfg.dtype))

        if bool(cfg.pretrained):
            self.tokenizer = CLIPTokenizer.from_pretrained(str(cfg.model_name))
            self.model = CLIPModel.from_pretrained(str(cfg.model_name)).to(self.device)
        else:
            # Random weights for smoke/debug only. Not expected to be useful.
            self.tokenizer = CLIPTokenizer.from_pretrained(str(cfg.model_name))
            config = CLIPConfig(
                text_config=CLIPTextConfig(
                    vocab_size=49408,
                    hidden_size=128,
                    intermediate_size=256,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    max_position_embeddings=77,
                ),
                vision_config=CLIPVisionConfig(
                    image_size=224,
                    patch_size=16,
                    hidden_size=128,
                    intermediate_size=256,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                ),
                projection_dim=64,
            )
            self.model = CLIPModel(config).to(self.device)

        self.model.eval()

    @torch.no_grad()
    def text_features(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer([str(x) for x in texts], return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(device=self.device) for k, v in tokens.items()}
        feat = self.model.get_text_features(**tokens)
        feat = _l2_normalize(feat)
        return feat.detach().cpu().numpy().astype(np.float32, copy=False)

    @torch.no_grad()
    def project_image_features(self, pooled_features: np.ndarray) -> np.ndarray:
        """
        Project cached CLIPVisionModel pooled features into CLIP's joint embedding space.

        pooled_features: [T, 768] float array from `avs/vision/clip_vit.py` caches.
        Returns:         [T, 512] normalized features.
        """
        x = torch.from_numpy(np.asarray(pooled_features, dtype=np.float32))
        x = x.to(device=self.device, dtype=self.dtype)
        feat = self.model.visual_projection(x)
        feat = _l2_normalize(feat)
        return feat.detach().cpu().numpy().astype(np.float32, copy=False)

    def logit_scale(self) -> float:
        # Used for softmax temperature in zero-shot scoring.
        return float(self.model.logit_scale.detach().exp().cpu().item())

