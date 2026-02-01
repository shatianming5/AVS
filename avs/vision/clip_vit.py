from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from transformers import CLIPVisionConfig, CLIPVisionModel


@dataclass(frozen=True)
class ClipVisionEncoderConfig:
    model_name: str = "openai/clip-vit-base-patch16"
    device: str = "cpu"
    dtype: str = "float32"
    pretrained: bool = True


def _clip_normalize(pixel_values: torch.Tensor) -> torch.Tensor:
    # OpenAI CLIP normalization constants.
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=pixel_values.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=pixel_values.device).view(1, 3, 1, 1)
    return (pixel_values - mean) / std


def preprocess_pil(images: list[Image.Image], *, resolution: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    batch = []
    for im in images:
        im = im.convert("RGB").resize((resolution, resolution), resample=Image.BICUBIC)
        arr = np.asarray(im, dtype=np.uint8).copy()
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        x = x.to(device=device, dtype=dtype) / 255.0
        batch.append(x)
    pixel_values = torch.stack(batch, dim=0)
    return _clip_normalize(pixel_values)


def _interpolate_positional_embeddings(pos_weight: torch.Tensor, *, old_grid: int, new_grid: int) -> torch.Tensor:
    """
    Interpolate CLIP position embeddings from (old_grid x old_grid) to (new_grid x new_grid).

    pos_weight: [1 + old_grid^2, dim]
    returns:    [1 + new_grid^2, dim]
    """
    if old_grid == new_grid:
        return pos_weight

    cls_pos = pos_weight[:1, :]
    patch_pos = pos_weight[1:, :]
    dim = patch_pos.shape[-1]
    patch_pos = patch_pos.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)  # 1,dim,H,W
    patch_pos = F.interpolate(patch_pos, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(new_grid * new_grid, dim)
    return torch.cat([cls_pos, patch_pos], dim=0)


class ClipVisionEncoder:
    def __init__(self, cfg: ClipVisionEncoderConfig = ClipVisionEncoderConfig()):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = getattr(torch, cfg.dtype)

        if cfg.pretrained:
            self.model = CLIPVisionModel.from_pretrained(cfg.model_name).to(self.device)
        else:
            # Small random model for smoke tests / offline environments.
            config = CLIPVisionConfig(
                image_size=224,
                patch_size=16,
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=2,
                num_attention_heads=4,
            )
            self.model = CLIPVisionModel(config).to(self.device)

        self.model.eval()

    @torch.no_grad()
    def encode(self, images: list[Image.Image], *, resolution: int) -> torch.Tensor:
        pixel_values = preprocess_pil(images, resolution=resolution, device=self.device, dtype=self.dtype)

        vision = self.model.vision_model
        emb = vision.embeddings

        target_dtype = emb.patch_embedding.weight.dtype
        pv = pixel_values.to(dtype=target_dtype)
        patch = emb.patch_embedding(pv).flatten(2).transpose(1, 2)
        batch_size = patch.shape[0]
        hidden = torch.cat([emb.class_embedding.expand(batch_size, 1, -1), patch], dim=1)

        old_grid = vision.config.image_size // vision.config.patch_size
        new_grid = resolution // vision.config.patch_size
        pos_weight = emb.position_embedding.weight
        pos = _interpolate_positional_embeddings(pos_weight, old_grid=old_grid, new_grid=new_grid).to(hidden.dtype)
        hidden = hidden + pos.unsqueeze(0)

        hidden = vision.pre_layrnorm(hidden)
        enc_out = vision.encoder(inputs_embeds=hidden, return_dict=True)
        last_hidden = enc_out.last_hidden_state
        pooled = vision.post_layernorm(last_hidden[:, 0, :])
        return pooled.detach().cpu()
