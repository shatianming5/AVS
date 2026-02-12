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


def _timm_preprocess_pil(
    images: list[Image.Image],
    *,
    resolution: int,
    crop_pct: float | None,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if crop_pct is None:
        crop_pct = 1.0
    crop_pct = float(crop_pct)
    if not (0.0 < crop_pct <= 1.0):
        crop_pct = 1.0

    resize_size = int(resolution)
    if crop_pct < 1.0:
        resize_size = int(round(float(resolution) / crop_pct))
        resize_size = max(int(resolution), int(resize_size))

    batch = []
    for im in images:
        im = im.convert("RGB").resize((resize_size, resize_size), resample=Image.BICUBIC)
        if resize_size != int(resolution):
            left = (resize_size - int(resolution)) // 2
            top = (resize_size - int(resolution)) // 2
            im = im.crop((left, top, left + int(resolution), top + int(resolution)))
        arr = np.asarray(im, dtype=np.uint8).copy()
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        x = x.to(device=device, dtype=dtype) / 255.0
        batch.append(x)

    pixel_values = torch.stack(batch, dim=0)
    mean_t = torch.tensor(list(mean), device=device, dtype=dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(list(std), device=device, dtype=dtype).view(1, 3, 1, 1)
    return (pixel_values - mean_t) / std_t


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
        self.backend = "hf_clip"
        self._timm_data_cfg: dict | None = None
        self._last_resolution: int | None = None

        if str(cfg.model_name).startswith("timm:"):
            import timm
            from timm.data import resolve_model_data_config

            self.backend = "timm"
            model_name = str(cfg.model_name).split(":", 1)[1]
            # Many timm ViTs support "token" pooling (CLS token), but some (e.g., SigLIP variants)
            # are trained without a class token and require average pooling.
            try:
                self.model = timm.create_model(
                    model_name,
                    pretrained=bool(cfg.pretrained),
                    num_classes=0,
                    global_pool="token",
                ).to(self.device)
            except AssertionError:
                # Fall back to the model's default global_pool (e.g., SigLIP uses attention pooling via global_pool='map').
                self.model = timm.create_model(
                    model_name,
                    pretrained=bool(cfg.pretrained),
                    num_classes=0,
                ).to(self.device)
            self._timm_data_cfg = resolve_model_data_config(self.model)
        else:
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
        if self.backend == "timm":
            assert self._timm_data_cfg is not None
            if self._last_resolution != int(resolution):
                # timm VisionTransformer supports dynamic img_size via positional embedding resampling.
                self.model.set_input_size(img_size=(int(resolution), int(resolution)))
                self._last_resolution = int(resolution)

            crop_pct = self._timm_data_cfg.get("crop_pct")
            mean = tuple(float(x) for x in self._timm_data_cfg.get("mean", (0.5, 0.5, 0.5)))
            std = tuple(float(x) for x in self._timm_data_cfg.get("std", (0.5, 0.5, 0.5)))
            pixel_values = _timm_preprocess_pil(
                images,
                resolution=int(resolution),
                crop_pct=float(crop_pct) if crop_pct is not None else None,
                mean=mean,  # type: ignore[arg-type]
                std=std,  # type: ignore[arg-type]
                device=self.device,
                dtype=self.dtype,
            )
            feats = self.model(pixel_values)
            return feats.detach().cpu()

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
