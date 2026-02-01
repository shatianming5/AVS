from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VitFlopsConfig:
    num_layers: int
    hidden_size: int
    intermediate_size: int
    patch_size: int = 16
    num_channels: int = 3
    include_class_token: bool = True


def estimate_vit_forward_flops(cfg: VitFlopsConfig, *, resolution: int) -> int:
    """
    Rough FLOPs estimate for a ViT-style encoder forward pass (single image).

    This is an approximation intended for *relative* comparisons across resolutions:
    - Counts QKV + output projections, attention matmuls, and MLP matmuls per layer.
    - Includes patch embedding convolutional projection.
    - Ignores LayerNorm, bias adds, softmax, GELU, and embedding interpolation.
    - Treats one multiply-add as 2 FLOPs.
    """
    res = int(resolution)
    if res <= 0:
        raise ValueError("resolution must be positive")
    if res % int(cfg.patch_size) != 0:
        raise ValueError(f"resolution {res} must be divisible by patch_size {cfg.patch_size}")

    grid = res // int(cfg.patch_size)
    n_patches = int(grid * grid)
    seq_len = int(n_patches + (1 if cfg.include_class_token else 0))

    d = int(cfg.hidden_size)
    m = int(cfg.intermediate_size)
    layers = int(cfg.num_layers)

    # Patch embedding: conv/linear projection from (patch_size^2 * C) -> d for each patch.
    patch_in = int(cfg.patch_size) * int(cfg.patch_size) * int(cfg.num_channels)
    patch_embed_macs = int(n_patches) * int(patch_in) * int(d)

    # Transformer block MACs per layer (very rough):
    # - QKV projections: 3 * (L*d*d)
    # - Output projection: 1 * (L*d*d)
    proj_macs = 4 * seq_len * d * d
    # - Attention score (QK^T) + weighted sum (AV): 2 * (L*L*d)
    attn_macs = 2 * seq_len * seq_len * d
    # - MLP: 2 linear layers: 2 * (L*d*m)
    mlp_macs = 2 * seq_len * d * m
    per_layer_macs = proj_macs + attn_macs + mlp_macs

    total_macs = patch_embed_macs + layers * per_layer_macs
    total_flops = 2 * total_macs
    return int(total_flops)

