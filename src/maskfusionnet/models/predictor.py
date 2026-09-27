"""Predictor head, Section III-B-7.

    "temporal upsampling and spatial averaging are performed, and the
    features are finally projected onto a 1D signal Y in R^T."

Used, unmodified, by both the pre-training model (on X_enc) and the
fine-tuning model (on the fusion encoder's output).

ASSUMPTION: the paper names "temporal upsampling" before "spatial
averaging" but does not give an exact op order or upsampling mode. We
average over the spatial axes first (order-invariant, since averaging and
upsampling are applied to independent axes) and then linearly upsample the
temporal axis from T_t tokens to the clip's original frame count, which is
the only order that makes the two operations composable without an
intermediate spatial upsample nobody describes.
"""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["Predictor"]


class Predictor(nn.Module):
    def __init__(self, embed_dim: int = 96, output_length: int = 160):
        super().__init__()
        self.output_length = output_length
        self.upsample = nn.Upsample(size=output_length, mode="linear", align_corners=False)
        self.proj = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        b, c, t, h, w = x.shape
        x = torch.mean(x, dim=(-2, -1))              # spatial average -> [B, C, T]
        x = self.upsample(x)                          # temporal upsample -> [B, C, output_length]
        x = self.proj(x.transpose(1, 2))               # [B, output_length, 1]
        return x.squeeze(-1)                           # [B, output_length]
