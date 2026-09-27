"""Tube random masking, Section III-B-3, Figure 3 (left panel).

    "the tube random masking strategy randomly selects masking positions
    in the spatial dimension and extends them across the entire temporal
    axis. This means that different frames share the same masking
    pattern. Tube random masking strategy ensures that the temporal
    neighbors of the masked positions are consistently masked."

Concretely: choose a random subset of (H_t, W_t) *spatial token* positions
to mask, then apply that identical spatial mask to every one of the T_t
temporal token positions -- i.e. mask[:, t, h, w] is the same boolean for
every t, for a fixed (h, w).

BUG FIXED: the original repository's `TubeMasking` flattened the mask
over `T*H*W` and applied `torch.randperm` to the *flattened* index list:

    mask = torch.ones(B, T*H*W); mask[:, :num_mask] = 0
    mask = mask[:, torch.randperm(mask.size(1))].reshape(B, T, H, W)

This is Figure 3's *right*-hand "completely random masking strategy" --
the exact baseline the paper masks against, not the tube strategy it
actually proposes. A fixed (h, w) is masked independently at every t.

ARCHITECTURAL NOTE: the paper's encoders (Phi_enc etc.) use conv-derived
Q/K/V (Eq. 5), which need a spatially coherent grid to operate over -- you
cannot simply drop 75% of the tokens from a [C,T,H,W] tensor and still run
a Conv3d over the remainder the way you could with a plain ViT sequence
encoder (VideoMAE's usual token-dropping trick). We therefore implement
masking as "replace masked positions with a learnable mask-token
embedding, keep the tensor's full shape," matching Figure 2's diagram
(same-size grid before/after masking, with "M" cubes marking masked
positions) rather than literally shortening the sequence. This is an
explicit, documented interpretation of an architectural detail the paper
does not spell out in full (see AUDIT.md, "Ambiguities").
"""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["TubeMasking"]


class TubeMasking(nn.Module):
    def __init__(self, embed_dim: int, mask_ratio: float = 0.75):
        super().__init__()
        assert 0.0 < mask_ratio < 1.0
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim, 1, 1, 1))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: token tensor [B, C, T, H, W] (already conv+embedded tokens).
        Returns:
            x_vis: same shape as x, masked positions replaced by mask_token.
            mask:  bool tensor [B, 1, T, H, W], True where masked. Constant
                   across T for a fixed (b, h, w), by construction.
        """
        b, c, t, h, w = x.shape
        num_tokens = h * w
        num_mask = int(round(num_tokens * self.mask_ratio))

        # One spatial mask per batch element, independent of T.
        spatial_mask = torch.zeros(b, num_tokens, device=x.device, dtype=torch.bool)
        for i in range(b):
            perm = torch.randperm(num_tokens, device=x.device)
            spatial_mask[i, perm[:num_mask]] = True
        spatial_mask = spatial_mask.view(b, 1, 1, h, w)          # [B,1,1,H,W]
        mask = spatial_mask.expand(b, 1, t, h, w).contiguous()   # replicate across T

        x_vis = torch.where(mask, self.mask_token.expand(b, c, t, h, w), x)
        return x_vis, mask
