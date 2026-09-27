"""Transformer encoder/decoder layers, Section III-B-4 and III-B-6.

Eq. (4):  X_dot^l = MSA(LN(X^{l-1})) + X^{l-1}
          X^l     = MLP(LN(X_dot^l)) + X_dot^l

Eq. (12): X_dot_dec^l = MSA_d(LN(X_dec^{l-1})) + X_dec^{l-1}
          X_dec^l     = MLP_d(LN(X_dot_dec^l)) + X_dot_dec^l

The encoder's MSA uses conv-derived Q/K/V (`MultiHeadSelfAttention`,
operates on the [C,T,H,W] grid so it can keep local spatial structure --
this is why it is NOT implemented as a plain sequence transformer).
The decoder's MSA_d uses *linear* projections (Eq. 13) on a flattened
token sequence, i.e. a conventional ViT-style block -- consistent with
the paper's statement that the decoder "consists of the vanilla
transformer layers [10], [44]" (ViT / VideoMAE).
"""
from __future__ import annotations

import torch
from torch import nn

from .attention import MultiHeadSelfAttention

__all__ = ["TransformerEncoderLayer", "TransformerDecoderLayer"]


class TransformerEncoderLayer(nn.Module):
    """One layer of Phi_enc / Theta_enc / Psi_enc / Gamma_enc (Eq. 4)."""

    def __init__(self, embed_dim: int = 96, num_heads: int = 4, mlp_ratio: float = 4.0,
                 dropout: float = 0.1, sigma: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.msa = MultiHeadSelfAttention(embed_dim, num_heads, sigma=sigma)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def _ln(self, x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        # LayerNorm is defined over the channel dim; x is [B, C, T, H, W].
        return norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        x = self.msa(self._ln(x, self.norm1)) + x
        x_norm = self._ln(x, self.norm2)
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm_flat = x_norm.flatten(2).transpose(1, 2)
        x_flat = self.mlp(x_norm_flat) + x_flat
        return x_flat.transpose(1, 2).reshape(b, c, t, h, w)


class _LinearMSA(nn.Module):
    """Vanilla linear-projection multi-head self-attention, Eq. (13)-(14)."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # NOTE ON A LIKELY PAPER TYPO: Eq. (13)/(14) as printed give the
        # per-head decoder dimension as C_d = sqrt(C) / M_d, which is not
        # an integer for the paper's own C=96, M_d=4 (sqrt(96)/4 ~= 2.45)
        # and cannot define a channel split. We treat this as a
        # transcription error and use the dimensionally-consistent
        # convention C_d = C / M_d (identical to the encoder's Eq. 6),
        # with attention scaled by 1/sqrt(head_dim) as Eq. (14) specifies.
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class TransformerDecoderLayer(nn.Module):
    """One layer of Phi_dec (Eq. 12), operating on a flattened token sequence."""

    def __init__(self, embed_dim: int = 96, num_heads: int = 4, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.msa = _LinearMSA(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]  (already flattened -- see pretraining.py)
        x = self.msa(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        return x
