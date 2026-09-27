"""Attention primitives matching Eq. (4)-(10) of the paper.

The original repository's `SelfAttention.py` computed

    attn_scores = einsum('bchwk,bchwk->bhwk', q, k)

which contracts Q and K over the *channel* axis at *matching* spatial
positions only. That is a per-position channel-wise dot product (a gating
scalar per token), never a token-to-token similarity matrix -- it cannot
attend from one spatio-temporal location to another at all. This is not
self-attention in any sense used by the paper; it silently degrades the
"transformer-based" architecture into a set of independent 1x1-ish gates.

This module implements genuine attention over the flattened token
sequence, as Eq. (6) specifies:

    SA^(i) = Softmax( Q_e^(i) (K_e^(i))^T / sigma ) V_e^(i)

with Q, K, V obtained by flattening the conv-projected [C, T, H, W] tensor
into a sequence of N = T*H*W tokens per Section III-B-4, i.e. the
attention matrix has shape [B*heads, N, N].
"""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["MultiHeadSelfAttention", "MultiHeadInteractiveAttention"]


def _flatten(x: torch.Tensor) -> torch.Tensor:
    """[B, C, T, H, W] -> [B, N, C], N = T*H*W."""
    b, c, t, h, w = x.shape
    return x.flatten(2).transpose(1, 2), (t, h, w)


def _unflatten(x: torch.Tensor, thw: tuple[int, int, int]) -> torch.Tensor:
    """[B, N, C] -> [B, C, T, H, W]."""
    b, n, c = x.shape
    t, h, w = thw
    return x.transpose(1, 2).reshape(b, c, t, h, w)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with conv-derived Q/K/V, Eq. (5)-(7).

    Eq. (5): Q_e = BN(Conv_{3x3x3}(x)); K_e = BN(Conv_{3x3x3}(x)); V_e = Conv_{1x1x1}(x)
    Eq. (6): SA^(i) = Softmax(Q_e^(i) (K_e^(i))^T / sigma) V_e^(i),  sigma = 2.0
    Eq. (7): MSA = Concat(SA^(1), ..., SA^(M_e)) W_e
    """

    def __init__(self, dim: int, num_heads: int = 4, sigma: float = 2.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sigma = sigma

        self.conv_q = nn.Conv3d(dim, dim, kernel_size=3, padding=1)
        self.bn_q = nn.BatchNorm3d(dim)
        self.conv_k = nn.Conv3d(dim, dim, kernel_size=3, padding=1)
        self.bn_k = nn.BatchNorm3d(dim)
        self.conv_v = nn.Conv3d(dim, dim, kernel_size=1)

        self.proj = nn.Linear(dim, dim)  # W_e in Eq. (7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        b, c, t, h, w = x.shape
        q = self.bn_q(self.conv_q(x))
        k = self.bn_k(self.conv_k(x))
        v = self.conv_v(x)

        q, _ = _flatten(q)  # [B, N, C]
        k, _ = _flatten(k)
        v, thw = _flatten(v)

        q = q.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, N, hd]
        k = k.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.sigma  # [B, heads, N, N]
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, heads, N, hd]

        out = out.transpose(1, 2).contiguous().view(b, -1, self.num_heads * self.head_dim)  # [B, N, C]
        out = self.proj(out)
        return _unflatten(out, thw)


class MultiHeadInteractiveAttention(nn.Module):
    """Interactive (cross-branch) attention used inside the MFB, Eq. (9)-(10).

    Eq. (9): Q_m = BN(Conv_{3x3x3}(SEB));  K_m = BN(Conv_{3x3x3}(ADB));  V_m = Conv_{1x1x1}(ADB)
    Eq. (10): IA^(i) = Softmax(Q_m^(i) (K_m^(i))^T / sigma) V_m^(i)
              MIA = Concat(IA^(1), ..., IA^(M_m)) W_m

    Query comes from the Segmented Branch (long-range / coarse scale),
    Key & Value come from the Adjacent Branch (short-range / fine scale) --
    NOT the reverse. The original repository's `MultiHeadSelfAttention1`
    was called as `MFB(y_ADB, SEB_features)` with `x1 -> Q`, `x2 -> K, V`,
    which projected ADB into the query and SEB into key/value: exactly
    backwards relative to Eq. (9). This class fixes that by taking
    explicitly named `query_source` (SEB) and `kv_source` (ADB) arguments.

    Output has the same [C, T, H, W] shape as `query_source` (i.e. SEB's
    shape), consistent with Eq. (8): X_f = MIA(...) + X_SEB.
    """

    def __init__(self, dim: int, num_heads: int = 4, sigma: float = 2.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sigma = sigma

        self.conv_q = nn.Conv3d(dim, dim, kernel_size=3, padding=1)
        self.bn_q = nn.BatchNorm3d(dim)
        self.conv_k = nn.Conv3d(dim, dim, kernel_size=3, padding=1)
        self.bn_k = nn.BatchNorm3d(dim)
        self.conv_v = nn.Conv3d(dim, dim, kernel_size=1)

        self.proj = nn.Linear(dim, dim)  # W_m in Eq. (10)

    def forward(self, query_source: torch.Tensor, kv_source: torch.Tensor) -> torch.Tensor:
        # query_source (SEB): [B, C, Tq, Hq, Wq]
        # kv_source     (ADB): [B, C, Tk, Hk, Wk]   (spatial dims must already match Tq/Hq/Wq;
        #                                             temporal alignment is the caller's job --
        #                                             see MultiScaleFusionBlock)
        b = query_source.shape[0]
        q = self.bn_q(self.conv_q(query_source))
        k = self.bn_k(self.conv_k(kv_source))
        v = self.conv_v(kv_source)

        q, thw_q = _flatten(q)
        k, _ = _flatten(k)
        v, _ = _flatten(v)

        q = q.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.sigma
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(b, -1, self.num_heads * self.head_dim)
        out = self.proj(out)
        return _unflatten(out, thw_q)
