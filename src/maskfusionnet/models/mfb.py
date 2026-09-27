"""Multi-Scale Fusion Block (MFB), Section III-B-5, Figure 4, Eq. (8)-(10).

    X_f = MIA(LN(X_SEB), LN(X_ADB)) + X_SEB                      (Eq. 8)
    Q_m = BN(Conv_{3x3x3}(LN(X_SEB)))                             (Eq. 9)
    K_m = BN(Conv_{3x3x3}(LN(X_ADB)))
    V_m = Conv_{1x1x1}(LN(X_ADB))
    IA^(i) = Softmax(Q_m^(i) (K_m^(i))^T / sigma) V_m^(i)         (Eq. 10)
    MIA = Concat(IA^(1..M_m)) W_m

Two bugs fixed relative to the original repository's
`MultiScaleFusionBlock` / `MultiHeadSelfAttention1`:

1. Query/Key-Value were swapped. The original called
   `self.MFB(y_ADB, SEB_features)` and inside, `x1 (ADB) -> query`,
   `x2 (SEB) -> key, value` -- backwards vs. Eq. (9), which projects the
   Segmented Branch into the query and the Adjacent Branch into key/value.
2. The residual connection `+ X_SEB` from Eq. (8) was missing entirely --
   the block just returned the raw attention output.

The paper also states (Section IV, "we use one MFB to combine the output
features of the fourth transformer layer... At the end... their features
are also fused using another MFB") -- i.e. TWO separate fusion points,
described with the word "another", which we read as two independently
parameterized `MultiScaleFusionBlock` instances (the original repo reused
a single `self.MFB` object -- and, worse, a single `self.encoder`
`nn.ModuleList` -- for both calls, meaning both fusion points and both
branches shared literal weight tensors). `maskfusionnet.py` instantiates
two separate `MultiScaleFusionBlock`s.

TEMPORAL ALIGNMENT (documented assumption): ADB's tube embedding has half
the temporal stride of SEB's (2 vs 4, Section III-A-2-b), so ADB produces
2x as many temporal tokens as SEB for the same input clip. The paper does
not spell out how the two are reconciled dimensionally before fusion; we
average-pool ADB's temporal axis by a factor of 2 (`nn.AvgPool3d((2,1,1))`)
immediately before each MFB call so its token grid matches SEB's, which is
also what the original repository did and is a reasonable reading of
Figure 2 (no extra module is drawn between the branches and the MFB other
than the fusion block itself).
"""
from __future__ import annotations

import torch
from torch import nn

from .attention import MultiHeadInteractiveAttention

__all__ = ["MultiScaleFusionBlock"]


class MultiScaleFusionBlock(nn.Module):
    def __init__(self, embed_dim: int = 96, num_heads: int = 4, sigma: float = 2.0):
        super().__init__()
        self.norm_seb = nn.LayerNorm(embed_dim)
        self.norm_adb = nn.LayerNorm(embed_dim)
        self.mia = MultiHeadInteractiveAttention(embed_dim, num_heads, sigma=sigma)

    @staticmethod
    def _ln(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        return norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

    def forward(self, x_seb: torch.Tensor, x_adb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seb: Segmented Branch features [B, C, T, H, W] -- provides the query.
            x_adb: Adjacent Branch features, already temporally aligned to
                   x_seb's T (see the AvgPool3d step in maskfusionnet.py) --
                   provides the key/value.
        Returns:
            X_f with the same shape as x_seb (Eq. 8).
        """
        q_in = self._ln(x_seb, self.norm_seb)
        kv_in = self._ln(x_adb, self.norm_adb)
        fused = self.mia(query_source=q_in, kv_source=kv_in)
        return fused + x_seb
