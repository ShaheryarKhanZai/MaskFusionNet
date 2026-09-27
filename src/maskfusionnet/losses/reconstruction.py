"""Spatio-temporal reconstruction loss, Section III-C-1, Eq. (15)-(17).

L_space (Eq. 15): MSE between X_tube and X_dec, computed ONLY over the
masked token set Omega:

    L_space = (1/|Omega|) * sum_{p in Omega} || X_tube(p) - X_dec(p) ||^2

L_time (Eq. 16): L1 loss between adjacent-frame differences of X_tube and
X_dec (temporal-smoothness / continuity constraint), over the full
sequence (the paper does not restrict this term to masked positions):

    X_tube^{i,i+1} = X_tube^{i+1} - X_tube^{i}   (same for X_dec)
    L_time = (1/(D-1)) * sum_i | X_tube^{i,i+1} - X_dec^{i,i+1} |

L_rec (Eq. 17): L_rec = alpha * L_space + beta * L_time

None of this existed in the original repository -- there was no loss
module, no reconstruction target, and no training loop that would ever
call it.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["spatial_reconstruction_loss", "temporal_smoothness_loss", "reconstruction_loss"]


def spatial_reconstruction_loss(x_tube: torch.Tensor, x_dec: torch.Tensor,
                                 mask: torch.Tensor) -> torch.Tensor:
    """Eq. (15). `mask` is bool, True at masked (reconstruction-target) positions,
    broadcastable to x_tube's shape [B, C, T, H, W]."""
    mask = mask.expand_as(x_tube)
    if mask.sum() == 0:
        return torch.zeros((), device=x_tube.device, dtype=x_tube.dtype)
    diff = (x_tube - x_dec)[mask]
    return torch.mean(diff ** 2)


def temporal_smoothness_loss(x_tube: torch.Tensor, x_dec: torch.Tensor) -> torch.Tensor:
    """Eq. (16). Difference along the temporal (dim=2) axis."""
    d_tube = x_tube[:, :, 1:] - x_tube[:, :, :-1]
    d_dec = x_dec[:, :, 1:] - x_dec[:, :, :-1]
    return F.l1_loss(d_dec, d_tube)


def reconstruction_loss(x_tube: torch.Tensor, x_dec: torch.Tensor, mask: torch.Tensor,
                         alpha: float = 1.0, beta: float = 1.0) -> dict:
    """Eq. (17). Returns component losses and the weighted total (both needed
    for logging + backprop)."""
    l_space = spatial_reconstruction_loss(x_tube, x_dec, mask)
    l_time = temporal_smoothness_loss(x_tube, x_dec)
    total = alpha * l_space + beta * l_time
    return {"l_space": l_space, "l_time": l_time, "l_rec": total}
