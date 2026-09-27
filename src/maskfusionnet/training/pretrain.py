"""Stage 1 step function: masked pretraining.

Wires `models.pretraining.MaskFusionNetPretrain` to
`losses.reconstruction.reconstruction_loss` and
`losses.rppg.prediction_loss` (Eq. 20):

    L_stage1 = L_rec + gamma * L_pre
             = alpha*L_space + beta*L_time + gamma*L_pre
"""
from __future__ import annotations

import torch

from ..losses.reconstruction import reconstruction_loss
from ..losses.rppg import prediction_loss
from ..signal.fft import psd_over_hr_bins

__all__ = ["pretrain_step"]


def pretrain_step(model, batch: dict, device: torch.device, epoch: int,
                   alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0) -> dict:
    """`batch` must provide: clip [B,3,T,H,W], bvp [B,T], fps (scalar or [B]),
    hr_bpm [B] (ground-truth mean HR for the clip, used only by the
    frequency-domain term of L_pre)."""
    clip = batch["clip"].to(device)
    bvp = batch["bvp"].to(device)
    hr_bpm = batch["hr_bpm"].to(device)
    fs = float(batch["fps"][0]) if torch.is_tensor(batch["fps"]) else float(batch["fps"])

    out = model(clip)
    rec = reconstruction_loss(out["x_tube"], out["x_dec"], out["mask"], alpha=alpha, beta=beta)

    logits = psd_over_hr_bins(out["hr_signal"], fs=fs, num_bins=140)
    pre = prediction_loss(out["hr_signal"], bvp, logits, hr_bpm, epoch=epoch)

    total = rec["l_rec"] + gamma * pre["l_pre"]
    return {"loss": total, **{f"rec_{k}": v for k, v in rec.items()},
            **{f"pre_{k}": v for k, v in pre.items() if torch.is_tensor(v)}}
