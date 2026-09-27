"""Stage 2 step function: rPPG fine-tuning.

Wires `models.maskfusionnet.MaskFusionNet` to `losses.rppg.prediction_loss`
(Eq. 21): L_stage2 = L_pre (no reconstruction term -- fine-tuning has no
decoder).
"""
from __future__ import annotations

import torch

from ..losses.rppg import prediction_loss
from ..signal.fft import psd_over_hr_bins

__all__ = ["finetune_step"]


def finetune_step(model, batch: dict, device: torch.device, epoch: int) -> dict:
    """`batch` must provide: clip [B,3,T,H,W], bvp [B,T], fps, hr_bpm [B]."""
    clip = batch["clip"].to(device)
    bvp = batch["bvp"].to(device)
    hr_bpm = batch["hr_bpm"].to(device)
    fs = float(batch["fps"][0]) if torch.is_tensor(batch["fps"]) else float(batch["fps"])

    pred_signal = model(clip)
    logits = psd_over_hr_bins(pred_signal, fs=fs, num_bins=140)
    pre = prediction_loss(pred_signal, bvp, logits, hr_bpm, epoch=epoch)

    return {"loss": pre["l_pre"], **{k: v for k, v in pre.items() if torch.is_tensor(v)},
            "pred_signal": pred_signal}
