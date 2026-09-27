"""Prediction loss, Section III-C-2, Eq. (18)-(19).

L_pre = lambda * L_temporal + mu * L_frequency
      = lambda * L_PN + mu * (L_CE + L_KL)

L_PN: negative Pearson correlation between predicted and ground-truth
signal (temporal-domain trend constraint).

L_CE, L_KL: cross-entropy and KL-divergence between the ground-truth and
predicted heart-rate *category* distributions, over 140 discrete bins
covering HR in [40, 180) bpm (Section III-C-2: "a represents a category of
the discrete distribution, corresponding to heart rate values in the range
of [40, 180), with a total of 140 categories").

Dynamic weighting, Eq. (19):
    lambda = 0.1
    mu = mu0 * eta^((e-1)/25)   if e < 25
    mu = mu0                    if e >= 25
    with mu0 = 1.0, eta = 5.0 (paper's Section IV-B values).

None of this existed in the original repository -- there was no loss
module for either stage.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "negative_pearson_loss",
    "hr_to_soft_label",
    "frequency_domain_loss",
    "prediction_loss",
    "loss_weight_schedule",
]

HR_MIN_BPM = 40
HR_MAX_BPM = 180
NUM_HR_BINS = 140  # [40, 180) in 1-bpm bins, per Section III-C-2


def negative_pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Eq. (18), L_PN. pred/target: [B, T]. Returns 1 - mean Pearson r
    (so 0 is perfect correlation, matching a loss to be minimized -- the
    paper's L_PN is written as the correlation coefficient itself and used
    as a loss to *maximize* correlation, which is the standard "negative
    Pearson loss" convention 1 - r)."""
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    num = (pred * target).sum(dim=1)
    den = torch.sqrt((pred ** 2).sum(dim=1) * (target ** 2).sum(dim=1) + eps)
    r = num / (den + eps)
    return (1 - r).mean()


def hr_to_soft_label(hr_bpm: torch.Tensor, num_bins: int = NUM_HR_BINS,
                      hr_min: int = HR_MIN_BPM, sigma: float = 1.0) -> torch.Tensor:
    """Ground-truth HR (bpm) -> a Gaussian-smoothed categorical distribution
    over `num_bins` bins, used as the target for both L_CE and L_KL. The
    paper does not specify hard-vs-soft labels explicitly; we use a
    narrow-sigma Gaussian around the true bin (a common, documented choice
    for HR-as-classification losses, e.g. PhysFormer / CVD) rather than a
    one-hot label, since KL-divergence against a one-hot target is
    numerically degenerate. `hr_bpm`: [B]."""
    device = hr_bpm.device
    bins = torch.arange(num_bins, device=device).float() + hr_min  # bin centers, bpm
    hr_bpm = hr_bpm.clamp(hr_min, hr_min + num_bins - 1).unsqueeze(1)  # [B,1]
    dist = torch.exp(-0.5 * ((bins.unsqueeze(0) - hr_bpm) / sigma) ** 2)
    return dist / dist.sum(dim=1, keepdim=True)


def frequency_domain_loss(pred_logits: torch.Tensor, target_hr_bpm: torch.Tensor) -> dict:
    """L_CE + L_KL, Eq. (18). `pred_logits`: [B, num_bins] raw scores (a
    linear head over the predicted signal's spectrum -- see
    training/finetune.py); `target_hr_bpm`: [B] ground-truth HR."""
    target_dist = hr_to_soft_label(target_hr_bpm, pred_logits.shape[1])
    log_q = F.log_softmax(pred_logits, dim=1)
    l_ce = -(target_dist * log_q).sum(dim=1).mean()
    l_kl = F.kl_div(log_q, target_dist, reduction="batchmean")
    return {"l_ce": l_ce, "l_kl": l_kl, "l_frequency": l_ce + l_kl}


def loss_weight_schedule(epoch: int, mu0: float = 1.0, eta: float = 5.0,
                          warmup_epochs: int = 25, lam: float = 0.1) -> tuple[float, float]:
    """Eq. (19). `epoch` is 1-indexed, matching the paper's `e`.

    NOTE ON SHAPE: taken literally, this schedule is NOT a monotonic
    ramp-up-to-mu0 warmup. Because eta=5.0 > 1, mu *overshoots* mu0 during
    e in [1, 24] (rising from mu0 at e=1 to mu0*eta^(23/25)~=4.5 at e=24),
    then drops back down to exactly mu0 at e=25 and stays there. We
    implement Eq. (19) exactly as printed rather than "fixing" it to a
    smoother-looking curve, since the paper gives no indication this is a
    transcription error (unlike the decoder head-dimension issue noted in
    transformer.py, where the printed formula is dimensionally impossible).
    If you intended a monotonic warmup instead, use a different `eta`
    (0 < eta < 1) or exponent sign -- both are one-line changes here.
    """
    if epoch < warmup_epochs:
        mu = mu0 * (eta ** ((epoch - 1) / warmup_epochs))
    else:
        mu = mu0
    return lam, mu


def prediction_loss(pred_signal: torch.Tensor, target_signal: torch.Tensor,
                     pred_hr_logits: torch.Tensor, target_hr_bpm: torch.Tensor,
                     epoch: int) -> dict:
    """Full L_pre, Eq. (18)-(19)."""
    lam, mu = loss_weight_schedule(epoch)
    l_pn = negative_pearson_loss(pred_signal, target_signal)
    freq = frequency_domain_loss(pred_hr_logits, target_hr_bpm)
    l_pre = lam * l_pn + mu * freq["l_frequency"]
    return {"l_pn": l_pn, **freq, "l_pre": l_pre, "lambda": lam, "mu": mu}
