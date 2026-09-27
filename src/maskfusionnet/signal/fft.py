"""FFT-based frequency analysis.

`dominant_frequency` / `spectrum` are used at inference time (real-time
demo, evaluation) to turn a filtered waveform into a BPM number and a
plottable spectrum. `psd_over_hr_bins` is used at *training* time to build
the categorical "predicted HR distribution" that Eq. (18)'s L_CE / L_KL
terms compare against the ground-truth soft label (see
losses/rppg.py:frequency_domain_loss) -- this construction (treat the
power spectrum, restricted to the physiological band and rebinned to
whole-bpm categories, as an unnormalized categorical distribution) is the
standard "frequency cross-entropy" trick used by PhysFormer / CVD, which
this paper's Eq. (18) cites and follows; the paper's text does not spell
out the exact binning procedure, so this is a documented, standard
implementation of it, done in a differentiable way (torch, not numpy) so
it can be backpropagated through during fine-tuning.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["spectrum", "dominant_frequency", "psd_over_hr_bins"]


def spectrum(sig: np.ndarray, fs: float):
    """Returns (freqs_hz, magnitude) of the one-sided FFT of a real signal."""
    sig = np.asarray(sig, dtype=float).reshape(-1)
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    mag = np.abs(np.fft.rfft(sig))
    return freqs, mag


def dominant_frequency(sig: np.ndarray, fs: float, low_hz: float, high_hz: float) -> float:
    """Peak frequency (Hz) within [low_hz, high_hz]."""
    freqs, mag = spectrum(sig, fs)
    band = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(band):
        raise ValueError("No FFT bins fall inside the requested frequency band.")
    freqs_band, mag_band = freqs[band], mag[band]
    return float(freqs_band[np.argmax(mag_band)])


def psd_over_hr_bins(pred_signal: torch.Tensor, fs: float, num_bins: int = 140,
                      hr_min_bpm: int = 40, hr_max_bpm: int = 180) -> torch.Tensor:
    """Differentiable power spectrum of `pred_signal` ([B, T]) resampled
    onto `num_bins` whole-bpm categories in [hr_min_bpm, hr_max_bpm),
    usable directly as logits for `losses.rppg.frequency_domain_loss`.
    """
    b, t = pred_signal.shape
    sig = pred_signal - pred_signal.mean(dim=1, keepdim=True)
    spec = torch.fft.rfft(sig, dim=1)
    power = (spec.real ** 2 + spec.imag ** 2)                       # [B, F]
    freqs = torch.fft.rfftfreq(t, d=1.0 / fs).to(pred_signal.device)  # [F]
    bpm = freqs * 60.0

    bin_centers = torch.arange(num_bins, device=pred_signal.device).float() + hr_min_bpm  # [num_bins]
    # Soft-assign each FFT bin's power to the nearest bpm category via a
    # narrow Gaussian kernel over |bpm - bin_center|, then sum -- a
    # differentiable histogram.
    dist = (bpm.unsqueeze(1) - bin_centers.unsqueeze(0)) ** 2  # [F, num_bins]
    weights = torch.exp(-dist / (2 * 1.0 ** 2))
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    logits = torch.matmul(power, weights)  # [B, num_bins]
    return logits
