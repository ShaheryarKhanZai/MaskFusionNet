"""Evaluation metrics, Section IV-C of the paper (Eq. 22-25) plus SNR.

MAE/RMSE/r are computed on per-clip heart-rate estimates (bpm), matching
how the paper reports them (Table I-IV). SNR is a waveform-quality metric
(not reported by the paper's tables, but standard in the rPPG literature,
e.g. de Haan & Jeanne 2013) reported separately so BVP-waveform quality
and HR-accuracy are never conflated into one number, per the task brief.
"""
from __future__ import annotations

import numpy as np

from ..signal.fft import spectrum

__all__ = ["mae", "rmse", "pearson_r", "snr", "hr_metrics_report"]


def mae(pred_bpm: np.ndarray, gt_bpm: np.ndarray) -> float:
    """Eq. (23)."""
    return float(np.mean(np.abs(np.asarray(pred_bpm) - np.asarray(gt_bpm))))


def rmse(pred_bpm: np.ndarray, gt_bpm: np.ndarray) -> float:
    """Eq. (24)."""
    return float(np.sqrt(np.mean((np.asarray(pred_bpm) - np.asarray(gt_bpm)) ** 2)))


def pearson_r(pred_bpm: np.ndarray, gt_bpm: np.ndarray) -> float:
    """Eq. (25)."""
    pred_bpm = np.asarray(pred_bpm, dtype=float)
    gt_bpm = np.asarray(gt_bpm, dtype=float)
    if pred_bpm.std() == 0 or gt_bpm.std() == 0:
        return 0.0
    return float(np.corrcoef(pred_bpm, gt_bpm)[0, 1])


def snr(sig: np.ndarray, fs: float, hr_freq_hz: float, band_hz: float = 0.1,
        harmonics: int = 2) -> float:
    """Signal-to-noise ratio (dB) of a filtered waveform: ratio of power in
    narrow bands around the true HR frequency (and its harmonics) to power
    elsewhere in the physiological band. Not part of the paper's Eq. set --
    included because the task brief separately asks for a waveform-quality
    metric distinct from HR-accuracy metrics."""
    freqs, mag = spectrum(sig, fs)
    power = mag ** 2
    signal_mask = np.zeros_like(freqs, dtype=bool)
    for h in range(1, harmonics + 1):
        signal_mask |= np.abs(freqs - h * hr_freq_hz) <= band_hz
    physiological = (freqs >= 40 / 60) & (freqs <= 180 / 60)
    noise_mask = physiological & ~signal_mask
    signal_power = power[signal_mask].sum()
    noise_power = power[noise_mask].sum()
    if noise_power <= 0:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power))


def hr_metrics_report(pred_bpm: np.ndarray, gt_bpm: np.ndarray) -> dict:
    return {"MAE": mae(pred_bpm, gt_bpm), "RMSE": rmse(pred_bpm, gt_bpm), "Pearson_r": pearson_r(pred_bpm, gt_bpm)}
