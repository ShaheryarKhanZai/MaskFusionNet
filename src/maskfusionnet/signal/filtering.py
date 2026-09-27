"""Bandpass filtering for rPPG/BVP signals.

Kept from the original repository's `postprocess.py` (this part was
reasonably implemented) but: made frequency band and filter order
configurable instead of hard-coded, moved the dead commented-out
alternate implementation out entirely (see AUDIT.md), and separated
"filter the signal" from "estimate the frequency" into two functions
(`bandpass_filter` here, `dominant_frequency` in fft.py) instead of one
function that silently does both and returns only a scalar BPM -- callers
that need the filtered waveform (e.g. for plotting predicted-vs-ground-
truth BVP) previously had no way to get it out.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

__all__ = ["bandpass_filter"]


def bandpass_filter(sig: np.ndarray, low_hz: float, high_hz: float, fs: float,
                     order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter.

    Args:
        sig: 1D signal.
        low_hz, high_hz: passband edges in Hz. For HR estimation these
            correspond to the paper's 40-180 bpm category range
            (0.667-3.0 Hz); for BVP waveform display a similar
            physiological band is typical.
        fs: sampling rate (video frame rate) in Hz.
        order: Butterworth filter order.
    """
    sig = np.asarray(sig, dtype=float).reshape(-1)
    nyq = 0.5 * fs
    low = low_hz / nyq
    high = high_hz / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, sig)
