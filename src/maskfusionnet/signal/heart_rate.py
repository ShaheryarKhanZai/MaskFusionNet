"""Waveform -> BPM, end to end.

Kept separate from the model: model output is a raw predicted rPPG/BVP
waveform; everything below is classical signal processing applied
*after* inference, matching the "Predictor -> rPPG -> Signal Processing ->
Heart Rate" pipeline described in the task brief. Frequency band is
configurable and documented, not a silent hard-coded assumption.
"""
from __future__ import annotations

import numpy as np

from .filtering import bandpass_filter
from .fft import dominant_frequency

__all__ = ["waveform_to_bpm", "DEFAULT_HR_BAND_HZ"]

# Matches the paper's 140-category classification range, [40, 180) bpm,
# Section III-C-2 -- expressed in Hz for filtering/FFT purposes.
DEFAULT_HR_BAND_HZ = (40 / 60, 180 / 60)


def waveform_to_bpm(sig: np.ndarray, fs: float,
                     band_hz: tuple[float, float] = DEFAULT_HR_BAND_HZ,
                     filter_order: int = 2) -> dict:
    """Filter a raw predicted signal and estimate a single BPM value.

    Returns a dict with the filtered signal (for plotting) and the
    estimated BPM (for display/evaluation), so callers never have to
    re-derive one from the other.
    """
    low_hz, high_hz = band_hz
    filtered = bandpass_filter(sig, low_hz, high_hz, fs, order=filter_order)
    freq_hz = dominant_frequency(filtered, fs, low_hz, high_hz)
    return {"filtered_signal": filtered, "bpm": freq_hz * 60.0}
