import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maskfusionnet.signal.filtering import bandpass_filter
from maskfusionnet.signal.fft import dominant_frequency, spectrum
from maskfusionnet.signal.heart_rate import waveform_to_bpm


def _make_sinusoid(bpm: float, fs: float, duration_s: float = 10.0, noise_std: float = 0.05):
    t = np.arange(0, duration_s, 1.0 / fs)
    freq_hz = bpm / 60.0
    rng = np.random.default_rng(0)
    return t, np.sin(2 * np.pi * freq_hz * t) + rng.normal(0, noise_std, size=t.shape)


def test_dominant_frequency_recovers_known_bpm():
    fs = 30.0
    true_bpm = 72.0
    _, sig = _make_sinusoid(true_bpm, fs)
    freq_hz = dominant_frequency(sig, fs, low_hz=40 / 60, high_hz=180 / 60)
    assert abs(freq_hz * 60 - true_bpm) < 3.0  # within 3 bpm on a clean synthetic signal


def test_waveform_to_bpm_end_to_end_pipeline():
    fs = 30.0
    true_bpm = 95.0
    _, sig = _make_sinusoid(true_bpm, fs)
    result = waveform_to_bpm(sig, fs=fs)
    assert abs(result["bpm"] - true_bpm) < 3.0
    assert result["filtered_signal"].shape == sig.shape


def test_bandpass_filter_removes_out_of_band_energy():
    fs = 30.0
    t = np.arange(0, 10, 1 / fs)
    in_band = np.sin(2 * np.pi * (75 / 60) * t)      # 75 bpm, inside [40,180) bpm band
    out_of_band = np.sin(2 * np.pi * 5.0 * t)         # 5 Hz = 300 bpm, well outside
    sig = in_band + out_of_band

    filtered = bandpass_filter(sig, low_hz=40 / 60, high_hz=180 / 60, fs=fs)
    freqs, mag = spectrum(filtered, fs)

    band_power = mag[(freqs >= 40 / 60) & (freqs <= 180 / 60)].sum()
    out_of_band_idx = np.argmin(np.abs(freqs - 5.0))
    assert mag[out_of_band_idx] < 0.1 * band_power


def test_multiple_bpms_across_physiological_range():
    fs = 30.0
    for true_bpm in [45.0, 60.0, 90.0, 120.0, 160.0]:
        _, sig = _make_sinusoid(true_bpm, fs, noise_std=0.02)
        result = waveform_to_bpm(sig, fs=fs)
        assert abs(result["bpm"] - true_bpm) < 3.0, f"failed for {true_bpm} bpm, got {result['bpm']}"
