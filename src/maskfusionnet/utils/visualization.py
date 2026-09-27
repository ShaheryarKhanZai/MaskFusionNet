"""Plotting helpers. Every function here draws only from data actually
passed in by the caller -- none of them fabricate numbers. Where a figure
in this repo is generated from synthetic data (e.g. `scripts/inspect_shapes.py`
demoing the masking pattern on random noise, since no dataset is bundled),
that is stated in the figure's own title, never silently presented as a
result from a trained model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..signal.fft import spectrum

__all__ = [
    "plot_tube_masking",
    "plot_bvp_comparison",
    "plot_spectrum",
    "plot_training_curves",
]


def plot_tube_masking(clip_frames: np.ndarray, mask_hw: np.ndarray, token_stride: int,
                       out_path: str | Path, num_frames_to_show: int = 4,
                       title_suffix: str = ""):
    """clip_frames: [T,H,W,3] uint8/float RGB frames (pixel space).
    mask_hw: [Ht,Wt] bool, the *spatial* tube mask (True = masked),
    identical for every shown frame -- this is the whole point of tube
    masking, and the figure is meant to make that visually obvious.
    token_stride: pixels-per-token, to upscale mask_hw to pixel resolution.
    """
    t_total = clip_frames.shape[0]
    idxs = np.linspace(0, t_total - 1, num=num_frames_to_show).astype(int)
    mask_px = np.kron(mask_hw, np.ones((token_stride, token_stride), dtype=bool))

    fig, axes = plt.subplots(1, len(idxs), figsize=(3 * len(idxs), 3.2))
    if len(idxs) == 1:
        axes = [axes]
    for ax, t in zip(axes, idxs):
        frame = clip_frames[t].astype(float)
        if frame.max() <= 1.0:
            frame = frame * 255.0
        frame = frame.astype(np.uint8).copy()
        overlay = frame.copy()
        overlay[mask_px] = [30, 30, 30]
        blended = (0.35 * frame + 0.65 * overlay).astype(np.uint8)
        ax.imshow(blended)
        ax.set_title(f"t={t}")
        ax.axis("off")
    fig.suptitle(f"Tube masking: same spatial mask at every t{title_suffix}")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bvp_comparison(pred: np.ndarray, gt: np.ndarray, fs: float, out_path: str | Path,
                         title: str = "Predicted vs. ground-truth BVP"):
    t = np.arange(len(gt)) / fs
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, (gt - gt.mean()) / (gt.std() + 1e-8), label="Ground truth (normalized)")
    ax.plot(t[:len(pred)], (pred - pred.mean()) / (pred.std() + 1e-8), label="Predicted (normalized)", alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (z-score)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_spectrum(sig: np.ndarray, fs: float, out_path: str | Path,
                   band_hz: tuple[float, float] = (40 / 60, 180 / 60),
                   estimated_bpm: float | None = None):
    freqs, mag = spectrum(sig, fs)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(freqs, mag)
    ax.axvspan(band_hz[0], band_hz[1], color="orange", alpha=0.15, label="Physiological band")
    if estimated_bpm is not None:
        ax.axvline(estimated_bpm / 60.0, color="red", linestyle="--",
                    label=f"Estimated HR = {estimated_bpm:.1f} bpm")
    ax.set_xlim(0, max(band_hz[1] * 1.5, freqs.max() * 0.2))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("FFT spectrum")
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curves(csv_path: str | Path, out_path: str | Path,
                          y_columns: tuple[str, ...] = ("train_loss", "val_loss")):
    import csv as csv_mod

    steps, series = [], {c: [] for c in y_columns}
    with open(csv_path) as f:
        for row in csv_mod.DictReader(f):
            steps.append(float(row["step"]))
            for c in y_columns:
                if c in row and row[c] not in ("", None):
                    series[c].append(float(row[c]))
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in y_columns:
        if series[c]:
            ax.plot(steps[: len(series[c])], series[c], label=c)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training curves")
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
