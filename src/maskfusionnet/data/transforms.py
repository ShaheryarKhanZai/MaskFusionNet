"""Optional clip-level augmentations for training.

The paper does not describe an augmentation policy for VIPL-HR/COHFACE/
PURE beyond the preprocessing already covered in `preprocessing.py`
(face-crop, resize, frame-rate standardization). These transforms are our
own, clearly-scoped addition for the UBFC-rPPG reproduction, applied
identically across all T frames of a clip so temporal signal content
(the thing being predicted) is not distorted -- e.g. horizontal flip is
applied to the whole clip, never per-frame independently, and brightness
jitter uses one sampled factor per clip rather than per frame, since
per-frame-independent color jitter would inject high-frequency noise
directly into the rPPG-relevant color channel.
"""
from __future__ import annotations

import numpy as np

__all__ = ["random_horizontal_flip", "random_brightness"]


def random_horizontal_flip(clip: np.ndarray, p: float = 0.5, rng: np.random.Generator | None = None) -> np.ndarray:
    """clip: [3, T, H, W]. Flips the W axis for the whole clip at once."""
    rng = rng or np.random.default_rng()
    if rng.random() < p:
        return clip[:, :, :, ::-1].copy()
    return clip


def random_brightness(clip: np.ndarray, max_delta: float = 0.1,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """clip: [3, T, H, W] float32 in [0, 1]. One additive brightness shift
    for the entire clip (not per-frame), then re-clipped to [0, 1]."""
    rng = rng or np.random.default_rng()
    delta = rng.uniform(-max_delta, max_delta)
    return np.clip(clip + delta, 0.0, 1.0).astype(np.float32)
