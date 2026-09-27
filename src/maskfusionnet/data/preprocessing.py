"""Video -> face-cropped, normalized tensor clip.

    video -> frame extraction -> face detection -> face crop -> resize
          -> normalization -> temporal clip creation -> model

The paper's own datasets (VIPL-HR / COHFACE / PURE) are preprocessed with
the FAN face detector (Section IV-B: "we used the FAN [4] face detector"),
resized to 128x128, and clipped to fixed-length segments (160 frames for
training clips, 30s/three 10s segments at test time -- Section IV-B/D).

DOCUMENTED DEVIATION FOR UBFC-rPPG: this project targets UBFC-rPPG, which
the paper never uses. FAN is a heavier dependency (requires a separate face
-alignment package + landmark model download this environment cannot
fetch); we default to OpenCV's Haar-cascade frontal-face detector (already
vendored in the original repository as
`haarcascade_frontalface_default.xml`) with the same downstream steps
(crop -> resize 128x128 -> normalize). This is strictly a preprocessing
substitution, not a methodology change, and is why results on UBFC-rPPG
must never be presented as reproducing the paper's VIPL-HR/COHFACE/PURE
numbers -- see README.md, "Dataset" section.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "FaceCropper", "normalize_frame", "read_video_frames",
    "load_bvp_ground_truth", "align_bvp_to_frames",
]


class FaceCropper:
    """Thin, swappable face detector wrapper. Defaults to Haar cascade
    (fast, no extra weights to download); pass `cascade_path` explicitly if
    the file lives elsewhere than this package expects."""

    def __init__(self, cascade_path: str | None = None, image_size: int = 128):
        import cv2  # local import: keep cv2 optional for users who only touch the model code

        self.cv2 = cv2
        self.image_size = image_size
        path = cascade_path or (cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.detector = cv2.CascadeClassifier(path)
        self._last_box = None  # (x, y, w, h) fallback if detection fails on a frame

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        cv2 = self.cv2
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            # Largest detected face, in case of spurious small detections.
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            self._last_box = (x, y, w, h)
        elif self._last_box is not None:
            # Missing-frame handling: reuse the last known box rather than
            # dropping the frame, so clip length stays fixed.
            x, y, w, h = self._last_box
        else:
            # No detection yet anywhere in the clip: fall back to a
            # center-crop so the pipeline never silently returns None.
            h_full, w_full = frame_bgr.shape[:2]
            side = min(h_full, w_full)
            x, y, w, h = (w_full - side) // 2, (h_full - side) // 2, side, side

        face = frame_bgr[y:y + h, x:x + w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        return cv2.resize(face_rgb, (self.image_size, self.image_size))


def normalize_frame(frame_rgb_uint8: np.ndarray) -> np.ndarray:
    """[H,W,3] uint8 -> [3,H,W] float32 in [0,1]. Kept as a plain min-max
    scale (matching the original repository) rather than ImageNet
    mean/std normalization, since the paper does not specify normalization
    statistics and the model is trained from scratch, not from an
    ImageNet-pretrained backbone."""
    return frame_rgb_uint8.transpose(2, 0, 1).astype(np.float32) / 255.0


def read_video_frames(video_path: str | Path):
    """Yields BGR frames from a video file, one at a time."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def load_bvp_ground_truth(gt_path: str | Path) -> np.ndarray:
    """UBFC-rPPG ground truth format: `ground_truth.txt` with 3
    whitespace-separated rows -- BVP waveform, HR trace, timestamps
    (seconds) -- one column per sample. Returns the BVP row as a 1D array.

    ASSUMPTION: this matches the UBFC-rPPG DATASET_2 layout that is the
    version most commonly redistributed; if a given file only contains a
    single BVP row (older DATASET_1 layout), that row is used directly.
    Callers should sanity-check `sample count` against the video's frame
    count -- see `align_bvp_to_frames` for the resampling step this feeds.
    """
    data = np.loadtxt(gt_path)
    if data.ndim == 1:
        return data
    return data[0]


def align_bvp_to_frames(bvp: np.ndarray, num_frames: int) -> np.ndarray:
    """Resample a BVP trace of arbitrary length onto exactly `num_frames`
    samples via linear interpolation on a normalized [0, 1] time axis.

    UBFC-rPPG's pulse-oximeter and camera are not guaranteed to emit the
    exact same number of samples for a given recording; this makes the
    common (undocumented, in most third-party UBFC loaders) assumption
    that both signals span the *same wall-clock duration*, so resampling
    on a shared [0,1] axis aligns them without needing separate absolute
    timestamps. If precise per-frame timestamps are available (UBFC's
    3-row `ground_truth.txt` layout includes them), prefer aligning on
    those directly instead of this fallback.
    """
    bvp = np.asarray(bvp, dtype=float).reshape(-1)
    src_t = np.linspace(0.0, 1.0, num=len(bvp))
    dst_t = np.linspace(0.0, 1.0, num=num_frames)
    return np.interp(dst_t, src_t, bvp)
