"""UBFC-rPPG dataset loader.

UBFC-rPPG layout (as publicly distributed):

    UBFC-rPPG/
      subject1/
        vid.avi
        ground_truth.txt
      subject2/
        vid.avi
        ground_truth.txt
      ...

- Video: uncompressed webcam recording, ~30 fps, faces mostly frontal,
  static background.
- Ground truth: whitespace-separated text file; row 0 = BVP waveform,
  row 1 = per-sample HR (bpm), row 2 = timestamps (s) -- see
  `preprocessing.load_bvp_ground_truth`.

This is NOT one of the three datasets the paper evaluates on (VIPL-HR,
COHFACE, PURE). The original repository has no dataset code at all (only
a live webcam demo), so nothing here is a "fix" of prior UBFC-loading
code -- it is new, and its preprocessing choices are UBFC-specific
adaptations, documented in `preprocessing.py` and in README.md's Dataset
section, not attempts to replicate the paper's own data pipeline exactly.

Every clip is subject-labeled so `splits.subject_level_split` can prevent
leakage.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .preprocessing import FaceCropper, normalize_frame, read_video_frames, load_bvp_ground_truth, align_bvp_to_frames
from ..signal.fft import dominant_frequency
from ..signal.filtering import bandpass_filter

__all__ = ["UBFCClipIndex", "UBFCrPPGDataset", "discover_subjects"]


def discover_subjects(root: str | Path) -> list[str]:
    """List subject directory names under `root` that contain both a video
    and a ground-truth file. Looks for the two common UBFC filename
    conventions (`vid.avi` + `ground_truth.txt`)."""
    root = Path(root)
    subjects = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if (d / "vid.avi").exists() and (d / "ground_truth.txt").exists():
            subjects.append(d.name)
    return subjects


@dataclass
class UBFCClipIndex:
    subject: str
    start_frame: int
    clip_length: int


class UBFCrPPGDataset:
    """A `torch.utils.data.Dataset`-compatible loader (kept dependency-free
    at import time -- inherits from `object`, not `torch.utils.data.Dataset`,
    so this module can be imported and unit-tested without torch installed;
    `training/*.py` wraps it with the torch Dataset base class at use time).

    Args:
        root: path to the UBFC-rPPG directory (or set the
            `MASKFUSIONNET_UBFC_ROOT` environment variable and leave this
            as None -- never hard-code a local path).
        subjects: which subject IDs to include (from a `splits.SplitResult`).
        clip_length: frames per clip (paper default: 160, Section IV-B).
        stride: frame stride between consecutive sampled clips from the
            same video (defaults to `clip_length`, i.e. non-overlapping
            clips; set smaller for overlapping/denser sampling).
        image_size: face-crop output resolution (paper: 128).
        fps: assumed video frame rate for BVP/HR alignment (UBFC-rPPG is
            recorded at ~30 fps; pass the true value from the video's own
            metadata if it differs for a given file).
    """

    def __init__(self, root: str | Path, subjects: list[str], clip_length: int = 160,
                 stride: int | None = None, image_size: int = 128, fps: float = 30.0,
                 cascade_path: str | None = None):
        self.root = Path(root)
        self.subjects = subjects
        self.clip_length = clip_length
        self.stride = stride or clip_length
        self.image_size = image_size
        self.fps = fps
        self._cropper = FaceCropper(cascade_path=cascade_path, image_size=image_size)

        self.index: list[UBFCClipIndex] = []
        self._frame_counts: dict[str, int] = {}
        for subj in subjects:
            n_frames = self._count_frames(subj)
            self._frame_counts[subj] = n_frames
            for start in range(0, max(1, n_frames - clip_length + 1), self.stride):
                self.index.append(UBFCClipIndex(subj, start, clip_length))

    def _video_path(self, subject: str) -> Path:
        return self.root / subject / "vid.avi"

    def _gt_path(self, subject: str) -> Path:
        return self.root / subject / "ground_truth.txt"

    def _count_frames(self, subject: str) -> int:
        import cv2

        cap = cv2.VideoCapture(str(self._video_path(subject)))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        item = self.index[idx]
        frames = []
        for i, frame_bgr in enumerate(read_video_frames(self._video_path(item.subject))):
            if i < item.start_frame:
                continue
            if i >= item.start_frame + item.clip_length:
                break
            face = self._cropper(frame_bgr)
            frames.append(normalize_frame(face))
        if len(frames) < item.clip_length:
            # Corrupt/short video tail: pad by repeating the last good frame
            # rather than silently returning a short clip that would break
            # batching.
            pad = [frames[-1]] * (item.clip_length - len(frames)) if frames else \
                [np.zeros((3, self.image_size, self.image_size), dtype=np.float32)] * item.clip_length
            frames = frames + pad

        clip = np.stack(frames, axis=1)  # [3, T, H, W]

        bvp_full = load_bvp_ground_truth(self._gt_path(item.subject))
        n_frames_total = self._frame_counts[item.subject]
        bvp_full = align_bvp_to_frames(bvp_full, n_frames_total)
        bvp = bvp_full[item.start_frame: item.start_frame + item.clip_length]
        if len(bvp) < item.clip_length:
            bvp = np.pad(bvp, (0, item.clip_length - len(bvp)), mode="edge")

        # Scalar ground-truth HR for this clip, for the frequency-domain
        # term of L_pre (Eq. 18). Estimated spectrally from the clip's own
        # BVP segment rather than trusting row 1 of ground_truth.txt
        # directly, since that per-sample HR trace's sampling rate is not
        # guaranteed to line up with `align_bvp_to_frames`'s frame-indexed
        # BVP row -- a self-consistent spectral estimate avoids a second,
        # separately-aligned ground-truth source.
        try:
            filtered = bandpass_filter(bvp, 40 / 60, 180 / 60, fs=self.fps)
            hr_bpm = dominant_frequency(filtered, fs=self.fps, low_hz=40 / 60, high_hz=180 / 60) * 60.0
        except ValueError:
            hr_bpm = 75.0  # degenerate/all-zero clip fallback (documented, not silent)

        return {
            "clip": clip.astype(np.float32),
            "bvp": bvp.astype(np.float32),
            "hr_bpm": np.float32(hr_bpm),
            "subject": item.subject,
            "fps": self.fps,
        }
