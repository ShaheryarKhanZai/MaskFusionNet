#!/usr/bin/env python3
"""Real-time webcam heart-rate demo.

Pipeline (matches the task brief exactly):

    Webcam -> Face detection -> Rolling temporal buffer -> MaskFusionNet
           -> rPPG -> Bandpass -> FFT -> BPM

FIXES relative to the original repository's `Main.py`:

- The original injected literal random jitter into the *displayed* number:
      if random.uniform(1,5) > 3:
          if random.uniform(1,5) > 3:
              hr1 = hr + random.randint(-1,1)
  That code path is gone. The BPM shown on screen is always either the
  freshly computed estimate or the actual smoothed value described below
  -- never a randomly perturbed one.
- Smoothing, when used, is a real, documented method: a simple moving
  average over the last `--smooth_window` raw BPM estimates (default 5),
  clearly separate from the "instant" (unsmoothed) estimate, which is also
  displayed so the two are never confused.
- `--checkpoint` is loaded with an explicit missing/unexpected-key report
  (see utils/checkpoints.py) instead of a silent `strict=False`.
- "Signal quality" is the actual SNR of the filtered waveform (see
  utils/metrics.py:snr), not a made-up label.
- FPS shown is the actual measured capture-loop rate.

This is a single-threaded loop rather than the original's three-process
`multiprocessing.Manager` design: at ~1 inference per second on a rolling
buffer, multiprocessing bought no real latency benefit here and made the
"is this number real" question much harder to audit (state passed through
shared `Manager.Value`s from three separate processes). If you need higher
throughput, the pipeline stages are still cleanly separated by function
below and can be moved back onto separate threads/processes.
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import torch

from maskfusionnet.data.preprocessing import FaceCropper, normalize_frame
from maskfusionnet.models import MaskFusionNet
from maskfusionnet.signal.heart_rate import waveform_to_bpm
from maskfusionnet.utils import load_checkpoint
from maskfusionnet.utils.metrics import snr as compute_snr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--camera_index", type=int, default=0)
    ap.add_argument("--clip_length", type=int, default=100)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--embed_dim", type=int, default=96)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--infer_every_n_frames", type=int, default=15)
    ap.add_argument("--smooth_window", type=int, default=5,
                     help="Moving-average window (in # of estimates) for the displayed BPM.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = MaskFusionNet(embed_dim=args.embed_dim, num_heads=args.num_heads, clip_length=args.clip_length)
    load_checkpoint(args.checkpoint, model, map_location=args.device)
    model.to(args.device).eval()

    cropper = FaceCropper(image_size=args.image_size)
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera_index}.")

    frame_buffer: deque = deque(maxlen=args.clip_length)
    bpm_history: deque = deque(maxlen=args.smooth_window)
    fps_history: deque = deque(maxlen=30)

    status = "Capturing frames"
    instant_bpm, smoothed_bpm, quality_db, measured_fps = None, None, None, 0.0
    frame_count = 0

    print("Starting webcam demo. Press 'q' to quit. Make sure your face is visible with consistent lighting.")

    try:
        while True:
            t_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            face = cropper(frame)
            frame_buffer.append(normalize_frame(face))
            frame_count += 1

            if len(frame_buffer) == args.clip_length and frame_count % args.infer_every_n_frames == 0:
                status = "Predicting"
                clip = np.stack(list(frame_buffer), axis=1).astype(np.float32)
                clip_t = torch.from_numpy(clip).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    pred_signal = model(clip_t).cpu().numpy()[0]

                fs_est = float(np.mean(fps_history)) if fps_history else 30.0
                try:
                    result = waveform_to_bpm(pred_signal, fs=fs_est)
                    instant_bpm = result["bpm"]
                    quality_db = compute_snr(result["filtered_signal"], fs_est, instant_bpm / 60.0)
                    bpm_history.append(instant_bpm)
                    smoothed_bpm = float(np.mean(bpm_history))
                    status = "Predicted"
                except ValueError:
                    status = "Signal out of band -- adjust lighting/position"
            elif len(frame_buffer) < args.clip_length:
                status = f"Filling buffer ({len(frame_buffer)}/{args.clip_length})"

            loop_fps = 1.0 / max(1e-6, time.time() - t_start)
            fps_history.append(loop_fps)
            measured_fps = float(np.mean(fps_history))

            display = frame.copy()
            if smoothed_bpm is not None:
                cv2.putText(display, f"Heart Rate: {smoothed_bpm:.1f} bpm (instant: {instant_bpm:.1f})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                quality_txt = f"{quality_db:.1f} dB" if np.isfinite(quality_db) else "n/a"
                cv2.putText(display, f"Signal quality (SNR): {quality_txt}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Status: {status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"FPS: {measured_fps:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("MaskFusionNet - Heart Rate Monitor", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
