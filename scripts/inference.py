#!/usr/bin/env python3
"""Run inference on a single video file (offline, not the live webcam
demo -- see realtime_demo.py for that) and report an estimated BPM, plus
optionally save the predicted-vs-ground-truth BVP plot and spectrum plot.

Usage:
    python scripts/inference.py --checkpoint checkpoints/finetune/best.pt \
        --video path/to/vid.avi --fps 30 [--ground_truth path/to/ground_truth.txt]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from maskfusionnet.data.preprocessing import FaceCropper, normalize_frame, read_video_frames, \
    load_bvp_ground_truth, align_bvp_to_frames
from maskfusionnet.models import MaskFusionNet
from maskfusionnet.signal.heart_rate import waveform_to_bpm
from maskfusionnet.utils import load_checkpoint
from maskfusionnet.utils.visualization import plot_bvp_comparison, plot_spectrum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--ground_truth", type=str, default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--clip_length", type=int, default=160)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--embed_dim", type=int, default=96)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--out_dir", type=str, default="results/figures")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cropper = FaceCropper(image_size=args.image_size)
    frames = []
    for frame_bgr in read_video_frames(args.video):
        frames.append(normalize_frame(cropper(frame_bgr)))
        if len(frames) == args.clip_length:
            break
    if len(frames) < args.clip_length:
        raise SystemExit(f"Video only has {len(frames)} frames; need {args.clip_length}.")

    clip = np.stack(frames, axis=1).astype(np.float32)  # [3,T,H,W]
    clip_t = torch.from_numpy(clip).unsqueeze(0).to(args.device)

    model = MaskFusionNet(embed_dim=args.embed_dim, num_heads=args.num_heads, clip_length=args.clip_length)
    load_checkpoint(args.checkpoint, model, map_location=args.device)
    model.to(args.device).eval()

    with torch.no_grad():
        pred_signal = model(clip_t).cpu().numpy()[0]

    result = waveform_to_bpm(pred_signal, fs=args.fps)
    print(f"Estimated heart rate: {result['bpm']:.1f} bpm")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    plot_spectrum(result["filtered_signal"], args.fps, Path(args.out_dir) / "inference_spectrum.png",
                  estimated_bpm=result["bpm"])

    if args.ground_truth:
        gt_full = load_bvp_ground_truth(args.ground_truth)
        gt_full = align_bvp_to_frames(gt_full, len(frames))
        plot_bvp_comparison(pred_signal, gt_full, args.fps, Path(args.out_dir) / "inference_bvp_comparison.png")
        gt_result = waveform_to_bpm(gt_full, fs=args.fps)
        print(f"Ground-truth heart rate (from provided BVP): {gt_result['bpm']:.1f} bpm")

    print(f"Figures written to {args.out_dir}/")


if __name__ == "__main__":
    main()
