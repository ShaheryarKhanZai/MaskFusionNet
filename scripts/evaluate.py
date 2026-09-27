#!/usr/bin/env python3
"""Evaluate a fine-tuned checkpoint on the held-out (subject-independent)
UBFC-rPPG test split. Reports MAE / RMSE / Pearson r on per-clip HR
estimates, and SNR on the filtered waveform, per README's metrics table.

Usage:
    python scripts/evaluate.py --config configs/finetune.yaml --checkpoint checkpoints/finetune/best.pt
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from maskfusionnet.data import discover_subjects, subject_level_split, UBFCrPPGDataset
from maskfusionnet.models import MaskFusionNet
from maskfusionnet.signal.heart_rate import waveform_to_bpm
from maskfusionnet.utils import hr_metrics_report, snr, load_config, load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_json", type=str, default="results/tables/eval_results.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_cfg, model_cfg = cfg["data"], cfg["model"]

    root = data_cfg["root"]
    if not root or root.startswith("${"):
        raise SystemExit("data.root is unset. Set MASKFUSIONNET_UBFC_ROOT or edit the config.")

    subjects = discover_subjects(root)
    split = subject_level_split(subjects, data_cfg["train_frac"], data_cfg["val_frac"], data_cfg["split_seed"])
    test_ds = UBFCrPPGDataset(root, split.test, clip_length=data_cfg["clip_length"],
                               stride=data_cfg["clip_length"],  # non-overlapping at test time
                               image_size=data_cfg["image_size"], fps=data_cfg["fps"])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model = MaskFusionNet(embed_dim=model_cfg["embed_dim"], num_heads=model_cfg["num_heads"],
                           branch_layers=model_cfg["branch_layers"], fusion_layers=model_cfg["fusion_layers"],
                           mfb_fuse_at=model_cfg["mfb_fuse_at"], clip_length=data_cfg["clip_length"])
    load_checkpoint(args.checkpoint, model, map_location=args.device)
    model.to(args.device).eval()

    pred_bpms, gt_bpms, snrs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            clip = batch["clip"].to(args.device)
            pred_signal = model(clip).cpu().numpy()[0]
            gt_signal = batch["bvp"].numpy()[0]
            fps = float(batch["fps"][0])

            pred_result = waveform_to_bpm(pred_signal, fs=fps)
            gt_result = waveform_to_bpm(gt_signal, fs=fps)

            pred_bpms.append(pred_result["bpm"])
            gt_bpms.append(gt_result["bpm"])
            snrs.append(snr(pred_result["filtered_signal"], fps, pred_result["bpm"] / 60.0))

    if not pred_bpms:
        raise SystemExit("Test split produced zero clips -- check clip_length vs. video lengths.")

    report = hr_metrics_report(np.array(pred_bpms), np.array(gt_bpms))
    report["SNR_dB_mean"] = float(np.mean([s for s in snrs if np.isfinite(s)]))
    report["n_test_clips"] = len(pred_bpms)
    report["test_subjects"] = split.test

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved to {args.out_json}")


if __name__ == "__main__":
    main()
