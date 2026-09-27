#!/usr/bin/env python3
"""Stage 1 CLI: masked pretraining on UBFC-rPPG.

Usage:
    python scripts/train_pretrain.py --config configs/pretrain.yaml
    python scripts/train_pretrain.py --config configs/pretrain.yaml --resume checkpoints/pretrain/epoch_50.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader

from maskfusionnet.data import discover_subjects, subject_level_split, UBFCrPPGDataset
from maskfusionnet.models import MaskFusionNetPretrain
from maskfusionnet.training import Trainer, pretrain_step
from maskfusionnet.utils import load_config


def build_step_fn(cfg):
    loss_cfg = cfg.get("loss", {})

    def step(model, batch, device, epoch):
        return pretrain_step(model, batch, device, epoch,
                              alpha=loss_cfg.get("alpha", 1.0),
                              beta=loss_cfg.get("beta", 1.0),
                              gamma=loss_cfg.get("gamma", 1.0))
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_cfg, model_cfg, train_cfg, out_cfg = cfg["data"], cfg["model"], cfg["training"], cfg["output"]

    root = data_cfg["root"]
    if not root or root.startswith("${"):
        raise SystemExit(
            "data.root is unset. Set the MASKFUSIONNET_UBFC_ROOT environment variable, "
            "or edit configs/pretrain.yaml, to point at your UBFC-rPPG directory."
        )

    subjects = discover_subjects(root)
    if len(subjects) < 3:
        raise SystemExit(f"Found only {len(subjects)} subject(s) under {root}; need >= 3 for a split.")
    split = subject_level_split(subjects, data_cfg["train_frac"], data_cfg["val_frac"], data_cfg["split_seed"])

    def make_loader(subj_list, shuffle):
        ds = UBFCrPPGDataset(root, subj_list, clip_length=data_cfg["clip_length"],
                              stride=data_cfg["stride"], image_size=data_cfg["image_size"],
                              fps=data_cfg["fps"])
        return DataLoader(ds, batch_size=train_cfg["batch_size"], shuffle=shuffle,
                           num_workers=train_cfg.get("num_workers", 0))

    train_loader = make_loader(split.train, shuffle=True)
    val_loader = make_loader(split.val, shuffle=False) if split.val else None

    model = MaskFusionNetPretrain(
        embed_dim=model_cfg["embed_dim"], num_heads=model_cfg["num_heads"],
        mask_ratio=model_cfg["mask_ratio"], encoder_layers=model_cfg["encoder_layers"],
        decoder_layers=model_cfg["decoder_layers"], decoder_heads=model_cfg["decoder_heads"],
        tube_size=tuple(model_cfg["tube_size"]), clip_length=data_cfg["clip_length"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"],
                                  weight_decay=train_cfg["weight_decay"])

    trainer = Trainer(model, optimizer, build_step_fn(cfg), device=args.device,
                       use_amp=train_cfg.get("use_amp", False),
                       grad_accum_steps=train_cfg.get("grad_accum_steps", 1),
                       checkpoint_dir=out_cfg["checkpoint_dir"], log_dir=out_cfg["log_dir"],
                       config=cfg)
    if args.resume:
        trainer.resume(args.resume)

    trainer.fit(train_loader, val_loader, epochs=train_cfg["epochs"],
                checkpoint_every=train_cfg.get("checkpoint_every", 1))


if __name__ == "__main__":
    main()
