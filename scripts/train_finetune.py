#!/usr/bin/env python3
"""Stage 2 CLI: rPPG fine-tuning on UBFC-rPPG, optionally initialized from
a Stage-1 pretraining checkpoint via the paper's Section III-A-2-a
weight-loading rule (see models/maskfusionnet.py:load_pretrained_encoder).

Usage:
    python scripts/train_finetune.py --config configs/finetune.yaml
    python scripts/train_finetune.py --config configs/finetune.yaml --no_pretrain
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader

from maskfusionnet.data import discover_subjects, subject_level_split, UBFCrPPGDataset
from maskfusionnet.models import MaskFusionNet, MaskFusionNetPretrain
from maskfusionnet.training import Trainer, finetune_step
from maskfusionnet.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--no_pretrain", action="store_true",
                     help="Skip loading a Stage-1 checkpoint (ablation: fine-tune from scratch).")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_cfg, model_cfg, train_cfg, out_cfg = cfg["data"], cfg["model"], cfg["training"], cfg["output"]

    root = data_cfg["root"]
    if not root or root.startswith("${"):
        raise SystemExit(
            "data.root is unset. Set the MASKFUSIONNET_UBFC_ROOT environment variable, "
            "or edit configs/finetune.yaml, to point at your UBFC-rPPG directory."
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

    model = MaskFusionNet(
        embed_dim=model_cfg["embed_dim"], num_heads=model_cfg["num_heads"],
        branch_layers=model_cfg["branch_layers"], fusion_layers=model_cfg["fusion_layers"],
        mfb_fuse_at=model_cfg["mfb_fuse_at"], clip_length=data_cfg["clip_length"],
    )

    pretrain_cfg = cfg.get("pretrained_encoder", {})
    ckpt_path = pretrain_cfg.get("checkpoint")
    if ckpt_path and not args.no_pretrain:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise SystemExit(
                f"pretrained_encoder.checkpoint={ckpt_path} does not exist. "
                f"Run train_pretrain.py first, or pass --no_pretrain to fine-tune from scratch."
            )
        pre = MaskFusionNetPretrain(
            embed_dim=model_cfg["embed_dim"], num_heads=model_cfg["num_heads"],
            clip_length=data_cfg["clip_length"],
        )
        payload = torch.load(ckpt_path, map_location="cpu")
        pre.load_state_dict(payload["model_state_dict"])
        report = model.load_pretrained_encoder(pre, strict_shapes=pretrain_cfg.get("strict_shapes", True))
        print(f"[train_finetune] Loaded pretrained weights: {report}")
    else:
        print("[train_finetune] Fine-tuning from randomly initialized weights "
              "(no_pretrain=True or no checkpoint configured).")

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"],
                                  weight_decay=train_cfg["weight_decay"])
    trainer = Trainer(model, optimizer, finetune_step, device=args.device,
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
