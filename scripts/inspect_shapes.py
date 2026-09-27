#!/usr/bin/env python3
"""Run a dummy input through both stages of MaskFusionNet and print every
major tensor shape -- the "complete tensor-shape trace" the task brief
asks for. Uses no dataset; pure synthetic input, so it always runs.

Usage:
    python scripts/inspect_shapes.py                  # paper-scale defaults
    python scripts/inspect_shapes.py --T 32 --H 64 --W 64 --batch 1  # faster on CPU
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from maskfusionnet.models.backbone import ConvBlock, EmbeddingLayer
from maskfusionnet.models import MaskFusionNetPretrain, MaskFusionNet


def trace_pretrain(x, embed_dim, num_heads):
    print("\n=== Stage 1: MaskFusionNetPretrain ===")
    model = MaskFusionNetPretrain(embed_dim=embed_dim, num_heads=num_heads, clip_length=x.shape[2])
    model.eval()

    with torch.no_grad():
        x_inter = model.conv_block(x)
        print(f"Input:                 {tuple(x.shape)}")
        print(f"After ConvBlock:       {tuple(x_inter.shape)}")
        x_tube = model.embedding(x_inter)
        print(f"After Embedding (tube):{tuple(x_tube.shape)}")
        x_vis, mask = model.masking(x_tube)
        print(f"After TubeMasking:     {tuple(x_vis.shape)}  (mask ratio={mask.float().mean():.3f})")
        x_enc = x_vis
        for layer in model.encoder:
            x_enc = layer(x_enc)
        print(f"After Encoder (12 lyr):{tuple(x_enc.shape)}")
        hr_signal = model.predictor(x_enc)
        print(f"Predictor output:      {tuple(hr_signal.shape)}")
        out = model(x)
        print(f"Decoder output x_dec:  {tuple(out['x_dec'].shape)}  (== x_tube shape: {out['x_dec'].shape == x_tube.shape})")
    return out


def trace_finetune(x, embed_dim, num_heads):
    print("\n=== Stage 2: MaskFusionNet (fine-tuning) ===")
    model = MaskFusionNet(embed_dim=embed_dim, num_heads=num_heads, clip_length=x.shape[2])
    model.eval()
    with torch.no_grad():
        adb = model.adb_embed(model.adb_conv(x))
        seb = model.seb_embed(model.seb_conv(x))
        print(f"Input:                 {tuple(x.shape)}")
        print(f"ADB tokens (tube 2,4,4): {tuple(adb.shape)}")
        print(f"SEB tokens (tube 4,4,4): {tuple(seb.shape)}")

        for i in range(model.mfb_fuse_at):
            adb = model.adb_encoder[i](adb)
        for i in range(model.mfb_fuse_at):
            seb = model.seb_encoder[i](seb)
        print(f"ADB after 4 layers:    {tuple(adb.shape)}")
        print(f"SEB after 4 layers:    {tuple(seb.shape)}")

        adb_aligned = model.adb_temporal_align(adb)
        print(f"ADB temporally aligned to SEB: {tuple(adb_aligned.shape)}")
        seb = model.mfb_mid(x_seb=seb, x_adb=adb_aligned)
        print(f"After MFB #1 (-> new SEB stream): {tuple(seb.shape)}")

        for i in range(model.mfb_fuse_at, model.branch_layers):
            adb = model.adb_encoder[i](adb)
        for i in range(model.mfb_fuse_at, model.branch_layers):
            seb = model.seb_encoder[i](seb)
        print(f"ADB after 8 layers:    {tuple(adb.shape)}")
        print(f"SEB after 8 layers:    {tuple(seb.shape)}")

        adb_aligned = model.adb_temporal_align(adb)
        fused = model.mfb_final(x_seb=seb, x_adb=adb_aligned)
        print(f"After MFB #2 (final fusion): {tuple(fused.shape)}")

        for layer in model.fusion_encoder:
            fused = layer(fused)
        print(f"After Fusion Encoder (4 lyr): {tuple(fused.shape)}")

        pred = model.predictor(fused)
        print(f"Predictor output (rPPG): {tuple(pred.shape)}")
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=160, help="clip length (paper default: 160)")
    ap.add_argument("--H", type=int, default=128, help="face crop height (paper default: 128)")
    ap.add_argument("--W", type=int, default=128, help="face crop width (paper default: 128)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--embed_dim", type=int, default=96)
    ap.add_argument("--num_heads", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(0)
    x = torch.randn(args.batch, 3, args.T, args.H, args.W)

    trace_pretrain(x, args.embed_dim, args.num_heads)
    trace_finetune(x, args.embed_dim, args.num_heads)


if __name__ == "__main__":
    main()
