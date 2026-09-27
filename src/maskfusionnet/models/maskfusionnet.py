"""Stage 2: multi-scale dual-stream fine-tuning model.

Section III-A-2 / Figure 2 (bottom half). Two independent streams:

    Adjacent Branch  (ADB, Psi_*):   tube (2,4,4), 8 transformer layers
    Segmented Branch (SEB, Theta_*): tube (4,4,4), 8 transformer layers

fused twice by a Multi-Scale Fusion Block (once after each branch's 4th
layer, once after each branch's 8th/last layer), then encoded by a 4-layer
Fusion Feature Encoder (Gamma_enc) and read out by the Predictor.

BUGS FIXED relative to the original repository's `MaskFusionNet.py`:

1. `self.ADB_encoder`, `self.SEB_encoder`, and `self.fusion_encoder` were
   all built by slicing the *same* `nn.ModuleList` of 12 `TransformerLayer`
   objects (`self.encoder`). Because `nn.Module` stores submodules by
   reference, ADB layers 0-7 and SEB layers 0-7 were literally the same
   parameter tensors -- gradients from both branches accumulated on one
   shared set of weights every step, and the "fusion encoder" was reusing
   the tail of that same list. This is not "the same architecture," it
   silently merges three modules the paper describes as independent. Here,
   ADB, SEB, and the fusion encoder are built from three separate
   `nn.ModuleList`s with their own parameters.
2. ADB and SEB also shared a single `ConvBlock` instance in the original
   code (`x = self.conv_block(x)` computed once, then fed to both
   `self.embedding1` (ADB) and `self.embedding` (SEB)). The paper names
   distinct blocks Psi_conv and Theta_conv; we give each branch its own
   `ConvBlock`.
3. The two `self.MFB(...)` calls reused one `MultiScaleFusionBlock`
   instance; here they are two independently parameterized instances
   (see mfb.py docstring for the textual justification).
4. `load_pretrained_encoder()` below implements the loading rule from
   Section III-A-2-a, which the original repository did not implement at
   all (no checkpoint bridging code existed): the first 8 of Phi_enc's 12
   layers are copied into Theta_enc (SEB), the remaining 4 into Gamma_enc
   (fusion encoder), and Psi_enc (ADB) is deliberately left randomly
   initialized because its tube size (2,4,4) differs from the
   pre-training tube size (4,4,4), so its token/channel layout does not
   match Phi_enc's learned weights.
"""
from __future__ import annotations

import torch
from torch import nn

from .backbone import ConvBlock, EmbeddingLayer
from .transformer import TransformerEncoderLayer
from .mfb import MultiScaleFusionBlock
from .predictor import Predictor
from .pretraining import MaskFusionNetPretrain

__all__ = ["MaskFusionNet"]

ADB_TUBE = (2, 4, 4)
SEB_TUBE = (4, 4, 4)


class MaskFusionNet(nn.Module):
    def __init__(
        self,
        embed_dim: int = 96,
        num_heads: int = 4,
        branch_layers: int = 8,
        fusion_layers: int = 4,
        mfb_fuse_at: int = 4,
        clip_length: int = 160,
    ):
        super().__init__()
        assert 0 < mfb_fuse_at < branch_layers
        self.mfb_fuse_at = mfb_fuse_at
        self.branch_layers = branch_layers

        # --- Adjacent Branch (Psi_*) ---
        self.adb_conv = ConvBlock(in_channels=3, out_channels=embed_dim)
        self.adb_embed = EmbeddingLayer(embed_dim, embed_dim, ADB_TUBE)
        self.adb_encoder = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads) for _ in range(branch_layers)
        ])
        # ADB has 2x SEB's temporal token count (tube stride 2 vs 4);
        # align before each fusion call. See mfb.py docstring.
        self.adb_temporal_align = nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

        # --- Segmented Branch (Theta_*) ---
        self.seb_conv = ConvBlock(in_channels=3, out_channels=embed_dim)
        self.seb_embed = EmbeddingLayer(embed_dim, embed_dim, SEB_TUBE)
        self.seb_encoder = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads) for _ in range(branch_layers)
        ])

        # --- Multi-Scale Fusion Blocks (two independent instances) ---
        self.mfb_mid = MultiScaleFusionBlock(embed_dim, num_heads)
        self.mfb_final = MultiScaleFusionBlock(embed_dim, num_heads)

        # --- Fusion Feature Encoder (Gamma_*) ---
        self.fusion_encoder = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads) for _ in range(fusion_layers)
        ])

        self.predictor = Predictor(embed_dim, output_length=clip_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: raw clip [B, 3, T, H, W] -> predicted rPPG signal [B, T]."""
        adb = self.adb_embed(self.adb_conv(x))   # [B,C,T/2,H/8/4,W/8/4]
        seb = self.seb_embed(self.seb_conv(x))   # [B,C,T/4, same H,W]

        for i in range(self.mfb_fuse_at):
            adb = self.adb_encoder[i](adb)
        for i in range(self.mfb_fuse_at):
            seb = self.seb_encoder[i](seb)

        adb_aligned = self.adb_temporal_align(adb)
        seb = self.mfb_mid(x_seb=seb, x_adb=adb_aligned)   # -> new SEB stream (Eq. 8)

        for i in range(self.mfb_fuse_at, self.branch_layers):
            adb = self.adb_encoder[i](adb)
        for i in range(self.mfb_fuse_at, self.branch_layers):
            seb = self.seb_encoder[i](seb)

        adb_aligned = self.adb_temporal_align(adb)
        fused = self.mfb_final(x_seb=seb, x_adb=adb_aligned)

        for layer in self.fusion_encoder:
            fused = layer(fused)

        return self.predictor(fused)

    def load_pretrained_encoder(self, pretrained: "MaskFusionNetPretrain | dict",
                                 strict_shapes: bool = True) -> dict:
        """Section III-A-2-a weight-loading rule.

        "the parameters of the first 8 transformer layers in Phi_enc are
        loaded into Theta_enc [SEB]... the parameters of the remaining 4
        transformer layers in Phi_enc are loaded into Gamma_enc [fusion
        encoder]... Psi_enc [ADB] does not perform parameter loading" due
        to the tube-size mismatch.

        Unlike the original code's blanket
        `load_state_dict(..., strict=False)` (which silently swallows any
        shape mismatch anywhere in the whole model with no report), this
        method only ever touches Theta_enc and Gamma_enc, loads with
        `strict=True` for each of those two subsets, and returns a report
        so missing/unexpected keys are never silent.
        """
        if isinstance(pretrained, MaskFusionNetPretrain):
            pre_encoder = pretrained.encoder
        else:
            # Accept a plain state_dict keyed like "encoder.<idx>.<...>"
            pre_encoder = pretrained

        report = {"seb_layers_loaded": 0, "fusion_layers_loaded": 0, "adb_layers_loaded": 0}

        def _layer_state(idx: int) -> dict:
            if isinstance(pre_encoder, nn.ModuleList):
                return pre_encoder[idx].state_dict()
            prefix = f"encoder.{idx}."
            return {k[len(prefix):]: v for k, v in pre_encoder.items() if k.startswith(prefix)}

        n_pretrain_layers = len(pre_encoder) if isinstance(pre_encoder, nn.ModuleList) else \
            (max(int(k.split(".")[1]) for k in pre_encoder if k.startswith("encoder.")) + 1)
        assert n_pretrain_layers >= len(self.seb_encoder) + len(self.fusion_encoder), (
            "Pretrained encoder has fewer layers than SEB+FusionEncoder require."
        )

        for i, layer in enumerate(self.seb_encoder):
            layer.load_state_dict(_layer_state(i), strict=strict_shapes)
            report["seb_layers_loaded"] += 1

        offset = len(self.seb_encoder)
        for i, layer in enumerate(self.fusion_encoder):
            layer.load_state_dict(_layer_state(offset + i), strict=strict_shapes)
            report["fusion_layers_loaded"] += 1

        # ADB (Psi_enc) intentionally left as-is: tube size (2,4,4) != (4,4,4),
        # so its per-layer tensor shapes do not match Phi_enc's.
        return report
