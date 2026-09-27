"""Stage 1: masking-reconstruction pre-training model.

Section III-A-1 / Figure 2 (top half):

    X --Phi_conv--> X_inter --Phi_emb--> X_tube --Phi_mask--> X_vis
      --Phi_enc--> X_enc --Phi_dec--> X_dec  (reconstruction target: X_tube)
      X_enc --Predictor--> heart-rate signal (prediction loss target)

This is the piece that was entirely missing from the original repository:
`TubeMasking` and a pretraining-only `Predictor` existed as dead classes,
but `MaskFusionNet.forward()` never called masking, there was no decoder
module anywhere, and there was no training script that would ever compute
a reconstruction loss. Concretely, Stage 1 as described by the paper was
not implemented -- only a (buggy) Stage-2-shaped forward pass existed.

Encoder: 12 transformer layers (Phi_enc), Section III-B-4 ("L in Phi_enc,
Psi_enc, Theta_enc, Gamma_enc [set] to 12, 8, 8, 4 respectively").
Decoder: 6 transformer layers (Phi_dec), Section III-B-6 ("L_d ... set to
6"), operating on the flattened token sequence with linear-projection
attention (see transformer.py: TransformerDecoderLayer).
"""
from __future__ import annotations

import torch
from torch import nn

from .backbone import ConvBlock, EmbeddingLayer
from .masking import TubeMasking
from .transformer import TransformerEncoderLayer, TransformerDecoderLayer
from .predictor import Predictor

__all__ = ["MaskFusionNetPretrain"]


class MaskFusionNetPretrain(nn.Module):
    def __init__(
        self,
        embed_dim: int = 96,
        num_heads: int = 4,
        mask_ratio: float = 0.75,
        encoder_layers: int = 12,
        decoder_layers: int = 6,
        decoder_heads: int = 4,
        tube_size: tuple[int, int, int] = (4, 4, 4),
        clip_length: int = 160,
    ):
        super().__init__()
        self.conv_block = ConvBlock(in_channels=3, out_channels=embed_dim)   # Phi_conv
        self.embedding = EmbeddingLayer(embed_dim, embed_dim, tube_size)     # Phi_emb
        self.masking = TubeMasking(embed_dim, mask_ratio)                   # Phi_mask
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads) for _ in range(encoder_layers)
        ])                                                                  # Phi_enc
        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, decoder_heads) for _ in range(decoder_layers)
        ])                                                                  # Phi_dec
        self.predictor = Predictor(embed_dim, output_length=clip_length)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: raw clip [B, 3, T, H, W]
        Returns:
            dict with:
              x_tube:      embedded tokens before masking [B,C,Tt,Ht,Wt]
                           (reconstruction target)
              x_dec:       reconstructed tokens, same shape as x_tube
              mask:        bool mask [B,1,Tt,Ht,Wt], True at masked positions
              hr_signal:   predicted 1D signal from X_enc, [B, T]  (for L_pre)
        """
        x_inter = self.conv_block(x)
        x_tube = self.embedding(x_inter)                 # [B,C,Tt,Ht,Wt]
        x_vis, mask = self.masking(x_tube)

        x_enc = x_vis
        for layer in self.encoder:
            x_enc = layer(x_enc)

        hr_signal = self.predictor(x_enc)

        b, c, t, h, w = x_enc.shape
        dec_in = x_enc.flatten(2).transpose(1, 2)         # [B, N, C] for the linear decoder
        for layer in self.decoder:
            dec_in = layer(dec_in)
        x_dec = dec_in.transpose(1, 2).reshape(b, c, t, h, w)

        return {"x_tube": x_tube, "x_dec": x_dec, "mask": mask, "hr_signal": hr_signal}
