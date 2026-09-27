"""Convolutional stem and tube-embedding layer.

Implements Section III-B-1 and III-B-2 of the paper:

    "MaskFusionNet: A Dual-Stream Fusion Model With Masked Pre-Training
    Mechanism for rPPG Measurement" (Zhang et al., TCSVT 2024).

Eq. (1):   X_inter = Phi_conv(X)
Eq. (2):   X_tube  = Phi_emb(X_inter),  with T_t = floor(T_i / T_hat) etc.

Three independent convolution blocks exist in the paper (Phi_conv for the
pre-training model, Theta_conv for the Segmented Branch / SEB, Psi_conv for
the Adjacent Branch / ADB) and three independent embedding layers
(Phi_emb, Theta_emb, Psi_emb). The paper states the three ConvBlocks "have
identical structure" but does not say they share weights -- each stream
(pretrain / ADB / SEB) gets its **own instance** with independently learned
parameters. This module intentionally does not implement any weight
sharing; callers must instantiate one `ConvBlock` per stream. This directly
fixes a bug in the original repository where `MaskFusionNet.forward()`
reused a single `ConvBlock` instance for both ADB and SEB.
"""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["ConvBlock", "EmbeddingLayer"]


class ConvBlock(nn.Module):
    """Shallow 3D convolutional stem, Section III-B-1.

    "each convolution block consists of three convolutional layers with
    kernel sizes of (1x5x5), (3x3x3) and (3x3x3) respectively. Each
    convolutional layer is cascaded with batch normalization (BN), ReLU
    activation, and MaxPool operation." The paper does not spell out the
    exact pooling stride. We assume (as is standard for rPPG transformer
    stems such as PhysFormer's) a spatial-only 2x downsample per layer via
    MaxPool3d(kernel=(1, 2, 2)), which preserves full temporal resolution
    for the tube-embedding step that follows and yields an overall 8x
    spatial reduction across the three layers.

    ASSUMPTION (documented, not silently invented): pooling kernel/stride
    = (1, 2, 2) at every layer. If the authors' released code specifies a
    different stride, only the `pool` kernel argument here needs to change.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 96,
                 pool_kernel: tuple[int, int, int] = (1, 2, 2)):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 5, 5), padding=(0, 2, 2))
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=1)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.pool = nn.MaxPool3d(pool_kernel, pool_kernel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T, H, W]
        x = self.relu(self.bn1(self.pool(self.conv1(x))))
        x = self.relu(self.bn2(self.pool(self.conv2(x))))
        x = self.relu(self.bn3(self.pool(self.conv3(x))))
        return x  # [B, C_out, T, H/8, W/8]


class EmbeddingLayer(nn.Module):
    """Tube-token embedding, Section III-B-2, Eq. (2).

    A single 3D convolution with kernel == stride == tube size
    (T_hat, H_hat, W_hat), turning the dense feature map into a grid of
    non-overlapping spatio-temporal tokens.

    Paper's configured tube sizes (Section III-A-2-b):
        pre-training (Phi_emb): (4, 4, 4)
        SEB (Theta_emb):        (4, 4, 4)
        ADB (Psi_emb):          (2, 4, 4)
    """

    def __init__(self, in_channels: int = 96, out_channels: int = 96,
                 tube_size: tuple[int, int, int] = (4, 4, 4)):
        super().__init__()
        self.tube_size = tube_size
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=tube_size, stride=tube_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T_i, H_i, W_i] -> [B, C_out, T_t, H_t, W_t]
        return self.proj(x)
