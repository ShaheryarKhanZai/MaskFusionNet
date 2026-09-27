import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maskfusionnet.models import MaskFusionNetPretrain, MaskFusionNet


def _tiny_clip(batch=2, T=16, H=32, W=32):
    torch.manual_seed(0)
    return torch.randn(batch, 3, T, H, W)


def test_pretrain_forward_shapes():
    x = _tiny_clip()
    model = MaskFusionNetPretrain(embed_dim=16, num_heads=4, clip_length=x.shape[2])
    out = model(x)
    assert out["x_dec"].shape == out["x_tube"].shape
    assert out["hr_signal"].shape == (x.shape[0], x.shape[2])
    assert not torch.isnan(out["x_dec"]).any()
    assert not torch.isnan(out["hr_signal"]).any()


def test_finetune_forward_shape():
    x = _tiny_clip()
    model = MaskFusionNet(embed_dim=16, num_heads=4, clip_length=x.shape[2])
    pred = model(x)
    assert pred.shape == (x.shape[0], x.shape[2])
    assert not torch.isnan(pred).any()


def test_finetune_backward_produces_gradients_everywhere():
    x = _tiny_clip()
    model = MaskFusionNet(embed_dim=16, num_heads=4, clip_length=x.shape[2])
    pred = model(x)
    pred.sum().backward()
    no_grad_params = [name for name, p in model.named_parameters()
                       if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    assert not no_grad_params, f"These parameters received no gradient: {no_grad_params}"


def test_adb_seb_fusion_encoder_are_independent_modules():
    """Regression test for the original repository's core bug: ADB, SEB,
    and the fusion encoder must NOT be the same underlying layer objects."""
    model = MaskFusionNet(embed_dim=16, num_heads=4, clip_length=16)
    assert model.adb_encoder[0] is not model.seb_encoder[0]
    assert model.seb_encoder[0] is not model.fusion_encoder[0]
    assert model.adb_conv is not model.seb_conv
    for p_adb, p_seb in zip(model.adb_encoder[0].parameters(), model.seb_encoder[0].parameters()):
        assert p_adb is not p_seb


def test_pretrained_encoder_loading_report():
    pre = MaskFusionNetPretrain(embed_dim=16, num_heads=4, clip_length=16)
    ft = MaskFusionNet(embed_dim=16, num_heads=4, clip_length=16)
    report = ft.load_pretrained_encoder(pre)
    assert report == {"seb_layers_loaded": 8, "fusion_layers_loaded": 4, "adb_layers_loaded": 0}
