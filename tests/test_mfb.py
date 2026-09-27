import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maskfusionnet.models.mfb import MultiScaleFusionBlock


def test_mfb_output_shape_matches_seb():
    seb = torch.randn(2, 16, 5, 4, 4)
    adb = torch.randn(2, 16, 5, 4, 4)  # caller must pre-align temporal dim
    mfb = MultiScaleFusionBlock(embed_dim=16, num_heads=4)
    out = mfb(x_seb=seb, x_adb=adb)
    assert out.shape == seb.shape


def test_mfb_has_residual_connection_to_seb():
    """Eq. 8: X_f = MIA(...) + X_SEB. If we zero out the interactive
    attention's output projection, the block's output must reduce to
    (approximately) X_SEB itself."""
    torch.manual_seed(0)
    mfb = MultiScaleFusionBlock(embed_dim=8, num_heads=2)
    with torch.no_grad():
        mfb.mia.proj.weight.zero_()
        mfb.mia.proj.bias.zero_()
    seb = torch.randn(1, 8, 3, 4, 4)
    adb = torch.randn(1, 8, 3, 4, 4)
    out = mfb(x_seb=seb, x_adb=adb)
    assert torch.allclose(out, seb, atol=1e-5), (
        "With the attention branch zeroed, MFB output should collapse to the "
        "residual X_SEB (Eq. 8) -- if this fails, the residual connection is missing."
    )


def test_mfb_no_nans_and_gradients_flow_to_both_branches():
    seb = torch.randn(2, 16, 4, 4, 4, requires_grad=True)
    adb = torch.randn(2, 16, 4, 4, 4, requires_grad=True)
    mfb = MultiScaleFusionBlock(embed_dim=16, num_heads=4)
    out = mfb(x_seb=seb, x_adb=adb)
    assert not torch.isnan(out).any()
    out.sum().backward()
    assert seb.grad is not None and seb.grad.abs().sum() > 0
    assert adb.grad is not None and adb.grad.abs().sum() > 0
