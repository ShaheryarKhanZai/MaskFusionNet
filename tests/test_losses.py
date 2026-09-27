import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maskfusionnet.losses.reconstruction import spatial_reconstruction_loss, temporal_smoothness_loss, reconstruction_loss
from maskfusionnet.losses.rppg import negative_pearson_loss, hr_to_soft_label, loss_weight_schedule


def test_spatial_reconstruction_loss_zero_when_identical():
    x = torch.randn(2, 4, 3, 3, 3)
    mask = torch.rand_like(x[:, :1]) > 0.5
    loss = spatial_reconstruction_loss(x, x.clone(), mask)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_spatial_reconstruction_loss_ignores_unmasked_positions():
    torch.manual_seed(0)
    x_tube = torch.zeros(1, 2, 2, 2, 2)
    x_dec = torch.zeros(1, 2, 2, 2, 2)
    mask = torch.zeros(1, 1, 2, 2, 2, dtype=torch.bool)
    mask[0, 0, 0, 0, 0] = True  # only one masked position

    x_dec_wrong_elsewhere = x_dec.clone()
    x_dec_wrong_elsewhere[0, :, 1, 1, 1] = 100.0  # large error OUTSIDE the mask
    loss = spatial_reconstruction_loss(x_tube, x_dec_wrong_elsewhere, mask)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    x_dec_wrong_inside = x_dec.clone()
    x_dec_wrong_inside[0, :, 0, 0, 0] = 1.0  # error INSIDE the mask
    loss2 = spatial_reconstruction_loss(x_tube, x_dec_wrong_inside, mask)
    assert loss2 > 0


def test_temporal_smoothness_loss_zero_for_identical_sequences():
    x = torch.randn(2, 4, 5, 3, 3)
    assert torch.isclose(temporal_smoothness_loss(x, x.clone()), torch.tensor(0.0), atol=1e-6)


def test_reconstruction_loss_combines_components():
    x_tube = torch.randn(1, 4, 4, 2, 2)
    x_dec = x_tube + 0.1 * torch.randn_like(x_tube)
    mask = torch.rand(1, 1, 4, 2, 2) > 0.5
    out = reconstruction_loss(x_tube, x_dec, mask, alpha=2.0, beta=3.0)
    assert torch.isclose(out["l_rec"], 2.0 * out["l_space"] + 3.0 * out["l_time"], atol=1e-5)


def test_negative_pearson_loss_zero_for_perfectly_correlated_signals():
    t = torch.linspace(0, 4 * 3.14159, 50)
    sig = torch.sin(t).unsqueeze(0)
    loss = negative_pearson_loss(sig, sig.clone())
    assert loss.item() < 1e-4


def test_negative_pearson_loss_high_for_uncorrelated_signals():
    torch.manual_seed(0)
    a = torch.randn(1, 100)
    b = torch.randn(1, 100)
    loss = negative_pearson_loss(a, b)
    assert loss.item() > 0.5  # not guaranteed but overwhelmingly likely for random signals


def test_hr_soft_label_peaks_at_true_bin():
    dist = hr_to_soft_label(torch.tensor([75.0]), num_bins=140, hr_min=40)
    assert dist.shape == (1, 140)
    assert torch.isclose(dist.sum(), torch.tensor(1.0), atol=1e-4)
    assert dist.argmax().item() == 75 - 40


def test_loss_weight_schedule_matches_paper_eq19():
    """Eq. (19) is intentionally non-monotonic (see rppg.py docstring):
    mu overshoots mu0 during warm-up (e < 25) and then drops back to
    exactly mu0 at e=25 onward. lambda is constant at 0.1 throughout."""
    lam1, mu_epoch1 = loss_weight_schedule(1, mu0=1.0, eta=5.0, warmup_epochs=25)
    lam15, mu_epoch15 = loss_weight_schedule(15, mu0=1.0, eta=5.0, warmup_epochs=25)
    lam25, mu_epoch25 = loss_weight_schedule(25, mu0=1.0, eta=5.0, warmup_epochs=25)
    lam100, mu_epoch100 = loss_weight_schedule(100, mu0=1.0, eta=5.0, warmup_epochs=25)
    assert lam1 == lam15 == lam25 == lam100 == 0.1
    assert mu_epoch1 == 1.0                 # eta^0 == 1 at e=1
    assert mu_epoch15 > mu_epoch1           # overshoots mu0 mid-warmup
    assert mu_epoch25 == 1.0                # drops back to mu0 exactly at e=25
    assert mu_epoch100 == 1.0               # stays at mu0 afterward
