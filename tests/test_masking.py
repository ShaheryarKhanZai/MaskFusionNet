import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maskfusionnet.models.masking import TubeMasking


def test_mask_ratio_is_approximately_correct():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 4, 6, 6)  # B,C,T,H,W -> 36 spatial tokens
    masking = TubeMasking(embed_dim=8, mask_ratio=0.75)
    _, mask = masking(x)
    ratio = mask.float().mean().item()
    assert abs(ratio - 0.75) < 0.05


def test_mask_is_constant_across_time_tube_strategy():
    """This is the core requirement: for a fixed (b, h, w), the masking
    decision must be identical for every t (Figure 3, tube strategy)."""
    torch.manual_seed(1)
    x = torch.randn(3, 8, 5, 6, 6)
    masking = TubeMasking(embed_dim=8, mask_ratio=0.75)
    _, mask = masking(x)  # [B,1,T,H,W]
    b, _, t, h, w = mask.shape
    for bi in range(b):
        first_t = mask[bi, 0, 0]
        for ti in range(1, t):
            assert torch.equal(mask[bi, 0, ti], first_t), (
                f"Mask at t={ti} differs from t=0 for batch {bi}: tube masking "
                f"requires identical spatial mask across all frames."
            )


def test_mask_differs_across_batch_items_generally():
    """Not a hard requirement of the paper, but a sanity check that masks
    are actually randomized per sample rather than accidentally identical
    across the whole batch (which would indicate a bug, e.g. a fixed seed
    reused per-call)."""
    torch.manual_seed(2)
    x = torch.randn(4, 8, 3, 8, 8)
    masking = TubeMasking(embed_dim=8, mask_ratio=0.5)
    _, mask = masking(x)
    all_same = all(torch.equal(mask[0, 0, 0], mask[i, 0, 0]) for i in range(1, 4))
    assert not all_same


def test_masked_positions_replaced_with_mask_token_not_zero():
    torch.manual_seed(3)
    x = torch.randn(1, 8, 2, 4, 4)
    masking = TubeMasking(embed_dim=8, mask_ratio=0.75)
    masking.mask_token.data.fill_(1.234)
    x_vis, mask = masking(x)
    masked_vals = x_vis[mask.expand_as(x_vis)]
    assert torch.allclose(masked_vals, torch.full_like(masked_vals, 1.234))


def test_visible_positions_are_unchanged():
    torch.manual_seed(4)
    x = torch.randn(1, 8, 2, 4, 4)
    masking = TubeMasking(embed_dim=8, mask_ratio=0.75)
    x_vis, mask = masking(x)
    visible = ~mask.expand_as(x)
    assert torch.equal(x_vis[visible], x[visible])
