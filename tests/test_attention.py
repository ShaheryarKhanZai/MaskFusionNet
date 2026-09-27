import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maskfusionnet.models.attention import MultiHeadSelfAttention, MultiHeadInteractiveAttention


def test_self_attention_output_shape():
    x = torch.randn(2, 16, 3, 4, 4)
    attn = MultiHeadSelfAttention(dim=16, num_heads=4)
    out = attn(x)
    assert out.shape == x.shape


def test_self_attention_no_nans():
    x = torch.randn(2, 16, 3, 4, 4)
    attn = MultiHeadSelfAttention(dim=16, num_heads=4)
    out = attn(x)
    assert not torch.isnan(out).any()


def test_self_attention_actually_mixes_tokens():
    """The bug this guards against: the original repo's attention
    contracted Q/K over the channel axis at MATCHING spatial positions
    only, so changing one token could not influence another token's
    output at all. Here we perturb a single input token and confirm the
    change propagates to *other* output token positions -- proof that the
    attention matrix genuinely spans the full token sequence (Eq. 6)."""
    torch.manual_seed(0)
    attn = MultiHeadSelfAttention(dim=8, num_heads=2)
    attn.eval()
    x = torch.randn(1, 8, 2, 3, 3)
    with torch.no_grad():
        out1 = attn(x)
        x2 = x.clone()
        x2[0, :, 0, 0, 0] += 5.0  # perturb exactly one token
        out2 = attn(x2)
    # Some OTHER token position must have changed too.
    other_positions_changed = not torch.allclose(out1[0, :, 1, 1, 1], out2[0, :, 1, 1, 1], atol=1e-5)
    assert other_positions_changed, (
        "Perturbing one token had no effect on other tokens' attention output -- "
        "this indicates attention is not actually mixing information across "
        "the token sequence (the bug this test targets)."
    )


def test_interactive_attention_output_matches_query_shape():
    seb = torch.randn(2, 16, 4, 4, 4)
    adb = torch.randn(2, 16, 4, 4, 4)  # already temporally aligned by the caller
    mia = MultiHeadInteractiveAttention(dim=16, num_heads=4)
    out = mia(query_source=seb, kv_source=adb)
    assert out.shape == seb.shape


def test_interactive_attention_depends_on_both_sources():
    torch.manual_seed(0)
    mia = MultiHeadInteractiveAttention(dim=8, num_heads=2)
    mia.eval()
    seb = torch.randn(1, 8, 2, 3, 3)
    adb = torch.randn(1, 8, 2, 3, 3)
    with torch.no_grad():
        out_base = mia(seb, adb)
        out_seb_changed = mia(seb + 1.0, adb)
        out_adb_changed = mia(seb, adb + 1.0)
    assert not torch.allclose(out_base, out_seb_changed)
    assert not torch.allclose(out_base, out_adb_changed)
