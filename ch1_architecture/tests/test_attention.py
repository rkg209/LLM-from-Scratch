"""Causal-mask correctness: position t must never see position t+1 or later (A2)."""

from __future__ import annotations

import torch
from ch1_architecture.model.attention import MultiHeadSelfAttention

D, H, MAX_SEQ_LEN = 16, 4, 10


def test_causal_mask_blocks_future_positions() -> None:
    torch.manual_seed(0)
    attn = MultiHeadSelfAttention(D, H, MAX_SEQ_LEN)
    attn.eval()

    T = 6
    x = torch.randn(1, T, D)
    x_perturbed = x.clone()
    x_perturbed[0, -1] += 10.0  # perturb only the last position

    with torch.no_grad():
        out = attn(x)
        out_perturbed = attn(x_perturbed)

    # Every position except the perturbed one itself must be unaffected: it can only
    # attend to positions <= itself, and the perturbed token is the last one.
    assert torch.allclose(out[0, :-1], out_perturbed[0, :-1], atol=1e-6)
    assert not torch.allclose(out[0, -1], out_perturbed[0, -1], atol=1e-6)


def test_causal_mask_is_upper_triangular() -> None:
    attn = MultiHeadSelfAttention(D, H, MAX_SEQ_LEN)
    mask = attn.causal_mask
    for i in range(MAX_SEQ_LEN):
        for j in range(MAX_SEQ_LEN):
            assert mask[i, j].item() == (j > i)
