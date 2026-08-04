"""Hand-written causal multi-head self-attention (A2).

`nn.MultiheadAttention` and `F.scaled_dot_product_attention` are banned (CON-5): the
scaled dot product itself — matmul, scale, mask, softmax, matmul — is one of the two or
three things this chapter exists to prove was understood, not called.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from ch1_architecture.kv_cache import KVCache


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # Built once: causal_mask[i, j] is True where position i may NOT attend to j.
        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

    def _scaled_dot_product(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_offset: int
    ) -> torch.Tensor:
        # q: [B, H, T, Dh], k/v: [B, H, S, Dh] -> [B, H, T, Dh]
        T, S = q.size(2), k.size(2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, T, S]
        # Sliced from `seq_offset` so a cached decode step (T=1, S=t+1) asks whether the
        # single query at absolute position `seq_offset` may see each of the S keys —
        # the same mask, no separate cached-path logic.
        mask = self.causal_mask[seq_offset : seq_offset + T, :S]
        scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, v)  # [B, H, T, Dh]

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        # x: [B, T, D] -> [B, T, D]
        B, T, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        q = self.w_q(x).view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]
        k = self.w_k(x).view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]
        v = self.w_v(x).view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]

        seq_offset = 0
        if cache is not None:
            assert layer_idx is not None, "layer_idx is required when cache is not None"
            seq_offset = cache.current_len
            k, v = cache.update(layer_idx, k, v)  # [B, H, S, Dh] each

        out = self._scaled_dot_product(q, k, v, seq_offset)  # [B, H, T, Dh]
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, D]
        return self.w_o(out)
