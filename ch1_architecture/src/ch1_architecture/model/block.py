"""One pre-norm transformer block: attention + MLP, each behind a residual (A2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from ch1_architecture.model.attention import MultiHeadSelfAttention
from ch1_architecture.model.mlp import MLP
from ch1_architecture.model.norm import LayerNorm

if TYPE_CHECKING:
    from ch1_architecture.kv_cache import KVCache


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int) -> None:
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, max_seq_len)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        # x: [B, T, D] -> [B, T, D]
        x = x + self.attn(self.ln1(x), cache, layer_idx)
        x = x + self.mlp(self.ln2(x))
        return x
