"""Token and sinusoidal positional embeddings (A2)."""

from __future__ import annotations

import math

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T] -> [B, T, D]
        return self.embedding(input_ids)


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        position = torch.arange(max_seq_len).unsqueeze(1)  # [max_seq_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # [D/2]
        table = torch.zeros(max_seq_len, d_model)
        table[:, 0::2] = torch.sin(position * div_term)
        table[:, 1::2] = torch.cos(position * div_term)
        # Not a parameter: the sinusoidal table is fixed, not learned.
        self.register_buffer("table", table.unsqueeze(0))  # [1, max_seq_len, D]

    def forward(self, seq_len: int, offset: int = 0) -> torch.Tensor:
        # offset is the absolute position of the first token in this forward pass.
        # It is always 0 in the uncached path, and cache.current_len once a KV-cache
        # is live (A4) — without it, every cached decode step would embed as position 0.
        return self.table[:, offset : offset + seq_len, :]  # [1, T, D]
