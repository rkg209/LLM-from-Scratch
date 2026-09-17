"""The full GPT model: embeddings, a stack of blocks, final norm, and a tied head (A2)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

from ch1_architecture.config import GPTConfig
from ch1_architecture.model.block import TransformerBlock
from ch1_architecture.model.embeddings import PositionalEmbedding, TokenEmbedding
from ch1_architecture.model.norm import LayerNorm

if TYPE_CHECKING:
    from ch1_architecture.kv_cache import KVCache


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = TokenEmbedding(config.vocab_size, config.d_model)
        self.pos_emb = PositionalEmbedding(config.seq_len, config.d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config.d_model, config.n_heads, config.seq_len)
                for _ in range(config.n_layers)
            ]
        )
        self.ln_f = LayerNorm(config.d_model)

        self.apply(self._init_weights)
        # The two projections that write back into the residual stream get a smaller
        # std, scaled by the depth they are summed over (GPT-2 §2.3). Without it the
        # residual variance grows with n_layers, and a deeper model starts worse than
        # a shallow one for no reason a reader could see in the code.
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for name, param in self.named_parameters():
            if name.endswith(("w_o.weight", "fc2.weight")):
                nn.init.normal_(param, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-2's initialization: N(0, 0.02) on every weight matrix, zero biases.

        Torch's `nn.Embedding` default is N(0, 1), and this model ties its output head
        to the token embedding — so the default put the logits at std ~23 and the loss
        at initialization at ~386, against ln(vocab_size) ~ 8.3 for uniform guessing.
        The first 200 steps of a run were spent climbing back down to worse-than-random
        rather than learning anything, which is what the A3 GPU run (job 402127) showed.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        # input_ids: [B, T] -> logits: [B, T, V]
        B, T = input_ids.shape
        offset = cache.current_len if cache is not None else 0

        x = self.token_emb(input_ids) + self.pos_emb(T, offset=offset)  # [B, T, D]
        for layer_idx, block in enumerate(self.blocks):
            x = block(x, cache=cache, layer_idx=layer_idx)
        x = self.ln_f(x)  # [B, T, D]

        if cache is not None:
            # Advanced once per forward pass, after every layer has written its slice
            # — not once per layer, or layer N would read a window layer 0 never wrote.
            cache.advance(T)

        # Tied output head: reuse the token embedding weight instead of a second
        # [D, V] matrix. Note this keeps the head itself outside `quantize_model`
        # in A5 — it is a matmul against an nn.Embedding weight, not an nn.Linear.
        return x @ self.token_emb.weight.T  # [B, T, V]

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
