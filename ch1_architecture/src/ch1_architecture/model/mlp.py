"""The position-wise feed-forward block (A2)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# The 4x expansion ratio is a fixed architectural constant (planning/03), not a
# hyperparameter — it does not vary per config the way d_model or n_heads do.
_EXPANSION_RATIO = 4


class MLP(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, _EXPANSION_RATIO * d_model)
        self.fc2 = nn.Linear(_EXPANSION_RATIO * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> [B, T, 4D] -> [B, T, D]
        return self.fc2(F.gelu(self.fc1(x)))
