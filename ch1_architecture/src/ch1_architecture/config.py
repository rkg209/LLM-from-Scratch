"""Chapter 1 config. Fields are fixed in planning/03-system-design.md §1.1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eval.config import load_config as _load_config


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    seq_len: int
    batch_size: int
    max_steps: int
    learning_rate: float
    clip_grad_norm: float
    log_every: int
    seed: int
    device: str  # "cpu" | "cuda"
    run_name: str  # W&B run name

    def __post_init__(self) -> None:
        # Caught here or caught as a silently scrambled head dimension three hours into
        # a training run. The head split is d_model -> n_heads x head_dim; it must divide.
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.device not in {"cpu", "cuda"}:
            raise ValueError(f"device must be 'cpu' or 'cuda', got {self.device!r}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


def load_gpt_config(path: Path | str) -> GPTConfig:
    return _load_config(path, GPTConfig)
