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
    corpus_path: str
    tokenizer_path: str
    warmup_steps: int
    lr_min_ratio: float
    ckpt_every: int
    sample_every: int
    sample_prompt: str
    max_new_tokens: int
    temperature: float
    top_k: int
    checkpoint_path: str
    wandb_project: str
    wandb_mode: str  # "disabled" | "offline" | "online"

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


@dataclass(frozen=True)
class BenchmarkConfig:
    checkpoint_path: str
    modes: list[str]  # subset of quantize.MODES, e.g. ["fp32", "fp16", "int8", "int4"]
    prompt: str
    n_steps: int  # total generation steps measured; the loop length itself
    warmup_steps: int
    seed: int
    device: str  # "cpu" | "cuda"
    eval_corpus_path: str
    results_path: str
    plots_dir: str

    def __post_init__(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError(f"device must be 'cpu' or 'cuda', got {self.device!r}")
        if self.warmup_steps >= self.n_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be < n_steps ({self.n_steps})"
            )


def load_benchmark_config(path: Path | str) -> BenchmarkConfig:
    return _load_config(path, BenchmarkConfig)
