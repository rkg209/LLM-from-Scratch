"""Chapter 1 config: the committed profiles load, and the head-split invariant is enforced."""

from __future__ import annotations

from pathlib import Path

import pytest
from ch1_architecture.config import (
    BenchmarkConfig,
    GPTConfig,
    load_benchmark_config,
    load_gpt_config,
)

CONFIGS = Path("ch1_architecture/configs")


@pytest.mark.parametrize("profile", ["smoke", "full"])
def test_committed_config_loads(profile: str) -> None:
    config = load_gpt_config(CONFIGS / f"{profile}.yaml")
    assert config.run_name == f"ch1-{profile}"
    assert config.seed == 42


def test_smoke_config_is_actually_small() -> None:
    """The smoke profile has to finish on a CPU in seconds, or nobody will run it (NFR-1)."""
    config = load_gpt_config(CONFIGS / "smoke.yaml")

    assert config.device == "cpu"
    assert config.max_steps <= 100
    assert config.d_model <= 128
    assert config.n_layers <= 4


def test_head_dim_divides() -> None:
    config = load_gpt_config(CONFIGS / "smoke.yaml")
    assert config.head_dim == config.d_model // config.n_heads


def test_indivisible_head_split_is_rejected() -> None:
    """d_model % n_heads != 0 silently scrambles the head dimension. Catch it at load."""
    with pytest.raises(ValueError, match="divisible"):
        GPTConfig(
            vocab_size=256,
            d_model=65,  # not divisible by 2
            n_heads=2,
            n_layers=2,
            seq_len=64,
            batch_size=4,
            max_steps=50,
            learning_rate=1e-3,
            clip_grad_norm=1.0,
            log_every=10,
            seed=42,
            device="cpu",
            run_name="bad",
            corpus_path="ch1_architecture/tests/fixtures/corpus_smoke.txt",
            tokenizer_path="outputs/ch1/tokenizer_smoke.json",
            warmup_steps=5,
            lr_min_ratio=0.1,
            ckpt_every=25,
            sample_every=25,
            sample_prompt="First Citizen:",
            max_new_tokens=20,
            temperature=0.8,
            top_k=20,
            checkpoint_path="outputs/ch1/model_smoke.pt",
            wandb_project="ch1-architecture",
            wandb_mode="disabled",
            val_fraction=0.1,
            val_every=5,
        )


@pytest.mark.parametrize("profile", ["benchmark_smoke", "benchmark_full"])
def test_committed_benchmark_config_loads(profile: str) -> None:
    config = load_benchmark_config(CONFIGS / f"{profile}.yaml")
    assert config.seed == 42
    assert set(config.modes) <= {"fp32", "fp16", "int8", "int4"}


def test_benchmark_config_rejects_warmup_at_or_above_n_steps() -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        BenchmarkConfig(
            checkpoint_path="outputs/ch1/model_smoke.pt",
            modes=["fp32"],
            prompt="hi",
            n_steps=5,
            warmup_steps=5,
            seed=42,
            device="cpu",
            eval_corpus_path="ch1_architecture/tests/fixtures/corpus_smoke.txt",
            val_fraction=0.1,
            results_path="eval/results/ch1_benchmark.json",
            plots_dir="eval/results/plots",
        )


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="device"):
        GPTConfig(
            vocab_size=256,
            d_model=64,
            n_heads=2,
            n_layers=2,
            seq_len=64,
            batch_size=4,
            max_steps=50,
            learning_rate=1e-3,
            clip_grad_norm=1.0,
            log_every=10,
            seed=42,
            device="tpu",
            run_name="bad",
            corpus_path="ch1_architecture/tests/fixtures/corpus_smoke.txt",
            tokenizer_path="outputs/ch1/tokenizer_smoke.json",
            warmup_steps=5,
            lr_min_ratio=0.1,
            ckpt_every=25,
            sample_every=25,
            sample_prompt="First Citizen:",
            max_new_tokens=20,
            temperature=0.8,
            top_k=20,
            checkpoint_path="outputs/ch1/model_smoke.pt",
            wandb_project="ch1-architecture",
            wandb_mode="disabled",
            val_fraction=0.1,
            val_every=5,
        )
