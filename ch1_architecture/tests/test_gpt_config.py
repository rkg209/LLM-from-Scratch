"""Chapter 1 config: the committed profiles load, and the head-split invariant is enforced."""

from __future__ import annotations

from pathlib import Path

import pytest
from ch1_architecture.config import GPTConfig, load_gpt_config

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
        )
