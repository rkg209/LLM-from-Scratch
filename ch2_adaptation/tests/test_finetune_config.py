"""Chapter 2 config: the profiles load, the base model is locked, smoke stays on CPU."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from ch2_adaptation.config import LOCKED_MODEL_TAG, SMOKE_MODEL_TAG, load_finetune_config

CONFIGS = Path("ch2_adaptation/configs")


@pytest.mark.parametrize("profile", ["smoke", "full"])
def test_committed_config_loads(profile: str) -> None:
    config = load_finetune_config(CONFIGS / f"{profile}.yaml")
    assert config.seed == 42


def test_smoke_profile_runs_on_cpu() -> None:
    """bitsandbytes needs CUDA, so the CPU path must not ask for 4-bit (NFR-1)."""
    config = load_finetune_config(CONFIGS / "smoke.yaml")

    assert config.use_4bit is False
    assert config.is_smoke is True
    assert config.max_steps <= 10


def test_smoke_profile_uses_the_tiny_stand_in_model() -> None:
    """A first-time load of the real 1.5B model can't reliably finish in the smoke budget."""
    config = load_finetune_config(CONFIGS / "smoke.yaml")
    assert config.model_tag == SMOKE_MODEL_TAG


def test_full_profile_is_the_gpu_recipe() -> None:
    config = load_finetune_config(CONFIGS / "full.yaml")

    assert config.use_4bit is True
    assert config.model_tag == LOCKED_MODEL_TAG
    assert config.lora_r == 16
    assert config.lora_alpha == 32


def test_base_model_is_locked_for_the_full_profile() -> None:
    """Swapping the base model invalidates every measured baseline (CON-11)."""
    config = load_finetune_config(CONFIGS / "full.yaml")

    with pytest.raises(ValueError, match="locked"):
        replace(config, model_tag="meta-llama/Llama-3.2-1B")


def test_smoke_profile_rejects_any_tag_other_than_the_allowlisted_stand_in() -> None:
    config = load_finetune_config(CONFIGS / "smoke.yaml")

    with pytest.raises(ValueError, match="smoke fine-tune runs"):
        replace(config, model_tag=LOCKED_MODEL_TAG)


def test_lora_needs_a_target() -> None:
    config = load_finetune_config(CONFIGS / "full.yaml")

    with pytest.raises(ValueError, match="target_modules"):
        replace(config, target_modules=[])


def test_attention_projections_are_the_lora_targets() -> None:
    config = load_finetune_config(CONFIGS / "full.yaml")
    assert set(config.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}
