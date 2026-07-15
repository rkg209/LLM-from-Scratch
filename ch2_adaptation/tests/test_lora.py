"""build_lora_config matches the locked recipe exactly; attach_lora's trainable-fraction
guard actually stops a misconfigured attach (spec C4, task 2)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")

import torch  # noqa: E402
from ch2_adaptation.config import load_finetune_config  # noqa: E402
from ch2_adaptation.lora import (  # noqa: E402
    attach_lora,
    build_lora_config,
    check_trainable_fraction,
    compute_trainable_fraction,
)

CONFIGS = Path("ch2_adaptation/configs")


class TinyAttention(torch.nn.Module):
    """A model shaped enough like an attention block for peft to target by name: four
    small projections plus one much larger "rest of the base model" layer, so a frozen
    vs. unfrozen base produces a clearly different trainable fraction."""

    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(dim, dim)
        self.k_proj = torch.nn.Linear(dim, dim)
        self.v_proj = torch.nn.Linear(dim, dim)
        self.o_proj = torch.nn.Linear(dim, dim)
        self.rest_of_the_model = torch.nn.Linear(dim, dim * 200)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))

    def prepare_inputs_for_generation(self, *args: object, **kwargs: object) -> None:
        """`PeftModelForCausalLM` expects a HF-style causal-LM interface; this model only
        needs to exist well enough for `get_peft_model` to wrap it, never to actually
        generate."""
        raise NotImplementedError


def test_build_lora_config_matches_locked_recipe() -> None:
    config = load_finetune_config(CONFIGS / "full.yaml")
    lora_config = build_lora_config(config)

    assert lora_config.r == config.lora_r
    assert lora_config.lora_alpha == config.lora_alpha
    assert lora_config.lora_dropout == config.lora_dropout
    assert set(lora_config.target_modules) == set(config.target_modules)
    assert lora_config.bias == "none"
    assert lora_config.task_type.value == "CAUSAL_LM"


def test_build_lora_config_uses_smoke_hyperparameters_for_smoke_config() -> None:
    config = load_finetune_config(CONFIGS / "smoke.yaml")
    lora_config = build_lora_config(config)

    assert lora_config.r == 4
    assert lora_config.lora_alpha == 8


def test_compute_trainable_fraction_matches_simple_arithmetic() -> None:
    model = torch.nn.Linear(10, 10)
    for param in model.parameters():
        param.requires_grad_(False)
    model.weight.requires_grad_(True)

    fraction = compute_trainable_fraction(model)

    expected = model.weight.numel() / (model.weight.numel() + model.bias.numel())
    assert fraction == pytest.approx(expected)


def test_check_trainable_fraction_accepts_the_expected_range() -> None:
    check_trainable_fraction(0.005)  # must not raise


def test_check_trainable_fraction_rejects_near_full_fine_tuning() -> None:
    with pytest.raises(RuntimeError, match="outside the expected"):
        check_trainable_fraction(0.9)


def test_check_trainable_fraction_rejects_zero() -> None:
    with pytest.raises(RuntimeError, match="outside the expected"):
        check_trainable_fraction(0.0)


def test_attach_lora_succeeds_with_the_locked_recipe() -> None:
    """`peft.get_peft_model` freezes every non-LoRA parameter itself, so a normal attach
    (small r, four attention-only targets) always lands in the expected low-single-digit
    trainable range -- this is the recipe's actual shape, not a special setup."""
    config = load_finetune_config(CONFIGS / "smoke.yaml")
    model = TinyAttention()

    peft_model = attach_lora(model, config)

    fraction = compute_trainable_fraction(peft_model)
    assert 0.0 < fraction < 0.5


def test_attach_lora_raises_when_target_modules_cover_nearly_the_whole_model() -> None:
    """AC-3's real trigger: `target_modules` (mis)configured to also cover the bulk of the
    model, not "forgetting to freeze" -- peft handles freezing on its own, so that scenario
    can't actually happen through `get_peft_model`."""
    config = load_finetune_config(CONFIGS / "smoke.yaml")
    config = replace(config, target_modules=[*config.target_modules, "rest_of_the_model"])
    model = TinyAttention()

    with pytest.raises(RuntimeError, match="outside the expected"):
        attach_lora(model, config)


def test_attach_lora_on_the_real_smoke_model_clears_the_floor_with_margin() -> None:
    """Regression guard for the exact bug the code review caught: the locked recipe's four
    target modules are a much larger share of SMOKE_MODEL_TAG's ~2.4M parameters than of
    the full 1.5B model, but still land at ~0.018% -- close enough to a tight floor that a
    bound calibrated only against the full recipe would falsely fail every smoke run."""
    transformers = pytest.importorskip("transformers")
    config = load_finetune_config(CONFIGS / "smoke.yaml")
    model = transformers.AutoModelForCausalLM.from_pretrained(config.model_tag)

    peft_model = attach_lora(model, config)

    fraction = compute_trainable_fraction(peft_model)
    assert fraction > 0.0001  # comfortably above the floor, not just technically inside it
