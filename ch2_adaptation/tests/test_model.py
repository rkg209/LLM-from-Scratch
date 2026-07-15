"""make_hf_generator / load_base_model_for_training against the real, tiny SMOKE_MODEL_TAG.

Skipped in the base+dev CI environment (no torch/transformers installed) via
pytest.importorskip -- exercised locally and whenever the ch2 extra is present.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from ch2_adaptation.config import load_baseline_config, load_finetune_config  # noqa: E402
from ch2_adaptation.model import (  # noqa: E402
    load_base_model_for_training,
    make_hf_generator,
)

CONFIGS = Path("ch2_adaptation/configs")


def test_make_hf_generator_returns_one_output_per_prompt() -> None:
    cfg = load_baseline_config(CONFIGS / "baseline_smoke.yaml")
    cfg = replace(cfg, max_new_tokens=4)
    generate = make_hf_generator(cfg)

    outputs = generate(["review this code", "and this code too"])

    assert len(outputs) == 2
    assert all(isinstance(output, str) for output in outputs)


def test_load_base_model_for_training_smoke_branch_loads_on_cpu() -> None:
    cfg = load_finetune_config(CONFIGS / "smoke.yaml")

    model, tokenizer = load_base_model_for_training(cfg)

    assert next(model.parameters()).dtype.__str__() == "torch.float32"
    assert tokenizer is not None


def test_load_base_model_for_training_smoke_branch_never_touches_bitsandbytes_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patches `transformers.BitsAndBytesConfig` itself to blow up if constructed --
    tied directly to the exact class the 4-bit branch imports and builds, rather than
    the import machinery (`transformers` probes bitsandbytes availability via
    `importlib.util.find_spec`, not `__import__`, so a `builtins.__import__` patch
    doesn't actually intercept that check, as confirmed by testing it directly)."""
    import transformers

    def raise_if_constructed(*args: object, **kwargs: object) -> None:
        raise AssertionError("smoke path must never construct BitsAndBytesConfig")

    monkeypatch.setattr(transformers, "BitsAndBytesConfig", raise_if_constructed)
    cfg = load_finetune_config(CONFIGS / "smoke.yaml")

    load_base_model_for_training(cfg)  # must not raise
