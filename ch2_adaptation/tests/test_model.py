"""make_hf_generator against the real, tiny SMOKE_MODEL_TAG.

Skipped in the base+dev CI environment (no torch/transformers installed) via
pytest.importorskip -- exercised locally and whenever the ch2 extra is present.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from ch2_adaptation.config import load_baseline_config  # noqa: E402
from ch2_adaptation.model import make_hf_generator  # noqa: E402

CONFIGS = Path("ch2_adaptation/configs")


def test_make_hf_generator_returns_one_output_per_prompt() -> None:
    cfg = load_baseline_config(CONFIGS / "baseline_smoke.yaml")
    cfg = replace(cfg, max_new_tokens=4)
    generate = make_hf_generator(cfg)

    outputs = generate(["review this code", "and this code too"])

    assert len(outputs) == 2
    assert all(isinstance(output, str) for output in outputs)
