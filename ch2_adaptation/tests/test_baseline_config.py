"""BaselineConfig is what keeps CON-11 honest: the full baseline must use the locked
model, and smoke must use the one allowlisted stand-in and never touch the published
results file."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from ch2_adaptation.config import LOCKED_MODEL_TAG, SMOKE_MODEL_TAG, load_baseline_config

CONFIGS = Path("ch2_adaptation/configs")


def test_full_profile_loads_with_locked_model_tag() -> None:
    config = load_baseline_config(CONFIGS / "baseline_full.yaml")
    assert config.base_model_tag == LOCKED_MODEL_TAG
    assert config.is_smoke is False
    assert config.results_path == "eval/results/baselines.json"


def test_smoke_profile_loads_with_smoke_model_tag() -> None:
    config = load_baseline_config(CONFIGS / "baseline_smoke.yaml")
    assert config.base_model_tag == SMOKE_MODEL_TAG
    assert config.is_smoke is True
    assert config.results_path != "eval/results/baselines.json"


def test_full_profile_rejects_a_non_locked_model_tag() -> None:
    config = load_baseline_config(CONFIGS / "baseline_full.yaml")
    with pytest.raises(ValueError, match="locked"):
        replace(config, base_model_tag="meta-llama/Llama-3.2-1B")


def test_smoke_profile_rejects_any_tag_other_than_the_allowlisted_stand_in() -> None:
    config = load_baseline_config(CONFIGS / "baseline_smoke.yaml")
    with pytest.raises(ValueError, match="smoke baseline runs"):
        replace(config, base_model_tag=LOCKED_MODEL_TAG)


def test_smoke_profile_rejects_writing_the_published_results_path() -> None:
    config = load_baseline_config(CONFIGS / "baseline_smoke.yaml")
    with pytest.raises(ValueError, match="published"):
        replace(config, results_path="eval/results/baselines.json")


def test_frontier_provider_selects_smoke_vs_full() -> None:
    full = load_baseline_config(CONFIGS / "baseline_full.yaml")
    smoke = load_baseline_config(CONFIGS / "baseline_smoke.yaml")
    assert full.frontier_provider == "gemini"
    assert smoke.frontier_provider == "stub"


def test_baseline_holdout_profile_scores_the_frozen_holdout_not_the_stub_set() -> None:
    """spec C5: re-scoring base/frontier on the identical set the fine-tuned model is
    scored on is just a new config -- baseline.py itself needs no changes."""
    config = load_baseline_config(CONFIGS / "baseline_holdout.yaml")

    assert config.stub_set_path == "eval/holdout/holdout.jsonl"
    assert config.base_model_tag == LOCKED_MODEL_TAG
    assert config.results_path != "eval/results/baselines.json"
