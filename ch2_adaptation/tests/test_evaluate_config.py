"""EvaluateConfig: the smoke/full split and the published-results guard, mirroring
BaselineConfig/DataGenConfig's established pattern (spec C5, task 4)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from ch2_adaptation.config import LOCKED_MODEL_TAG, SMOKE_MODEL_TAG, load_evaluate_config

CONFIGS = Path("ch2_adaptation/configs")


def test_full_profile_loads_with_locked_model_tag() -> None:
    config = load_evaluate_config(CONFIGS / "eval_full.yaml")
    assert config.model_tag == LOCKED_MODEL_TAG
    assert config.is_smoke is False
    assert config.holdout_path == "eval/holdout/holdout.jsonl"
    assert config.results_path == "eval/results/finetuned.json"


def test_smoke_profile_loads_with_smoke_model_tag() -> None:
    config = load_evaluate_config(CONFIGS / "eval_smoke.yaml")
    assert config.model_tag == SMOKE_MODEL_TAG
    assert config.is_smoke is True
    assert config.holdout_path != "eval/holdout/holdout.jsonl"
    assert config.results_path != "eval/results/finetuned.json"


def test_full_profile_rejects_a_non_locked_model_tag() -> None:
    config = load_evaluate_config(CONFIGS / "eval_full.yaml")
    with pytest.raises(ValueError, match="locked"):
        replace(config, model_tag="meta-llama/Llama-3.2-1B")


def test_smoke_profile_rejects_any_tag_other_than_the_allowlisted_stand_in() -> None:
    config = load_evaluate_config(CONFIGS / "eval_smoke.yaml")
    with pytest.raises(ValueError, match="smoke eval runs"):
        replace(config, model_tag=LOCKED_MODEL_TAG)


def test_smoke_profile_rejects_writing_the_published_results_path() -> None:
    config = load_evaluate_config(CONFIGS / "eval_smoke.yaml")
    with pytest.raises(ValueError, match="published"):
        replace(config, results_path="eval/results/finetuned.json")
