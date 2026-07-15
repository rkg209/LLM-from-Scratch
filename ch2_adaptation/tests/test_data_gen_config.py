"""DataGenConfig is what keeps C3's budget guard and provenance honest: a smoke run must
never write the committed provenance.json, and the numeric fields must be sane."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from ch2_adaptation.config import load_data_gen_config

CONFIGS = Path("ch2_adaptation/configs")


def test_full_profile_loads_with_gemini_provider() -> None:
    config = load_data_gen_config(CONFIGS / "datagen_full.yaml")
    assert config.frontier_provider == "gemini"
    assert config.is_smoke is False
    assert config.provenance_path == "ch2_adaptation/data/provenance.json"


def test_smoke_profile_loads_with_stub_provider() -> None:
    config = load_data_gen_config(CONFIGS / "datagen_smoke.yaml")
    assert config.frontier_provider == "stub"
    assert config.is_smoke is True
    assert config.provenance_path != "ch2_adaptation/data/provenance.json"


def test_smoke_profile_rejects_writing_the_published_provenance_path() -> None:
    config = load_data_gen_config(CONFIGS / "datagen_smoke.yaml")
    with pytest.raises(ValueError, match="published"):
        replace(config, provenance_path="ch2_adaptation/data/provenance.json")


def test_val_frac_must_be_in_open_unit_interval() -> None:
    config = load_data_gen_config(CONFIGS / "datagen_smoke.yaml")
    with pytest.raises(ValueError, match="val_frac"):
        replace(config, val_frac=0.0)
    with pytest.raises(ValueError, match="val_frac"):
        replace(config, val_frac=1.0)


def test_n_snippets_must_be_positive() -> None:
    config = load_data_gen_config(CONFIGS / "datagen_smoke.yaml")
    with pytest.raises(ValueError, match="n_snippets"):
        replace(config, n_snippets=0)


def test_variants_per_snippet_must_be_positive() -> None:
    config = load_data_gen_config(CONFIGS / "datagen_smoke.yaml")
    with pytest.raises(ValueError, match="variants_per_snippet"):
        replace(config, variants_per_snippet=0)


def test_max_budget_usd_must_be_positive() -> None:
    config = load_data_gen_config(CONFIGS / "datagen_smoke.yaml")
    with pytest.raises(ValueError, match="max_budget_usd"):
        replace(config, max_budget_usd=0.0)


def test_full_profile_target_size_matches_spec_amendment() -> None:
    config = load_data_gen_config(CONFIGS / "datagen_full.yaml")
    assert config.n_snippets * config.variants_per_snippet >= 270
