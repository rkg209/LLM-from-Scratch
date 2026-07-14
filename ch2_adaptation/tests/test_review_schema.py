"""ReviewOutput must accept exactly what eval/schema.json accepts, and the drift guard
must actually catch drift rather than passing by construction."""

from __future__ import annotations

import copy

import pytest
from ch2_adaptation.schema import ReviewOutput, SchemaDriftError, _assert_matches_eval_schema
from pydantic import ValidationError

from eval.harness import load_schema

VALID = {
    "severity": "major",
    "category": "resource-leak",
    "line": 42,
    "issue": "FileInputStream is never closed if read() throws.",
    "suggested_fix": "Use try-with-resources so the stream closes on any exit path.",
}


def test_valid_record_accepted() -> None:
    ReviewOutput(**VALID)


def test_extra_key_rejected() -> None:
    with pytest.raises(ValidationError):
        ReviewOutput(**VALID, confidence=0.9)


def test_bad_severity_rejected() -> None:
    with pytest.raises(ValidationError):
        ReviewOutput(**{**VALID, "severity": "warning"})


def test_line_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        ReviewOutput(**{**VALID, "line": 0})


def test_empty_category_rejected() -> None:
    with pytest.raises(ValidationError):
        ReviewOutput(**{**VALID, "category": ""})


def test_drift_guard_passes_against_real_schema() -> None:
    _assert_matches_eval_schema(load_schema())


def test_drift_guard_raises_on_mutated_severity_enum() -> None:
    mutated = copy.deepcopy(load_schema())
    mutated["properties"]["severity"]["enum"] = ["critical", "major"]
    with pytest.raises(SchemaDriftError):
        _assert_matches_eval_schema(mutated)


def test_drift_guard_raises_on_mutated_additional_properties() -> None:
    mutated = copy.deepcopy(load_schema())
    mutated["additionalProperties"] = True
    with pytest.raises(SchemaDriftError):
        _assert_matches_eval_schema(mutated)


def test_drift_guard_raises_on_mutated_required() -> None:
    mutated = copy.deepcopy(load_schema())
    mutated["required"] = ["severity", "category"]
    with pytest.raises(SchemaDriftError):
        _assert_matches_eval_schema(mutated)


def test_model_json_schema_agrees_with_eval_schema_on_load_bearing_keys() -> None:
    schema = load_schema()
    pydantic_schema = ReviewOutput.model_json_schema()
    assert set(pydantic_schema["required"]) == set(schema["required"])
    assert pydantic_schema["additionalProperties"] is schema["additionalProperties"]
    assert (
        pydantic_schema["properties"]["severity"]["enum"]
        == schema["properties"]["severity"]["enum"]
    )
