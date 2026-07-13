"""The JSON contract must accept a conforming review and reject each way of breaking it.

Every rejection case here is a way a real model actually fails: it invents a severity,
it drops a field, it adds a chatty extra one, it reports line 0.
"""

from __future__ import annotations

import json

import pytest
from jsonschema import Draft7Validator

from eval.harness import load_schema

VALID = {
    "severity": "major",
    "category": "resource-leak",
    "line": 42,
    "issue": "FileInputStream is never closed if read() throws.",
    "suggested_fix": "Use try-with-resources so the stream closes on any exit path.",
}


@pytest.fixture(scope="module")
def validator() -> Draft7Validator:
    return Draft7Validator(load_schema())


def test_conforming_output_passes(validator: Draft7Validator) -> None:
    assert validator.is_valid(VALID)


@pytest.mark.parametrize("severity", ["critical", "major", "minor", "info"])
def test_every_allowed_severity_passes(validator: Draft7Validator, severity: str) -> None:
    assert validator.is_valid({**VALID, "severity": severity})


@pytest.mark.parametrize("field", list(VALID))
def test_missing_required_field_fails(validator: Draft7Validator, field: str) -> None:
    incomplete = {key: value for key, value in VALID.items() if key != field}
    assert not validator.is_valid(incomplete)


def test_extra_field_fails(validator: Draft7Validator) -> None:
    # additionalProperties: false — a model that helpfully adds "confidence" is out of contract.
    assert not validator.is_valid({**VALID, "confidence": 0.9})


@pytest.mark.parametrize("severity", ["CRITICAL", "warning", "high", ""])
def test_disallowed_severity_fails(validator: Draft7Validator, severity: str) -> None:
    assert not validator.is_valid({**VALID, "severity": severity})


@pytest.mark.parametrize("line", [0, -1, "42", 1.5])
def test_invalid_line_fails(validator: Draft7Validator, line: object) -> None:
    assert not validator.is_valid({**VALID, "line": line})


@pytest.mark.parametrize("field", ["category", "issue", "suggested_fix"])
def test_empty_string_field_fails(validator: Draft7Validator, field: str) -> None:
    assert not validator.is_valid({**VALID, field: ""})


def test_overlong_field_fails(validator: Draft7Validator) -> None:
    assert not validator.is_valid({**VALID, "category": "x" * 65})


def test_schema_is_the_locked_contract() -> None:
    """The five fields are the contract. Changing them invalidates every published number."""
    schema = load_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"severity", "category", "line", "issue", "suggested_fix"}
    assert schema["properties"]["severity"]["enum"] == ["critical", "major", "minor", "info"]


def test_valid_example_serializes(validator: Draft7Validator) -> None:
    assert validator.is_valid(json.loads(json.dumps(VALID)))
