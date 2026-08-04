"""O2: `validate_output` -- exactly two parse attempts, then an honest failure."""

from __future__ import annotations

import pytest
from ch3_operation.api.validator import (
    OutputParseError,
    OutputSchemaError,
    validate_output,
)

_VALID = '{"severity": "minor", "category": "x", "line": 3, "issue": "y", "suggested_fix": "z"}'


def test_clean_bare_json_validates() -> None:
    result = validate_output(_VALID)
    assert result.severity == "minor"
    assert result.line == 3


def test_json_inside_a_json_fence_is_extracted_and_validates() -> None:
    raw = f"```json\n{_VALID}\n```"
    result = validate_output(raw)
    assert result.category == "x"


def test_json_inside_a_bare_fence_is_extracted_and_validates() -> None:
    raw = f"```\n{_VALID}\n```"
    result = validate_output(raw)
    assert result.category == "x"


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        "",
        "the model rambled about the bug but never wrote json",
        '{"severity": "minor", "category": "x", "line": 3',  # truncated brace
    ],
)
def test_unparsable_output_raises_output_parse_error(raw: str) -> None:
    with pytest.raises(OutputParseError):
        validate_output(raw)


def test_prose_before_and_after_a_fence_still_extracts() -> None:
    raw = f"Here is my review:\n```json\n{_VALID}\n```\nHope that helps!"
    result = validate_output(raw)
    assert result.line == 3


def test_extra_sixth_field_is_a_schema_error_not_a_parse_error() -> None:
    raw = (
        '{"severity": "minor", "category": "x", "line": 3, "issue": "y", '
        '"suggested_fix": "z", "confidence": 0.9}'
    )
    with pytest.raises(OutputSchemaError) as exc_info:
        validate_output(raw)
    assert "confidence" in exc_info.value.detail or "extra" in exc_info.value.detail.lower()


def test_wrong_severity_literal_is_a_schema_error() -> None:
    raw = (
        '{"severity": "catastrophic", "category": "x", "line": 3, "issue": "y", '
        '"suggested_fix": "z"}'
    )
    with pytest.raises(OutputSchemaError) as exc_info:
        validate_output(raw)
    assert "severity" in exc_info.value.detail


def test_schema_error_carries_the_pydantic_message_verbatim() -> None:
    raw = '{"severity": "minor", "category": "x", "line": 0, "issue": "y", "suggested_fix": "z"}'
    with pytest.raises(OutputSchemaError) as exc_info:
        validate_output(raw)
    assert "line" in exc_info.value.detail


def test_no_third_attempt_is_made_for_a_second_malformed_fence() -> None:
    """Only the first fence is tried -- a second, valid fence later in the text is not a
    third attempt in disguise."""
    raw = f"```json\n{{not json}}\n```\n\nActually, try this:\n```json\n{_VALID}\n```"
    with pytest.raises(OutputParseError):
        validate_output(raw)
