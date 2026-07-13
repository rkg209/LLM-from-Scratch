"""The two metrics behind the headline table. Pure functions, no side effects.

They live apart from the harness so they can be tested in isolation — these two numbers
are the project's entire published claim, and a bug in either is a bug in the claim.
"""

from __future__ import annotations

import json
from typing import Any

from jsonschema import Draft7Validator

# A caught bug must be reported on roughly the right line. Two lines of slack absorbs
# off-by-one differences in how a snippet was extracted, without letting a model claim
# credit for pointing anywhere in the file.
LINE_TOLERANCE = 2


def parse_output(raw: str) -> dict[str, Any] | None:
    """Parse a model's raw output as a JSON object, or None if it is not one."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def is_schema_valid(raw: str, schema: dict[str, Any]) -> bool:
    """True if `raw` parses as JSON and satisfies the ReviewOutput schema."""
    parsed = parse_output(raw)
    if parsed is None:
        return False
    return Draft7Validator(schema).is_valid(parsed)


def compute_schema_validity(outputs: list[str], schema: dict[str, Any]) -> float:
    """Fraction of outputs that satisfy the schema.

    The denominator is every output the model produced — an output that failed to parse
    is a failure, not a sample to exclude (NFR-7).
    """
    if not outputs:
        return 0.0
    return sum(is_schema_valid(raw, schema) for raw in outputs) / len(outputs)


def is_bug_caught(raw: str, ground_truth: dict[str, Any], schema: dict[str, Any]) -> bool:
    """True if the output is schema-valid and names a line within tolerance of the bug.

    Invalid output catches nothing: a review nobody can parse has not found anything,
    however right it might have been in prose.
    """
    if not is_schema_valid(raw, schema):
        return False
    parsed = parse_output(raw)
    assert parsed is not None  # guaranteed by is_schema_valid
    return abs(int(parsed["line"]) - int(ground_truth["line"])) <= LINE_TOLERANCE


def compute_bug_catch_rate(
    outputs: list[str],
    holdout: list[dict[str, Any]],
    schema: dict[str, Any],
) -> float:
    """Fraction of holdout bugs the model identified, over the whole holdout set."""
    if len(outputs) != len(holdout):
        raise ValueError(
            f"outputs and holdout must align 1:1, got {len(outputs)} and {len(holdout)}"
        )
    if not holdout:
        return 0.0
    caught = sum(
        is_bug_caught(raw, truth, schema) for raw, truth in zip(outputs, holdout, strict=True)
    )
    return caught / len(holdout)
