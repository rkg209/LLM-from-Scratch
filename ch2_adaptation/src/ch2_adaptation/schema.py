"""ReviewOutput — the Pydantic mirror of eval/schema.json.

Any model output (base, frontier, or fine-tuned) is validated against this before it is
scored. The drift guard below is what stops this file and eval/schema.json from silently
diverging: if either changes without the other, import fails loudly.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from eval.harness import load_schema


class SchemaDriftError(RuntimeError):
    """eval/schema.json and ReviewOutput have diverged on a load-bearing constraint.

    Not AssertionError: `assert` is stripped under `python -O`, which would silently
    disable the one guard whose entire job is to not be silent.
    """


class ReviewOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: Literal["critical", "major", "minor", "info"]
    category: str = Field(min_length=1, max_length=64)
    line: int = Field(ge=1)
    issue: str = Field(min_length=1, max_length=512)
    suggested_fix: str = Field(min_length=1, max_length=1024)


def _load_bearing_keys(schema: dict[str, Any]) -> dict[str, Any]:
    props = schema["properties"]
    return {
        "required": set(schema["required"]),
        "additionalProperties": schema["additionalProperties"],
        "severity_enum": props["severity"]["enum"],
        "category_min_max": (props["category"]["minLength"], props["category"]["maxLength"]),
        "line_minimum": props["line"]["minimum"],
        "issue_min_max": (props["issue"]["minLength"], props["issue"]["maxLength"]),
        "suggested_fix_min_max": (
            props["suggested_fix"]["minLength"],
            props["suggested_fix"]["maxLength"],
        ),
    }


def _assert_matches_eval_schema(schema: dict[str, Any] | None = None) -> None:
    """Compare ReviewOutput against eval/schema.json on every load-bearing key.

    A naive `model_json_schema() == json.load(...)` does not work: Pydantic emits a
    `title` key per property and no top-level `$schema`. So only the keys that actually
    constrain scoring are compared — the same set eval/tests/test_schema.py pins.
    """
    reference = schema if schema is not None else load_schema()
    pydantic_schema = ReviewOutput.model_json_schema()

    expected = _load_bearing_keys(reference)
    props = pydantic_schema["properties"]
    actual = {
        "required": set(pydantic_schema["required"]),
        "additionalProperties": pydantic_schema.get("additionalProperties", False),
        "severity_enum": props["severity"]["enum"],
        "category_min_max": (props["category"]["minLength"], props["category"]["maxLength"]),
        "line_minimum": props["line"]["minimum"],
        "issue_min_max": (props["issue"]["minLength"], props["issue"]["maxLength"]),
        "suggested_fix_min_max": (
            props["suggested_fix"]["minLength"],
            props["suggested_fix"]["maxLength"],
        ),
    }

    if actual != expected:
        raise SchemaDriftError(
            "ReviewOutput has diverged from eval/schema.json on a load-bearing constraint:\n"
            f"  eval/schema.json: {expected}\n"
            f"  ReviewOutput:     {actual}\n"
            "Update whichever one is stale before trusting any score computed against it."
        )


_assert_matches_eval_schema()
