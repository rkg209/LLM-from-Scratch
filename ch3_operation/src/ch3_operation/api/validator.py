"""O2: turn a model's raw text into a validated `ReviewOutput`, or an honest failure.

Exactly two parse attempts, in order: a bare `json.loads(raw)`, then extraction from a
```json fenced block. No third attempt, no retrying the model, no coercion, no defaults --
`prompts.SYSTEM_PROMPT` already tells the model not to use a fence, so a model that fences
anyway is still given one chance to be read, but nothing beyond that. Retrying inflates the
schema-validity rate this project publishes (CON-7, NFR-20); this module's whole job is to
report exactly what happened, not to make the model look better than it was.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from ch3_operation.api.schema import ReviewOutput

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class OutputParseError(ValueError):
    """Neither parse attempt produced a JSON object at all."""


class OutputSchemaError(ValueError):
    """Parsed as JSON, but did not satisfy `ReviewOutput` -- `detail` is Pydantic's message."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def _extract_from_fence(raw: str) -> dict[str, Any] | None:
    """Pull a JSON object out of the first ```json ... ``` (or bare ``` ... ```) fence."""
    match = _FENCE.search(raw)
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    """Attempt one: the whole response is a bare JSON object."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def validate_output(raw: str) -> ReviewOutput:
    """Parse `raw` (exactly two attempts) and validate it against `ReviewOutput`.

    Raises `OutputParseError` if neither attempt yields a JSON object, or
    `OutputSchemaError` if one did but Pydantic rejected its shape.
    """
    parsed = _parse_json_object(raw)
    if parsed is None:
        parsed = _extract_from_fence(raw)
    if parsed is None:
        raise OutputParseError(
            "model output was not valid JSON and no ```json fenced block could be extracted"
        )

    try:
        return ReviewOutput.model_validate(parsed)
    except ValidationError as exc:
        raise OutputSchemaError(detail=str(exc)) from exc
