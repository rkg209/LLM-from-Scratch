"""Wire shapes for `/v1/review` and `/health`.

`ReviewResponse` (O0's `{"raw": ...}` wrapper) is gone as of O2: the 200 body is now
`ReviewOutput` itself (`ch3_operation.api.schema`) -- a caller gets the validated review, not
the model's unparsed text.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator

# The exact substring `ch3_operation.api.errors.CODE_FIELD_EMPTY_MARKER` matches against,
# to tell an empty `code` apart from every other request-validation failure inside the same
# `RequestValidationError` handler.
_CODE_FIELD_EMPTY_MESSAGE = "code must not be empty or whitespace-only"


class ReviewRequest(BaseModel):
    code: str
    context: str = ""

    @field_validator("code")
    @classmethod
    def _code_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(_CODE_FIELD_EMPTY_MESSAGE)
        return value


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
