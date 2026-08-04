"""O2: the error envelope shape and `error_response` in isolation, no FastAPI app needed."""

from __future__ import annotations

import json

import pytest
from ch3_operation.api.errors import ErrorCode, ErrorResponse, error_response


def test_error_response_envelope_shape() -> None:
    response = error_response(
        ErrorCode.VALIDATION_FAILED,
        "model output was not valid JSON",
        422,
        "a3f2c1d4-8b7e-4f9a-b2c3-d4e5f6a7b8c9",
    )

    assert response.status_code == 422
    body = json.loads(response.body)
    assert body == {
        "error": {
            "code": "VALIDATION_FAILED",
            "message": "model output was not valid JSON",
            "detail": None,
        },
        "request_id": "a3f2c1d4-8b7e-4f9a-b2c3-d4e5f6a7b8c9",
    }
    assert response.headers["X-Request-Id"] == "a3f2c1d4-8b7e-4f9a-b2c3-d4e5f6a7b8c9"


def test_error_response_carries_detail_when_given() -> None:
    response = error_response(
        ErrorCode.SCHEMA_VIOLATION, "schema violation", 422, "req-1", detail="line: bad value"
    )

    body = json.loads(response.body)
    assert body["error"]["detail"] == "line: bad value"


@pytest.mark.parametrize(
    "code",
    [
        ErrorCode.INVALID_REQUEST,
        ErrorCode.CODE_FIELD_EMPTY,
        ErrorCode.PAYLOAD_TOO_LARGE,
        ErrorCode.VALIDATION_FAILED,
        ErrorCode.SCHEMA_VIOLATION,
        ErrorCode.RATE_LIMITED,
        ErrorCode.INFERENCE_ERROR,
        ErrorCode.INTERNAL_ERROR,
        ErrorCode.MODEL_NOT_READY,
    ],
)
def test_every_registry_code_round_trips_through_error_response(code: ErrorCode) -> None:
    response = error_response(code, "message", 400, "req-1")
    body = json.loads(response.body)
    assert body["error"]["code"] == code.value


def test_error_response_model_validates_its_own_shape() -> None:
    """`ErrorResponse` itself round-trips the envelope planning/05-api-design.md §3.1 locks."""
    parsed = ErrorResponse.model_validate(
        {
            "error": {"code": "MODEL_NOT_READY", "message": "loading", "detail": None},
            "request_id": "req-1",
        }
    )
    assert parsed.error.code == ErrorCode.MODEL_NOT_READY
