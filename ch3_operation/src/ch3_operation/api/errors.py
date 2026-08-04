"""O2: the error envelope, the error-code registry, and the exception handlers.

Registry and envelope shape are locked in `planning/05-api-design.md §3`. `RATE_LIMITED` is
defined here but never raised in v1.0 (plan D-6) -- it is not behind any O2 acceptance
criterion, and HF Spaces throttles at the infrastructure level.

O3 addition: every handler below also records the request as a monitoring outcome
(`app.state.metrics`/`app.state.monitor`, AC-4) -- these handlers are the only source of a
4xx/5xx response in this app, and all of them currently fire only for `/v1/review` (or its
`/review` alias), since `/health` reports its own status via a plain `Response` rather than
raising, and `/metrics` never fails in a way that reaches here.
"""

from __future__ import annotations

import logging
import time
from enum import StrEnum

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from ch3_operation.api.validator import OutputParseError, OutputSchemaError

logger = logging.getLogger(__name__)


class ErrorCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    CODE_FIELD_EMPTY = "CODE_FIELD_EMPTY"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    SCHEMA_VIOLATION = "SCHEMA_VIOLATION"
    RATE_LIMITED = "RATE_LIMITED"  # defined, not implemented in v1.0 (plan D-6)
    INFERENCE_ERROR = "INFERENCE_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    MODEL_NOT_READY = "MODEL_NOT_READY"


class ErrorBody(BaseModel):
    code: ErrorCode
    message: str
    detail: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
    request_id: str


def error_response(
    code: ErrorCode,
    message: str,
    status_code: int,
    request_id: str,
    detail: str | None = None,
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorBody(code=code, message=message, detail=detail), request_id=request_id
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(),
        headers={"X-Request-Id": request_id},
    )


# The sentinel `ReviewRequest.code`'s field_validator raises with -- distinguishes an empty
# `code` (400 CODE_FIELD_EMPTY) from every other request-validation failure (400
# INVALID_REQUEST) inside the same RequestValidationError handler.
CODE_FIELD_EMPTY_MARKER = "code must not be empty or whitespace-only"


def _request_id(request: Request) -> str:
    # Set unconditionally by RequestIdMiddleware; the getattr default only protects the
    # handful of TestClient calls in tests that build a bare app without the middleware.
    return getattr(request.state, "request_id", "unknown")


def _record_outcome(request: Request, valid: bool) -> None:
    """Record a failed `/v1/review` outcome for `/metrics` (AC-4).

    `request.state.request_start` is set by `RequestTimingMiddleware` before `call_next`;
    the getattr default lets tests that build a bare app (no O3 wiring) exercise these
    handlers without crashing. `app.state.metrics`/`monitor` are likewise optional so O2's
    own tests, which predate O3, keep working unmodified.
    """
    metrics = getattr(request.app.state, "metrics", None)
    monitor = getattr(request.app.state, "monitor", None)
    if metrics is None or monitor is None:
        return
    start = getattr(request.state, "request_start", None)
    elapsed = time.perf_counter() - start if start is not None else 0.0
    metrics.record_request(elapsed, valid)
    monitor.record(valid)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        _record_outcome(request, valid=False)
        request_id = _request_id(request)
        message = str(exc.errors()[0]["msg"]) if exc.errors() else "invalid request body"
        if CODE_FIELD_EMPTY_MARKER in message:
            return error_response(
                ErrorCode.CODE_FIELD_EMPTY,
                "Field 'code' must not be empty or whitespace-only.",
                status.HTTP_400_BAD_REQUEST,
                request_id,
            )
        return error_response(
            ErrorCode.INVALID_REQUEST,
            "Request body is not valid, or a required field is missing.",
            status.HTTP_400_BAD_REQUEST,
            request_id,
            detail=message,
        )

    @app.exception_handler(OutputParseError)
    async def handle_output_parse_error(request: Request, exc: OutputParseError) -> JSONResponse:
        _record_outcome(request, valid=False)
        return error_response(
            ErrorCode.VALIDATION_FAILED,
            "Model output did not conform to ReviewOutput schema after 2 parse attempts.",
            422,
            _request_id(request),
        )

    @app.exception_handler(OutputSchemaError)
    async def handle_output_schema_error(request: Request, exc: OutputSchemaError) -> JSONResponse:
        _record_outcome(request, valid=False)
        return error_response(
            ErrorCode.SCHEMA_VIOLATION,
            "Model output parsed as JSON but failed ReviewOutput validation.",
            422,
            _request_id(request),
            detail=exc.detail,
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        # 503 is reserved for MODEL_NOT_READY, 500 for a backend that raised mid-inference
        # (INFERENCE_ERROR) -- both are raised as plain HTTPException by main.py's `review`
        # route, distinguished only by status code, since that route is the only source of
        # either (`planning/05-api-design.md §3.2`).
        code = {503: ErrorCode.MODEL_NOT_READY, 500: ErrorCode.INFERENCE_ERROR}.get(
            exc.status_code, ErrorCode.INTERNAL_ERROR
        )
        _record_outcome(request, valid=False)
        return error_response(code, str(exc.detail), exc.status_code, _request_id(request))

    @app.exception_handler(Exception)
    async def handle_unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled exception")
        _record_outcome(request, valid=False)
        return error_response(
            ErrorCode.INTERNAL_ERROR,
            "An unexpected server error occurred.",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            _request_id(request),
        )
