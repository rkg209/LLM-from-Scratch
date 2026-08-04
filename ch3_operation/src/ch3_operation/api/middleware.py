"""`RequestIdMiddleware`, `ContentSizeLimitMiddleware` (O2), and `RequestTimingMiddleware` (O3).

`RequestTimingMiddleware` only stashes a start timestamp on `request.state.request_start`,
before `call_next` -- it never records anything itself. Exception handlers run *inside*
`call_next` (Starlette's `ExceptionMiddleware` sits below any middleware added via
`add_middleware`), so a handler for the 422/500 paths can already compute elapsed time from
`request.state.request_start` and call `metrics.record_request()` itself; if timing were
instead computed only *after* `call_next` returns, every error path would have already
finished responding by the time that number existed, and AC-4 ("every /review outcome,
including 422 and 500") would be unreachable from the error handlers.
"""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ch3_operation.api.errors import ErrorBody, ErrorCode, ErrorResponse


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Mint a UUID v4 per request; stash it on `request.state` and echo it as a header on
    every response, success or error alike (plan D-2)."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Stash the request's start time on `request.state.request_start`. Records nothing --
    the route handler and the error handlers each compute their own elapsed time from it and
    call `metrics.record_request()` themselves, once they know the outcome (AC-5)."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request.state.request_start = time.perf_counter()
        return await call_next(request)


class ContentSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject request bodies over `max_bytes` with `413 PAYLOAD_TOO_LARGE`.

    Checks `Content-Length` first (cheap, no body read); when it is absent, falls back to
    counting streamed bytes so a caller cannot dodge the limit by omitting the header
    (AC-8, `planning/05-api-design.md §2.3`).
    """

    def __init__(self, app, max_bytes: int) -> None:  # noqa: ANN001 - Starlette's own signature
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:

        request_id = getattr(request.state, "request_id", "unknown")
        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > self._max_bytes:
            return self._too_large(request_id)

        if content_length is None:
            body = await request.body()
            if len(body) > self._max_bytes:
                return self._too_large(request_id)

        return await call_next(request)

    def _too_large(self, request_id: str) -> JSONResponse:

        body = ErrorResponse(
            error=ErrorBody(
                code=ErrorCode.PAYLOAD_TOO_LARGE,
                message=f"Request body exceeds the {self._max_bytes}-byte limit.",
                detail=None,
            ),
            request_id=request_id,
        )
        return JSONResponse(
            status_code=413, content=body.model_dump(), headers={"X-Request-Id": request_id}
        )
