"""The FastAPI app: `POST /v1/review` (+ deprecated `/review` alias), `GET /health`.

O2 turns O0's raw pass-through into a guardrailed endpoint: the backend's raw text is parsed
and validated against `ReviewOutput` before anything is returned as a 200. A model failure to
produce valid structured output is an honest `422`, never silently reshaped into something
that looks like success.

`create_app` never constructs a real backend itself -- it only wires routes around whatever
`Backend` it is given, reading it from `request.app.state` on every call so the same routes
work whether the backend was supplied synchronously (tests, `serve.py`) or loaded later by
`_load_backend_on_startup` (the bare `uvicorn ...:app` path below). That split is what keeps
importing this module free of side effects: the real `LlamaCppBackend` loads once, at ASGI
startup, never merely from `import`.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ch3_operation.api.errors import register_exception_handlers
from ch3_operation.api.metrics import PrometheusMetrics
from ch3_operation.api.middleware import (
    ContentSizeLimitMiddleware,
    RequestIdMiddleware,
    RequestTimingMiddleware,
)
from ch3_operation.api.schema import ReviewOutput
from ch3_operation.api.schemas import HealthResponse, ReviewRequest
from ch3_operation.api.validator import validate_output
from ch3_operation.config import ServeConfig, load_serve_config
from ch3_operation.model_backend import Backend, LlamaCppBackend
from ch3_operation.prompts import format_zero_shot
from monitoring.drift import RollingQualityMonitor

logger = logging.getLogger(__name__)


def create_app(backend: Backend | None, config: ServeConfig, *, lifespan=None) -> FastAPI:
    app = FastAPI(title="Java Code Reviewer", lifespan=lifespan)
    app.state.backend = backend
    app.state.config = config
    app.state.metrics = PrometheusMetrics()
    app.state.monitor = RollingQualityMonitor(config.metrics_window, app.state.metrics)
    register_exception_handlers(app)
    # Order matters: Starlette runs middleware in reverse of add order, so
    # ContentSizeLimitMiddleware (added last) runs first and can reject an oversized body
    # before RequestIdMiddleware's/RequestTimingMiddleware's state would otherwise be
    # needed downstream. RequestId and RequestTiming both write to `request.state` that the
    # error handlers read, so they must wrap everything inside them.
    app.add_middleware(RequestTimingMiddleware)
    app.add_middleware(ContentSizeLimitMiddleware, max_bytes=config.max_request_bytes)
    app.add_middleware(RequestIdMiddleware)

    @app.get("/health")
    def health(request: Request, response: Response) -> HealthResponse:
        # `model_loaded` isn't part of the Backend Protocol -- LlamaCppBackend either
        # loads at startup or the app never comes up, so it has no way to report
        # "degraded" post-startup today. The getattr default only lets fakes drive the
        # 503 branch for route-logic testing; a real post-startup liveness signal is out
        # of scope for O0/O2 (that's O3's monitoring, if it's ever needed). /health keeps
        # its own body shape (HealthResponse) rather than the error envelope -- it is a
        # liveness probe, not a request that failed.
        backend = request.app.state.backend
        model_loaded = backend is not None and getattr(backend, "model_loaded", True)
        response.status_code = 200 if model_loaded else 503
        return HealthResponse(
            status="ok" if model_loaded else "degraded", model_loaded=model_loaded
        )

    def review(payload: ReviewRequest, request: Request) -> ReviewOutput:
        state = request.app.state
        if state.backend is None or not getattr(state.backend, "model_loaded", True):
            raise HTTPException(status_code=503, detail="Model is still loading.")

        prompt = format_zero_shot(payload.code, payload.context)
        try:
            raw = state.backend.generate(prompt, state.config.max_tokens)
        except Exception as exc:
            logger.exception("backend.generate failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        # One inference, two parse attempts (validate_output), then an honest failure --
        # no retry of the model. Retrying would inflate the schema-validity rate this
        # project publishes. A parse/schema failure here propagates to the OutputParseError/
        # OutputSchemaError handlers in errors.py, which record the (invalid) outcome
        # themselves -- this function only records the success path (AC-4).
        result = validate_output(raw)

        start = getattr(request.state, "request_start", None)
        elapsed = time.perf_counter() - start if start is not None else 0.0
        state.metrics.record_request(elapsed, valid=True)
        state.monitor.record(valid=True)
        return result

    app.add_api_route("/v1/review", review, methods=["POST"])
    app.add_api_route(
        "/review",
        review,
        methods=["POST"],
        deprecated=True,
        description="Deprecated alias for POST /v1/review.",
    )

    @app.get("/metrics")
    def metrics(request: Request) -> Response:
        # Never triggers inference and always responds fast (planning/05-api-design.md's
        # own description of this route) -- it just serializes the registry's current state.
        payload = generate_latest(request.app.state.metrics.registry)
        return Response(content=payload, media_type=CONTENT_TYPE_LATEST)

    return app


@asynccontextmanager
async def _load_backend_on_startup(app: FastAPI) -> AsyncIterator[None]:
    """Loads the real model once, when uvicorn actually starts serving -- not at import."""
    app.state.backend = LlamaCppBackend(app.state.config)
    yield


_config_path = os.getenv("CONFIG_PATH", "ch3_operation/configs/full.yaml")
_config = load_serve_config(_config_path)
app = create_app(backend=None, config=_config, lifespan=_load_backend_on_startup)
