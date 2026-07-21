"""The FastAPI app: `POST /review`, `GET /health`.

No schema validation, no repair, no metrics here — `/review` returns whatever the model
produced raw. That is O2's job. O0 only proves the serving path carries bytes end to end.

`create_app` never constructs a real backend itself -- it only wires routes around
whatever `Backend` it is given, reading it from `request.app.state` on every call so the
same routes work whether the backend was supplied synchronously (tests, `serve.py`) or
loaded later by `_load_backend_on_startup` (the bare `uvicorn ...:app` path below). That
split is what keeps importing this module free of side effects: the real
`LlamaCppBackend` loads once, at ASGI startup, never merely from `import`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response

from ch3_operation.api.schemas import HealthResponse, ReviewRequest, ReviewResponse
from ch3_operation.config import ServeConfig, load_serve_config
from ch3_operation.model_backend import Backend, LlamaCppBackend
from ch3_operation.prompts import format_zero_shot

logger = logging.getLogger(__name__)


def create_app(backend: Backend | None, config: ServeConfig, *, lifespan=None) -> FastAPI:
    app = FastAPI(title="Java Code Reviewer", lifespan=lifespan)
    app.state.backend = backend
    app.state.config = config

    @app.get("/health")
    def health(request: Request, response: Response) -> HealthResponse:
        # `model_loaded` isn't part of the Backend Protocol -- LlamaCppBackend either
        # loads at startup or the app never comes up, so it has no way to report
        # "degraded" post-startup today. The getattr default only lets fakes drive the
        # 503 branch for route-logic testing; a real post-startup liveness signal is out
        # of scope for O0 (that's O3's monitoring, if it's ever needed).
        backend = request.app.state.backend
        model_loaded = backend is not None and getattr(backend, "model_loaded", True)
        response.status_code = 200 if model_loaded else 503
        return HealthResponse(
            status="ok" if model_loaded else "degraded", model_loaded=model_loaded
        )

    @app.post("/review")
    def review(payload: ReviewRequest, request: Request) -> ReviewResponse:
        state = request.app.state
        prompt = format_zero_shot(payload.code, payload.context)
        try:
            raw = state.backend.generate(prompt, state.config.max_tokens)
        except Exception as exc:
            logger.exception("backend.generate failed")
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return ReviewResponse(raw=raw)

    return app


@asynccontextmanager
async def _load_backend_on_startup(app: FastAPI) -> AsyncIterator[None]:
    """Loads the real model once, when uvicorn actually starts serving -- not at import."""
    app.state.backend = LlamaCppBackend(app.state.config)
    yield


_config_path = os.getenv("CONFIG_PATH", "ch3_operation/configs/full.yaml")
_config = load_serve_config(_config_path)
app = create_app(backend=None, config=_config, lifespan=_load_backend_on_startup)
