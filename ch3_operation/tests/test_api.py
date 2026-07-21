"""Route behaviour of the O0 FastAPI app, via `TestClient` against fakes -- no real model."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

import ch3_operation.api.main as api_main  # noqa: E402
from ch3_operation.api.main import create_app  # noqa: E402
from ch3_operation.config import ServeConfig, load_serve_config  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

CONFIGS = Path("ch3_operation/configs")


class FakeBackend:
    model_loaded = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def generate(self, prompt: str, max_tokens: int) -> str:
        self.calls.append((prompt, max_tokens))
        return "fake review output"


class DegradedBackend:
    model_loaded = False

    def generate(self, prompt: str, max_tokens: int) -> str:
        raise RuntimeError("model not loaded")


class FailingBackend:
    model_loaded = True

    def generate(self, prompt: str, max_tokens: int) -> str:
        raise RuntimeError("inference blew up")


@pytest.fixture
def smoke_config() -> ServeConfig:
    return load_serve_config(CONFIGS / "smoke.yaml")


def test_health_ok_when_model_loaded(smoke_config: ServeConfig) -> None:
    client = TestClient(create_app(FakeBackend(), smoke_config))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_health_degraded_when_backend_reports_unloaded(smoke_config: ServeConfig) -> None:
    client = TestClient(create_app(DegradedBackend(), smoke_config))

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "model_loaded": False}


def test_review_returns_the_backend_output(smoke_config: ServeConfig) -> None:
    client = TestClient(create_app(FakeBackend(), smoke_config))

    response = client.post("/review", json={"code": "public void f() {}", "context": "Spring"})

    assert response.status_code == 200
    assert response.json() == {"raw": "fake review output"}


def test_review_rejects_empty_code(smoke_config: ServeConfig) -> None:
    client = TestClient(create_app(FakeBackend(), smoke_config))

    response = client.post("/review", json={"code": "", "context": ""})

    assert response.status_code == 422


def test_review_backend_failure_returns_503_and_process_survives(
    smoke_config: ServeConfig,
) -> None:
    client = TestClient(create_app(FailingBackend(), smoke_config))

    response = client.post("/review", json={"code": "public void f() {}", "context": ""})
    assert response.status_code == 503

    # a second request proves the failure didn't take the worker down with it
    health = client.get("/health")
    assert health.status_code == 200


def test_prompt_handed_to_backend_carries_code_and_context(smoke_config: ServeConfig) -> None:
    backend = FakeBackend()
    client = TestClient(create_app(backend, smoke_config))

    client.post("/review", json={"code": "public void f() {}", "context": "Spring Boot"})

    prompt, max_tokens = backend.calls[0]
    assert "public void f() {}" in prompt
    assert "Spring Boot" in prompt
    assert max_tokens == smoke_config.max_tokens


def test_real_app_loads_the_backend_once_at_startup_not_per_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercises the actual `uvicorn ch3_operation.api.main:app` wiring (FR-20): the real
    backend loads exactly once, at lifespan startup, never again across requests."""
    construction_count = 0

    class CountingBackend:
        model_loaded = True

        def __init__(self, config: ServeConfig) -> None:
            nonlocal construction_count
            construction_count += 1

        def generate(self, prompt: str, max_tokens: int) -> str:
            return "counted"

    monkeypatch.setattr(api_main, "LlamaCppBackend", CountingBackend)

    with TestClient(api_main.app) as client:
        client.post("/review", json={"code": "public void f() {}", "context": ""})
        client.post("/review", json={"code": "public void g() {}", "context": ""})

    assert construction_count == 1
