"""O3: `PrometheusMetrics` and `GET /metrics`."""

from __future__ import annotations

from pathlib import Path

import pytest
from ch3_operation.api.metrics import PrometheusMetrics
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

pytest.importorskip("fastapi")

from ch3_operation.api.main import create_app  # noqa: E402
from ch3_operation.config import ServeConfig, load_serve_config  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

CONFIGS = Path("ch3_operation/configs")


class FakeBackend:
    model_loaded = True

    def generate(self, prompt: str, max_tokens: int) -> str:
        return (
            '{"severity": "minor", "category": "x", "line": 1, "issue": "y", '
            '"suggested_fix": "z"}'
        )


@pytest.fixture
def smoke_config() -> ServeConfig:
    return load_serve_config(CONFIGS / "smoke.yaml")


def test_metrics_own_a_fresh_registry_each_instance() -> None:
    """Two instances must not collide -- each test in this file constructs its own app,
    and the global default registry would raise "duplicated timeseries" on the second."""
    first = PrometheusMetrics()
    second = PrometheusMetrics()
    assert first.registry is not second.registry


def test_record_request_increments_counters_and_observes_latency() -> None:
    metrics = PrometheusMetrics()

    metrics.record_request(0.2, valid=True)
    metrics.record_request(1.5, valid=False)

    output = generate_latest(metrics.registry).decode()
    assert 'review_requests_total{status="success"} 1.0' in output
    assert 'review_requests_total{status="error"} 1.0' in output
    assert "review_request_latency_seconds_count 2.0" in output


def test_metrics_endpoint_returns_prometheus_content_type(smoke_config: ServeConfig) -> None:
    client = TestClient(create_app(FakeBackend(), smoke_config))

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(CONTENT_TYPE_LATEST.split(";")[0])


def test_metrics_endpoint_reflects_review_traffic(smoke_config: ServeConfig) -> None:
    client = TestClient(create_app(FakeBackend(), smoke_config))

    client.post("/v1/review", json={"code": "public void f() {}", "context": ""})

    body = client.get("/metrics").text
    assert 'review_requests_total{status="success"} 1.0' in body
    assert "review_schema_validity_rate 1.0" in body


def test_metrics_endpoint_reflects_a_failed_review(smoke_config: ServeConfig) -> None:
    class UnparsableBackend:
        model_loaded = True

        def generate(self, prompt: str, max_tokens: int) -> str:
            return "not json"

    client = TestClient(create_app(UnparsableBackend(), smoke_config))

    client.post("/v1/review", json={"code": "public void f() {}", "context": ""})

    body = client.get("/metrics").text
    assert 'review_requests_total{status="error"} 1.0' in body
    assert "review_schema_validity_rate 0.0" in body
