"""O4 T3: build the image, run it, hit `/health`, `/metrics`, `POST /v1/review`.

Skipped unless `DOCKER=1` -- a real `docker build` is minutes-long and needs a daemon this
project's CI/dev environment may not have. This is the one test that actually retires the
"does the container serve real traffic" risk; everything else in O4 is Dockerfile/script
review. Uses `ch3_operation/tests/fixtures/tiny.gguf` as the model so it needs no network
access beyond what building the image itself already requires.

    DOCKER=1 uv run pytest ch3_operation/tests/test_docker.py -q
"""

from __future__ import annotations

import os
import subprocess
import time

import pytest

requests = pytest.importorskip("requests")

pytestmark = pytest.mark.skipif(
    os.environ.get("DOCKER") != "1", reason="set DOCKER=1 to run the real docker build/run"
)

IMAGE_TAG = "ch3-operation-test:latest"
CONTAINER_NAME = "ch3-operation-test-container"
PORT = 8091


@pytest.fixture(scope="module")
def running_container():
    subprocess.run(
        ["docker", "build", "-f", "docker/Dockerfile", "-t", IMAGE_TAG, "."],
        check=True,
    )
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], check=False)
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            CONTAINER_NAME,
            "-p",
            f"{PORT}:8000",
            "-v",
            f"{os.getcwd()}/ch3_operation/tests/fixtures/tiny.gguf:/models/model.gguf:ro",
            "-e",
            "MODEL_PATH=/models/model.gguf",
            "-e",
            "CONFIG_PATH=/app/ch3_operation/configs/smoke.yaml",
            IMAGE_TAG,
        ],
        check=True,
    )
    try:
        _wait_for_health()
        yield f"http://localhost:{PORT}"
    finally:
        subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], check=False)


def _wait_for_health(timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if requests.get(f"http://localhost:{PORT}/health", timeout=2).status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"container did not become healthy within {timeout_s}s")


def test_health_is_ok(running_container: str) -> None:
    response = requests.get(f"{running_container}/health", timeout=5)
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_metrics_is_prometheus_text(running_container: str) -> None:
    response = requests.get(f"{running_container}/metrics", timeout=5)
    assert response.status_code == 200
    assert "review_requests_total" in response.text


def test_review_returns_a_response(running_container: str) -> None:
    response = requests.post(
        f"{running_container}/v1/review",
        json={"code": "public void f() {}", "context": ""},
        timeout=30,
    )
    # tiny.gguf cannot produce schema-valid JSON -- 422 VALIDATION_FAILED is the correct,
    # expected outcome here (same as the manual curl check in the umbrella plan's
    # Verification section). The point of this test is that the container answered at all.
    assert response.status_code in (200, 422)
    assert response.headers["X-Request-Id"]
