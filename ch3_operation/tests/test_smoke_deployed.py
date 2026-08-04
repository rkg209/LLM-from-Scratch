"""O5: `scripts/smoke_deployed.py`, with `urllib.request.urlopen` faked out entirely."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS = Path("scripts")


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_scripts_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


smoke_deployed = _load_script("smoke_deployed")


class _FakeResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def _fake_urlopen(responses: dict[str, tuple[int, bytes]]):
    def urlopen(request, timeout=None):  # noqa: ANN001
        url = request.full_url if hasattr(request, "full_url") else request
        for key, (status, body) in responses.items():
            if key in url:
                return _FakeResponse(status, body)
        raise AssertionError(f"unexpected URL: {url}")

    return urlopen


def test_check_health_passes_when_model_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    body = json.dumps({"status": "ok", "model_loaded": True}).encode()
    monkeypatch.setattr(
        smoke_deployed.urllib.request, "urlopen", _fake_urlopen({"/health": (200, body)})
    )

    smoke_deployed.check_health("https://example.test", timeout_s=5)


def test_check_health_raises_when_model_not_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    body = json.dumps({"status": "degraded", "model_loaded": False}).encode()
    monkeypatch.setattr(
        smoke_deployed.urllib.request, "urlopen", _fake_urlopen({"/health": (200, body)})
    )

    with pytest.raises(smoke_deployed.SmokeCheckError, match="model_loaded"):
        smoke_deployed.check_health("https://example.test", timeout_s=5)


def test_check_health_raises_on_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        smoke_deployed.urllib.request, "urlopen", _fake_urlopen({"/health": (503, b"{}")})
    )

    with pytest.raises(smoke_deployed.SmokeCheckError, match="503"):
        smoke_deployed.check_health("https://example.test", timeout_s=5)


def test_check_metrics_requires_the_requests_total_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        smoke_deployed.urllib.request,
        "urlopen",
        _fake_urlopen({"/metrics": (200, b"# nothing useful here")}),
    )

    with pytest.raises(smoke_deployed.SmokeCheckError, match="review_requests_total"):
        smoke_deployed.check_metrics("https://example.test", timeout_s=5)


def test_check_review_returns_the_parsed_result_on_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = json.dumps(
        {
            "severity": "critical",
            "category": "npe",
            "line": 2,
            "issue": "y",
            "suggested_fix": "z",
        }
    ).encode()
    monkeypatch.setattr(
        smoke_deployed.urllib.request, "urlopen", _fake_urlopen({"/v1/review": (200, body)})
    )

    result = smoke_deployed.check_review("https://example.test", timeout_s=5)

    assert result["severity"] == "critical"


def test_check_review_raises_on_422(monkeypatch: pytest.MonkeyPatch) -> None:
    body = json.dumps({"error": {"code": "VALIDATION_FAILED"}}).encode()
    monkeypatch.setattr(
        smoke_deployed.urllib.request, "urlopen", _fake_urlopen({"/v1/review": (422, body)})
    )

    with pytest.raises(smoke_deployed.SmokeCheckError, match="422"):
        smoke_deployed.check_review("https://example.test", timeout_s=5)
