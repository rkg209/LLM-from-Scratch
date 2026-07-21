"""The inference seam: Protocol conformance, CPU-only, the lock, and a real fixture load."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

import pytest
from ch3_operation.config import load_serve_config
from ch3_operation.model_backend import Backend, LlamaCppBackend

CONFIGS = Path("ch3_operation/configs")


class FakeBackend:
    def generate(self, prompt: str, max_tokens: int) -> str:
        return "fake output"


def test_fake_backend_satisfies_the_protocol() -> None:
    assert isinstance(FakeBackend(), Backend)


class _StubLlama:
    """Stands in for `llama_cpp.Llama`, recording the kwargs it was constructed with."""

    last_init_kwargs: dict[str, Any] | None = None

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_init_kwargs = kwargs

    def __call__(self, prompt: str, max_tokens: int, temperature: float, echo: bool) -> Any:
        return {"choices": [{"text": "stubbed"}]}


def _install_stub_llama_cpp(monkeypatch: pytest.MonkeyPatch) -> type[_StubLlama]:
    stub_module = type(sys)("llama_cpp")
    stub_module.Llama = _StubLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", stub_module)
    return _StubLlama


def test_n_gpu_layers_is_hardcoded_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """CON-12: n_gpu_layers is never read from config."""
    stub = _install_stub_llama_cpp(monkeypatch)
    config = load_serve_config(CONFIGS / "smoke.yaml")

    LlamaCppBackend(config)

    assert stub.last_init_kwargs["n_gpu_layers"] == 0


class _LockCheckingLlama:
    def __init__(self, **kwargs: Any) -> None:
        self._backend_lock: threading.Lock | None = None

    def __call__(self, prompt: str, max_tokens: int, temperature: float, echo: bool) -> Any:
        assert self._backend_lock is not None and self._backend_lock.locked()
        return {"choices": [{"text": "locked"}]}


def test_generate_holds_the_lock_during_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_module = type(sys)("llama_cpp")
    stub_module.Llama = _LockCheckingLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", stub_module)
    config = load_serve_config(CONFIGS / "smoke.yaml")

    backend = LlamaCppBackend(config)
    backend._llm._backend_lock = backend._lock

    assert backend.generate("hello", max_tokens=4) == "locked"


llama_cpp = pytest.importorskip("llama_cpp")


def test_real_backend_loads_the_fixture_and_generates() -> None:
    """The one test that actually retires the integration risk."""
    config = load_serve_config(CONFIGS / "smoke.yaml")

    backend = LlamaCppBackend(config)
    output = backend.generate("Once upon a time", max_tokens=4)

    assert isinstance(output, str)
    assert not output.startswith("Once upon a time")
