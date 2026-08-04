"""Chapter 3 config: the profiles load, CPU-only holds, and ${MODEL_PATH} resolves."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from ch3_operation.config import N_GPU_LAYERS, load_serve_config

CONFIGS = Path("ch3_operation/configs")


@pytest.mark.parametrize("profile", ["smoke", "full"])
def test_committed_config_loads(profile: str) -> None:
    config = load_serve_config(CONFIGS / f"{profile}.yaml")
    assert config.port == 8000
    assert config.metrics_window == 100
    assert config.max_request_bytes == 32768
    assert config.api_version == "v1"


def test_serving_is_cpu_only() -> None:
    """The 'edge' claim is a CPU claim. This is a constant, not a knob (CON-12)."""
    assert N_GPU_LAYERS == 0


def test_smoke_config_points_at_the_test_fixture() -> None:
    config = load_serve_config(CONFIGS / "smoke.yaml")
    assert config.model_path.endswith("tiny.gguf")
    assert config.resolved_model_path() == Path("ch3_operation/tests/fixtures/tiny.gguf")


def test_full_config_resolves_model_path_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The GGUF is pulled from HF Hub at container start, not baked into the image."""
    config = load_serve_config(CONFIGS / "full.yaml")
    monkeypatch.setenv("MODEL_PATH", "/models/model-Q4_K_M.gguf")

    assert config.resolved_model_path() == Path("/models/model-Q4_K_M.gguf")


def test_unset_model_path_raises_rather_than_serving_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_serve_config(CONFIGS / "full.yaml")
    monkeypatch.delenv("MODEL_PATH", raising=False)

    with pytest.raises(KeyError, match="MODEL_PATH"):
        config.resolved_model_path()


def test_invalid_metrics_window_is_rejected() -> None:
    config = load_serve_config(CONFIGS / "smoke.yaml")

    with pytest.raises(ValueError, match="metrics_window"):
        replace(config, metrics_window=0)


def test_invalid_max_request_bytes_is_rejected() -> None:
    config = load_serve_config(CONFIGS / "smoke.yaml")

    with pytest.raises(ValueError, match="max_request_bytes"):
        replace(config, max_request_bytes=0)
