"""O3 T5: `scripts/benchmark_serving.py`, with the HTTP call monkeypatched out."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import urllib.error
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


benchmark_serving = _load_script("benchmark_serving")


def test_benchmark_computes_p50_p99_and_throughput(monkeypatch: pytest.MonkeyPatch) -> None:
    latencies = iter([0.1, 0.2, 0.3, 0.4, 0.5])
    monkeypatch.setattr(
        benchmark_serving,
        "send_one_request",
        lambda url, timeout_s=60.0: (next(latencies), 200),
    )

    result = benchmark_serving.benchmark("http://example.test", n_requests=5)

    assert result["n_requests"] == 5
    assert result["p50_s"] == pytest.approx(0.3)
    assert result["throughput_rps"] > 0
    assert result["n_errors"] == 0
    assert result["error_rate"] == 0.0


def test_benchmark_rejects_a_nonpositive_request_count() -> None:
    with pytest.raises(ValueError, match="n_requests"):
        benchmark_serving.benchmark("http://example.test", n_requests=0)


def test_benchmark_survives_a_non_2xx_response_and_counts_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 422 (the model failing to emit schema-valid JSON) is a documented, expected
    response for this API -- it must be timed and counted as an error, never abort the
    run the way a bare `urlopen` call (which raises `HTTPError` on any status >= 400)
    would."""
    monkeypatch.setattr(
        benchmark_serving, "send_one_request", lambda url, timeout_s=60.0: (0.2, 422)
    )

    result = benchmark_serving.benchmark("http://example.test", n_requests=3)

    assert result["n_requests"] == 3
    assert result["n_errors"] == 3
    assert result["error_rate"] == pytest.approx(1.0)
    assert result["p50_s"] == pytest.approx(0.2)


def test_send_one_request_reports_http_error_status_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_422(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.HTTPError(
            url="http://example.test/v1/review",
            code=422,
            msg="Unprocessable Entity",
            hdrs=None,
            fp=io.BytesIO(b"{}"),
        )

    monkeypatch.setattr(benchmark_serving.urllib.request, "urlopen", _raise_422)

    latency, status = benchmark_serving.send_one_request("http://example.test")

    assert status == 422
    assert latency >= 0.0


def test_load_benchmark_config_reads_results_path() -> None:
    config = benchmark_serving.load_benchmark_config("ch3_operation/configs/benchmark_smoke.yaml")
    assert config.results_path == "outputs/smoke/serving_metrics.json"


def test_main_writes_the_results_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    latencies = iter([0.1, 0.2, 0.1])
    monkeypatch.setattr(
        benchmark_serving,
        "send_one_request",
        lambda url, timeout_s=60.0: (next(latencies), 200),
    )
    results_path = tmp_path / "serving_metrics.json"
    monkeypatch.setattr(
        benchmark_serving,
        "load_benchmark_config",
        lambda _path: benchmark_serving.BenchmarkConfig(results_path=str(results_path)),
    )

    class Args:
        url = "http://example.test"
        requests = 3
        config = Path("ch3_operation/configs/benchmark_smoke.yaml")

    monkeypatch.setattr(benchmark_serving, "parse_args", lambda: Args())

    benchmark_serving.main()

    assert results_path.exists()
    written = json.loads(results_path.read_text())
    assert written["url"] == "http://example.test"
    assert written["measured_at"].endswith("+00:00")
