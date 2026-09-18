"""O3 T5: fire a fixed prompt set at a deployed URL; report p50/p99/throughput.

This is what produces the numbers in the README's serving scorecard -- run against a real
deployed Space (O5), never against a laptop, so the published numbers mean what they say.

Usage:
    uv run python scripts/benchmark_serving.py --url http://localhost:8000 --requests 30 \
      --config ch3_operation/configs/benchmark_smoke.yaml
"""

from __future__ import annotations

import argparse
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from eval.config import load_config
from eval.harness import write_json_atomic

# One representative snippet is enough to measure serving latency -- benchmarking is not
# re-scoring quality (that's ch3_operation.evaluate); it only needs *a* request the model
# will actually spend real inference time on.
_BENCHMARK_CODE = (
    "public String getUserName(User user) {\n    return user.getProfile().getName();\n}"
)


@dataclass(frozen=True)
class BenchmarkConfig:
    results_path: str


def load_benchmark_config(path: Path | str) -> BenchmarkConfig:
    return load_config(path, BenchmarkConfig)


def send_one_request(url: str, timeout_s: float = 60.0) -> tuple[float, int]:
    """POST one review request; return its (wall-clock latency in seconds, HTTP status).

    A non-2xx response (e.g. the documented 422 the API returns when the model fails to
    emit schema-valid JSON) is not a network failure here -- it is a real, expected outcome
    of this API and must be timed and counted, not raised past this function.
    """
    payload = f'{{"code": {_BENCHMARK_CODE!r}}}'.replace("'", '"').encode()
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/review",
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        error.read()
        status = error.code
    return time.perf_counter() - start, status


def benchmark(url: str, n_requests: int) -> dict[str, float]:
    """Fire `n_requests` sequentially and report latency percentiles and error rate.

    `throughput_rps` is sequential throughput (n_requests / total wall time of one
    request after another) -- what NFR-3's "single-user load" asks for, not a
    concurrent-load figure.
    """
    if n_requests < 1:
        raise ValueError(f"n_requests must be positive, got {n_requests}")

    results = [send_one_request(url) for _ in range(n_requests)]
    latencies = [latency for latency, _status in results]
    n_errors = sum(1 for _latency, status in results if status >= 400)
    total_time = sum(latencies)

    return {
        "n_requests": n_requests,
        "p50_s": statistics.median(latencies),
        "p99_s": statistics.quantiles(latencies, n=100)[98] if n_requests >= 2 else latencies[0],
        "throughput_rps": n_requests / total_time if total_time > 0 else 0.0,
        "n_errors": n_errors,
        "error_rate": n_errors / n_requests,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a deployed reviewer endpoint.")
    parser.add_argument("--url", required=True, help="base URL of the deployed service")
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_benchmark_config(args.config)

    print(f"[benchmark_serving] sending {args.requests} requests to {args.url}")
    measured_at = datetime.now(UTC).isoformat(timespec="seconds")
    # The numbers mean nothing without where and when they were taken -- a laptop and a
    # free-tier host differ by an order of magnitude, so the artifact carries both itself.
    result = {**benchmark(args.url, args.requests), "url": args.url, "measured_at": measured_at}
    write_json_atomic(result, config.results_path)

    print(
        f"[benchmark_serving] p50={result['p50_s']:.2f}s p99={result['p99_s']:.2f}s "
        f"throughput={result['throughput_rps']:.2f} req/s"
    )
    print(f"[benchmark_serving] wrote {config.results_path}")


if __name__ == "__main__":
    main()
