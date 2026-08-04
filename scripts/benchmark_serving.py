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
import urllib.request
from dataclasses import dataclass
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


def send_one_request(url: str, timeout_s: float = 60.0) -> float:
    """POST one review request; return its wall-clock latency in seconds."""
    payload = f'{{"code": {_BENCHMARK_CODE!r}}}'.replace("'", '"').encode()
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/review",
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        response.read()
    return time.perf_counter() - start


def benchmark(url: str, n_requests: int) -> dict[str, float]:
    if n_requests < 1:
        raise ValueError(f"n_requests must be positive, got {n_requests}")

    latencies = [send_one_request(url) for _ in range(n_requests)]
    total_time = sum(latencies)

    return {
        "n_requests": n_requests,
        "p50_s": statistics.median(latencies),
        "p99_s": statistics.quantiles(latencies, n=100)[98] if n_requests >= 2 else latencies[0],
        "throughput_rps": n_requests / total_time if total_time > 0 else 0.0,
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
    result = benchmark(args.url, args.requests)
    write_json_atomic(result, config.results_path)

    print(
        f"[benchmark_serving] p50={result['p50_s']:.2f}s p99={result['p99_s']:.2f}s "
        f"throughput={result['throughput_rps']:.2f} req/s"
    )
    print(f"[benchmark_serving] wrote {config.results_path}")


if __name__ == "__main__":
    main()
