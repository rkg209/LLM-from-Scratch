"""O3: the three Prometheus metrics behind `/metrics`.

Owns its **own** `CollectorRegistry` rather than the global default -- tests construct a
fresh `PrometheusMetrics` per test case, and registering the same metric name against the
global default registry twice raises. Buckets, label values, and metric shapes are locked in
`planning/05-api-design.md §6` (`review_request_latency_seconds`,
`review_requests_total{status}`, `review_schema_validity_rate`).
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# planning/05-api-design.md's own example histogram -- not tunable per-deploy, since a
# changed bucket set makes p50/p99 across deployments incomparable.
LATENCY_BUCKETS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)


class PrometheusMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()

        self.request_latency_seconds = Histogram(
            "review_request_latency_seconds",
            "End-to-end request latency in seconds",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.requests_total = Counter(
            "review_requests_total",
            "Total requests by outcome",
            labelnames=("status",),
            registry=self.registry,
        )
        self.schema_validity_rate = Gauge(
            "review_schema_validity_rate",
            "Rolling schema validity rate over the configured window",
            registry=self.registry,
        )

    def record_request(self, latency_s: float, valid: bool) -> None:
        """Record one `/v1/review` outcome: its latency, and success/error by `valid`."""
        self.request_latency_seconds.observe(latency_s)
        self.requests_total.labels(status="success" if valid else "error").inc()
