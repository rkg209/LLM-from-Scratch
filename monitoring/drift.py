"""O3: `RollingQualityMonitor` -- "is the model still producing valid output?" from outside
the process.

Lives at the repo root, not inside `ch3_operation`, on the same footing as `eval/`: a small,
dependency-light module a route handler calls into on every request. Backed by a bounded
`collections.deque` so memory never grows unbounded across a long-running server's lifetime,
and thread-safe by construction -- `deque.append`/iteration under the GIL don't need an
explicit lock for this access pattern (single append per call, no partial reads).
"""

from __future__ import annotations

from collections import deque

from ch3_operation.api.metrics import PrometheusMetrics


class RollingQualityMonitor:
    """Tracks the fraction of the last `window` requests that were schema-valid."""

    def __init__(self, window: int, metrics: PrometheusMetrics) -> None:
        if window < 1:
            raise ValueError(f"window must be positive, got {window}")
        self._window: deque[bool] = deque(maxlen=window)
        self._metrics = metrics

    def record(self, valid: bool) -> None:
        self._window.append(valid)
        self._metrics.schema_validity_rate.set(self.current_rate)

    @property
    def current_rate(self) -> float:
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)
