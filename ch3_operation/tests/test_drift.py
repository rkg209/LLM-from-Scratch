"""O3: `RollingQualityMonitor` -- the rolling schema-validity window."""

from __future__ import annotations

import pytest
from ch3_operation.api.metrics import PrometheusMetrics

from monitoring.drift import RollingQualityMonitor


def test_current_rate_is_zero_with_no_history() -> None:
    monitor = RollingQualityMonitor(window=5, metrics=PrometheusMetrics())
    assert monitor.current_rate == 0.0


def test_current_rate_tracks_a_known_valid_invalid_sequence() -> None:
    monitor = RollingQualityMonitor(window=4, metrics=PrometheusMetrics())

    for valid in (True, True, False, True):
        monitor.record(valid)

    assert monitor.current_rate == 0.75


def test_window_rolls_at_maxlen() -> None:
    monitor = RollingQualityMonitor(window=3, metrics=PrometheusMetrics())

    for valid in (True, True, True, False, False):
        monitor.record(valid)

    # only the last 3 records (True, False, False) count
    assert monitor.current_rate == pytest.approx(1 / 3)


def test_record_updates_the_gauge_in_the_shared_metrics() -> None:
    metrics = PrometheusMetrics()
    monitor = RollingQualityMonitor(window=2, metrics=metrics)

    monitor.record(True)
    monitor.record(False)

    assert metrics.schema_validity_rate._value.get() == 0.5


def test_nonpositive_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="window"):
        RollingQualityMonitor(window=0, metrics=PrometheusMetrics())
