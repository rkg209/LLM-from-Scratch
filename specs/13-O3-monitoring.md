# O3 — Monitoring

| | |
|---|---|
| **State** | draft |
| **Depends on** | O2 |
| **Requirements** | FR-23, NFR-3, NFR-17 |
| **W&B run** | — |

## Problem

"I trained a model" is table stakes. "I operated one" is the claim this chapter makes, and the thing that distinguishes them is knowing — from the outside, without attaching a debugger — whether the deployed model is still doing its job.

Two questions a real operator must be able to answer at any moment: *how slow is it for the worst-off user* (p99, not the mean — the mean hides the tail, and the tail is what people actually complain about), and *is the model still producing valid output, or has something drifted*. A rolling schema-validity rate is the cheapest possible quality signal and it needs no labels: the model tells you it is failing by failing to produce parseable JSON.

## Scope

Prometheus-compatible metrics: a latency histogram, a request counter by status, and a rolling schema-validity gauge, exposed at `GET /metrics`.

## Acceptance criteria

1. `GET /metrics` returns Prometheus text format (`text/plain; version=0.0.4`) containing all three metric families (FR-23a).
2. `request_latency_seconds` is a Histogram with buckets `[0.1, 0.5, 1, 2, 5, 10, 30]` — bucketed for CPU inference latency, where a 30-second request is realistic and a 100 ms one is not.
3. `requests_total` is a Counter labelled by `status` (`success` / `error`).
4. `schema_validity_rate` is a Gauge over a **rolling window** of the last `metrics_window` (default 100) requests, updated on **every** request (FR-23c).
5. Latency is measured in middleware but recorded in the route handler, so it is associated with the request's valid/error outcome rather than counted blind.
6. **p50 and p99 latency, and throughput (req/s), are measured on the deployed HF Spaces CPU instance and recorded in the README's metrics scorecard** (FR-23b, NFR-3) — real numbers from the real deployment, not from a laptop.
7. Throughput of **at least 1 req/s** under single-user load is confirmed on HF Spaces CPU, or the actual number is reported if it falls short (NFR-3).
8. The drift monitor is thread-safe (`collections.deque` with `maxlen`).

## Out of scope

- Alerting, a Prometheus server, or Grafana dashboards. The endpoint exposes the metrics; standing up a scrape stack is beyond a free-tier portfolio deployment.
- Statistical drift detection on the input distribution (KS tests, embedding drift). The rolling schema-validity rate is the quality signal; the write-up names the limitation.
- Persisting request logs. The service holds no request history by design (`planning/04-database-design.md` §3.10).

## Clarifications

*(filled by `/clarify`)*

## Technical plan

*(filled by `/plan`)*

## Tasks

*(filled by `/tasks`)*
