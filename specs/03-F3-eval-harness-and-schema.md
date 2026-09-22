# F3 — Shared eval harness + schema

| | |
|---|---|
| **State** | done |
| **Depends on** | F1 |
| **Requirements** | FR-5, FR-6, NFR-6, NFR-7, NFR-20 |
| **W&B run** | — |

## Problem

The project's headline claim is a **comparison** — fine-tuned versus base versus frontier API. A comparison is only worth reading if all three systems were scored by the same code, on the same data, against the same contract. If each chapter scores itself, the numbers are three unrelated anecdotes.

This has to exist *before* Chapter 2 begins, for a reason that is easy to get backwards: the scoring rules must be fixed while nobody knows which system they will favor. A harness written after the results are in is a harness that was tuned, however unconsciously, to produce them.

The `eval/holdout/` directory is reserved here and left **empty** — the set itself is curated in C2. What F3 establishes is that it is sacred ground.

## Scope

`eval/schema.json` (the canonical JSON contract), `eval/harness.py` (scores any model's outputs against a holdout set), `eval/metrics.py` (the two metric functions), and the reserved `eval/holdout/` and `eval/results/` directories.

## Acceptance criteria

1. `eval/schema.json` is a Draft-07 JSON Schema requiring exactly `severity`, `category`, `line`, `issue`, `suggested_fix`, with `additionalProperties: false`, matching `planning/02-architecture.md` §2.4 verbatim.
2. `severity` accepts only `critical`/`major`/`minor`/`info`; `line` is an integer ≥ 1; the three string fields have the specified length bounds.
3. A conforming output passes validation; an output that omits any required field, adds an extra field, or uses a disallowed `severity` **fails** — one test per failure mode.
4. `eval/harness.py::score_outputs(outputs, holdout_path, schema_path) -> EvalResult` returns `schema_validity_rate`, `bug_catch_rate`, `n_samples`, `n_valid`, `n_caught`, and `per_sample`.
5. A bug counts as caught only if the output is schema-valid **and** `abs(output.line - ground_truth.line) <= 2`.
6. On a fixture with a known mix of valid/invalid and caught/missed outputs, `score_outputs` returns exactly the expected rates.
7. `eval/holdout/` exists, is empty of data at project start, and its README states that the leakage guard protects it.

## Out of scope

- Curating the holdout set itself → **C2**.
- Producing the comparison table and plots → **C5** (via the `eval-table` skill).
- The Pydantic mirrors of this schema in ch2/ch3 → **C1** and **O2**. `eval/schema.json` stays the single source of truth (NFR-20); those are runtime enforcement of it.

## Clarifications

- **Q:** Why is `bug_catch_rate` a line-proximity match rather than semantic comparison of the `issue` text? → **A:** Line ±2 is objective and cheap. Semantic matching would need a judge model, which adds cost, a dependency, and a new source of bias to the headline metric. *(2026-07-13)*
- **Q:** Should `n_caught` be counted among invalid outputs? → **A:** No. `n_caught <= n_valid` — an output that does not parse has not caught anything. *(2026-07-13)*

## Technical plan

**Files** — `eval/schema.json`, `eval/harness.py`, `eval/metrics.py`, `eval/holdout/README.md`, `eval/tests/`.

**Approach** — `jsonschema.Draft7Validator` against the loaded schema; `metrics.py` holds the two pure functions (`compute_schema_validity`, `compute_bug_catch_rate`) that `score_outputs` calls. `EvalResult` is a dataclass; `per_sample` carries `{id, valid, caught, raw_output}` so any published number can be traced to the record that produced it.

## Tasks

- [x] **T1** — `eval/schema.json` + validation tests for each failure mode · verify: `uv run pytest -q eval/tests/test_schema.py`
- [x] **T2** — `eval/metrics.py` pure metric functions + tests · verify: `uv run pytest -q eval/tests/test_metrics.py`
- [x] **T3** — `eval/harness.py::score_outputs` + fixture test · verify: `uv run pytest -q eval/tests/test_harness.py`
- [x] **T4** — reserve `eval/holdout/` and `eval/results/` with READMEs · verify: directories exist, no data
