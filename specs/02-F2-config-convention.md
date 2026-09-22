# F2 — Smoke/full config convention

| | |
|---|---|
| **State** | done |
| **Depends on** | F1 |
| **Requirements** | FR-3, FR-4, NFR-1, NFR-4, NFR-12 |
| **W&B run** | — |

## Problem

Two rules the whole project depends on are only enforceable if configs are a first-class concept. The first: **nothing heavy runs in a session** — which requires every entrypoint to have a CPU profile that finishes in seconds, so there is always a legitimate thing to run instead of the full job. The second: **no magic numbers** — a hyperparameter buried in source is a hyperparameter nobody can reproduce, sweep, or defend in an interview.

A shared loader also makes missing keys loud. Silent Python-side defaults are how a run ends up using `lr=3e-4` when the config said nothing at all.

## Scope

A shared config loader used by all three packages, plus a `configs/smoke.yaml` and `configs/full.yaml` in each. Frozen dataclasses per package (`GPTConfig`, `FinetuneConfig`, `ServeConfig`) with the fields fixed in `planning/03-system-design.md`. Every entrypoint takes `--config`.

## Acceptance criteria

1. `eval/config.py::load_config(path, cls)` reads a YAML file and returns a **frozen** dataclass instance of `cls`.
2. A YAML file missing any field of the target dataclass raises `KeyError` naming the missing field — **no Python-side defaults** for hyperparameters.
3. Each package defines its frozen config dataclass with exactly the fields in `planning/03-system-design.md` §1.1/§1.2/§1.3.
4. `GPTConfig` validation rejects a config where `d_model % n_heads != 0`.
5. Each package has `configs/smoke.yaml` and `configs/full.yaml`; every entrypoint accepts `--config <path>`.
6. Every smoke entrypoint completes on a CPU-only machine in **under 120 seconds** (NFR-1).
7. Every config includes an explicit `seed` (NFR-4).

## Out of scope

- What the entrypoints actually *do* — the ch1 training loop is **A3**, the fine-tune is **C4**, the server is **O0**. F2 delivers config-loading entrypoint stubs only.
- Hydra-style composition or CLI overrides. A plain YAML file per profile is enough; add composition only if a spec needs it.

## Clarifications

- **Q:** FR-4 says the loader is shared, but `planning/02-architecture.md` gives each package its own `config.py`. Which? → **A:** Both. The generic loader lives in `eval/config.py` (shared); each package's `config.py` defines its own frozen dataclass and calls it. *(2026-07-13)*

## Technical plan

**Files** — `eval/config.py`; `ch{1,2,3}_*/src/*/config.py`; `ch{1,2,3}_*/configs/{smoke,full}.yaml`.

**Approach** — `load_config(path: Path, cls: type[T]) -> T` reads YAML, checks that the key set covers every `dataclasses.fields(cls)` name, raises `KeyError` on any gap, and constructs `cls(**data)`. Per-package invariants (`d_model % n_heads == 0`) are enforced in the dataclass's `__post_init__`.

**Config** — smoke values from `planning/02-architecture.md` §2.1–2.3: ch1 smoke is `vocab=256, d_model=64, n_heads=2, n_layers=2, seq_len=64, batch=4, max_steps=50, seed=42`.

## Tasks

- [x] **T1** — `eval/config.py` shared loader + tests · verify: `uv run pytest -q eval/tests`
- [x] **T2** — three package config dataclasses + smoke/full YAMLs · verify: `uv run pytest -q`
- [x] **T3** — `--config` entrypoint stubs per package · verify: each smoke command exits 0
