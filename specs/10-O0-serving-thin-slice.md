# O0 — Serving thin slice (base model)

| | |
|---|---|
| **State** | done |
| **Depends on** | F3 |
| **Requirements** | FR-20, CON-12, NFR-1 |
| **W&B run** | — |

## Problem

Chapter 3 is where portfolio projects die. The model gets trained, the notebook has a nice number in it, and the deployment is "future work" — because deployment turns out to be a hundred small integration problems that nobody budgeted for, all discovered in the last week.

The fix is structural and it is why this spec sits early, out of dependency order: **stand up the whole serving path now, against the base model, before Chapter 2 finishes.** Every integration problem — llama.cpp on CPU, GGUF loading, FastAPI startup, the request/response shape, the test fixture — gets found while there is still time. When the fine-tuned model lands at O1, it is a config change to a pipeline that already works.

## Scope

A FastAPI app serving a **base** GGUF model on CPU via llama-cpp-python: `POST /review`, `GET /health`, and the test fixture and smoke path that let it run in CI without a real model.

## Acceptance criteria

1. `uvicorn ch3_operation.api.main:app` starts without error and loads the model **once at startup**, not per request (FR-20).
2. `POST /review` with `{"code": "...", "context": "..."}` returns a response; `GET /health` returns `{"status": "ok", "model_loaded": bool}`.
3. `uv run pytest -q ch3_operation/tests` passes **on CPU, in CI, with no real model** — against the committed `tests/fixtures/tiny.gguf` (NFR-1).
4. `n_gpu_layers=0` is **hardcoded, not configurable** (CON-12). CPU-only is the whole point of the "edge" claim.
5. `LlamaCppBackend.generate()` is wrapped in a `threading.Lock` — llama-cpp-python is not thread-safe, and a concurrent request will crash the process without it.
6. All serving parameters (`n_ctx`, `n_threads`, `max_tokens`, `temperature`, `host`, `port`) come from `ServeConfig`.
7. The smoke config points at the test fixture; the full config reads `MODEL_PATH` from the environment.

## Out of scope

- Serving the **fine-tuned** model → **O1**. This slice deliberately serves the base model.
- Schema validation and the repair strategy → **O2**. This slice returns whatever the model produced.
- Metrics → **O3**. Docker → **O4**. Deployment → **O5**.

## Clarifications

Resolved by the accepted plan (`.claude/plans/10-O0-serving-thin-slice.md`):

1. **`tiny.gguf` fixture** = `stories260K` from `ggml-org/models-moved` (the repo `ggml-org/models`
   was renamed/moved to this ID upstream; Karpathy llama2.c, MIT, ~1.1 MB) — the canonical
   llama.cpp CI fixture. Fetched once by `scripts/fetch_tiny_gguf.py` (idempotent,
   sha256-verified), committed with provenance in `ch3_operation/tests/fixtures/README.md`.
   This is the one sanctioned exception to non-negotiable #6, already pre-authorised in
   `.gitignore` and `commit_hygiene.py`.
2. **CI syncs `--extra ch3`.** CI currently runs `uv sync --extra dev` only, so `fastapi` and
   `llama_cpp` are absent and ch3 tests would silently skip via `importorskip`, making AC-3 a
   formality rather than a real check. Both are CPU wheels and install quickly, so CI adds
   `--extra ch3`.
3. **Route shape follows this spec, not `planning/05-openapi.yaml`.** O0 ships bare
   `POST /review` / `GET /health` → `{"status", "model_loaded"}`. The `/v1` prefix, bearer
   auth, and the extended `HealthResponse` (uptime, validity rate, version) arrive with the
   guardrails in O2/O3. This divergence is logged in `progress_report.md` so O2 closes it.

## Technical plan

Full design lives in `.claude/plans/10-O0-serving-thin-slice.md`. Summary:

- `ch3_operation/src/ch3_operation/prompts.py` — `format_zero_shot(code, context)`, ch3's own
  copy of the ch2 zero-shot prompt, built from `eval/harness.load_schema()` — same shared
  contract, no ch2 import (chapters stay independent).
- `ch3_operation/src/ch3_operation/model_backend.py` — `Backend` Protocol +
  `LlamaCppBackend`: lazy `llama_cpp` import inside `__init__` (so the module imports in the
  base env), `n_gpu_layers=N_GPU_LAYERS` (hardcoded 0, CON-12), a `threading.Lock` held across
  every `generate()` call (llama-cpp-python is not thread-safe).
- `ch3_operation/src/ch3_operation/api/{__init__,schemas,main}.py` — Pydantic
  `ReviewRequest`/`HealthResponse` wire shapes, and `create_app(backend, config)` factory:
  `POST /review` → `format_zero_shot` → `backend.generate` → `{"raw": ...}` (503 on backend
  failure); `GET /health` → `{"status", "model_loaded"}` (200/503). Routes read
  `request.app.state.backend`/`.config` rather than closing over fixed values, so the same
  routes serve a synchronously-supplied backend (tests, `serve.py`) or one loaded later.
  **Revised during implementation:** the module-level `app` does not build its real
  `LlamaCppBackend` at import time — that made merely importing the module (from tests, or
  from `serve.py` needing the `create_app` symbol) force a real model load, and crash outright
  if `$MODEL_PATH` was unset. Instead the real backend loads once inside a FastAPI `lifespan`
  callback, run when uvicorn actually starts serving. `serve.py` builds its own
  `LlamaCppBackend` directly from `--config` and never touches this module's lifespan at all.
- `scripts/fetch_tiny_gguf.py` + committed `ch3_operation/tests/fixtures/{tiny.gguf,README.md}`
  — the CI fixture and its provenance.
- `ch3_operation/src/ch3_operation/serve.py` — stop printing "SCAFFOLD"; wire
  `uvicorn.run(create_app(...))`, add `--check` (load config + model, report, exit 0) for a
  non-blocking CI smoke step.
- `.github/workflows/ci.yml` — `uv sync --extra dev --extra ch3`; smoke step becomes
  `... --config ch3_operation/configs/smoke.yaml --check`.

No schema validation, no repair, no metrics in this slice — `/review` returns raw model text.
Every serving parameter comes from `ServeConfig`; no magic numbers.

## Tasks

- [x] **T1** — Ch3's own zero-shot prompt builder · files: `ch3_operation/src/ch3_operation/prompts.py`, `ch3_operation/tests/test_prompts.py` · verify: `uv run pytest -q ch3_operation/tests/test_prompts.py` (asserts the built system message embeds the live `eval/schema.json`, no ch2 import)
- [x] **T2** — CI GGUF fixture, fetched and committed · files: `scripts/fetch_tiny_gguf.py`, `ch3_operation/tests/fixtures/tiny.gguf`, `ch3_operation/tests/fixtures/README.md` · verify: `uv run python scripts/fetch_tiny_gguf.py` reports sha256 match; `git add ch3_operation/tests/fixtures/tiny.gguf && git status --short` shows it staged, not blocked by `commit_hygiene.py`
- [x] **T3** — Inference backend seam · files: `ch3_operation/src/ch3_operation/model_backend.py`, `ch3_operation/tests/test_model_backend.py` · verify: `uv run pytest -q ch3_operation/tests/test_model_backend.py` (Protocol/lock/N_GPU_LAYERS tests always run; real-fixture load test runs when `--extra ch3` is installed, else skips via `importorskip`)
- [x] **T4** — FastAPI app: schemas + factory + routes · files: `ch3_operation/src/ch3_operation/api/__init__.py`, `ch3_operation/src/ch3_operation/api/schemas.py`, `ch3_operation/src/ch3_operation/api/main.py`, `ch3_operation/tests/test_api.py` · verify: `uv run pytest -q ch3_operation/tests/test_api.py` (health ok/degraded, review 200/422/503, load-once via counting fake, prompt carries code+context)
- [x] **T5** — Wire `serve.py` to the real app; CI installs `--extra ch3` · files: `ch3_operation/src/ch3_operation/serve.py`, `.github/workflows/ci.yml` · verify: `uv run python -m ch3_operation.serve --config ch3_operation/configs/smoke.yaml --check` exits 0, prints `n_gpu_layers=0`
- [x] **T6** — Close the loop: `STATUS.md` → `building`→ update to reflect O0 completion, `progress_report.md` entry (fixture choice, CI extra, route-shape divergence for O2) · files: `specs/STATUS.md`, `progress_report.md` · verify: `uv run ruff check . && uv run black --check . && uv run pytest -q` all green
