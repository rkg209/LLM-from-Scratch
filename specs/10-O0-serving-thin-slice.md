# O0 — Serving thin slice (base model)

| | |
|---|---|
| **State** | draft |
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

*(filled by `/clarify` — open question: where the `tiny.gguf` CI fixture comes from. It must be a genuinely tiny, permissively-licensed GGUF, committed to git as the one exception to the no-weights rule.)*

## Technical plan

*(filled by `/plan`)*

## Tasks

*(filled by `/tasks`)*
