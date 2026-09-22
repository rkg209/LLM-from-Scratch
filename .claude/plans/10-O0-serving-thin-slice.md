# Plan — O0: Serving thin slice (base model)

> **First step after approval:** copy this file to `.claude/plans/10-O0-serving-thin-slice.md`
> (where `05-C1` … `09-C5` plans live), then run `/tasks O0` against it.

## Context

Chapter 3 is where portfolio projects die: the model trains, the notebook has a nice number,
and "deployment" turns out to be a hundred small integration problems discovered in the last
week. `specs/10-O0-serving-thin-slice.md` sits deliberately out of dependency order to defuse
that — stand the *whole* serving path up now, against a **base** model, before Chapter 2
finishes. When the fine-tuned GGUF lands at O1 it should be a config change to a pipeline that
already works.

Today `ch3_operation/` is a config-only scaffold:

- `ch3_operation/src/ch3_operation/config.py` — `ServeConfig` (frozen dataclass), the
  `N_GPU_LAYERS = 0` constant, and `resolved_model_path()` which expands `${MODEL_PATH}`.
  **Already complete and correct for O0. Do not change it.**
- `ch3_operation/src/ch3_operation/serve.py` — prints what it *would* serve and says
  "SCAFFOLD: no FastAPI app yet — that is spec O0".
- `ch3_operation/configs/{smoke,full}.yaml` — smoke already points at
  `ch3_operation/tests/fixtures/tiny.gguf`; full already uses `${MODEL_PATH}`.
- `ch3_operation/tests/test_serve_config.py` — 6 passing config tests.
- `.gitignore` already whitelists `!ch3_operation/tests/fixtures/tiny.gguf`, and
  `.claude/hooks/commit_hygiene.py` already exempts `tests/fixtures/*.gguf` (5 MB cap).

So O0 is: the backend, the app, the fixture, and the wiring — nothing else.

### Decisions taken (resolves the spec's open Clarification)

1. **`tiny.gguf` = `stories260K` from `ggml-org/models`** (Karpathy llama2.c, MIT, ~1 MB) —
   the canonical llama.cpp CI fixture. Downloaded once by a helper script, committed, with
   provenance + sha256 recorded next to it. This is the one sanctioned exception to
   non-negotiable #6, and it is already pre-authorised in `.gitignore` and the hook.
2. **CI syncs `--extra ch3`.** Today CI runs `uv sync --extra dev`, so `fastapi` and
   `llama_cpp` are absent and every app test would silently skip — AC-3 would be a formality.
   Both are CPU wheels and install quickly. `pytest.importorskip` stays as a *local*-env guard.
3. **Route shape follows the spec, not `planning/05-openapi.yaml`.** O0 ships bare
   `POST /review` and `GET /health` → `{"status", "model_loaded"}`. The `/v1` prefix, bearer
   auth, and the extended `HealthResponse` (uptime, validity rate, version) arrive with the
   guardrails in O2/O3. Note this divergence in `progress_report.md` so O2 knows to close it.

### Constraints that shape the design

- `n_gpu_layers=0` hardcoded, never a config key (CON-12) — the constant already exists.
- `llama-cpp-python` is not thread-safe: one `threading.Lock` around every `generate()` call.
- Model loads **once at startup**, not per request (FR-20).
- Chapters stay independent: `ch3_operation` must **not** import from `ch2_adaptation`. The
  prompt is duplicated, exactly as `planning/03-system-design.md §1.3` prescribes for
  `schema.py` ("separate files, not a shared package; the JSON Schema file is the shared
  contract").
- No schema validation, no repair, no metrics in this slice — `/review` returns raw text.
- Every serving parameter comes from `ServeConfig`; no magic numbers.

## Files

### New

| Path | Purpose |
|---|---|
| `ch3_operation/src/ch3_operation/prompts.py` | Ch3's own `format_zero_shot(code, context)`; builds the system message from `eval/harness.load_schema()` — same shared contract as ch2, no ch2 import. |
| `ch3_operation/src/ch3_operation/model_backend.py` | `Backend` Protocol + `LlamaCppBackend` (lazy `llama_cpp` import, `n_gpu_layers=N_GPU_LAYERS`, `threading.Lock`). |
| `ch3_operation/src/ch3_operation/api/__init__.py` | Package marker. |
| `ch3_operation/src/ch3_operation/api/schemas.py` | Pydantic `ReviewRequest` / `HealthResponse` — the wire shape only. |
| `ch3_operation/src/ch3_operation/api/main.py` | `create_app(backend, config)` factory + module-level `app` built from `CONFIG_PATH`. |
| `ch3_operation/tests/fixtures/tiny.gguf` | ~1 MB stories260K GGUF (committed). |
| `ch3_operation/tests/fixtures/README.md` | Provenance: source URL, licence, sha256, why it is exempt from the no-weights rule. |
| `scripts/fetch_tiny_gguf.py` | Idempotent download + sha256 verify; how the fixture is reproduced. |
| `ch3_operation/tests/test_model_backend.py` | Backend contract, lock, CPU-only, real-fixture load. |
| `ch3_operation/tests/test_api.py` | Route behaviour via `TestClient` against a `FakeBackend`. |

### Modified

| Path | Change |
|---|---|
| `ch3_operation/src/ch3_operation/serve.py` | Stop printing "SCAFFOLD"; build the backend and run `uvicorn.run(create_app(...), host=config.host, port=config.port)`. Keep `--config`. Add `--check` (load config + model, report, exit 0) so the CI smoke step stays non-blocking. |
| `.github/workflows/ci.yml` | `uv sync --extra dev --extra ch3`; ch3 smoke step becomes `... --config ch3_operation/configs/smoke.yaml --check`. |
| `specs/10-O0-serving-thin-slice.md` | Fill **Clarifications** (fixture decision) and **Technical plan**; flip State `draft` → `planned`. |
| `specs/STATUS.md` | O0 row → `planned`, then `building` once `/tasks` lands. |
| `progress_report.md` | An appended entry per task (newest at bottom, never rewrite). |

## Approach

### 1. `prompts.py` — the zero-shot prompt, owned by ch3

Mirror `ch2_adaptation/src/ch2_adaptation/prompts.py:83` (`format_zero_shot`) but take
`context` too, since `ReviewRequest` carries it and ch2's training format does not:

```python
def format_zero_shot(code: str, context: str = "") -> str:
    """System message, optional caller context, then the code under review."""
```

The system message is built the same way ch2 builds it — `json.dumps(load_schema(), indent=2)`
from `eval/harness.py` — so both chapters bind to `eval/schema.json`, not to each other.
No markdown-fence instruction drift: copy ch2's wording verbatim.

A test asserts the ch3 system prompt embeds the live `eval/schema.json`, so schema edits can't
silently desync the served prompt.

### 2. `model_backend.py` — the inference seam

```python
class Backend(Protocol):
    def generate(self, prompt: str, max_tokens: int) -> str: ...

class LlamaCppBackend:
    def __init__(self, config: ServeConfig) -> None:
        from llama_cpp import Llama          # lazy: base env has no llama_cpp
        self._config = config
        self._lock = threading.Lock()
        self._llm = Llama(
            model_path=str(config.resolved_model_path()),
            n_ctx=config.n_ctx,
            n_threads=config.n_threads,
            n_gpu_layers=N_GPU_LAYERS,        # 0, hardcoded (CON-12)
            verbose=False,
        )

    def generate(self, prompt: str, max_tokens: int) -> str:
        with self._lock:                      # llama-cpp-python is not thread-safe
            result = self._llm(prompt, max_tokens=max_tokens,
                               temperature=self._config.temperature, echo=False)
        return result["choices"][0]["text"]
```

Why a `Protocol` rather than `LlamaCppBackend` everywhere: it is what lets `create_app` take a
`FakeBackend` in tests without patching `llama_cpp`, and it is the seam O1 swaps the fine-tuned
GGUF through. `echo=False` handles the "strip the prompt echo" invariant from the design doc —
assert it in a test rather than trusting the flag.

Import of `llama_cpp` is **inside `__init__`**, not module-level, so `model_backend.py` imports
in the base env and the Protocol/config tests run without the ch3 extra.

### 3. `api/main.py` — app factory

```python
def create_app(backend: Backend, config: ServeConfig) -> FastAPI:
    app = FastAPI(title="Java Code Reviewer")
    app.state.backend = backend
    app.state.config = config
    ...
    return app
```

- `POST /review` → `format_zero_shot(req.code, req.context)` →
  `backend.generate(prompt, config.max_tokens)` → `{"raw": <text>}`. **No parsing, no
  validation** — that is O2, and the spec says this slice returns whatever the model produced.
  Wrap `generate` in try/except → HTTP 503 (matches the OpenAPI error model for inference
  failure) so one bad request cannot kill the worker.
- `GET /health` → `{"status": "ok" if loaded else "degraded", "model_loaded": bool}`, HTTP 200
  when ok / 503 when degraded, per the OpenAPI health contract.
- Module-level `app = create_app(LlamaCppBackend(cfg), cfg)` where
  `cfg = load_serve_config(os.getenv("CONFIG_PATH", "ch3_operation/configs/full.yaml"))`, so
  AC-1's `uvicorn ch3_operation.api.main:app` works. The factory is what actually loads the
  model, and it runs exactly once at import — no `@app.on_event("startup")` (deprecated in
  current FastAPI) and no per-request load.

The `create_app` split is the whole reason the tests can run with no real model while AC-1's
import path still exercises the real one.

### 4. The fixture

`scripts/fetch_tiny_gguf.py`: download `stories260K.gguf` from `ggml-org/models` via
`urllib.request`, verify a pinned sha256, write to `ch3_operation/tests/fixtures/`, skip if
already present and matching. Run it once, commit the result plus `fixtures/README.md`
(source URL, MIT licence, sha256, size, and a sentence on why non-negotiable #6 is exempted
here). No new dependency — stdlib only.

### 5. Tests

`test_model_backend.py`
- `FakeBackend` satisfies the `Backend` Protocol (`isinstance` with `runtime_checkable`).
- `N_GPU_LAYERS == 0` is passed through to `Llama(...)` — assert on a stubbed `Llama` via
  `monkeypatch.setitem(sys.modules, "llama_cpp", ...)`, so this runs with no wheel installed.
- The lock is held across `generate()`: stub `Llama.__call__` to assert
  `backend._lock.locked()` from inside the call.
- **Real-load test**, `pytest.importorskip("llama_cpp")`: `LlamaCppBackend(smoke_config)`
  loads `tiny.gguf` and `generate("hello", max_tokens=4)` returns a `str` that does not start
  with the prompt. This is the one test that actually retires the integration risk.

`test_api.py` (`pytest.importorskip("fastapi")`)
- `TestClient(create_app(FakeBackend(), smoke_config))`.
- `GET /health` → 200, `{"status": "ok", "model_loaded": True}`; degraded backend → 503.
- `POST /review` with `{"code", "context"}` → 200, body carries the fake's output.
- Empty `code` → 422 (`min_length=1` on the Pydantic field).
- Backend raising → 503, process survives.
- **Load-once**: a counting fake proves two `POST /review` calls construct the backend once.
- The prompt handed to the backend contains the submitted code *and* the context string.

## Task decomposition (for `/tasks`, one commit each)

1. **T1** — `prompts.py` + test. No serving deps; runs anywhere.
2. **T2** — `scripts/fetch_tiny_gguf.py`, run it, commit `tiny.gguf` + `fixtures/README.md`.
3. **T3** — `model_backend.py` + `test_model_backend.py` (needs T2 for the real-load test).
4. **T4** — `api/schemas.py`, `api/main.py`, `test_api.py`.
5. **T5** — `serve.py` rewrite (`uvicorn.run` + `--check`) and the CI workflow `--extra ch3`.
6. **T6** — spec Clarifications/Technical plan filled, `STATUS.md` updated, `progress_report.md`
   entry covering the three decisions above and the OpenAPI divergence.

## Verification

```bash
# 1. Full local check (the project default)
uv sync --extra dev --extra ch3
uv run ruff check . && uv run black --check . && uv run pytest -q

# 2. Ch3 tests specifically — AC-3
uv run pytest -q ch3_operation/tests
#    expect: config tests + backend tests + api tests, zero skips with --extra ch3

# 3. The smoke path — AC-7, and the CI step
uv run python -m ch3_operation.serve --config ch3_operation/configs/smoke.yaml --check
#    expect: loads tiny.gguf, prints n_gpu_layers=0, exits 0

# 4. AC-1/AC-2 end-to-end against the real fixture
CONFIG_PATH=ch3_operation/configs/smoke.yaml \
  uv run uvicorn ch3_operation.api.main:app --port 8000 &
curl -s localhost:8000/health
#    expect: {"status":"ok","model_loaded":true}
curl -s -X POST localhost:8000/review -H 'content-type: application/json' \
  -d '{"code":"public User get(Long id){return repo.findById(id).get();}","context":"Spring Boot"}'
#    expect: 200 with a "raw" string. stories260K writes children's stories, not JSON —
#    that is EXPECTED and correct. O0 proves the path carries bytes end to end; output
#    quality is O1's job and validation is O2's.

# 5. Concurrency — AC-5
#    two parallel curl POSTs must both return 200 and the server must still be alive
#    afterwards (this is the failure the lock exists to prevent).

# 6. Hygiene — the fixture must be committable
git add ch3_operation/tests/fixtures/tiny.gguf && git status --short
#    commit_hygiene.py allows tests/fixtures/*.gguf under 5 MB; confirm the commit is not blocked
```

## Acceptance-criteria trace

| AC | Where it is met | Verified by |
|---|---|---|
| 1 — uvicorn starts, model loads once | `api/main.py` module-level `app` + factory | step 4; counting-fake test |
| 2 — `/review`, `/health` shapes | `api/main.py`, `api/schemas.py` | `test_api.py`; step 4 |
| 3 — CPU CI, no real model | `create_app(FakeBackend, ...)` + `--extra ch3` in CI | step 2 |
| 4 — `n_gpu_layers=0` hardcoded | `N_GPU_LAYERS` in `config.py` (exists) | stubbed-`Llama` test |
| 5 — `threading.Lock` on generate | `LlamaCppBackend.generate` | lock-held test; step 5 |
| 6 — params from `ServeConfig` | backend + `serve.py` read only from config | existing config tests |
| 7 — smoke→fixture, full→`MODEL_PATH` | configs already correct | `test_serve_config.py` (passing) |
