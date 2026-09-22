# Plan — Chapter 3 (Operation): O1 → O6

> Covers `specs/11-O1` … `specs/16-O6`. Written against the repo as of `32a0a28` (O0 done).
> Per-spec plans are still expected to be pasted into each spec's **Technical plan** section
> before `/tasks <ID>`; this file is the umbrella design so the six specs are built as one
> coherent pipeline instead of six local decisions.

---

## Context

Chapter 3 is the chapter that turns a checkpoint into a thing a stranger can use. O0 already
stood up the whole serving path against a **base** GGUF — `create_app()`, the `Backend`
protocol, `LlamaCppBackend`, `ServeConfig`, `tiny.gguf` in CI. What remains is everything that
makes it trustworthy rather than merely alive:

- **O1** — get the *fine-tuned* model to the edge (merge → f16 GGUF → Q4_K_M) and prove the
  quantized model is not worse than the model the eval table brags about.
- **O2** — stop returning raw model text. Parse, validate, one repair, honest 422.
- **O3** — make it observable from the outside: latency histogram, request counter, rolling
  schema-validity gauge, and real p50/p99/throughput from the deployed box.
- **O4** — a self-contained CPU image that downloads its weights at start.
- **O5** — two live public URLs in the README, with a worked example.
- **O6** — *(stretch, cut-line)* the same model running client-side in a browser tab.

Today `/review` returns `{"raw": "<whatever the model said>"}` and there is no `/metrics`, no
`validator.py`, no `Dockerfile`, and no `scripts/merge_adapter.py`. The intended outcome of
this plan is: a public URL that answers `POST /v1/review` with schema-valid JSON from a
quantized model whose eval numbers were re-measured *after* quantization.

### Decisions taken up front

| # | Decision | Why |
|---|---|---|
| **D-1** | O2 mounts the reviewer at **`/v1/review`** and keeps **`/review`** as a deprecated alias. `/health` and `/metrics` stay unversioned. | Closes the O0 divergence from `planning/05-openapi.yaml` without breaking O0's tests or any README curl. |
| **D-2** | `request_id` (UUID v4) is on **every** response as the `X-Request-Id` header, and additionally as a body field inside the error envelope. | Satisfies O2 AC-7 *and* the OpenAPI doc, which types the 200 body as a bare `ReviewOutput`. |
| **D-3** | O1's scripts are built and tested **now**, against the tiny smoke models; the real merge → GGUF → re-eval is a manual GPU-box run once C4/C5 land. | The critical path does not stall behind a GPU run. O2–O4 develop against the base GGUF and only swap `MODEL_PATH` afterwards. |
| **D-4** | `scripts/` may import `ch2_adaptation` **and** `ch3_operation`; the packages still never import each other. | The merge step is inherently a chapter seam. Putting it in `scripts/` keeps the "chapters communicate only through files" rule intact where it matters — inside the packages. |
| **D-5** | Ch3 gets its **own** `schema.py` (`ReviewOutput` mirror + import-time drift guard against `eval/schema.json`), copied in shape from `ch2_adaptation/schema.py`. | `planning/03-system-design.md §1.3` prescribes exactly this: separate files, the JSON Schema is the shared contract. Ch2 is a torch-adjacent package; ch3 must stay importable with only the `ch3` extra. |
| **D-6** | Rate limiting (`429 RATE_LIMITED`) is defined in the error-code registry but **not implemented** in v1.0. | It is not in any O2 acceptance criterion, and HF Spaces throttles at the infrastructure level (`planning/05-api-design.md §2`). |
| **D-7** | O6 is planned as a sketch only. | It is a declared cut-line; file-level design before O5 is live is waste. |

### Constraints that shape everything below

- `N_GPU_LAYERS = 0` stays a module constant, never a config key (CON-12).
- Every entrypoint takes `--config`; every number comes from YAML. `eval.config.load_config`
  raises `KeyError` on missing *or unknown* keys — **adding a config key means editing both
  `smoke.yaml` and `full.yaml` plus `test_serve_config.py` in the same commit.**
- The schema is never relaxed to make a model look better (CON-7, NFR-20).
- `eval/holdout/` reads require `EVAL_CONTEXT=1`; writes are denied unconditionally.
- `.gguf` never enters git (the `tests/fixtures/tiny.gguf` exemption is the only one).
- The GPU-budget hook blocks any Bash command containing `configs/full.yaml`, `load_in_4bit`,
  etc. Full-path commands in this plan are **manual, out-of-session** runs (`ALLOW_FULL_RUN=1`).
- One task = one commit; `progress_report.md` gets an appended entry per task; `specs/STATUS.md`
  is updated when a spec's state changes.

---

## O1 — Merge + GGUF quantize

**Goal:** `outputs/model-Q4_K_M.gguf` exists, and its schema-validity rate on the C5 holdout
sample is **≥** the adapter's.

### New files

| Path | Purpose |
|---|---|
| `scripts/merge_adapter.py` | `--config` CLI: load base fp32/CPU → `PeftModel.from_pretrained` → `merge_and_unload()` → `save_pretrained(merged_dir)` + tokenizer. Sanity-generates on ~10 **training/staging** examples (never holdout) and prints the schema-valid fraction; non-zero exit if it collapses. |
| `scripts/export_gguf.py` | Wraps `llama.cpp/convert_hf_to_gguf.py --outtype f16`. Locates llama.cpp via config, verifies the output file exists and is non-trivial in size. |
| `scripts/quantize_gguf.py` | Wraps `llama-quantize <f16> <out> <quant_type>`; `quant_type` comes from config (`Q4_K_M` default, `Q5_K_M`/`Q8_0` are the fallback ladder in AC-4). |
| `ch3_operation/src/ch3_operation/evaluate.py` | `python -m ch3_operation.evaluate --config …` — builds a `LlamaCppBackend` from the config, renders prompts with `prompts.format_zero_shot`, scores with `eval.harness.score_outputs`, writes `eval/results/gguf.json`, and **fails loudly** if validity < the adapter's rate in `eval/results/finetuned.json`. |
| `ch3_operation/configs/export_{smoke,full}.yaml` | `base_model_tag`, `adapter_path`, `merged_dir`, `gguf_f16_path`, `gguf_quant_path`, `quant_type`, `llama_cpp_dir`, `sanity_sample_path`, `sanity_n`. |
| `ch3_operation/configs/eval_{smoke,full}.yaml` | `model_path`, `n_ctx`, `n_threads`, `max_tokens`, `temperature`, `holdout_path`, `results_path`, `compare_to`, `n_samples`, `seed`. |
| `ch3_operation/tests/test_evaluate.py`, `tests/test_export_scripts.py` | Fake `Backend`; subprocess calls monkeypatched so no llama.cpp binary is needed in CI. |

### Reuse (do not re-write)

- `ch2_adaptation.model.load_base_model(model_tag, use_4bit=False)` — the fp32/CPU branch is
  exactly what a merge needs; merging onto a 4-bit base is not supported by that path.
- `ch2_adaptation.config.LOCKED_MODEL_TAG` / `SMOKE_MODEL_TAG` — the merge must use the tag the
  adapter was trained on (AC-1). Assert the adapter's `adapter_config.json.base_model_name_or_path`
  matches, and refuse otherwise.
- `eval.harness.score_outputs` / `EvalResult.to_json` / `write_result` (atomic write).
- `eval.config.load_config` + the frozen-dataclass + `__post_init__` + `is_smoke` +
  "smoke may never overwrite a published path" pattern from `ch2_adaptation/config.py`.
- `ch3_operation.prompts.format_zero_shot` — the served model must be evaluated with the
  **serving** prompt, not ch2's.

### Approach notes

- Two-step conversion is mandatory (AC-2): f16 first, then quantize, so a failure localizes.
- The three scripts are independently runnable and each is idempotent (skip if the output
  exists and `--force` was not passed).
- `evaluate.py` needs `EVAL_CONTEXT=1` to read the holdout — document that in the module
  docstring and in `scripts/README.md`.
- If the ladder (`Q4_K_M` → `Q5_K_M` → `Q8_0`) never clears the bar, that is **published in the
  README as a finding** (AC-4) — a new `progress_report.md` entry and a README paragraph, not a
  quiet re-run.
- Upload the winning `.gguf` to HF Hub; record repo id + filename + sha256 in
  `outputs/gguf_provenance.json` (gitignored) and in the README serving section (AC-6).

### Tasks

`T1` export configs + config dataclasses + tests · `T2` `merge_adapter.py` (+ base-tag guard,
sanity gen) · `T3` `export_gguf.py` · `T4` `quantize_gguf.py` · `T5` `ch3_operation/evaluate.py`
+ eval configs + tests · `T6` `scripts/README.md` + runbook for the manual GPU-box sequence ·
`T7` *(manual, post-C5)* run the real pipeline, record numbers, update README + STATUS.

---

## O2 — FastAPI serving + guardrails

**Goal:** the service never hands a malformed body to a caller and calls it a success.

### New files

| Path | Purpose |
|---|---|
| `api/schema.py` | `ReviewOutput` Pydantic mirror (`extra="forbid"`, the five fields with ch2's exact bounds) + `_assert_matches_eval_schema()` at import, raising `SchemaDriftError`. Copy the shape from `ch2_adaptation/schema.py`. |
| `api/validator.py` | `validate_output(raw) -> ReviewOutput`; `_extract_from_fence(raw) -> dict`. **Exactly two attempts**: `json.loads(raw)`, then the ```` ```json ```` fence extraction. Raises `OutputParseError` (both attempts failed to yield JSON) or `OutputSchemaError(detail=<pydantic message>)`. Never supplies a default, coerces a type, or fills a blank. |
| `api/errors.py` | `ErrorCode` enum (the §3.2 registry), `ErrorResponse`/`ErrorBody` Pydantic models, `error_response(code, message, status, request_id, detail=None)`, and the exception handlers registered in `create_app`. |
| `api/middleware.py` | `RequestIdMiddleware` (mint UUID v4 → `request.state.request_id` + `X-Request-Id` header on every response) and `ContentSizeLimitMiddleware` (413 `PAYLOAD_TOO_LARGE`; checks `Content-Length` and, when absent, counts streamed bytes). O3 adds `RequestTimingMiddleware` alongside these. |
| `tests/test_validator.py`, `tests/test_errors.py` | Table-driven raw-output cases: clean JSON, fenced JSON, prose-then-JSON, truncated brace, extra sixth field, wrong `severity` literal. |

### Changed files

- `api/main.py` — router mounted at `/v1` with an unprefixed alias (D-1). `/review` now:
  backend → `validate_output` → 200 `ReviewOutput`. Backend exception → **500 `INFERENCE_ERROR`**
  (O0 currently returns 503 for this; 503 is reserved for `MODEL_NOT_READY`). Backend absent /
  `model_loaded=False` → 503 `MODEL_NOT_READY`. Register the handlers from `errors.py`, including
  a `RequestValidationError` handler that returns **400** (not FastAPI's default 422) with
  `INVALID_REQUEST`, or `CODE_FIELD_EMPTY` when the marker below fires.
- `api/schemas.py` — `ReviewRequest.code` gets a `field_validator` that rejects whitespace-only
  with a sentinel message the handler maps to `CODE_FIELD_EMPTY`. `ReviewResponse` (the `{"raw":…}`
  wrapper) is deleted — the 200 body is now `ReviewOutput` itself.
- `config.py` + both YAMLs + `test_serve_config.py` — add `max_request_bytes: 32768` (AC-8; no
  magic numbers). Also add `api_version: "v1"` rather than hard-coding the prefix.
- `tests/test_api.py` — update for the new 200 shape and the new status codes; keep the
  "backend constructed exactly once" test.

### Approach notes

- The 422 split is load-bearing: `VALIDATION_FAILED` = neither attempt produced JSON;
  `SCHEMA_VIOLATION` = JSON parsed but Pydantic rejected it, and `error.detail` carries the
  Pydantic message verbatim (AC-9).
- No retry of the model. One inference, two parse attempts, then an honest failure — retrying
  would inflate the schema-validity rate the project publishes.
- Because ch3's `schema.py` guards itself against `eval/schema.json` at import, a schema change
  breaks the app at startup rather than in production (NFR-20).

### Tasks

`T1` `schema.py` + drift guard + tests · `T2` `validator.py` + tests · `T3` `errors.py`
envelope, registry, handlers · `T4` `middleware.py` (request id + size limit) + config keys ·
`T5` rewire `/v1/review` + `/review` alias, status-code corrections, `test_api.py` update ·
`T6` refresh `planning/05-openapi.yaml` drift notes + README API section.

---

## O3 — Monitoring

**Goal:** answer "how slow for the worst-off user" and "is it still producing valid output"
from outside the process.

### New files

| Path | Purpose |
|---|---|
| `api/metrics.py` | `PrometheusMetrics` owning its **own** `CollectorRegistry` (never the global default — tests would leak state between cases). `request_latency_seconds` Histogram, buckets `[0.1, 0.5, 1, 2, 5, 10, 30]`; `requests_total` Counter labelled `status` ∈ {`success`,`error`}; `schema_validity_rate` Gauge. `record_request(latency_s, valid)`. |
| `monitoring/drift.py` | `RollingQualityMonitor(window, metrics)` over `collections.deque(maxlen=window)`; `record(valid)` recomputes `sum/len` and sets the gauge; `current_rate` property. Thread-safe by construction (AC-8). |
| `scripts/benchmark_serving.py` | `--url --requests --config`: fires a fixed prompt set at a **deployed** URL, reports p50/p99/throughput, writes `eval/results/serving_metrics.json`. This is what produces the AC-6/AC-7 numbers. |
| `eval/report.py` | Shared `update_readme_section(readme_path, marker, markdown)` helper, generalized from `ch2_adaptation.evaluate.update_readme_results_section`. Used by ch3 for `<!-- SERVING_METRICS_START/END -->`. (Ch2 can adopt it later; not in this scope.) |
| `tests/test_metrics.py`, `tests/test_drift.py` | `/metrics` content-type + all three families present; gauge tracks a known valid/invalid sequence; window truly rolls at `maxlen`. |

### Changed files

- `api/middleware.py` — add `RequestTimingMiddleware`: `t0 = perf_counter()` before
  `call_next`, stash the elapsed time on `request.state.latency_s`. It **does not record**;
  the route handler does, so latency is attributed to the valid/error outcome (AC-5).
- `api/main.py` — construct `PrometheusMetrics` + `RollingQualityMonitor(config.metrics_window, …)`
  into `app.state`; call `metrics.record_request(...)` and `monitor.record(valid)` on **every**
  `/review` outcome including the 422 and 500 paths (AC-4); add
  `GET /metrics` → `generate_latest(registry)` with `CONTENT_TYPE_LATEST`.
- `README.md` — a metrics scorecard section between the markers, filled from the **deployed**
  run in O5, not from a laptop (AC-6). If throughput lands under 1 req/s, the real number is
  reported (AC-7) — that is a finding, not a failure to hide.

`metrics_window` already exists in `ServeConfig` and both YAMLs; O3 is the first consumer.

### Tasks

`T1` `metrics.py` + tests · `T2` `drift.py` + tests · `T3` `RequestTimingMiddleware` + wire
recording into every `/review` outcome · `T4` `GET /metrics` route + content-type test ·
`T5` `eval/report.py` + `benchmark_serving.py` + README scorecard skeleton.

---

## O4 — Containerize

**Goal:** `docker build` from a clean checkout, `docker run -p 8000:8000` serves — nothing on
the host but Docker.

### New files

| Path | Purpose |
|---|---|
| `docker/Dockerfile` | Multi-stage. **Builder**: `python:3.12-slim` + `build-essential`/`cmake` to compile the `llama-cpp-python` CPU wheel into a venv. **Final**: `python:3.12-slim`, copy the venv only — no compiler survives (AC-7). Non-root user, `EXPOSE 8000`. |
| `docker/entrypoint.sh` | Downloads the `.gguf` from HF Hub into `/models` at **container start** (AC-6, never baked into a layer), verifies sha256, then `exec uvicorn ch3_operation.api.main:app --host 0.0.0.0 --port ${PORT:-8000}`. Env: `MODEL_REPO`, `MODEL_FILE`, `MODEL_PATH`, `CONFIG_PATH`, `PORT`. |
| `docker/.dockerignore` | Exclude `outputs/`, `eval/holdout/`, `.git`, `*.gguf`, caches, `wandb/`. |
| `ch3_operation/tests/test_docker.py` | Skipped unless `DOCKER=1`: build, run, hit `/health`, `/metrics`, and `POST /v1/review`, assert schema-valid (AC-2, AC-3). |
| `.github/workflows/docker.yml` | `docker buildx build --platform linux/amd64`, no push. Triggered on changes under `docker/` and on `workflow_dispatch` — kept out of the main `check` job so every commit does not pay for a multi-minute build. |

### Approach notes

- `--platform linux/amd64` is not optional (AC-8): an arm64-only image built on this Mac will
  not start on HF Spaces. Build with buildx locally too.
- CPU-only base image, no CUDA anywhere (AC-5, CON-12).
- Because the model is fetched at start, first boot is slow — that cold-start behaviour is what
  O5 AC-5 requires documenting.

### Tasks

`T1` Dockerfile + `.dockerignore` · `T2` `entrypoint.sh` (download + sha verify + exec) ·
`T3` `test_docker.py` behind `DOCKER=1` · `T4` `docker.yml` amd64 build job · `T5` README
"Run it locally with Docker" section.

---

## O5 — Deploy

**Goal:** two live public URLs, real numbers, and a worked example in the README.

### New files

| Path | Purpose |
|---|---|
| `space/README.md` | HF Spaces front-matter (`sdk: docker`, `app_port: 8000`) + the Space landing copy. |
| `space/Dockerfile` | Thin `FROM`/copy of `docker/Dockerfile`, or a symlinked build context — one Dockerfile of record, not two that drift. |
| `docs/DEPLOY.md` | The runbook: build → push → set `MODEL_REPO`/`MODEL_FILE` Space variables → verify → benchmark. Includes the second host's steps. |
| `scripts/smoke_deployed.py` | Post-deploy check against a URL: `/health`, `/metrics`, one real `POST /v1/review` asserted schema-valid. Run against both hosts. |

### Approach notes

- **Host 1**: HF Spaces, Docker SDK, CPU basic (AC-1).
- **Host 2**: chosen at deploy time from Render / Fly.io / Railway — the spec deliberately
  leaves this open because free tiers keep moving. Whichever is picked gets its steps in
  `docs/DEPLOY.md` (AC-2). Two hosts because one sleeping free tier kills the portfolio's best
  artifact at the worst moment.
- Model pulled from HF Hub at start; **no weights in git** and **no secret in the Space**
  (AC-3, AC-7) — the model repo is public, so no `HF_TOKEN` is needed at runtime.
- Run `scripts/benchmark_serving.py` **against the live Space** and paste those numbers into the
  README scorecard (AC-4). Laptop numbers do not count.
- README gets: both links, the cold-start warning (AC-5), and a worked example — a real Java
  snippet in, the actual returned JSON out, copied from a real response (AC-6).

### Tasks

`T1` Space scaffold (`space/`) + build context wiring · `T2` deploy to HF Spaces, verify with
`smoke_deployed.py` · `T3` deploy to the second free tier, verify · `T4` benchmark the live
Space, fill the README scorecard · `T5` README demo section: links, cold start, worked example ·
`T6` STATUS.md → O5 done.

---

## O6 — In-browser WebLLM *(stretch — sketch only)*

**Do not start before O5 is live.** This is the first thing to drop if the schedule slips.

Approach: convert the merged fp16 model with MLC-LLM (`mlc_llm convert_weight` + `gen_config`,
q4f16_1), publish the artifacts to HF Hub, and serve a single static page (GitHub Pages or an
HF static Space) that loads it through `@mlc-ai/web-llm`. The page reuses `eval/schema.json` —
fetched and validated client-side with a small JSON-Schema check — so the browser demo enforces
the *same* contract (AC-3). Feature-detect `navigator.gpu`; when WebGPU is missing, show a link
to the hosted Space instead of hanging (AC-2, and the out-of-scope note). State the model
download size on the page before download starts. Link it in the README **alongside** the server
demo, and if the browser model scores worse, say so (AC-4 and out-of-scope note).

Likely tasks: `T1` MLC conversion + Hub upload · `T2` static page + WebGPU detect/degrade ·
`T3` client-side schema validation · `T4` README link + honest quality note.

---

## Verification

Per-task, before every commit:

```bash
uv run ruff check . && uv run black --check . && uv run pytest -q
```

Per spec:

```bash
# O1 — smoke path only in-session (the full path is a manual GPU-box run)
uv run python -m ch3_operation.evaluate --config ch3_operation/configs/eval_smoke.yaml

# O2/O3 — the app itself
uv run python -m ch3_operation.serve --config ch3_operation/configs/smoke.yaml --check
uv run python -m ch3_operation.serve --config ch3_operation/configs/smoke.yaml   # then:
curl -s localhost:8000/health
curl -s localhost:8000/metrics | grep -E 'request_latency_seconds|requests_total|schema_validity_rate'
curl -s -X POST localhost:8000/v1/review -H 'content-type: application/json' \
     -d '{"code":"public String f(User u){return u.getProfile().getName();}"}'
# tiny.gguf will NOT produce valid JSON — the correct result here is a 422 with the
# VALIDATION_FAILED envelope and an X-Request-Id header. That is the guardrail working.
head -c 40000 /dev/zero | tr '\0' 'a' > /tmp/big && \
  curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8000/v1/review \
    -H 'content-type: application/json' --data-binary @/tmp/big     # expect 413

# O4
docker buildx build --platform linux/amd64 -f docker/Dockerfile -t reviewer:local .
DOCKER=1 uv run pytest ch3_operation/tests/test_docker.py -q

# O5
uv run python scripts/smoke_deployed.py --url https://<space>.hf.space
uv run python scripts/benchmark_serving.py --url https://<space>.hf.space --requests 30
```

**Manual, out-of-session (GPU box, `ALLOW_FULL_RUN=1`)** — the O1 full path:

```
merge_adapter.py → export_gguf.py → quantize_gguf.py → EVAL_CONTEXT=1 ch3_operation.evaluate
```

Acceptance for that run is one number: GGUF schema-validity ≥ adapter schema-validity on the
same holdout sample. If it fails, walk the ladder (Q5_K_M, Q8_0); if it still fails, it goes in
the README as a finding.

---

## Risks / open items

1. **O1 is blocked on C4/C5.** Everything except the final numbers can be built and tested now
   (D-3), but O1 cannot be marked `done` until a real adapter exists.
2. **The GPU-budget hook will block the full-path commands** from inside a session (they match
   `configs/full.yaml` / `load_in_4bit`). That is intended — the runbook in `scripts/README.md`
   must say so explicitly rather than tempting a future session into `ALLOW_FULL_RUN=1`.
3. **Adding config keys is a three-file edit** (`config.py`, both YAMLs) plus tests, because
   `load_config` rejects unknown *and* missing keys.
4. **The second free tier is a moving target** — pick at deploy time, per the spec's own
   clarification note.
5. **`planning/02-architecture.md` says `python:3.11-slim`** while O4 AC-7 and `pyproject`
   (`py312`) say 3.12. Following the spec (3.12); note the doc divergence in `progress_report.md`.
6. **Cold start + free-tier sleep** may make the measured p99 look terrible if the benchmark
   catches a wake-up. Benchmark after a warm-up request and report both numbers honestly.
