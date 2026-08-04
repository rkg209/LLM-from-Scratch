# LLM Engineering: From Architecture to Edge

Three connected chapters on one theme: *understanding a model from its internals all the way to a deployed edge service.*

The chapters share a narrative, not a model. Chapter 1 proves understanding on a toy transformer built by hand. Chapters 2 and 3 take a real small open model and carry it to the edge.

| Chapter | What it does | Headline artifact |
|---|---|---|
| **1 — Architecture** (`ch1_architecture/`) | A GPT-style LM written from scratch in pure PyTorch — embeddings, multi-head attention, MLP, norm, training loop — then a KV-cache and quantization added on top. | tokens/sec speedup curve · perplexity-vs-quantization tradeoff |
| **2 — Adaptation** (`ch2_adaptation/`) | QLoRA fine-tune of `Qwen/Qwen2.5-Coder-1.5B-Instruct` into a Java/Spring code reviewer that emits structured JSON. | head-to-head eval table: fine-tuned vs base vs frontier-API few-shot |
| **3 — Operation** (`ch3_operation/`) | The fine-tuned model quantized to GGUF, served behind FastAPI with schema validation and latency/drift monitoring, containerized and deployed. | live CPU demo on HF Spaces |

## Architecture

*(spec X1 — the single diagram covering all three chapters goes here)*

## Results

**Chapter 2 — head-to-head on the frozen holdout set** *(spec C5)*

<!-- EVAL_TABLE_START -->
| System | Schema-validity | Bug-catch | n |
|---|---|---|---|
| Fine-tuned (QLoRA, Qwen2.5-Coder-1.5B) | — | — | — |
| Base model (zero-shot) | — | — | — |
| Frontier API (3-shot) | — | — | — |
<!-- EVAL_TABLE_END -->

**Chapter 1 — speedup and quantization cost** *(spec A5)*

*(speedup_curve.png and perplexity_tradeoff.png go here)*

**Chapter 3 — serving** *(spec O3, numbers filled in by O5 against the live deployment)*

<!-- SERVING_METRICS_START -->
*(p50/p99 latency, throughput, live link — filled by `scripts/benchmark_serving.py` against the deployed Space, not a laptop)*
<!-- SERVING_METRICS_END -->

## Getting started

```bash
uv sync --extra dev                              # build the environment
uv run ruff check . && uv run black --check . && uv run pytest -q
uv run python -m ch1_architecture.train --config ch1_architecture/configs/smoke.yaml
```

Everything has a `smoke` profile that runs on CPU in seconds. Full training runs are launched **manually on a GPU box** (Colab / Kaggle / cluster) and never from a development session — see `CLAUDE.md`.

## API

`ch3_operation` serves the reviewer over HTTP (spec O2). Locally:

```bash
uv run python -m ch3_operation.serve --config ch3_operation/configs/smoke.yaml
curl -s -X POST localhost:8000/v1/review -H 'content-type: application/json' \
  -d '{"code": "public String f(User u){return u.getProfile().getName();}"}'
```

| Method | Path | Notes |
|---|---|---|
| `POST` | `/v1/review` | Returns a validated `ReviewOutput` (`severity`, `category`, `line`, `issue`, `suggested_fix`) on `200`. `POST /review` is a deprecated alias for the same route. |
| `GET` | `/health` | `200` once the model is loaded, `503` while it's still starting. |
| `GET` | `/metrics` | Prometheus text format *(spec O3)*. |

A model failure to produce schema-valid JSON is an honest `422`, never silently reshaped into something that looks like success (one inference, two parse attempts, no retry). Every response — success or error — carries an `X-Request-Id` header; error bodies use one envelope:

```json
{"error": {"code": "VALIDATION_FAILED", "message": "...", "detail": null}, "request_id": "..."}
```

| Status | `error.code` | Trigger |
|---|---|---|
| `400` | `INVALID_REQUEST` / `CODE_FIELD_EMPTY` | Malformed body, or `code` missing/blank |
| `413` | `PAYLOAD_TOO_LARGE` | Body over `max_request_bytes` (32 KB) |
| `422` | `VALIDATION_FAILED` / `SCHEMA_VIOLATION` | Model output didn't parse, or parsed but failed the schema |
| `500` | `INFERENCE_ERROR` | The backend raised mid-generation |
| `503` | `MODEL_NOT_READY` | Request arrived before the model finished loading |

This is `planning/05-api-design.md`'s registry with two intentional differences, noted there as drift rather than silently diverging: routes are versioned at `/v1/review` (the design doc predates that decision and describes an unprefixed `/review`), and `429 RATE_LIMITED` is defined in the registry but not enforced in v1.0 — HF Spaces throttles at the infrastructure level.

## Run it locally with Docker

The image is CPU-only (no CUDA anywhere) and fetches its `.gguf` at container start rather than baking it into a layer, so the same image works against any HF Hub model repo:

```bash
docker buildx build --platform linux/amd64 -f docker/Dockerfile -t reviewer:local .
docker run -p 8000:8000 \
  -e MODEL_REPO=<hf-username>/<gguf-repo> \
  -e MODEL_FILE=model-Q4_K_M.gguf \
  reviewer:local
```

First boot is slow — the model download happens before `/health` reports ready. To run against the committed CI fixture instead of a real model (no network needed beyond the build):

```bash
docker run -p 8000:8000 \
  -v "$(pwd)/ch3_operation/tests/fixtures/tiny.gguf:/models/model.gguf:ro" \
  -e MODEL_PATH=/models/model.gguf \
  -e CONFIG_PATH=/app/ch3_operation/configs/smoke.yaml \
  reviewer:local
```

`DOCKER=1 uv run pytest ch3_operation/tests/test_docker.py -q` builds the image, runs it, and hits `/health`, `/metrics`, and `POST /v1/review` for real — skipped by default since it needs a Docker daemon and takes minutes.

## How this repo is built

Spec-driven development. The backlog lives in [`specs/STATUS.md`](specs/STATUS.md); each item has a spec file that flows `/specify → /clarify → /plan → /tasks → /implement`. The design documents it was derived from are in [`planning/`](planning/) and [`llm-engineering-architecture-to-edge.md`](llm-engineering-architecture-to-edge.md).

## Reproducing

*(spec X2)*
