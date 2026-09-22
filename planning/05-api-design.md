# api-design.md

> **Status:** v1.0 — Derived from `planning/02-architecture.md`, `planning/03-system-design.md`, and `planning/04-database-design.md`. All route names, request/response shapes, status codes, and field names are consistent with the locked architecture. This document is the authoritative contract for the Chapter 3 HTTP service and the shared eval harness HTTP interface.

---

## Table of Contents

1. [Scope](#1-scope)
2. [Authentication](#2-authentication)
3. [Error Handling](#3-error-handling)
4. [Pagination](#4-pagination)
5. [Versioning](#5-versioning)
6. [OpenAPI Specification](#6-openapi-specification)
7. [Request & Response Examples](#7-request--response-examples)
8. [Rate Limiting & Timeouts](#8-rate-limiting--timeouts)
9. [Design Decisions Log](#9-design-decisions-log)

---

## 1. Scope

This document covers the HTTP API surface of **Chapter 3 (`ch3_operation`)** only. Chapters 1 and 2 have no HTTP interface; they communicate exclusively through files. The Chapter 3 FastAPI application exposes three endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/review` | Submit Java code for review; receive structured `ReviewOutput` |
| `GET` | `/health` | Liveness and readiness probe |
| `GET` | `/metrics` | Prometheus metrics scrape endpoint |

No other endpoints exist. The API is intentionally minimal: it is a single-model inference service, not a general-purpose platform.

**Out of scope for this document:**
- The frontier API calls made by `ch2_adaptation/data_gen.py` and `ch2_adaptation/baseline.py` (those are outbound client calls, not an API this system exposes).
- The W&B and HF Hub MCP integrations (those are tool-call interfaces, not HTTP APIs owned by this system).
- Any future admin, authentication management, or dataset management endpoints (none are planned in v1.0).

---

## 2. Authentication

### 2.1 Decision: No Authentication in v1.0

**The v1.0 service has no authentication layer.**

**Rationale:**

The deployment target is HF Spaces free tier and equivalent CPU cloud free tiers. These environments are:

1. **Public by design.** HF Spaces are publicly accessible URLs. Adding token-based auth would require secret management infrastructure that is out of scope for a teaching project.
2. **Stateless inference only.** The service holds no user data, no persistent state, and no secrets beyond the model weights (which are public on HF Hub). There is no confidential data to protect.
3. **Rate-limited by the deployment platform.** HF Spaces enforces its own request throttling at the infrastructure level.
4. **CPU-bound and slow.** The inference latency (2–30 seconds per request on CPU) is itself a natural throttle against abuse.

### 2.2 Authentication Extension Points

The architecture is designed so that authentication can be added in a future version without breaking the API contract. The extension points are:

**Bearer token (future v1.1):**
```
Authorization: Bearer <token>
```
The `RequestTimingMiddleware` in `middleware.py` is the correct insertion point. A new `AuthMiddleware` would be added before `RequestTimingMiddleware` in the middleware stack. The `main.py` startup sequence would load a token from the `API_TOKEN` environment variable. Requests without a valid token would receive `401 Unauthorized` before reaching any route handler.

**HF Spaces built-in auth (future v1.1 alternative):**
HF Spaces supports a "private Space" mode where HF Hub OAuth gates all access. This requires zero application-level code changes and is the preferred path if the deployment target remains HF Spaces.

### 2.3 Current Security Posture

Since there is no authentication, the following mitigations apply in v1.0:

| Risk | Mitigation |
|------|-----------|
| Prompt injection via `code` field | Input is passed to the model as-is; the model output is validated against a strict schema before being returned. Malformed or adversarial outputs are rejected with `422`. |
| Resource exhaustion | `max_tokens` is capped server-side at the value in `ServeConfig`; the `threading.Lock` in `LlamaCppBackend` serializes requests, preventing parallel GPU/CPU exhaustion. |
| Data exfiltration | The service makes no outbound network calls during inference. The model is loaded from disk at startup. |
| Large payload abuse | Request body size is limited to **32 KB** enforced by a `ContentSizeLimitMiddleware` (see §8). |

---

## 3. Error Handling

### 3.1 Error Response Envelope

All error responses use a consistent JSON envelope:

```json
{
  "error": {
    "code": "VALIDATION_FAILED",
    "message": "Model output did not conform to ReviewOutput schema after 2 repair attempts.",
    "detail": null
  },
  "request_id": "a3f2c1d4-8b7e-4f9a-b2c3-d4e5f6a7b8c9"
}
```

| Field | Type | Always present | Description |
|-------|------|---------------|-------------|
| `error.code` | `string` | Yes | Machine-readable error code (see §3.2) |
| `error.message` | `string` | Yes | Human-readable description |
| `error.detail` | `string \| null` | No | Additional context; `null` when not applicable |
| `request_id` | `string` | Yes | UUID v4 generated per request; matches server logs |

**Design rationale:** The `request_id` field is present on both success and error responses. This allows callers to correlate client-side logs with server-side Prometheus metrics and structured logs without requiring a distributed tracing system.

### 3.2 Error Code Registry

| HTTP Status | `error.code` | Trigger condition |
|-------------|-------------|-------------------|
| `400` | `INVALID_REQUEST` | Request body is not valid JSON, or required fields (`code`) are missing |
| `400` | `CODE_FIELD_EMPTY` | `code` field is present but is an empty string or whitespace-only |
| `413` | `PAYLOAD_TOO_LARGE` | Request body exceeds 32 KB |
| `422` | `VALIDATION_FAILED` | Model produced output that failed both parse attempts in `validator.py` |
| `422` | `SCHEMA_VIOLATION` | Model output parsed as JSON but failed Pydantic `ReviewOutput` validation |
| `429` | `RATE_LIMITED` | Request rejected by rate limiter (see §8) |
| `500` | `INFERENCE_ERROR` | Unhandled exception inside `LlamaCppBackend.generate()` |
| `500` | `INTERNAL_ERROR` | Any other unhandled server-side exception |
| `503` | `MODEL_NOT_READY` | Request arrived before model finished loading at startup |

### 3.3 HTTP Status Code Policy

The service follows RFC 9110 semantics strictly:

- **`2xx`** — Request was received, processed, and the model produced a schema-valid output.
- **`4xx`** — The error is attributable to the caller (bad input, payload too large, rate limit exceeded). The caller should not retry without changing the request.
- **`5xx`** — The error is attributable to the server (inference failure, model not ready). The caller may retry with exponential backoff.

**`422` is a client error, not a server error.** When the model fails to produce a valid `ReviewOutput`, this is surfaced as `422` rather than `500`. The rationale: the caller submitted a valid request; the model's inability to produce structured output for that specific input is a known, expected failure mode of probabilistic inference. The caller can retry with a different or simplified `code` input, or accept that this snippet is not reviewable by the current model.

### 3.4 Validation Error Detail

When `error.code` is `SCHEMA_VIOLATION`, the `error.detail` field contains the Pydantic validation error message to help callers understand which field failed:

```json
{
  "error": {
    "code": "SCHEMA_VIOLATION",
    "message": "Model output parsed as JSON but failed ReviewOutput validation.",
    "detail": "1 validation error for ReviewOutput\nseverity\n  Input should be 'critical', 'major', 'minor' or 'info' [type=literal_error]"
  },
  "request_id": "b4c3d2e1-9a8b-4c7d-a1b2-c3d4e5f6a7b8"
}
```

### 3.5 Error Propagation Rules

```
LlamaCppBackend.generate()
    raises RuntimeError          →  500 INFERENCE_ERROR
    returns raw string
        │
        ▼
validator.validate_output(raw)
    raises json.JSONDecodeError  →  422 VALIDATION_FAILED
    raises ValueError            →  422 VALIDATION_FAILED  (no fence found)
    raises pydantic.ValidationError → 422 SCHEMA_VIOLATION
    returns ReviewOutput
        │
        ▼
route handler
    records metrics (valid=True)
    returns 200 ReviewOutput
```

Exceptions that escape the route handler are caught by a global exception handler registered in `main.py`:

```python
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # logs full traceback
    # records metrics (valid=False, status="error")
    return JSONResponse(status_code=500, content={
        "error": {"code": "INTERNAL_ERROR", "message": str(exc), "detail": None},
        "request_id": request.state.request_id,
    })
```

---

## 4. Pagination

### 4.1 Decision: No Pagination

**The v1.0 API has no paginated endpoints.**

**Rationale:**

All three endpoints return a single object:

- `POST /review` → one `ReviewOutput` object (or one error envelope)
- `GET /health` → one health status object
- `GET /metrics` → one Prometheus text-format document (a flat string, not a JSON collection)

There are no list endpoints, no collection resources, and no cursor-based or offset-based traversal. The service is a stateless inference endpoint, not a data retrieval API.

### 4.2 Pagination Extension Points

If a future version adds endpoints such as `GET /reviews` (history of past reviews) or `GET /results` (eval result listing), the following pagination strategy is pre-selected:

**Cursor-based pagination (not offset-based).**

```
GET /reviews?cursor=<opaque_token>&limit=20
```

| Parameter | Type | Default | Max |
|-----------|------|---------|-----|
| `cursor` | `string` | absent (first page) | — |
| `limit` | `integer` | `20` | `100` |

Response envelope (future):
```json
{
  "items": [...],
  "next_cursor": "eyJpZCI6IjEyMyIsInRzIjoiMjAyNS0wMS0wMVQwMDowMDowMFoifQ==",
  "has_more": true
}
```

**Rationale for cursor over offset:** The underlying storage is append-only JSONL files (or their SQLite equivalent per §4 of the database design). Offset pagination is unstable when new records are appended between pages. Cursor pagination using the record `id` (UUID v4, lexicographically sortable when using UUIDv7 in a future version) is stable under concurrent writes.

---

## 5. Versioning

### 5.1 Versioning Strategy: No URL Versioning in v1.0

**The v1.0 API has no version prefix in the URL path.**

Routes are:
```
POST /review
GET  /health
GET  /metrics
```

Not:
```
POST /v1/review   ← not used in v1.0
```

**Rationale:**

1. **Single deployment target.** The service runs as one instance on HF Spaces. There is no need to run v1 and v2 simultaneously.
2. **Teaching context.** The monorepo is a learning artifact. URL versioning adds boilerplate that obscures the core concepts being taught.
3. **Schema is the contract.** The `ReviewOutput` schema (defined in `eval/schema.json`) is the real versioning surface. Changes to the schema are breaking changes regardless of URL versioning.

### 5.2 Breaking vs. Non-Breaking Changes

| Change type | Breaking? | Version bump required? |
|-------------|-----------|----------------------|
| Adding an optional field to `ReviewOutput` | No | No — existing clients ignore unknown fields |
| Removing a field from `ReviewOutput` | **Yes** | Yes — bump to v2 |
| Changing a field type in `ReviewOutput` | **Yes** | Yes |
| Changing an enum value in `severity` | **Yes** | Yes |
| Adding a new endpoint | No | No |
| Changing an existing endpoint's URL | **Yes** | Yes |
| Adding a new optional query parameter | No | No |
| Changing an error code string | **Yes** | Yes — callers may match on error codes |
| Changing an HTTP status code | **Yes** | Yes |

### 5.3 Version Signaling

Although there is no URL version prefix, the API version is communicated in two ways:

**Response header (all responses):**
```
X-API-Version: 1.0.0
X-Schema-Version: <sha256_prefix_of_eval/schema.json>
```

The `X-Schema-Version` header contains the first 8 characters of the SHA-256 hash of `eval/schema.json` at server startup. This allows callers to detect schema drift without parsing the schema file itself.

**Health endpoint body:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "api_version": "1.0.0",
  "schema_version": "a3f2c1d4"
}
```

### 5.4 Deprecation Policy (future versions)

When a breaking change is introduced:

1. The new version is deployed at `/v2/review` (etc.) alongside the existing routes.
2. The old routes return a `Deprecation` response header: `Deprecation: true` and `Sunset: <ISO 8601 date>`.
3. Old routes are removed no sooner than 30 days after the `Sunset` date.
4. The `eval/schema.json` file is updated; its SHA-256 changes; `X-Schema-Version` changes automatically.

---

## 6. OpenAPI Specification

```yaml
openapi: "3.1.0"

info:
  title: "Java Code Reviewer"
  version: "1.0.0"
  description: |
    Single-model inference service that accepts a Java code snippet and returns
    a structured code review conforming to the ReviewOutput schema.

    The model is a QLoRA fine-tuned Qwen2.5-Coder-1.5B-Instruct, exported to
    GGUF format and served via llama-cpp-python on CPU.

    **Authentication:** None in v1.0. See api-design.md §2 for the extension plan.

    **Rate limiting:** 10 requests/minute per IP. See api-design.md §8.

    **Versioning:** No URL prefix in v1.0. API version is communicated via the
    `X-API-Version` response header and the `/health` endpoint body.
  contact:
    name: "Project Maintainer"
  license:
    name: "MIT"

servers:
  - url: "https://{space_name}.hf.space"
    description: "HF Spaces production deployment"
    variables:
      space_name:
        default: "java-code-reviewer"
        description: "HF Spaces subdomain"
  - url: "http://localhost:{port}"
    description: "Local development server"
    variables:
      port:
        default: "8000"
        enum: ["8000", "8080"]

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────

paths:

  /review:
    post:
      operationId: "submitReview"
      summary: "Submit Java code for review"
      description: |
        Accepts a Java code snippet and optional context string. Runs inference
        using the loaded GGUF model, validates the output against the ReviewOutput
        schema, and returns the structured review.

        **Latency:** Expect 2–30 seconds on CPU depending on snippet length and
        server load. The endpoint is synchronous; the connection is held open
        until inference completes.

        **Failure modes:**
        - `422 VALIDATION_FAILED`: The model could not produce a schema-valid
          output after two parse attempts. Retry with a shorter or simpler snippet.
        - `503 MODEL_NOT_READY`: The server is still loading the model at startup.
          Retry after a few seconds.
      tags:
        - "Inference"
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/ReviewRequest"
            examples:
              null_pointer:
                summary: "Null pointer dereference"
                value:
                  code: |
                    public String getUserName(User user) {
                        return user.getProfile().getName();
                    }
                  context: "UserService.java, called from REST controller"
              resource_leak:
                summary: "Unclosed resource"
                value:
                  code: |
                    public void writeLog(String msg) throws IOException {
                        FileWriter fw = new FileWriter("app.log", true);
                        fw.write(msg);
                    }
                  context: ""
      responses:
        "200":
          description: |
            Model produced a schema-valid ReviewOutput. The review identifies
            exactly one bug in the submitted code.
          headers:
            X-API-Version:
              $ref: "#/components/headers/X-API-Version"
            X-Schema-Version:
              $ref: "#/components/headers/X-Schema-Version"
            X-Request-Id:
              $ref: "#/components/headers/X-Request-Id"
            X-Latency-Ms:
              $ref: "#/components/headers/X-Latency-Ms"
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ReviewOutput"
              examples:
                null_pointer_response:
                  summary: "Null pointer dereference review"
                  value:
                    severity: "critical"
                    category: "NullPointerException"
                    line: 2
                    issue: "user.getProfile() may return null, causing a NullPointerException on getName()."
                    suggested_fix: "Add a null check: if (user.getProfile() == null) return \"unknown\";"
        "400":
          description: "Request body is malformed or required fields are missing."
          headers:
            X-API-Version:
              $ref: "#/components/headers/X-API-Version"
            X-Request-Id:
              $ref: "#/components/headers/X-Request-Id"
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"
              examples:
                missing_code:
                  summary: "Missing required field"
                  value:
                    error:
                      code: "INVALID_REQUEST"
                      message: "Field 'code' is required."
                      detail: null
                    request_id: "a3f2c1d4-8b7e-4f9a-b2c3-d4e5f6a7b8c9"
                empty_code:
                  summary: "Empty code field"
                  value:
                    error:
                      code: "CODE_FIELD_EMPTY"
                      message: "Field 'code' must not be empty or whitespace-only."
                      detail: null
                    request_id: "b5e3f2a1-7c6d-4e8b-a3c4-d5e6f7a8b9c0"
        "413":
          description: "Request body exceeds the 32 KB size limit."
          headers:
            X-API-Version:
              $ref: "#/components/headers/X-API-Version"
            X-Request-Id:
              $ref: "#/components/headers/X-Request-Id"
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"
              examples:
                payload_too_large:
                  value:
                    error:
                      code: "PAYLOAD_TOO_LARGE"
                      message: "Request body exceeds the 32768-byte limit."
                      detail: null
                    request_id: "c6f4a3b2-8d7e-4f9c-b4d5-e6f7a8b9c0d1"
        "422":
          description: |
            The model produced output that could not be parsed or validated as
            a ReviewOutput after two repair attempts. This is a client-attributable
            error: the submitted code snippet may be too complex, too short, or
            outside the model's training distribution.
          headers:
            X-API-Version:
              $ref: "#/components/headers/X-API-Version"
            X-Request-Id:
              $ref: "#/components/headers/X-Request-Id"
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"
              examples:
                validation_failed:
                  summary: "Both parse attempts failed"
                  value:
                    error:
                      code: "VALIDATION_FAILED"
                      message: "Model output did not conform to ReviewOutput schema after 2 repair attempts."
                      detail: null
                    request_id: "d7a5b4c3-9e8f-4a0d-c5e6-f7a8b9c0d1e2"
                schema_violation:
                  summary: "JSON parsed but schema invalid"
                  value:
                    error:
                      code: "SCHEMA_VIOLATION"
                      message: "Model output parsed as JSON but failed ReviewOutput validation."
                      detail: "1 validation error for ReviewOutput\nseverity\n  Input should be 'critical', 'major', 'minor' or 'info' [type=literal_error]"
                    request_id: "e8b6c5d4-0f9a-4b1e-d6f7-a8b9c0d1e2f3"
        "429":
          description: "Rate limit exceeded. See §8 for the rate limiting policy."
          headers:
            X-API-Version:
              $ref: "#/components/headers/X-API-Version"
            X-Request-Id:
              $ref: "#/components/headers/X-Request-Id"
            Retry-After:
              description: "Seconds until the rate limit window resets."
              schema:
                type: integer
                example: 47
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"
              examples:
                rate_limited:
                  value:
                    error:
                      code: "RATE_LIMITED"
                      message: "Rate limit exceeded: 10 requests per minute per IP."
                      detail: "Retry after 47 seconds."
                    request_id: "f9c7d6e5-1a0b-4c2f-e7a8-b9c0d1e2f3a4"
        "500":
          description: "Server-side error during inference or request processing."
          headers:
            X-API-Version:
              $ref: "#/components/headers/X-API-Version"
            X-Request-Id:
              $ref: "#/components/headers/X-Request-Id"
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"
              examples:
                inference_error:
                  summary: "llama-cpp-python raised an exception"
                  value:
                    error:
                      code: "INFERENCE_ERROR"
                      message: "Inference backend raised an unexpected error."
                      detail: "RuntimeError: llama_decode returned -1"
                    request_id: "a0d8e7f6-2b1c-4d3a-f8b9-c0d1e2f3a4b5"
        "503":
          description: "Model is still loading. Retry after a few seconds."
          headers:
            X-API-Version:
              $ref: "#/components/headers/X-API-Version"
            X-Request-Id:
              $ref: "#/components/headers/X-Request-Id"
            Retry-After:
              description: "Estimated seconds until model is ready."
              schema:
                type: integer
                example: 15
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"
              examples:
                model_not_ready:
                  value:
                    error:
                      code: "MODEL_NOT_READY"
                      message: "Model is still loading. Please retry in a few seconds."
                      detail: null
                    request_id: "b1e9f8a7-3c2d-4e4b-a9c0-d1e2f3a4b5c6"

  /health:
    get:
      operationId: "getHealth"
      summary: "Liveness and readiness probe"
      description: |
        Returns the current health status of the service. Intended for use by
        container orchestrators, load balancers, and HF Spaces health checks.

        - Returns `200` with `"status": "ok"` when the model is loaded and the
          service is ready to accept requests.
        - Returns `503` with `"status": "starting"` while the model is loading
          at startup.

        This endpoint never triggers model inference and always responds within
        100 ms.
      tags:
        - "Operations"
      responses:
        "200":
          description: "Service is healthy and ready to accept requests."
          headers:
            X-API-Version:
              $ref: "#/components/headers/X-API-Version"
            X-Schema-Version:
              $ref: "#/components/headers/X-Schema-Version"
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/HealthResponse"
              examples:
                healthy:
                  value:
                    status: "ok"
                    model_loaded: true
                    api_version: "1.0.0"
                    schema_version: "a3f2c1d4"
        "503":
          description: "Service is starting up; model is not yet loaded."
          headers:
            X-API-Version:
              $ref: "#/components/headers/X-API-Version"
            Retry-After:
              description: "Estimated seconds until model is ready."
              schema:
                type: integer
                example: 15
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/HealthResponse"
              examples:
                starting:
                  value:
                    status: "starting"
                    model_loaded: false
                    api_version: "1.0.0"
                    schema_version: "a3f2c1d4"

  /metrics:
    get:
      operationId: "getMetrics"
      summary: "Prometheus metrics scrape endpoint"
      description: |
        Returns current Prometheus metrics in the standard text exposition format
        (version 0.0.4). Intended for scraping by a Prometheus server or manual
        inspection via `curl`.

        **Exposed metrics:**

        | Metric name | Type | Labels | Description |
        |---|---|---|---|
        | `review_request_latency_seconds` | Histogram | — | End-to-end request latency in seconds |
        | `review_requests_total` | Counter | `status` | Total requests by outcome |
        | `review_schema_validity_rate` | Gauge | — | Rolling schema validity rate (window=100) |

        **Histogram buckets for `review_request_latency_seconds`:**
        `[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]`

        **`status` label values for `review_requests_total`:**
        `success` (model produced valid output) | `error` (any failure)

        This endpoint never triggers model inference and always responds within
        50 ms. It is not authenticated; Prometheus scrapers are assumed to be
        on a trusted network.
      tags:
        - "Operations"
      responses:
        "200":
          description: "Prometheus metrics in text exposition format."
          content:
            text/plain:
              schema:
                type: string
                description: "Prometheus text format (version 0.0.4)"
              example: |
                # HELP review_request_latency_seconds End-to-end request latency in seconds
                # TYPE review_request_latency_seconds histogram
                review_request_latency_seconds_bucket{le="0.1"} 0
                review_request_latency_seconds_bucket{le="0.5"} 2
                review_request_latency_seconds_bucket{le="1.0"} 8
                review_request_latency_seconds_bucket{le="2.0"} 19
                review_request_latency_seconds_bucket{le="5.0"} 34
                review_request_latency_seconds_bucket{le="10.0"} 41
                review_request_latency_seconds_bucket{le="30.0"} 43
                review_request_latency_seconds_bucket{le="+Inf"} 43
                review_request_latency_seconds_sum 87.432
                review_request_latency_seconds_count 43
                # HELP review_requests_total Total requests by outcome
                # TYPE review_requests_total counter
                review_requests_total{status="success"} 41
                review_requests_total{status="error"} 2
                # HELP review_schema_validity_rate Rolling schema validity rate over last 100 requests
                # TYPE review_schema_validity_rate gauge
                review_schema_validity_rate 0.95

# ─────────────────────────────────────────────────────────────
# COMPONENTS
# ─────────────────────────────────────────────────────────────

components:

  # ── Schemas ──────────────────────────────────────────────

  schemas:

    ReviewRequest:
      type: object
      required:
        - code
      additionalProperties: false
      properties:
        code:
          type: string
          minLength: 1
          maxLength: 32768
          description: |
            The Java source code snippet to review. Must be non-empty and
            non-whitespace-only. The model is trained on Java/Spring snippets
            of 5–100 lines; very short (<3 lines) or very long (>200 lines)
            snippets may produce lower-quality output.
          example: |
            public String getUserName(User user) {
                return user.getProfile().getName();
            }
        context:
          type: string
          maxLength: 512
          default: ""
          description: |
            Optional free-text description of the surrounding context: file name,
            class name, framework, or any other information that helps the model
            understand the snippet. May be an empty string.
          example: "UserService.java, called from REST controller"

    ReviewOutput:
      type: object
      required:
        - severity
        - category
        - line
        - issue
        - suggested_fix
      addit