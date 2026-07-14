# O2 — FastAPI serving + guardrails

| | |
|---|---|
| **State** | draft |
| **Depends on** | O1 |
| **Requirements** | FR-22, CON-7, NFR-20 |
| **W&B run** | — |

## Problem

A language model behind an HTTP endpoint is a service that returns a string. A *product* returns a contract. The difference is this spec.

The caller — a sibling project, a CI job, an IDE plugin — is going to `json.loads()` the response and index into `severity`. If the model wraps its JSON in a markdown fence, or hallucinates a sixth field, or emits prose before the object, the caller crashes. The service must never hand a malformed body to a client and call it a success: it parses, it validates, it makes exactly one repair attempt, and then it fails honestly with a 422.

The temptation to fix here is the temptation to *invent* — to fill in a missing `suggested_fix` with an empty string so the response validates. That converts a visible model failure into an invisible data-quality failure in someone else's system, and it also inflates the schema-validity metric the project publishes.

## Scope

`validator.py` (parse → Pydantic → single fence-extraction repair → fail), the `/review` route wired to it, the error envelope, and the request-size limit.

## Acceptance criteria

1. Every model output is validated against `eval/schema.json` (via the `ReviewOutput` Pydantic model) **before** it reaches the caller. An invalid output is never returned as-is (FR-22).
2. The repair strategy is **exactly two attempts, no more**: `json.loads(raw)`, then extraction from a ```` ```json ```` fence. Then it fails.
3. **The validator never fabricates a field.** It parses what the model returned; it does not supply defaults, coerce types, or fill blanks to force a pass.
4. A request that elicits malformed output returns **422** with the error envelope from `planning/05-api-design.md` §3.1 — `{error: {code, message, detail}, request_id}` — not the raw output, and not a 500 (FR-22a).
5. Error codes follow the registry in §3.2: `VALIDATION_FAILED`, `SCHEMA_VIOLATION`, `INVALID_REQUEST`, `CODE_FIELD_EMPTY`, `PAYLOAD_TOO_LARGE`, `INFERENCE_ERROR`, `MODEL_NOT_READY`.
6. A normal request returns **200** with a schema-valid `ReviewOutput` (FR-22b).
7. `request_id` (UUID v4) is present on **both** success and error responses, for log correlation.
8. Request bodies over **32 KB** are rejected with 413.
9. `SCHEMA_VIOLATION` responses include the Pydantic validation message in `error.detail`, so a caller can see which field failed.

## Out of scope

- Metrics and drift → **O3**. Docker → **O4**.
- Authentication. v1.0 has none, deliberately — see `planning/05-api-design.md` §2, which also names the extension point (an `AuthMiddleware` before `RequestTimingMiddleware`).
- Retrying the model on a malformed output. One inference, two parse attempts, then an honest failure — retrying hides the model's real schema-validity rate.

## Clarifications

*(filled by `/clarify`)*

## Technical plan

*(filled by `/plan`)*

## Tasks

*(filled by `/tasks`)*
