# O6 — In-browser WebLLM *(stretch)*

| | |
|---|---|
| **State** | draft |
| **Depends on** | O5 |
| **Requirements** | FR-26 |
| **W&B run** | — |

## Problem

The strongest possible version of the "edge" claim: the model runs **on the reader's own machine, in a browser tab, with no server at all**. No inference cost, no cold start, no data leaving the device — which happens to be the exact argument for a private, specialized code reviewer in the first place. A Java reviewer that runs entirely client-side is a genuinely compelling demo of why you would want a small specialist rather than an API call.

**This is a cut-line.** It is listed as a stretch goal in the project's own definition of done, and it is the first thing to drop if the schedule slips. It must never be started before O5 is live.

## Scope

Compile/convert the quantized model for WebLLM/MLC, serve a static page that runs inference client-side via WebGPU, and link it.

## Acceptance criteria

1. A public browser-based demo URL runs inference **client-side** with no server round-trip for generation — verifiable in the browser's network tab (FR-26).
2. The demo states the model download size and gracefully reports when WebGPU is unavailable, rather than hanging on a browser that cannot run it.
3. The output is validated against the same `eval/schema.json` contract in the client.
4. The demo is linked from the README **alongside** the server demo, not as a replacement for it.

## Out of scope

- Matching the server's quality. A further-quantized browser model may score worse; if it does, that is stated rather than glossed.
- Supporting browsers without WebGPU. Detect and degrade to a link to the hosted Space.

## Clarifications

*(filled by `/clarify`)*

## Technical plan

*(filled by `/plan`)*

## Tasks

*(filled by `/tasks`)*
