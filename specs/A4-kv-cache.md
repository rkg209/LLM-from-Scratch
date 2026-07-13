# A4 — KV-cache

| | |
|---|---|
| **State** | draft |
| **Depends on** | A3 |
| **Requirements** | FR-12, NFR-2, NFR-5 |
| **W&B run** | — |

## Problem

Naive autoregressive generation re-runs the entire sequence through every layer to produce one new token, recomputing keys and values it already computed on the previous step. The waste is quadratic in sequence length, and it is the single most important inference optimization in transformer serving — "why is naive attention slow, and what does a KV-cache actually cache" is a question that gets asked in interviews because the answer separates people who have read about it from people who have implemented it.

The deliverable is not the cache. It is the **measured speedup curve**, and that number is only meaningful if the cache is provably correct first.

## Scope

`KVCache` (per-layer pre-allocated K/V buffers with a write pointer), the cached path through `MultiHeadSelfAttention` and `generate()`, and the before/after tokens/sec measurement.

## Acceptance criteria

1. **Logit identity** — for the same prompt and seed, cached and uncached generation produce identical logits at every step, `max |Δ| < 1e-5` (FR-12a, NFR-5). If this fails, no speedup number from this spec may be reported.
2. Tokens/sec is measured with and without the cache under identical conditions (same seed, prompt, `max_new_tokens`, warm-up steps discarded), and the cached path is **faster** (FR-12b, NFR-2).
3. The speedup is recorded in `eval/results/ch1_benchmark.json` and the metrics scorecard.
4. `cache.reset()` is called at the start of each generation; a second generation on a reused cache produces the same output as the first on a fresh one.
5. `cache.update()` raises `IndexError` rather than corrupting memory when `current_len >= max_seq_len`.
6. Positional embeddings in the cached path use the **absolute** position in the sequence, not the length of the current input (which is always 1) — verified by criterion 1.

## Out of scope

- Quantization → **A5**.
- Cache eviction, paged attention, or sliding windows. One correct, pre-allocated cache.
- Batched generation with ragged lengths.

## Clarifications

*(filled by `/clarify`)*

## Technical plan

*(filled by `/plan`)*

## Tasks

*(filled by `/tasks`)*
