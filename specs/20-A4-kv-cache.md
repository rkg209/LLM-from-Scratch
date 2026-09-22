# A4 — KV-cache

| | |
|---|---|
| **State** | building (code + tests done — the logit-identity test passes at `atol=1e-5`; a meaningful, not-smoke-sized speedup number needs the full-config GPU run) |
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

- **Lazy allocation.** `KVCache(n_layers, max_seq_len, n_heads, head_dim)` keeps the locked 4-arg signature from `planning/03`, which gives `__init__` no batch argument, and allocates its `[B, H, max_seq_len, Dh]` buffers lazily on the first `update()`, taking `B` from the incoming tensor. `reset()` zeroes `current_len` only; it does not free buffers (AC-4 needs the second generation to reuse them).
- **Write-pointer timing.** `current_len` advances **once per forward pass** (a separate `cache.advance(T)` call after every layer has written), not once per layer — advancing per-layer would make layer 5 read a window layer 0 never wrote. `attention.forward` reads `cache.current_len` as the mask offset *before* `cache.update()` runs, so every layer in one forward pass sees the same offset.
- **Test methodology for the identity check.** Both paths decode greedily (`top_k=1`) from the same seed and prompt, so the sampled token sequence is identical by construction whenever the logits agree — comparing logits directly (rather than trying to keep two independent RNG streams in lock-step) is what AC-1 actually needs.

## Technical plan

See `.claude/plans/AfterOperation.md` §A4. `attention.forward`'s cached branch: `cache.update(layer_idx, k, v)` then attend `q` (`T=1`) against the full `[B,H,S,Dh]` history — the mask slice already written in A2 (`causal_mask[seq_offset:seq_offset+T, :S]`) needed no change. `gpt.forward` passes `offset=cache.current_len` to `PositionalEmbedding` when a cache is live, then calls `cache.advance(T)` once after the block loop. `generate.py` gains `_generate_cached`: `cache.reset()`, one prefill forward over the whole prompt, then one-token steps; sampling is shared with `_generate_uncached` so the two differ only in how the logits were produced.

## Tasks

- **A4-T1** — `kv_cache.py` + unit tests (lazy alloc, write pointer, `reset`, `IndexError`, view-not-copy). **Done.**
- **A4-T2** — cached branch in `attention.py` + positional offset in `gpt.py`. **Done.**
- **A4-T3** — `_generate_cached` + the logit-identity test. **Done — passes at `atol=1e-5`.**
- **A4-T4** — cached-vs-uncached tokens/sec into `ch1_benchmark.json` (writer is `benchmark.py`, A5). **Done** — `measure_tokens_per_sec_cached`/`measure_tokens_per_sec` both live in `benchmark.py` and both write into the `kv_cache` block of the JSON output.

Remaining: AC-2's cached path must actually be *faster* — on the smoke-sized model the effect is small (quadratic in sequence length), so the number that goes in the README comes from the full-config GPU run.
