# Why naive attention is slow, and what the KV-cache fixes

Source: `ch1_architecture/src/ch1_architecture/model/attention.py`,
`ch1_architecture/src/ch1_architecture/kv_cache.py`,
`ch1_architecture/src/ch1_architecture/benchmark.py::measure_tokens_per_sec[_cached]`,
`ch1_architecture/tests/test_kv_cache.py`.

## What is recomputed on every decode step, and why that is quadratic

Autoregressive generation produces one token at a time, and each new token needs the
model's forward pass to attend over every token generated so far. Without a cache,
`measure_tokens_per_sec` in this repo's benchmark shows the naive version of this
directly: at every decode step it re-runs the **whole growing sequence** through the
model from scratch —

```python
logits = model(tokens)          # tokens keeps growing: len 1, 2, 3, ..., N
next_token = logits[:, -1, :].argmax(...)
tokens = torch.cat([tokens, next_token], dim=1)
```

Step `t` costs `O(t)` work in the attention matmuls (`scores = q @ k.T`, shape `[T, S]`
with `T = S = t`), and generating `N` tokens sums that over `t = 1..N`, which is `O(N^2)`.
Every one of the first `t-1` key/value projections is thrown away and recomputed at step
`t`, identical to the projections computed at step `t-1` — pure wasted work, not new
information.

## What the cache stores, and what it does not

`KVCache` (`kv_cache.py`) stores exactly the two projections that step `t` would
otherwise recompute for every earlier position: the **K and V projections**, one buffer
per layer, per head, of shape `[B, H, max_seq_len, head_dim]`, filled in-place as
generation proceeds (`update()`, called once per layer per forward pass).

It does **not** store Q. The query for the current step is recomputed fresh from the one
new token every time — there is nothing to cache there, because a query is only ever used
once, at the step that produces it, and never revisited. The asymmetry is the whole
mechanism: K and V for position `i` are read at every later step `> i` (cache them once,
reuse forever); Q for position `i` is read exactly once, at step `i` (caching it would
save nothing).

## Why it does not help prefill

This is the detail that is easy to get backwards. Prefill — the first forward pass over
the prompt — already computes every position's attention **once**, in parallel, in a
single matmul. There is nothing to reuse yet, because nothing has been computed before.
What prefill actually does with the cache is *fill* it: `attention.forward` passes
`cache=cache, layer_idx=...` on the prefill call exactly as it does on every decode call,
and `KVCache.update` writes the prompt's K/V into the buffer at offset 0. The speedup
shows up entirely on the *decode* steps that follow, where each one now attends against
the cached history instead of recomputing it. A KV-cache does not make the first forward
pass faster; it makes the `N-1` forward passes after it unnecessary to redo.

## Why cached and uncached logits must be bit-identical

The cache is a pure optimization — it must change *how fast* the answer arrives, never
*what* the answer is. `_scaled_dot_product` in `attention.py` slices the causal mask from
`seq_offset` (`cache.current_len`, or `0` uncached) rather than always from `0`, so a
single-token decode step still asks "may this query see each of the `S` cached keys,"
the same question the uncached path asks about the same query embedded in a longer
sequence. If the cached path silently attended over a different, or wrongly-ordered, set
of keys, nothing would crash — the model would just generate confidently wrong text,
which is a much worse failure mode than a crash. That is why this repo does not eyeball
the generated samples to check the cache; `test_cached_and_uncached_generation_produce_identical_logits`
in `tests/test_kv_cache.py` runs both paths on the same prompt and asserts
`torch.allclose` on every generated logit, and nothing about the KV-cache speedup number
is trusted until that test is green.

One subtlety the tests also pin down: `current_len` advances **once per forward pass**
(`cache.advance(T)`, called after every layer has run), not once per layer inside
`update()`. If it advanced per-layer, layer 1 would see a different `current_len` than
layer 0 saw when computing its causal mask for the same forward pass — a bug that would
not show up as a crash, only as subtly wrong attention for every layer after the first.

## The memory cost that buys the speed

The cache is not free. Per sequence, it holds `2 · n_layers · n_heads · head_dim · seq_len`
scalars (the factor of 2 is K and V) — linear in sequence length, but linear in a
quantity that itself grows every step, and multiplied across every concurrent request a
server holds in memory at once. This is the real constraint at serving scale: the
KV-cache turns a compute problem (recompute everything) into a memory problem (hold
everything), and at high concurrency or long context, cache memory — not FLOPs — is what
runs out first. `ch3_operation`'s serving guardrails cap `n_ctx` for exactly this reason.

## This project's measured numbers

Measured 2026-09-18, PARAM Rudra job 402166, one A100 80GB PCIe, the 21.0M-parameter
checkpoint from the 3000-step run. Prompt `"First Citizen:"`, greedy decoding, the first
10 steps discarded as warm-up. Source: `eval/results/ch1_benchmark.json`.

| Decode length | cached tok/s | uncached tok/s | speedup |
|---|---|---|---|
| 32 | 273.6 | 271.6 | 1.008x |
| 64 | 274.3 | 271.4 | 1.010x |
| 128 | 273.7 | 267.9 | 1.022x |
| 250 | 272.9 | 265.0 | 1.030x |

**Read the two throughput columns, not the ratio.** Cached throughput is flat as the
sequence grows — 273.6 at 32 steps, 272.9 at 250 — because each cached step does the same
work regardless of how much history precedes it. Uncached throughput decays, 271.6 to
265.0, because every step re-encodes a prefix that keeps getting longer. That divergence
*is* the KV-cache doing its job, and it is visible even though the ratio never leaves the
third decimal place.

The ratio stays small because this model, at this size, on this hardware, is not
bottlenecked by the work the cache removes. A 21M-parameter model decoding at batch 1
spends roughly 4.3 ms per step on kernel-launch overhead and Python dispatch across six
layers, against something on the order of 0.08 ms of actual attention compute at 115
tokens on an A100. The cache is eliminating work that was already free. Its payoff scales
with the quadratic term — longer contexts, larger models, or hardware where compute is
the constraint — and none of those describe this benchmark.

That is why the honest way to report this is a curve rather than a single number. A lone
"1.03x" reads as a claim about KV-caches. It is a claim about a 21M model at 250 tokens
on an A100, and the trend line is what distinguishes the two.

**Caveat on the sequence-length ceiling.** 250 is as far as this can go: `seq_len` is 256
and the prompt consumes a few tokens. The trend is still rising at the right-hand edge of
the plot, so the curve is a lower bound on what a longer-context version of this model
would show, not a plateau.

## What I got wrong

The mask slice was the surprise. The original design note this code was built from said
the causal mask slice never needs to change between the uncached and cached paths — take
`mask[:T, :S]` and be done. Taken literally that is wrong: at a cached decode step `T=1`,
so `mask[:1, :S]` is *row 0* of the causal mask, which only unmasks column 0 — it would
silently blind every cached decode step to everything except the very first cached key,
while still running without error and producing plausible-looking (wrong) tokens. The fix
was slicing from `seq_offset` instead of `0`. It genuinely didn't occur to me until I sat
down to write the cached path that "no change needed" and "the slice indices are
literally identical" were different claims, and only the first one was actually true.
