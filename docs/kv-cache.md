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

<!-- filled once A4/A5's full-config GPU run lands; see eval/results/ch1_benchmark.json
     and the KV-cache line under README.md's "Chapter 1 — speedup and quantization cost".
     Placeholder per REPRODUCING.md's provenance table — no number is invented here ahead
     of the run that would produce it. -->
*(pending — A4 full GPU run: cached vs. uncached tokens/sec and the resulting speedup
factor, from `eval/results/ch1_benchmark.json`.)*

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
