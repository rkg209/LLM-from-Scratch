# A5 — Quantization (INT8 / 4-bit)

| | |
|---|---|
| **State** | building (code + tests done, verified end-to-end against a smoke checkpoint — JSON, both plots, and the README section all generate correctly; headline numbers need the full-config GPU run) |
| **Depends on** | A4 |
| **Requirements** | FR-13, CON-5, NFR-7 |
| **W&B run** | — |

## Problem

Quantization is the technique that makes edge deployment possible at all — it is what Chapter 3 does to the fine-tuned model to fit it on a free CPU tier. Chapter 1 earns the right to do that by implementing it by hand, in plain PyTorch, and measuring both sides of the bargain: the speedup **and** what it costs in quality.

The tradeoff curve is the deliverable, and reporting only the good half of it is the exact dishonesty this project is positioned against. A 4× speedup that doubles perplexity is a finding, not a failure — but it has to be published as one.

## Scope

`QuantizedLinear` and `quantize_model(model, mode)` for `fp16`, `int8`, and `int4` (absmax, plain PyTorch — no bitsandbytes), plus `benchmark.py` producing tokens/sec and perplexity at each precision and both plots.

## Acceptance criteria

1. `quantize_model` supports `fp16`, `int8`, and `int4`; quantized inference runs without error and returns correctly-shaped logits (FR-13a).
2. Absmax quantization is implemented as specified in `planning/03-system-design.md` §1.1: `scale = max(abs(W)) / 127` (int8) or `/ 7` (int4); dequantize on forward.
3. No `bitsandbytes` import appears in `ch1_architecture/src/` (CON-5).
4. `eval/results/plots/speedup_curve.png` plots tokens/sec across fp32 → fp16 → int8 → int4 (FR-13b).
5. `eval/results/plots/perplexity_tradeoff.png` plots perplexity across the same axis (FR-13c) — so the cost of each speedup sits next to it.
6. `eval/results/ch1_benchmark.json` records `{tokens_per_sec, perplexity, n_steps}` per mode, with the seed and checkpoint path.
7. Measurements exclude warm-up steps and use an identical seed, prompt, and step count across all modes.
8. Both plots are referenced in the README (FR-13d), **including any mode where quality degraded** (NFR-7).

## Out of scope

- GGUF and `llama.cpp` quantization → **O1**. This is hand-rolled quantization of the hand-rolled model; that is a production toolchain applied to a real one.
- Quantization-aware training, calibration sets, per-channel or group-wise schemes. Per-tensor absmax is the teaching case; note the limitation in the write-up rather than fixing it.

## Clarifications

- **Buffers, not parameters.** `QuantizedLinear` stores quantized weights as buffers — this is post-training quantization for inference, nothing here is trained.
- **`mode="fp32"` is a no-op.** `quantize_model(model, "fp32")` returns an untouched deep copy — the benchmark needs a baseline row, and a trivial branch is cleaner than special-casing the call site.
- **The tied head stays fp32 in every mode.** It is a matmul against the embedding weight (`x @ token_emb.weight.T`), not an `nn.Linear`, so `quantize_model`'s `named_children()` walk never touches it — worth noting in the write-up rather than treated as a bug.
- **int4 packing.** Two 4-bit two's-complement nibbles per `torch.uint8` (`(hi << 4) | lo`), zero-padded to even length before packing; the pre-pack shape is kept as a buffer so unpack can restore it exactly.
- **`BenchmarkConfig` dropped a redundant field.** The umbrella plan sketched both `max_new_tokens` and `n_steps`; only `n_steps` survived (it is both the generation-loop length and the tokens/sec measurement window, with `warmup_steps` discarded from the front) — a second, unused length field would have been dead config.
- **Benchmarked checkpoint.** `benchmark.py` reads `GPTConfig`, model weights, and the tokenizer's vocab/merges out of the checkpoint dict itself (no dependency on the corpus file or a separately-loaded tokenizer), so it can run against any checkpoint independent of how it was produced.

## Technical plan

See `.claude/plans/AfterOperation.md` §A5. Absmax per `planning/03` §1.1: `scale = W.abs().max() / 127` (int8) or `/7` (int4), `W_q = (W/scale).round().clamp(qmin, qmax)`, with a `scale == 0` guard for all-zero weights. `quantize_model` walks `named_modules()`/`named_children()` and swaps every `nn.Linear` via `setattr` on the parent. `benchmark.py`: `measure_tokens_per_sec` (wall clock over `n_steps` greedy-decode steps, `warmup_steps` discarded, no cache) and `measure_tokens_per_sec_cached` (same measurement through a `KVCache`, for the A4 block) and `compute_perplexity` (`exp` of mean cross-entropy over non-overlapping windows of `eval_corpus_path` — **not** `eval/holdout/`). Output via `eval.harness.write_json_atomic` to `eval/results/ch1_benchmark.json`: `{mode: {tokens_per_sec, perplexity, n_steps}}` plus `kv_cache`, `seed`, `checkpoint_path`, `device`. Plots via matplotlib `Agg`. README updated via `eval.report.update_readme_section` under a new `<!-- CH1_BENCHMARK_START/END -->` marker pair (added to `README.md` by hand, as the tool requires).

## Tasks

- **A5-T1** — `quantize.py`: `QuantizedLinear` fp16/int8 + `quantize_model` + tests. **Done.**
- **A5-T2** — int4 nibble packing/unpacking + tests. **Done.**
- **A5-T3** — `BenchmarkConfig` + both benchmark YAMLs + config tests. **Done.**
- **A5-T4** — `benchmark.py`: timing, perplexity, JSON output (including the A4 `kv_cache` block). **Done.**
- **A5-T5** — both plots + the README section. **Done** — verified end-to-end against the A3 smoke checkpoint (`fp32` 3385 tok/s → `int4` 1793 tok/s, KV-cache 1.28x on a 5 KB corpus / 50-step model); the README's `CH1_BENCHMARK` section was left as a placeholder rather than committed with these smoke numbers, since an undertrained toy model's perplexity (~856) is not a result worth publishing — the real table comes from the full-config GPU run.
