# A5 — Quantization (INT8 / 4-bit)

| | |
|---|---|
| **State** | draft |
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

*(filled by `/clarify`)*

## Technical plan

*(filled by `/plan`)*

## Tasks

*(filled by `/tasks`)*
