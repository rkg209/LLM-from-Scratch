# The quantization tradeoff

Source: `ch1_architecture/src/ch1_architecture/quantize.py`,
`ch1_architecture/src/ch1_architecture/benchmark.py`, `eval/results/ch1_benchmark.json`
and `eval/results/ch1_benchmark_cpu.json`, `.claude/skills/gguf-export/SKILL.md` for the
Chapter 3 tie-in.

## What absmax quantization actually does to a weight tensor

`quantize.py`'s `_quantize_absmax` computes one scale per tensor:

```python
scale = weight.abs().max() / qmax          # qmax = 127 for int8, 7 for int4
w_q = (weight / scale).round().clamp(-qmax - 1, qmax)
```

Every weight is divided by a single per-tensor scale derived from that tensor's single
largest-magnitude value, rounded to the nearest representable integer, and stored as
`int8` (or, for int4, packed two 4-bit two's-complement nibbles per byte via `_pack_int4`
— `(hi << 4) | lo`, unpacked with a sign correction for nibbles `>= 8`). `forward()`
dequantizes on the fly (`weight.float() * scale`) before the matmul — the model computes
in float, it just stores its weights compressed.

The cost this hides: **every weight in the tensor pays for the largest one.** A tensor
with one outlier weight ten times the size of everything else forces every other weight
into a coarser slice of the available integer range, since the scale is fixed by that one
outlier — the far more common case (`GPTModel`'s Linear weights, not adversarially
constructed) has moderate outliers, but the mechanism is the same regardless of degree:
resolution is spent uniformly across the tensor, priced by its extreme, not its typical
value.

## Why speed improves — and the honest counterexample this project measured

The intuitive story ("fewer bits, less compute") is not actually why quantization speeds
up inference here. At batch size 1 — which is what interactive serving looks like — the
model is **memory-bandwidth bound**, not compute bound: the bottleneck is moving weight
tensors from memory to the CPU, not the arithmetic once they're there. Fewer bytes per
weight means fewer bytes moved per forward pass, which is where int8/int4's speedup
actually comes from.

That framing predicts, correctly, the case where quantization does *not* help: `fp16`.
Halving the bytes of a `float32` weight should, by the bandwidth argument, speed things
up too — but there is no fast native CPU matmul path for `float16` the way there is for
`float32` or for the integer paths' dequantize-then-`float32`-matmul. `QuantizedLinear`'s
`fp16` branch (`F.linear(x.half(), self.weight, self.bias).float()`) still has to convert
back before the actual multiply runs on hardware that has no fast fp16 kernel. **This
prediction was written before the run and the run confirmed it**: fp16 measures 0.55x on
CPU (22.2 tok/s against fp32's 40.2) and 0.84x on the A100 — slower on both, precisely
because the "fewer bits -> faster" intuition assumes a fast low-precision compute path
that plain PyTorch doesn't have here. Reported as a finding, not tuned away.

## Why quality degrades, and which capability degrades first

Rounding every weight to a coarser grid is lossy, and the loss compounds through every
layer it passes through — at int4's 4-bit range (`[-8, 7]`), the quantization step size is
16x coarser than int8's, so the same outlier-driven scale problem above bites much harder.

The capability this project expects to degrade **first** is structured output, not
fluency. Valid JSON is a long chain of low-entropy, high-stakes token choices — the next
character after `{"severity": "` is one of four exact enum strings, a missing closing
`}` invalidates the entire object, a `line` field that isn't an integer fails the schema.
Each of those decisions has very little margin: a small logit perturbation that would be
invisible in ordinary prose (a slightly less likely but still fluent word choice) can flip
a low-margin structural decision at exactly the token that determines schema validity.
Fluent-sounding-but-invalid JSON is the expected failure mode of aggressive quantization
on this task, well before the text becomes obviously worse to read — which is exactly why
`ch3_operation`'s GGUF export pipeline re-validates schema-validity rate after quantizing,
not just perplexity (see `.claude/skills/gguf-export/SKILL.md` step 4), and refuses to
ship a quantization level that regresses it.

## This project's measured perplexity delta per precision level

Measured 2026-09-18, PARAM Rudra job 402166, one A100 80GB PCIe, the 21.0M-parameter
checkpoint from the 3000-step run. Perplexity is over 16,287 held-out tokens — the 10%
tail of the corpus, split off before the tokenizer was fitted, so the training run never
saw it. Source: `eval/results/ch1_benchmark.json`.

| Mode | tokens/sec | vs fp32 | perplexity | vs fp32 |
|---|---|---|---|---|
| fp32 | 237.4 | — | 61.45 | — |
| fp16 | 200.2 | 0.84x | 61.45 | +0.00% |
| int8 | 219.8 | 0.93x | 61.70 | +0.42% |
| int4 | 107.6 | 0.45x | 84.11 | +36.9% |

**On this device, every mode is slower than fp32** — the prediction above about fp16 held,
and held for int8 and int4 too. int4 is the clearest case: 2.2x slower for a 37%
perplexity increase, a loss on both axes at once. The CPU section below shows int8
reversing, which is the more useful half of the result.

The cause is this implementation, not the technique. `QuantizedLinear` stores absmax-
quantized weights and dequantizes them inside `forward`, in ordinary PyTorch ops, because
the chapter's premise is that nothing here comes from `bitsandbytes` or a fused kernel.
So every matmul pays a dequantization pass that fp32 never pays, and int4 additionally
pays bit-unpacking. What quantization buys in this implementation is memory; what it
costs is latency. On an A100 — where memory was never the binding constraint — that is a
straight loss.

## The same checkpoint on CPU: int8 is 1.51x FASTER

Job 402179, identical weights and code, only `device` changed. Source:
`eval/results/ch1_benchmark_cpu.json`.

| Mode | A100 tok/s | CPU tok/s | CPU vs fp32 | perplexity |
|---|---|---|---|---|
| fp32 | 237.4 | 40.2 | — | 61.45 |
| fp16 | 200.2 | 22.2 | 0.55x | 61.45 |
| int8 | 219.8 | **60.5** | **1.51x** | 61.70 |
| int4 | 107.6 | 22.8 | 0.57x | 84.11 |

**int8 reverses.** On the A100 it was a 0.93x loss; on CPU it is a 1.51x win, for the same
+0.42% perplexity. The dequantization cost did not change — what changed is what it is
competing against. A CPU fp32 matmul at this size is bandwidth-bound, so halving the bytes
moved buys more than the extra arithmetic costs. On an A100 with memory to spare, the
arithmetic is pure overhead. Same code, same weights, opposite verdicts: quantization is a
claim about a bottleneck, not about a model.

int4 stays a loss on both (0.45x and 0.57x) because 4-bit values must be unpacked before
any arithmetic can touch them, and the unpacking is per-forward-pass Python work that no
amount of saved bandwidth pays for. fp16 is the worst case on CPU (0.55x): CPUs have no
native fp16 arithmetic path here, so it converts to fp32 to compute and pays the
conversion for nothing.

Two things worth taking from that rather than filing it as a failure. First, int8's
quality cost is genuinely negligible (+0.42% perplexity), so the *quality* half of the
technique is confirmed even where the speed half is not: on hardware with int8 tensor
cores and a fused kernel, that trade is the good one. Second, this is the concrete reason
Chapter 3 serves GGUF through `llama.cpp` instead of shipping this code — the same idea,
implemented where the kernels exist, is what makes CPU-only serving viable at all. Having
measured the naive version is what makes that a reasoned choice rather than a borrowed
one.

## What I got wrong

I expected quantization to be a straightforward speed-for-quality trade where every step
down the precision ladder is faster. Writing `QuantizedLinear`'s fp16 path and thinking
through what would actually execute on the benchmark hardware is what surfaced that
`fp16` breaks that assumption on CPU — it costs quality-adjacent complexity (a `.half()` /
`.float()` round-trip) for **no** speed benefit there, because the speed benefit was never
about bit-width in the abstract, it was about bandwidth *and* having a fast compute
kernel for the resulting dtype, and CPU PyTorch only has the second for `float32` and the
integer-then-dequantize paths. It's a reminder that "smaller dtype" and "faster" are
correlated, not synonymous, and the mechanism connecting them matters more than the
correlation.
