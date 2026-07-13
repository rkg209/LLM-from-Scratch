---
name: from-scratch-guard
description: Rules and sanity checks for writing Chapter 1 transformer code. Use whenever creating or editing anything under ch1_architecture/src/ — attention, MLP, layer norm, embeddings, the training loop, the KV-cache, or quantization. Also use when reviewing ch1 code or debugging why a ch1 model will not learn.
---

# Chapter 1: from scratch means from scratch

The entire point of Chapter 1 is proving the author understands transformers rather than that they can call one. A single convenient import destroys that claim. Everything below is non-negotiable in `ch1_architecture/src/`.

## Banned

| Banned | Write instead |
|---|---|
| `import transformers` (anything from HF) | the module by hand |
| `nn.Transformer`, `nn.TransformerEncoder*` | `TransformerBlock` in `model/block.py` |
| `nn.MultiheadAttention` | `MultiHeadSelfAttention` in `model/attention.py` |
| `F.scaled_dot_product_attention` | explicit `torch.matmul` → scale → mask → softmax → `matmul` |
| `nn.LayerNorm` | `LayerNorm` in `model/norm.py`: `(x - mean) / sqrt(var + eps) * gamma + beta` |
| `bitsandbytes` | `quantize.py` — absmax quantization in plain PyTorch |
| `tokenizers` / `sentencepiece` | the BPE merge loop in `tokenizer.py` |

Allowed: `torch` tensors and autograd, `nn.Module`, `nn.Linear`, `nn.Embedding`, `nn.Parameter`, `F.gelu`, `F.softmax`, `F.cross_entropy`, and the optimizer.

`nn.Linear` is allowed because a matmul plus a bias is not the thing being demonstrated. Attention, normalization, the cache, and quantization are.

## Shape-check every tensor operation

Shape bugs in attention are silent — the code runs, the loss goes down a little, and the model is subtly wrong. Broadcasting will hide a transposed head dimension for an entire training run.

Annotate every step with the shape it produces, using `B` (batch), `T` (query positions), `S` (key positions — differs from `T` when a cache is live), `D` (`d_model`), `H` (heads), `Dh` (`D // H`):

```python
q = self.w_q(x).view(B, T, H, Dh).transpose(1, 2)   # [B, H, T, Dh]
scores = q @ k.transpose(-2, -1) / math.sqrt(Dh)    # [B, H, T, S]
scores = scores.masked_fill(mask[:T, :S], float("-inf"))
```

Then assert them in `tests/test_shapes.py`. Every module gets a test that feeds a known input shape and asserts the output shape — that is the cheapest bug-catcher in the chapter.

## Overfit a tiny batch before believing anything

**A model that cannot memorize eight sequences is broken, and no amount of training will fix it.** Before claiming any module works, drive the loss to near zero on a handful of examples:

```
loss on a batch of ≤ 8 sequences must fall below 0.1 within 100 steps
```

This is `tests/test_overfit.py`, and it is the single most valuable test in the package. If it fails, the bug is in the model, not the hyperparameters. The usual culprits, in the order they occur:

1. The causal mask is inverted — the model can see the future, or nothing at all.
2. Labels are not shifted by one, so the model is predicting the token it was just given.
3. A `view` where a `transpose` was meant, silently scrambling the head dimension.
4. The residual connection was dropped, so gradients cannot reach the early layers.
5. Weight tying was applied with the wrong transpose.

## KV-cache correctness is an identity, not a vibe

The cache is an optimization; it must change *nothing* about the output. The test is exact:

```
generate(prompt, use_cache=True) and generate(prompt, use_cache=False)
produce identical logits at every step (atol=1e-5)
```

If they differ, the cache is wrong, and any speedup number measured from it is meaningless. Fix the cache before you benchmark it. Common cause: the positional embedding is computed from the length of the *current* input (always 1 in the cached path) instead of the absolute position in the sequence.

## Benchmark honestly

Speed numbers are the deliverable, so measure them the way a skeptic would:

- Discard the first ~10 steps as warm-up. Never include model construction or the first CUDA sync in the timing.
- Same seed, same prompt, same `max_new_tokens` across every variant compared.
- Report tokens/sec *and* perplexity at each quantization level. A speedup that ruins quality is a finding, not a failure — report the tradeoff curve, not just the good end of it.
