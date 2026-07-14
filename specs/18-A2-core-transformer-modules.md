# A2 — Core transformer modules

| | |
|---|---|
| **State** | draft |
| **Depends on** | A1 |
| **Requirements** | FR-10, CON-5, NFR-9, NFR-11 |
| **W&B run** | — |

## Problem

This is the spec the whole first chapter exists for. Importing `nn.Transformer` and importing `nn.MultiheadAttention` produce the same model and prove nothing; writing scaled dot-product attention by hand — the scale, the causal mask, the head split, the softmax, the merge back — is what separates "I have used transformers" from "I understand them." The tokens/sec and quantization results later in the chapter are only *interesting* because the thing being optimized is a model the author actually wrote.

Shape bugs here are the real risk. They do not crash; they train to a mediocre loss and produce a model that is quietly wrong.

## Scope

Hand-written, in pure PyTorch: `TokenEmbedding`, sinusoidal `PositionalEmbedding`, `MultiHeadSelfAttention`, `PositionwiseMLP`, `LayerNorm`, `TransformerBlock`, and `GPTModel` (pre-norm residual layout, weight-tied output head).

## Acceptance criteria

1. None of `transformers`, `nn.Transformer`, `nn.MultiheadAttention`, `nn.LayerNorm`, or `F.scaled_dot_product_attention` appears anywhere in `ch1_architecture/src/` — verified by a grep-based test, not by inspection (FR-10a/b, CON-5).
2. Attention is explicit `torch.matmul` → scale by `1/sqrt(Dh)` → causal mask → softmax → `matmul`.
3. `LayerNorm` computes `(x - mean) / sqrt(var + eps) * gamma + beta` with `gamma`/`beta` as `nn.Parameter`s.
4. A shape test per module asserts the documented output shape at documented input dimensions: `[B,T,D] → [B,T,D]` for the block, `[B,T] → [B,T,V]` for the model (FR-10c).
5. **Overfit-a-tiny-batch**: loss falls below 0.1 on a batch of ≤ 8 sequences within 100 steps (FR-10d). This is the spec's central test — a model that cannot memorize eight sequences is broken.
6. The causal mask is verified directly: position `t` attends to positions `≤ t` and to no position `> t`.
7. `d_model % n_heads == 0` is enforced at config load, not assumed.

## Out of scope

- KV-cache → **A4** (though `MultiHeadSelfAttention.forward` takes the `cache`/`layer_idx` parameters now, so A4 does not have to reopen this file).
- Quantization → **A5**. The training loop → **A3**.
- Attention variants (GQA, MQA, flash). One correct implementation, hand-written.

## Clarifications

*(filled by `/clarify`)*

## Technical plan

*(filled by `/plan`)*

## Tasks

*(filled by `/tasks`)*
