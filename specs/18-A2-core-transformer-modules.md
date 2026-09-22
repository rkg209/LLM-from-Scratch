# A2 — Core transformer modules

| | |
|---|---|
| **State** | done |
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

- **File layout.** `model/{__init__,norm,embeddings,mlp,attention,block,gpt}.py`, per `planning/02-architecture.md` §2.1 — locked, not revisited.
- **`cache`/`layer_idx` threading.** `MultiHeadSelfAttention.forward` and `TransformerBlock.forward` accept `cache`/`layer_idx`, defaulting to `None`, unused until A4. This means A4 only adds a KV-cache branch inside `attention.py`; it never has to touch the block/model call signature.
- **Mask slicing.** `masked_fill` uses `causal_mask[seq_offset : seq_offset + T, :S]` rather than a literal `[:T, :S]` — the offset generalizes correctly to the cached decode step (`T=1`, `S=t+1`, offset `= cache.current_len`) without any change to this line in A4.

## Technical plan

See `.claude/plans/AfterOperation.md` §A2. Shape vocabulary: `B` batch, `T` query positions, `S` key positions, `D` `d_model`, `H` heads, `Dh` `D // H`. `LayerNorm`: `gamma`/`beta` as `nn.Parameter`s (ones/zeros), `unbiased=False` variance. `TokenEmbedding` wraps `nn.Embedding` and exposes `.weight` for tying; `PositionalEmbedding` precomputes a sinusoidal table as a buffer, `forward(T, offset=0)` slices `[offset:offset+T]` — the `offset` argument A4 needs, added now so A4 never reopens this file. `MLP`: `Linear(D,4D) → gelu → Linear(4D,D)`, the 4x ratio a module constant. `MultiHeadSelfAttention`: causal mask built once as a `[max_seq_len, max_seq_len]` bool buffer; explicit `matmul → scale → masked_fill → softmax → matmul`. `TransformerBlock`: pre-norm residuals. `GPTModel`: embeddings, `nn.ModuleList` of blocks, final norm, tied head via `x @ token_emb.weight.T` (kept outside `quantize_model` in A5 since it is not an `nn.Linear`).

## Tasks

- **A2-T1** — `norm.py` + `embeddings.py` (with `offset`) + shape tests. **Done.**
- **A2-T2** — `mlp.py` + shape test. **Done.**
- **A2-T3** — `attention.py` with `cache`/`layer_idx` accepted and unused; shape tests + causal-mask test. **Done.**
- **A2-T4** — `block.py` + `gpt.py` (pre-norm, weight tying, `get_num_params`) + shape tests. **Done.**
- **A2-T5** — `tests/test_overfit.py`. **Done** — loss < 0.1 within 100 steps at `lr=3e-3` on an 8-sequence random batch.
