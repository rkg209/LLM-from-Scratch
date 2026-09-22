# Plan — Chapter 1 (Architecture): A1 → A5

> Covers `specs/17-A1` … `specs/21-A5`. Written against the repo as of `2906b8d` (O5 T1 done).
> Companion to `.claude/plans/Operation.md`. Per-spec plans still get pasted into each spec's
> **Technical plan** section before `/tasks <ID>`; this file is the umbrella design so the five
> specs are built as one coherent model instead of five local decisions.

---

## Context

Chapter 1 is the only chapter that is still a stub. `specs/STATUS.md` says it plainly:
*"Chapter 1 is unstarted: `train.py` is a config-loading stub with no model code behind it."*
Everything under `ch1_architecture/src/ch1_architecture/` today is F2's scaffold — `config.py`
(a 13-field frozen `GPTConfig`) and a `train.py` whose `main()` prints what it *would* train.

The project's headline claim is "from architecture to edge." Chapters 2 and 3 are largely
built — a QLoRA reviewer and a served, quantized, monitored endpoint — but they are both
*applications of* libraries. Chapter 1 is the part that proves the author understands what
those libraries do: a hand-rolled BPE, hand-written attention, a hand-written training loop,
a KV-cache whose speedup is measured, and quantization whose quality cost is published next
to its speedup. Nothing in Chapters 2/3 depends on it — it is a parallel track — but it is in
the definition of done, and X1 (README) and X3 (interview notes) both block on A5.

**Intended outcome:** `outputs/ch1/model.pt` is a GPT the author wrote from an empty file that
generates recognizable English; `eval/results/ch1_benchmark.json` plus
`eval/results/plots/speedup_curve.png` and `perplexity_tradeoff.png` report tokens/sec and
perplexity across fp32 → fp16 → int8 → int4, with the cache speedup alongside — including any
mode where quality got worse.

### Decisions taken up front

| # | Decision | Why |
|---|---|---|
| **D-1** | The corpus is a **committed public-domain text file** at `ch1_architecture/data/corpus.txt` (~200–500 KB, TinyShakespeare slice), path read from config. A ~5 KB slice at `ch1_architecture/tests/fixtures/corpus_smoke.txt` serves the smoke config and tests. | Every other smoke path in this repo runs offline; a network fetch in `data.py` would make CI and `/smoke ch1` flaky. Well under the commit-hygiene 5 MB limit, and `.gitignore` only excludes `outputs/` and `*.pt`, so a `.txt` commits cleanly. |
| **D-2** | Training-path fields join **`GPTConfig`**; A5's benchmark gets its own frozen **`BenchmarkConfig`** with `configs/benchmark_smoke.yaml` / `benchmark_full.yaml`. | Matches ch2/ch3, which use one dataclass per purpose (`FinetuneConfig`, `EvalConfig`, `ExportConfig`). Keeps benchmark-only keys out of the train YAMLs — which matters because `eval.config.load_config` raises `KeyError` on **unknown** keys, not just missing ones. |
| **D-3** | The model lives in a **`model/` subpackage** (`embeddings.py`, `attention.py`, `mlp.py`, `norm.py`, `block.py`, `gpt.py`) exactly as `planning/02-architecture.md` §2.1 locks it. | The file tree is already locked in an accepted planning doc. Deviating buys nothing. |
| **D-4** | `MultiHeadSelfAttention.forward` and `TransformerBlock.forward` take `cache` / `layer_idx` **in A2**, defaulting to `None` and unused until A4. | A2's own out-of-scope note asks for this so A4 never reopens `attention.py`. |
| **D-5** | One grep-based **`tests/test_purity.py`** created in A1 with the *full* ban list (tokenizers, sentencepiece, transformers, `nn.Transformer`, `nn.MultiheadAttention`, `nn.LayerNorm`, `F.scaled_dot_product_attention`, bitsandbytes), scanning every `.py` under `ch1_architecture/src/`. | A1 AC-6 and A2 AC-1 are the same test with different rows. Writing it once, first, means the guard exists before the code it guards. It passes trivially on an empty tree — that is fine, it is a ratchet. |
| **D-6** | `KVCache` keeps the locked 4-arg signature `(n_layers, max_seq_len, n_heads, head_dim)` and **allocates its buffers lazily on the first `update()`**, taking `B` from the incoming tensor. | `planning/03` specifies buffers of shape `[B, H, max_seq_len, Dh]` but gives `__init__` no batch argument. Lazy allocation resolves the contradiction without changing a locked signature. `reset()` zeroes `current_len`; it does not free buffers. |
| **D-7** | int4 stores **two nibbles packed per `torch.uint8`**, unpacked and dequantized on forward. | `planning/03` §1.1 says "packed `torch.uint8`". Storing int4 values one-per-byte would make the memory claim a lie, and memory is half the point of the mode. |
| **D-8** | `benchmark.py` reads a checkpoint path **from `BenchmarkConfig`**, and its tests build a tiny model in-memory rather than depending on `outputs/ch1/model.pt`. | `outputs/` is gitignored and no checkpoint exists in CI. The tests must not depend on a training run having happened. |
| **D-9** | The BPE merge loop runs **once** and its result is cached to `outputs/ch1/tokenizer.json`; `train.py` loads the cache when it exists and the corpus hash matches. | A3 AC-1 is a hard 120 s CPU budget, and A1 explicitly declines to optimize the merge loop. Training a 256-entry vocab on a 5 KB corpus every smoke run would spend that budget on the wrong thing. |
| **D-10** | fp16 on CPU is expected to be **slower** than fp32 and will be reported as such. | PyTorch CPU fp16 matmul is not a fast path. A5's whole thesis is that the tradeoff curve gets published as measured; a mode that loses is a finding. The full benchmark is a GPU-box run where the number is meaningful. |

### Constraints that shape everything below

- `ch1_architecture/src/` is **pure PyTorch**. Allowed: `nn.Module`, `nn.Linear`, `nn.Embedding`,
  `nn.Parameter`, `F.gelu`, `F.softmax`, `F.cross_entropy`, autograd, the optimizer. Everything
  in D-5's ban list is not. `.claude/skills/from-scratch-guard/SKILL.md` is the authority.
- Every number comes from YAML. **Adding a config key means editing `smoke.yaml`, `full.yaml`,
  and `tests/test_gpt_config.py` in the same commit** — the loader rejects both missing and
  unknown keys.
- Type hints on every function; functions under ~40 lines; a test per new module.
- The GPU-budget hook blocks any Bash command naming `configs/full.yaml` or `--max_steps ≥ 100`.
  Every full-path command in this plan is a **manual, out-of-session** GPU-box run.
- `eval/results/` is one-writer-per-file: `ch1_benchmark.json` and `plots/` belong to
  `benchmark.py` and nothing else.
- One task = one commit; append a `progress_report.md` entry per task; update `specs/STATUS.md`
  when a spec changes state. **Never** a `Co-Authored-By:` trailer.

---

## A1 — Tokenizer + data pipeline

**Files:** `data/corpus.txt` *(new)*, `tests/fixtures/corpus_smoke.txt` *(new)*,
`src/ch1_architecture/tokenizer.py` *(new)*, `src/ch1_architecture/data.py` *(new)*,
`src/ch1_architecture/config.py` *(edit)*, both `configs/*.yaml` *(edit)*,
`tests/test_purity.py`, `tests/test_tokenizer.py`, `tests/test_data.py` *(new)*.

`BPETokenizer` holds `vocab: dict[int, bytes]`, `merges: list[tuple[bytes, bytes]]`,
`encoder: dict[bytes, int]`, and a `_trained: bool`. `train(corpus, vocab_size)` seeds the vocab
with all 256 byte values, then repeatedly counts adjacent pair frequencies and merges the most
frequent until `len(vocab) == vocab_size` — a plain Python loop, deliberately unoptimized.
Seeding with all 256 bytes is what makes the round-trip property hold for bytes never seen in
training (A1 AC-1); `decode` goes id → bytes → `bytes.decode("utf-8")`.

`encode`/`decode`/`save` raise `RuntimeError` when `_trained` is false (AC-3). `save` writes JSON
with byte sequences hex-encoded (JSON has no bytes type); `load` is a classmethod that restores
`_trained = True`. Round-tripping identical ids after save/load is AC-4.

`CorpusDataset(torch.utils.data.Dataset)` stores `token_ids` and `seq_len`;
`__getitem__(i)` returns `(ids[i:i+seq_len], ids[i+1:i+seq_len+1])` as `torch.long` tensors, and
`__len__` is `len(ids) - seq_len` so the shifted slice never runs off the end.
`make_dataloader(dataset, config)` wraps it with `batch_size` from config, `shuffle=True`, and a
`torch.Generator` seeded from `config.seed` so AC-7 and NFR-4 both hold.

**Config additions:** `corpus_path: str`, `tokenizer_path: str`.

### Tasks

- **A1-T1** — `tests/test_purity.py` (full ban list, globs `src/**/*.py`) + `data/corpus.txt` +
  `tests/fixtures/corpus_smoke.txt` + the two `GPTConfig` fields, both YAMLs, and the
  `test_gpt_config.py` update. *One commit, because the loader makes config changes atomic.*
- **A1-T2** — `tokenizer.py`: `train` / `encode` / `decode`, `RuntimeError` guards; round-trip
  tests over ASCII, non-ASCII (emoji, accents), and whitespace runs.
- **A1-T3** — `save` / `load` + the reload-encodes-identically test.
- **A1-T4** — `data.py`: `CorpusDataset`, `make_dataloader`, shift-by-one and shape tests.

---

## A2 — Core transformer modules

**Files:** `src/ch1_architecture/model/{__init__,norm,embeddings,mlp,attention,block,gpt}.py`
*(new)*, `tests/test_shapes.py`, `tests/test_attention.py`, `tests/test_overfit.py` *(new)*.

Shape annotations use the skill's vocabulary — `B` batch, `T` query positions, `S` key positions
(differs from `T` once a cache is live), `D` `d_model`, `H` heads, `Dh` `D // H` — on every line
that changes a shape.

- **`norm.py`** — `LayerNorm(d_model, eps)`: `gamma`/`beta` as `nn.Parameter` (ones/zeros),
  `(x - mean) / sqrt(var + eps) * gamma + beta` over the last dim with `unbiased=False`.
  `eps` is a constructor argument with the standard `1e-5` at the call site, not a bare literal.
- **`embeddings.py`** — `TokenEmbedding` wraps `nn.Embedding` and exposes `.weight` for tying.
  `PositionalEmbedding` precomputes the sinusoidal table in `__init__`, `register_buffer`s it
  (not a parameter), and `forward(T, offset=0)` returns `[1, T, D]` sliced at `offset:offset+T`.
  **The `offset` argument is what A4 needs** and is the single most common cache bug — added now.
- **`mlp.py`** — `Linear(D, 4D) → F.gelu → Linear(4D, D)`. The 4× is a module constant per
  `planning/03`, not a config key.
- **`attention.py`** — `MultiHeadSelfAttention(d_model, n_heads, max_seq_len)`. `w_q/w_k/w_v/w_o`
  are `nn.Linear`. Causal mask built once in `__init__` as a `[max_seq_len, max_seq_len]` bool
  buffer. `forward(x, cache=None, layer_idx=None)`: project → `view(B,T,H,Dh).transpose(1,2)` →
  (A4: `cache.update`) → `_scaled_dot_product` → `transpose(1,2).contiguous().view(B,T,D)` → `w_o`.
  `_scaled_dot_product` is explicit `matmul` → `/ sqrt(Dh)` → `masked_fill(mask[:T,:S], -inf)` →
  `F.softmax` → `matmul`. The mask is sliced `[:T, :S]` from the start so the cached path (where
  `T=1`, `S=t+1`) needs no change in A4.
- **`block.py`** — pre-norm: `x = x + attn(ln1(x), cache, layer_idx)`; `x = x + mlp(ln2(x))`.
- **`gpt.py`** — `GPTModel(config)`: embeddings, `nn.ModuleList` of blocks, final `LayerNorm`,
  and a tied head computed as `x @ token_emb.weight.T`. `forward(input_ids, cache=None)` passes
  `layer_idx=i` per block and derives the positional offset from `cache.current_len` when a cache
  is live. `get_num_params()` counts tied parameters once.

**Tests.** `test_shapes.py` asserts documented output shapes per module (`[B,T,D]→[B,T,D]` for
the block, `[B,T]→[B,T,V]` for the model). `test_attention.py` verifies the causal mask directly:
feed a sequence, perturb the token at position `t+1`, and assert the output at position `t` is
unchanged — position `t` attends to `≤ t` and nothing above. `test_overfit.py` is the spec's
central test: 8 random sequences, ≤ 100 AdamW steps, `assert loss < 0.1`. When it fails, read the
five suspects in `from-scratch-guard` in order — inverted mask, unshifted labels, `view` where a
`transpose` was meant, dropped residual, wrong transpose on the tied head.

`d_model % n_heads == 0` (AC-7) is already enforced in `GPTConfig.__post_init__`; A2 adds no
new check, only a test asserting the model constructor trusts it.

### Tasks

- **A2-T1** — `norm.py` + `embeddings.py` (with `offset`) + shape tests.
- **A2-T2** — `mlp.py` + shape test.
- **A2-T3** — `attention.py` with the `cache`/`layer_idx` parameters accepted and unused; shape
  tests and the causal-mask test.
- **A2-T4** — `block.py` + `gpt.py` (pre-norm, weight tying, `get_num_params`) + shape tests.
- **A2-T5** — `tests/test_overfit.py`.

---

## A3 — Custom training loop

**Files:** `src/ch1_architecture/train.py` *(rewrite)*, `src/ch1_architecture/generate.py`
*(new)*, `config.py` + both YAMLs *(edit)*, `tests/test_train.py`, `tests/test_generate.py` *(new)*.

`train.py` replaces the scaffold body while keeping its `parse_args`/`main` shape. `main()` stays
under 40 lines by delegating to `build_tokenizer_and_loader`, `build_model`, `build_optimizer`,
`build_scheduler`, `training_loop`, and `save_checkpoint`.

- **Optimizer/schedule** — `torch.optim.AdamW(lr=config.learning_rate)` with a `LambdaLR` that
  ramps linearly over `warmup_steps` then cosine-decays to `learning_rate * lr_min_ratio`.
  A hand-written lambda rather than `CosineAnnealingLR` because warmup and `eta_min` together
  are clearer as four lines than as two schedulers.
- **Step** — forward, `F.cross_entropy(logits.view(-1, V), labels.view(-1))`, backward,
  `clip_grad_norm_(max_norm=config.clip_grad_norm)`, `optimizer.step()`, `scheduler.step()`.
- **W&B** — inline and config-gated, copying `ch2_adaptation/finetune.py:79-105`. There is no
  shared helper in this repo. Every `config.log_every` steps, log
  `{step, train_loss, perplexity, lr, tokens_per_sec}` where `perplexity = exp(loss)` and
  `tokens_per_sec` is `batch_size * seq_len / step_wall_time`. Every `config.sample_every` steps,
  generate and log a text sample so AC-7's judgment is on the record. `wandb_mode: disabled` in
  smoke, `offline` when `WANDB_API_KEY` is absent, `online` in full — the smoke path never
  touches the network.
- **Checkpoint** — `outputs/ch1/model.pt` every `config.ckpt_every` steps and at the end, holding
  `{model_state, optimizer_state, config (as dict), step, tokenizer_vocab, tokenizer_merges}` —
  enough to reload and generate without the corpus (AC-5). Write via a temp file and `os.replace`
  so an interrupted save cannot corrupt the previous checkpoint.
- **`generate.py`** — `generate(model, prompt_ids, max_new, use_cache=True, temperature, top_k)`.
  A3 implements only the `use_cache=False` path: re-run the whole sequence each step, take
  `logits[:, -1, :]`, apply temperature, top-k filter, `torch.multinomial`. A private
  `_generate_uncached` returns per-step logits alongside the tokens — A4's identity test needs
  logits, not sampled tokens, and retrofitting that later would mean touching this file twice.

**Config additions:** `warmup_steps: int`, `lr_min_ratio: float`, `ckpt_every: int`,
`sample_every: int`, `sample_prompt: str`, `max_new_tokens: int`, `temperature: float`,
`top_k: int`, `checkpoint_path: str`, `wandb_project: str`, `wandb_mode: str`.

**Tests** run 2–3 steps on a toy config and assert: loss is finite and decreases; the checkpoint
round-trips (save → load → identical logits); the LR curve matches the closed-form cosine at
sampled steps; the same seed twice gives a bit-identical loss list (AC-6); `top_k=1` at
`temperature` → deterministic greedy output.

**The 120-second budget (AC-1)** is the risk here. D-9's tokenizer cache is the main lever; if
the smoke run is still slow, cut `max_steps` before cutting `n_layers` — the number of steps is
the cheapest honest knob.

### Tasks

- **A3-T1** — the config fields, both YAMLs, `test_gpt_config.py` update.
- **A3-T2** — tokenizer caching (`outputs/ch1/tokenizer.json` keyed by corpus hash) +
  `build_tokenizer_and_loader`.
- **A3-T3** — the training loop: optimizer, scheduler, clipping, loss/perplexity, determinism test.
- **A3-T4** — checkpointing (atomic write, round-trip test).
- **A3-T5** — `generate.py` uncached path + top-k/temperature tests.
- **A3-T6** — W&B wiring + periodic sample logging; smoke run under 120 s recorded in
  `progress_report.md`.

---

## A4 — KV-cache

**Files:** `src/ch1_architecture/kv_cache.py` *(new)*,
`src/ch1_architecture/model/attention.py`, `model/gpt.py`, `generate.py` *(edit)*,
`tests/test_kv_cache.py` *(new)*.

`KVCache(n_layers, max_seq_len, n_heads, head_dim)` holds `_k: list[Tensor|None]`,
`_v: list[Tensor|None]`, and `current_len: int`. On the first `update(layer_idx, k, v)` it
allocates `[B, H, max_seq_len, Dh]` per layer from the incoming `k`'s batch and dtype (D-6),
slice-assigns at `[..., current_len : current_len + k.size(2), :]`, and returns
`_k[layer_idx][..., :new_len, :]` — a view, zero-copy. `current_len` advances **once per token,
after the last layer writes**, not once per layer; getting that wrong makes layer 5 read a
window layer 0 never wrote. `update` raises `IndexError` when the write would exceed
`max_seq_len` (AC-5). `reset()` sets `current_len = 0` and keeps the buffers (AC-4).

`attention.forward` gains its cached branch: when `cache is not None`, call
`cache.update(layer_idx, k, v)` and attend `q` (`T=1`) against the full `[B,H,S,Dh]` history.
The mask slice `mask[:T, :S]` already written in A2 is correct here — with `T=1` and `S=t+1` the
single query row is all-visible, which is what a cached decode step means.

`gpt.forward` passes `offset=cache.current_len` to `PositionalEmbedding` when a cache is live.
**This is AC-6 and the classic bug**: the cached path's input length is always 1, so a position
derived from it would embed every generated token as position 0. The A2 `offset` parameter exists
for exactly this.

`generate.py` gains `_generate_cached`: `cache.reset()`, one prefill forward over the whole
prompt (which fills the cache), then one-token steps. Sampling is shared with the uncached path
so the two differ only in how the logits were produced.

**Tests.** AC-1 is the gate: run both paths with the same seed and prompt, collect per-step
logits, `assert (cached - uncached).abs().max() < 1e-5`. **If this fails, no speedup number from
this spec may be reported** — fix the cache first. Then: `reset()` reuse produces the same output
as a fresh cache; `update` past `max_seq_len` raises `IndexError`; the returned tensor is a view
of the buffer, not a copy.

**Measurement** (AC-2/AC-3) reuses A5's timing helper, so tokens/sec with and without the cache
is written by `benchmark.py` into `eval/results/ch1_benchmark.json` under a `kv_cache` key. Same
seed, prompt, `max_new_tokens`; first 10 steps discarded as warm-up. On a smoke-sized model over
a short sequence the speedup may be small or negative — the effect is quadratic in sequence
length, so the number that goes in the README comes from the full-config GPU run.

### Tasks

- **A4-T1** — `kv_cache.py` + unit tests (lazy alloc, write pointer, `reset`, `IndexError`, view).
- **A4-T2** — the cached branch in `attention.py` and the positional offset in `gpt.py`.
- **A4-T3** — `_generate_cached` + **the logit-identity test**.
- **A4-T4** — cached-vs-uncached tokens/sec into `ch1_benchmark.json` (writer is `benchmark.py`).

---

## A5 — Quantization

**Files:** `src/ch1_architecture/quantize.py`, `src/ch1_architecture/benchmark.py`,
`configs/benchmark_{smoke,full}.yaml` *(new)*, `config.py` *(edit — `BenchmarkConfig`)*,
`tests/test_quantize.py`, `tests/test_benchmark.py` *(new)*, `README.md` *(edit)*.

`QuantizedLinear(weight, bias, mode)` stores quantized weights as **buffers, not parameters** —
this is post-training quantization for inference; nothing here is trained.

| mode | storage | forward |
|---|---|---|
| `fp16` | `weight.half()` | `F.linear(x.half(), w, b).float()` |
| `int8` | `torch.int8` + scalar `scale` | `F.linear(x, w.float() * scale, b)` |
| `int4` | packed `torch.uint8` (two nibbles/byte) + `scale` | unpack → `float() * scale` → `F.linear` |

Absmax per `planning/03` §1.1: `scale = W.abs().max() / 127` (int8) or `/ 7` (int4);
`W_q = (W / scale).round().clamp(qmin, qmax)`. Guard `scale == 0` for an all-zero weight.
int4 packing stores `(hi << 4) | lo` over the flattened, zero-padded-to-even weight, with the
original shape kept as a buffer so unpack can restore it.

`quantize_model(model, mode)` walks `named_modules()` and swaps every `nn.Linear` for a
`QuantizedLinear` via `setattr` on the parent. **`mode="fp32"` returns the model untouched** —
the benchmark needs a baseline row and a no-op branch is cleaner than a special case at the call
site. The tied output head is *not* an `nn.Linear` (it is a matmul against the embedding weight),
so it is untouched — worth a sentence in the write-up, since it means the head stays fp32 in
every mode.

`benchmark.py` takes `--config`, loads a checkpoint at `BenchmarkConfig.checkpoint_path`, and for
each mode in `config.modes` deep-copies the fp32 model, quantizes, then measures
`measure_tokens_per_sec` (wall clock over `n_steps` generation steps, first `warmup_steps`
discarded) and `compute_perplexity` (`exp` of mean cross-entropy over non-overlapping windows of
a held-out corpus slice — **not** `eval/holdout/`, which is Chapter 2's Java holdout and is
hook-blocked anyway). Same seed, prompt, and step count across every mode (AC-7).

Output goes through `eval.harness.write_json_atomic` to `eval/results/ch1_benchmark.json`:
`{mode: {tokens_per_sec, perplexity, n_steps}}` plus `seed`, `checkpoint_path`, `device`, and the
A4 `kv_cache` block. Plots use matplotlib (already in the `ch1` extra) with the `Agg` backend,
writing `eval/results/plots/speedup_curve.png` and `perplexity_tradeoff.png`.

`README.md` gets both plots and a table, via `eval.report.update_readme_section` so it stays
idempotent. **Every mode is reported, including the ones that got worse** — fp16-on-CPU being
slower (D-10) and int4 perplexity climbing are the interesting rows, not embarrassments.

**Tests.** Round-trip: quantize a known weight, dequantize, assert error is within the mode's
quantization step. int4 pack/unpack is exactly invertible. Every mode runs and returns
`[B,T,V]`-shaped logits. int8 perplexity on a tiny model is within a sane factor of fp32 — a
mode that produces `nan` or garbage is a bug, not a tradeoff. `test_purity.py` already covers the
`bitsandbytes` ban (AC-3). Benchmark tests build a tiny model in-memory (D-8) and write to `tmp_path`.

### Tasks

- **A5-T1** — `quantize.py`: `QuantizedLinear` fp16/int8 + `quantize_model` + tests.
- **A5-T2** — int4 nibble packing/unpacking + tests.
- **A5-T3** — `BenchmarkConfig` + both benchmark YAMLs + config tests.
- **A5-T4** — `benchmark.py`: timing, perplexity, JSON output (including A4's `kv_cache` block).
- **A5-T5** — both plots + the README section.

---

## Verification

**Per task**, the standing gate:

```bash
uv run ruff check . && uv run black --check . && uv run pytest -q
```

**The five checks that actually decide whether Chapter 1 is real** — in order, because each is
meaningless if the one above it fails:

```bash
# 1. purity — the whole chapter's claim, one grep
uv run pytest ch1_architecture/tests/test_purity.py -q

# 2. shapes — the cheapest bug-catcher
uv run pytest ch1_architecture/tests/test_shapes.py -q

# 3. the model can learn at all
uv run pytest ch1_architecture/tests/test_overfit.py -q          # loss < 0.1 in <= 100 steps

# 4. the cache changes nothing
uv run pytest ch1_architecture/tests/test_kv_cache.py -q          # max |delta logits| < 1e-5

# 5. quantized inference runs in every mode
uv run pytest ch1_architecture/tests/test_quantize.py -q
```

**End-to-end on CPU, in session** (must finish under 120 s — time it):

```bash
time uv run python -m ch1_architecture.train --config ch1_architecture/configs/smoke.yaml
uv run python -m ch1_architecture.benchmark --config ch1_architecture/configs/benchmark_smoke.yaml
```

Then confirm `outputs/ch1/model.pt` exists and reloads, `eval/results/ch1_benchmark.json` has a
row per mode plus the `kv_cache` block, and both PNGs exist under `eval/results/plots/`.

**Out of session, on the GPU box** (`ALLOW_FULL_RUN=1`, the hook blocks these from a session):
the full train config, then the full benchmark. The full run is what produces the numbers that
go in the README and the W&B run IDs that go in `specs/STATUS.md` — AC-7 of A3 (samples are
recognizably English) and the meaningful KV-cache speedup both need it, since neither is
observable at smoke size.

## Sequencing note

A1 → A2 → A3 → A4 → A5 is a hard chain; there is no parallelism to exploit inside Chapter 1.
Nothing in Chapters 2 or 3 blocks on any of it, so this track can be interleaved with C1–C5 and
O1–O6 freely. X1 (README) and X3 (interview notes) block on A5.

Per CLAUDE.md, all five specs are still `draft` with empty **Clarifications** sections. The
workflow before writing code is `/clarify <ID>` → human accepts → `/plan <ID>` (paste that spec's
section of this file into its **Technical plan**) → `/tasks <ID>` → `loop(/implement)`. This file
is the umbrella design, not a substitute for that sequence.
