# system-design.md

> **Status:** v1.0 — Derived from `planning/02-architecture.md`. All names, types, and constraints are consistent with the locked architecture. This document adds the *how* (internal workflows, state machines, patterns) that the architecture document leaves implicit.

---

## Table of Contents

1. [Modules](#1-modules)
2. [Services](#2-services)
3. [Internal Workflows](#3-internal-workflows)
4. [Event Flows](#4-event-flows)
5. [State Transitions](#5-state-transitions)
6. [Design Patterns](#6-design-patterns)
7. [Integration Points](#7-integration-points)

---

## 1. Modules

This section catalogs every module in the monorepo, its public interface, its internal structure, and the invariants it must maintain.

---

### 1.1 Chapter 1 Modules (`ch1_architecture/src/ch1_architecture/`)

#### `config.py` — `ConfigLoader`

```
ConfigLoader
├── load_config(path: Path) → GPTConfig          # public entry point
└── _validate(raw: dict) → None                  # raises if any key missing
```

**`GPTConfig` dataclass (frozen):**

```python
@dataclass(frozen=True)
class GPTConfig:
    vocab_size:  int
    d_model:     int
    n_heads:     int
    n_layers:    int
    seq_len:     int
    batch_size:  int
    max_steps:   int
    seed:        int
    device:      str          # "cpu" | "cuda"
    run_name:    str          # used as W&B run name
```

**Invariants:**
- `d_model % n_heads == 0` — enforced in `_validate`.
- No field has a Python-side default; every value must be present in the YAML or `load_config` raises `KeyError`.
- `load_config` is the only constructor path; direct dataclass instantiation is not used outside tests.

---

#### `tokenizer.py` — `BPETokenizer`

```
BPETokenizer
├── train(corpus: str, vocab_size: int) → None
├── encode(text: str) → list[int]
├── decode(ids: list[int]) → str
├── save(path: Path) → None
└── load(path: Path) → BPETokenizer          # classmethod
```

**Internal state:**
- `vocab: dict[int, bytes]` — id → byte sequence.
- `merges: list[tuple[bytes, bytes]]` — ordered merge rules.
- `encoder: dict[bytes, int]` — reverse of vocab.

**Invariants:**
- `decode(encode(text)) == text` for any text in the training distribution (round-trip property; tested in `test_tokenizer.py`).
- `train()` must be called before `encode()`/`decode()`; raises `RuntimeError` otherwise.
- No dependency on the `tokenizers` library; merge loop is a plain Python `for` loop over pair frequencies.

---

#### `data.py` — `CorpusDataset` + `make_dataloader`

```
CorpusDataset(torch.utils.data.Dataset)
├── __init__(token_ids: list[int], seq_len: int)
├── __len__() → int
└── __getitem__(idx: int) → tuple[Tensor, Tensor]   # (input_ids, labels)

make_dataloader(dataset: CorpusDataset, config: GPTConfig) → DataLoader
```

**Sliding-window logic:**
- `__getitem__(i)` returns `(token_ids[i : i+seq_len], token_ids[i+1 : i+seq_len+1])`.
- Labels are the input shifted right by one; cross-entropy loss is computed over the full sequence.

---

#### `model/embeddings.py` — `TokenEmbedding`, `PositionalEmbedding`

```
TokenEmbedding(nn.Module)
└── forward(x: Tensor[B, T]) → Tensor[B, T, D]

PositionalEmbedding(nn.Module)          # sinusoidal, not learned
└── forward(T: int) → Tensor[1, T, D]  # broadcast-ready
```

**Sinusoidal formula (locked):**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
Pre-computed in `__init__`, registered as a buffer (not a parameter).

---

#### `model/attention.py` — `MultiHeadSelfAttention`

```
MultiHeadSelfAttention(nn.Module)
├── __init__(d_model: int, n_heads: int)
├── forward(
│     x:     Tensor[B, T, D],
│     cache: KVCache | None = None,
│     layer_idx: int | None = None
│   ) → Tensor[B, T, D]
└── _scaled_dot_product(
      q: Tensor[B, H, T, Dh],
      k: Tensor[B, H, S, Dh],
      v: Tensor[B, H, S, Dh],
      mask: Tensor[T, S]
    ) → Tensor[B, H, T, Dh]
```

**Implementation constraints:**
- Uses explicit `torch.matmul` (not `F.scaled_dot_product_attention`).
- Causal mask is a lower-triangular boolean tensor created once in `__init__` and registered as a buffer.
- When `cache` is not `None`: calls `cache.update(layer_idx, k, v)` to retrieve the full key/value history before computing attention.
- Head split: `D → H × Dh` via `view` + `transpose`; merged back via `transpose` + `contiguous` + `view`.

---

#### `model/mlp.py` — `PositionwiseMLP`

```
PositionwiseMLP(nn.Module)
└── forward(x: Tensor[B, T, D]) → Tensor[B, T, D]
```

Internal: `Linear(D, 4D) → GELU → Linear(4D, D)`. The 4× expansion factor is hardcoded (not configurable) per standard GPT-2 architecture.

---

#### `model/norm.py` — `LayerNorm`

```
LayerNorm(nn.Module)
└── forward(x: Tensor[...]) → Tensor[...]
```

Hand-written: `(x - mean) / sqrt(var + eps) * gamma + beta`. Does **not** use `nn.LayerNorm`. `gamma` and `beta` are `nn.Parameter` tensors initialized to ones and zeros respectively.

---

#### `model/block.py` — `TransformerBlock`

```
TransformerBlock(nn.Module)
└── forward(
      x:         Tensor[B, T, D],
      cache:     KVCache | None = None,
      layer_idx: int | None = None
    ) → Tensor[B, T, D]
```

**Residual pre-norm layout (locked):**
```
x = x + Attention(LayerNorm(x), cache, layer_idx)
x = x + MLP(LayerNorm(x))
```

---

#### `model/gpt.py` — `GPTModel`

```
GPTModel(nn.Module)
├── __init__(config: GPTConfig)
├── forward(
│     input_ids: Tensor[B, T],
│     cache:     KVCache | None = None
│   ) → Tensor[B, T, V]          # logits over vocab
└── get_num_params() → int
```

**Forward pass:**
```
token_emb  = TokenEmbedding(input_ids)          # [B, T, D]
pos_emb    = PositionalEmbedding(T)             # [1, T, D]
x          = token_emb + pos_emb
for i, block in enumerate(self.blocks):
    x = block(x, cache=cache, layer_idx=i)
logits = LayerNorm(x) @ token_embedding.weight.T   # weight tying
```

Weight tying: the output projection shares weights with `TokenEmbedding.weight` (standard GPT-2 practice; reduces parameters).

---

#### `kv_cache.py` — `KVCache`

```
KVCache
├── __init__(n_layers: int, max_seq_len: int, n_heads: int, head_dim: int)
├── update(layer_idx: int, k: Tensor[B,H,1,Dh], v: Tensor[B,H,1,Dh])
│         → tuple[Tensor[B,H,S,Dh], Tensor[B,H,S,Dh]]   # full history
├── reset() → None
└── current_len: int                                       # property
```

**Internal storage:** Two `list[Tensor]` of length `n_layers`; each tensor is pre-allocated to `[B, H, max_seq_len, Dh]` and filled via slice assignment. `current_len` tracks the write pointer.

**Invariants:**
- `reset()` must be called between sequences; `generate.py` calls it at the start of each new generation.
- `update()` raises `IndexError` if `current_len >= max_seq_len`.
- The returned tensors are slices of the pre-allocated buffer (zero-copy).

---

#### `quantize.py` — `QuantizedLinear`, `quantize_model`

```
QuantizedLinear(nn.Module)
├── __init__(weight: Tensor, bias: Tensor | None, mode: str)
└── forward(x: Tensor) → Tensor

quantize_model(model: GPTModel, mode: str) → GPTModel
    # mode ∈ {"fp16", "int8", "int4"}
    # in-place replacement of nn.Linear layers
```

**Quantization schemes:**

| Mode | Storage dtype | Compute dtype | Scale factor |
|------|--------------|---------------|--------------|
| `fp16` | `torch.float16` | `torch.float16` | none |
| `int8` | `torch.int8` | `torch.float32` | per-tensor absmax |
| `int4` | packed `torch.uint8` | `torch.float32` | per-tensor absmax |

**Absmax formula:**
```
scale = max(abs(W)) / 127          # int8
scale = max(abs(W)) / 7            # int4
W_q   = round(W / scale).clamp(min, max)
```

Dequantization on forward: `output = F.linear(x, W_q.float() * scale, bias)`.

No `bitsandbytes` dependency; all arithmetic is plain PyTorch.

---

#### `train.py` — training `__main__`

```
main()
├── load_config(args.config) → GPTConfig
├── seed_everything(config.seed)
├── build_tokenizer_and_dataset() → (BPETokenizer, DataLoader)
├── build_model(config) → GPTModel
├── build_optimizer(model, config) → AdamW
├── build_scheduler(optimizer, config) → CosineAnnealingLR
├── wandb.init(project="ch1", name=config.run_name, config=asdict(config))
├── training_loop(model, loader, optimizer, scheduler, config)
└── save_checkpoint(model, "outputs/ch1/model.pt")
```

**Training loop invariants:**
- Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.
- LR schedule: cosine decay from `lr_max` to `lr_min=lr_max/10` over `max_steps`.
- W&B logs: `{step, train_loss, perplexity, lr, tokens_per_sec}` every 10 steps.
- Checkpoint saved every 500 steps and at end of training.

---

#### `generate.py` — `generate`

```
generate(
    model:      GPTModel,
    prompt_ids: list[int],
    max_new:    int,
    use_cache:  bool = True,
    temperature: float = 1.0,
    top_k:      int = 50,
) → list[int]
```

**Cached path:** Prefill all prompt tokens in one forward pass (no cache), then generate one token at a time using `KVCache`.

**Uncached path:** Re-run the full sequence on every step (for correctness verification).

**Invariant (tested in `test_kv_cache.py`):** For the same prompt and seed, `generate(..., use_cache=True)` and `generate(..., use_cache=False)` produce identical logits at every step (atol=1e-5).

---

#### `benchmark.py` — `__main__`

```
main()
├── load_model_from_checkpoint("outputs/ch1/model.pt")
├── for mode in ["fp32", "fp16", "int8", "int4"]:
│     q_model = quantize_model(model.copy(), mode)
│     tps     = measure_tokens_per_sec(q_model, n_steps=100)
│     ppl     = compute_perplexity(q_model, test_slice)
│     results[mode] = {"tokens_per_sec": tps, "perplexity": ppl}
├── write_json(results, "eval/results/ch1_benchmark.json")
└── plot_results(results)   # writes speedup_curve.png, perplexity_tradeoff.png

measure_tokens_per_sec(model, n_steps: int) → float
    # wall-clock time / total tokens; seeded; warm-up 10 steps discarded

compute_perplexity(model, token_ids: list[int]) → float
    # exp(mean cross-entropy over non-overlapping windows)
```

---

### 1.2 Chapter 2 Modules (`ch2_adaptation/src/ch2_adaptation/`)

#### `config.py` — `ConfigLoader` (Ch2)

Same pattern as Ch1. Produces `FinetuneConfig`:

```python
@dataclass(frozen=True)
class FinetuneConfig:
    model_tag:       str      # "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    lora_r:          int      # 4 (smoke) | 16 (full)
    lora_alpha:      int      # 8 (smoke) | 32 (full)
    lora_dropout:    float
    target_modules:  list[str]
    max_steps:       int
    batch_size:      int
    learning_rate:   float
    use_4bit:        bool     # False (smoke) | True (full)
    seed:            int
    output_dir:      str
    wandb_project:   str
```

---

#### `schema.py` — `ReviewOutput`

```python
from pydantic import BaseModel, Field
from typing import Literal

class ReviewOutput(BaseModel):
    severity:      Literal["critical", "major", "minor", "info"]
    category:      str = Field(min_length=1, max_length=64)
    line:          int = Field(ge=1)
    issue:         str = Field(min_length=1, max_length=512)
    suggested_fix: str = Field(min_length=1, max_length=1024)

    model_config = ConfigDict(extra="forbid")   # mirrors additionalProperties: false
```

**Startup validation:** On module import, `ReviewOutput.model_json_schema()` is compared structurally against `eval/schema.json`; raises `AssertionError` if they diverge. This makes `schema.py` the single runtime source of truth while `eval/schema.json` remains the canonical file-based contract.

---

#### `data_gen.py` — `__main__`

```
main()
├── load_snippets_from_github()     # via GitHub MCP; returns list[str]
├── estimate_cost(snippets) → float
├── if cost > 1.50: abort("Budget guard: estimated ${cost:.2f} > $1.50")
├── for snippet in snippets:
│     raw = frontier_api.complete(PROMPT_TEMPLATE.format(code=snippet))
│     try:
│         record = ReviewOutput.model_validate_json(raw)
│         write_jsonl(record, "data/train.jsonl")
│     except ValidationError:
│         log_skip(snippet, raw)
├── split_val(source="data/train.jsonl", dest="data/val.jsonl", frac=0.10)
└── write_provenance("data/provenance.json", model, prompt, date, cost, n)
```

**`PROMPT_TEMPLATE` (locked structure):**
```
You are a Java code reviewer. Analyze the following Java code snippet and
identify exactly one bug. Respond with a single JSON object matching this
schema: {schema_json}. Code: {code}
```

**Budget guard logic:**
```
estimated_tokens = sum(len(s.split()) * 1.3 for s in snippets)
estimated_cost   = estimated_tokens / 1000 * PRICE_PER_1K_TOKENS
```
`PRICE_PER_1K_TOKENS` is read from an environment variable, not hardcoded.

---

#### `data_loader.py` — `ReviewDataset`, `format_prompt`

```
format_prompt(record: dict) → str
    # Produces Qwen2.5 chat-template format:
    # <|im_start|>system\n...<|im_end|>\n
    # <|im_start|>user\n{code}<|im_end|>\n
    # <|im_start|>assistant\n{json_output}<|im_end|>

ReviewDataset(torch.utils.data.Dataset)
├── __init__(jsonl_path: Path, tokenizer, config: FinetuneConfig)
├── __len__() → int
└── __getitem__(idx: int) → dict   # {"input_ids", "attention_mask", "labels"}
```

**Label masking:** Tokens in the prompt (system + user turns) have their labels set to `-100` so loss is computed only on the assistant's JSON output.

---

#### `model.py` — `load_base_model`

```
load_base_model(config: FinetuneConfig) → tuple[AutoModelForCausalLM, AutoTokenizer]
```

**Full config path:**
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
```

**Smoke config path:** `from_pretrained(..., torch_dtype=torch.float32, device_map="cpu")` — no `BitsAndBytesConfig`.

---

#### `lora.py` — `apply_lora`

```
apply_lora(model: AutoModelForCausalLM, config: FinetuneConfig) → PeftModel
```

```python
lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=config.lora_dropout,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
return get_peft_model(model, lora_config)
```

After `apply_lora`, `model.print_trainable_parameters()` is called and logged; expected ~0.5% of total parameters are trainable.

---

#### `finetune.py` — `__main__`

```
main()
├── config = load_config(args.config)
├── model, tokenizer = load_base_model(config)
├── model = apply_lora(model, config)
├── train_ds = ReviewDataset("data/train.jsonl", tokenizer, config)
├── val_ds   = ReviewDataset("data/val.jsonl",   tokenizer, config)
├── trainer  = SFTTrainer(
│       model=model,
│       train_dataset=train_ds,
│       eval_dataset=val_ds,
│       data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
│       args=TrainingArguments(
│           output_dir=config.output_dir,
│           max_steps=config.max_steps,
│           per_device_train_batch_size=config.batch_size,
│           learning_rate=config.learning_rate,
│           report_to="wandb",
│           ...
│       ),
│   )
├── wandb.init(project=config.wandb_project)
├── trainer.train()
└── model.save_pretrained("outputs/adapter/")
```

**Invariant:** `finetune.py` never reads from `eval/holdout/`. The leakage guard hook enforces this at the filesystem level.

---

#### `evaluate.py` — `score_model`

```
score_model(
    adapter_path: Path,
    holdout_path: Path,
    config:       FinetuneConfig,
) → EvalResult

_run_inference(model, tokenizer, code: str) → str   # raw model output
```

Delegates metric computation to `eval/harness.py::score_outputs`. Returns an `EvalResult` dataclass and writes it to `eval/results/finetuned.json`.

---

#### `baseline.py` — `__main__`

```
main()
├── score_base_model(holdout_path, config)
│     # zero-shot: loads Qwen2.5-Coder-1.5B-Instruct, no adapter
│     # → baselines.json["base_model"]
└── score_frontier_api(holdout_path)
      # 3-shot: fixed examples prepended to prompt
      # → baselines.json["frontier_api"]
```

Both paths write to `eval/results/baselines.json` atomically (write to `.tmp`, then rename).

---

### 1.3 Chapter 3 Modules (`ch3_operation/`)

#### `config.py` — `ConfigLoader` (Ch3)

Produces `ServeConfig`:

```python
@dataclass(frozen=True)
class ServeConfig:
    model_path:       str    # path or env var MODEL_PATH
    n_ctx:            int    # context window for llama-cpp
    n_threads:        int    # CPU thread count
    max_tokens:       int    # max generation tokens
    temperature:      float
    host:             str
    port:             int
    metrics_window:   int    # rolling window size for drift monitor (default 100)
```

---

#### `model_backend.py` — `LlamaCppBackend`

```
LlamaCppBackend
├── __init__(config: ServeConfig)
│     # loads model once: Llama(model_path, n_ctx=config.n_ctx,
│     #                         n_gpu_layers=0, n_threads=config.n_threads)
├── generate(prompt: str, max_tokens: int) → str
│     # calls self._llm(prompt, max_tokens=max_tokens, temperature=config.temperature)
│     # returns completion text only (strips prompt echo)
└── _llm: llama_cpp.Llama   # private; loaded at __init__
```

**Invariants:**
- `n_gpu_layers=0` is hardcoded, not configurable. CPU-only is a hard constraint.
- Model is loaded once at process startup; `generate()` is stateless with respect to model weights.
- Thread safety: `llama-cpp-python` is not thread-safe; a `threading.Lock` wraps every `generate()` call.

---

#### `schema.py` — `ReviewOutput` (Ch3)

Identical Pydantic model to `ch2_adaptation/schema.py`. Both import from `eval/schema.json` for startup validation. They are separate files (not a shared package) to keep chapters independent; the JSON Schema file is the shared contract.

---

#### `validator.py` — `validate_output`

```
validate_output(raw: str) → ReviewOutput
    # Attempt 1: json.loads(raw) → ReviewOutput.model_validate(parsed)
    # Attempt 2: _extract_from_fence(raw) → ReviewOutput.model_validate(parsed)
    # Fail:      raise ValidationError(detail=...)

_extract_from_fence(raw: str) → dict
    # regex: ```json\n(.*?)\n``` (DOTALL)
    # raises ValueError if no fence found
```

**Repair strategy is strictly two attempts, no more.** The validator never generates or hallucinates missing fields; it only parses what the model returned.

---

#### `metrics.py` — `PrometheusMetrics`

```
PrometheusMetrics
├── __init__()
│     # registers:
│     #   request_latency_seconds: Histogram(buckets=[0.1,0.5,1,2,5,10,30])
│     #   requests_total:          Counter(labels=["status"])   # status: success|error
│     #   schema_validity_rate:    Gauge()
├── record_request(latency_s: float, valid: bool) → None
└── registry: CollectorRegistry   # passed to /metrics endpoint
```

---

#### `drift.py` — `RollingQualityMonitor`

```
RollingQualityMonitor
├── __init__(window: int, metrics: PrometheusMetrics)
│     # self._window = deque(maxlen=window)   # thread-safe via collections.deque
├── record(valid: bool) → None
│     # appends valid to deque
│     # rate = sum(deque) / len(deque)
│     # metrics.schema_validity_rate.set(rate)
└── current_rate: float   # property
```

**Thread safety:** `collections.deque` with `maxlen` is used; individual `append` and `len` operations are atomic in CPython. No explicit lock needed for the deque itself; the Gauge update is also atomic.

---

#### `main.py` — FastAPI application

```
app = FastAPI(title="Java Code Reviewer")

# Startup
@app.on_event("startup")
async def startup():
    config  = load_config(os.getenv("CONFIG_PATH", "configs/full.yaml"))
    backend = LlamaCppBackend(config)
    monitor = RollingQualityMonitor(config.metrics_window, metrics)
    # stored in app.state

# Routes
POST /review
    body:    ReviewRequest(code: str, context: str)
    returns: ReviewOutput | ErrorResponse

GET /health
    returns: {"status": "ok", "model_loaded": bool}

GET /metrics
    returns: Prometheus text format (text/plain; version=0.0.4)
```

---

#### `middleware.py` — `RequestTimingMiddleware`

```
RequestTimingMiddleware(BaseHTTPMiddleware)
└── dispatch(request, call_next)
      # t0 = time.perf_counter()
      # response = await call_next(request)
      # latency = time.perf_counter() - t0
      # stored in request.state.latency_s for use by route handler
```

Latency is attached to `request.state` rather than recorded directly in middleware, so the route handler can associate it with the `valid/error` outcome before calling `metrics.record_request`.

---

### 1.4 Shared Eval Modules (`eval/`)

#### `harness.py` — `score_outputs`

```
score_outputs(
    outputs:      list[str],
    holdout_path: Path,
    schema_path:  Path = Path("eval/schema.json"),
) → EvalResult

_load_holdout(path: Path) → list[dict]
_validate_against_schema(output: str, schema: dict) → bool
_check_bug_caught(output: ReviewOutput, ground_truth: dict) → bool
    # line match: abs(output.line - gt.line) <= 2
```

**`EvalResult` dataclass:**
```python
@dataclass
class EvalResult:
    schema_validity_rate: float
    bug_catch_rate:       float
    n_samples:            int
    n_valid:              int
    n_caught:             int
    per_sample:           list[dict]   # {id, valid, caught, raw_output}
```

#### `metrics.py` — `compute_schema_validity`, `compute_bug_catch_rate`

```
compute_schema_validity(outputs: list[str], schema: dict) → float
compute_bug_catch_rate(outputs: list[ReviewOutput], holdout: list[dict]) → float
```

These are pure functions with no side effects; they are called by `harness.py::score_outputs`.

---

### 1.5 Claude Code Hook Modules (`.claude/hooks/`)

#### `leakage_guard.py`

```
# PreToolUse hook on Read, Write, Edit, Glob, Grep
# Blocks any tool call whose path argument resolves under eval/holdout/
# unless the calling command is /eval or the calling agent is experiment-analyst

def check(tool_name: str, tool_input: dict) -> HookResult:
    path = tool_input.get("path", "")
    if "eval/holdout" in path and not _is_eval_context():
        return HookResult.block("Leakage guard: eval/holdout/ is read-only outside /eval")
    return HookResult.allow()
```

#### `gpu_budget_guard.py`

```
# PreToolUse hook on Bash
# Intercepts commands that would launch training (python finetune.py, python train.py)
# Checks WANDB run count and estimated GPU-hours against a budget

def check(tool_name: str, tool_input: dict) -> HookResult:
    cmd = tool_input.get("command", "")
    if _is_training_command(cmd):
        hours = _estimate_gpu_hours(cmd)
        if hours > GPU_BUDGET_HOURS:
            return HookResult.block(f"GPU budget guard: estimated {hours:.1f}h > {GPU_BUDGET_HOURS}h")
    return HookResult.allow()
```

#### `auto_format_smoke.py`

```
# PostToolUse hook after Write, Edit
# Runs: ruff check --fix <file> && black <file>
# If the file is in a package with a smoke test, runs: pytest <pkg>/tests/ -k smoke -x -q
```

#### `commit_hygiene.py`

```
# PreToolUse hook on git commit
# Blocks commit if:
#   - any file in eval/holdout/ is staged
#   - any .py file has ruff errors
#   - commit message is fewer than 10 characters
```

---

##