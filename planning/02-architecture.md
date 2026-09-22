# architecture.md

> **Status:** v1.0 — Implementation-ready architecture document.
> **Audience:** Engineers building directly from this document. Every decision is locked. Where a choice exists, one option is selected and the rationale is stated once.

---

## 1. System Overview

This system is a three-chapter monorepo that progresses a single narrative: from a hand-built transformer to a fine-tuned specialist to a deployed edge service. The three chapters are **technically independent packages** that share infrastructure (eval harness, schema contract, config loader, Claude Code scaffold) but have no runtime dependency on each other.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    llm-engineering-architecture-to-edge              │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  ch1_architecture│  │  ch2_adaptation  │  │  ch3_operation   │  │
│  │                  │  │                  │  │                  │  │
│  │  Pure-PyTorch    │  │  QLoRA fine-tune │  │  GGUF serve +    │  │
│  │  GPT from        │  │  Qwen2.5-Coder   │  │  FastAPI +       │  │
│  │  scratch         │  │  → Java reviewer │  │  monitor +       │  │
│  │  + KV-cache      │  │  → JSON output   │  │  deploy          │  │
│  │  + quantization  │  │                  │  │                  │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  eval/   schema.json · holdout/ · harness · results/         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  specs/  STATUS.md · per-spec .md files · _template.md       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  .claude/  CLAUDE.md · commands/ · skills/ · agents/         │   │
│  │            hooks/ · settings.json                            │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

**Runtime topology:** Chapter 1 runs entirely locally (CPU or GPU, no network). Chapter 2 training runs on a GPU box (Colab/Kaggle/cluster); its data-generation step calls a frontier API once. Chapter 3 runs as a long-lived HTTP service on CPU (HF Spaces + cloud free tier). There is no shared runtime state between chapters; they communicate only through files (model artifacts, eval results) stored on disk or Hugging Face Hub.

---

## 2. Component Architecture

### 2.1 Chapter 1 — Architecture Package (`ch1_architecture/`)

All components are pure Python + PyTorch. No component may import `transformers` or use `nn.Transformer`.

```
ch1_architecture/
├── configs/
│   ├── smoke.yaml          # CPU, vocab=256, d_model=64, n_heads=2, n_layers=2,
│   │                       # seq_len=64, batch=4, max_steps=50, seed=42
│   └── full.yaml           # GPU, vocab=50257, d_model=512, n_heads=8, n_layers=6,
│                           # seq_len=256, batch=32, max_steps=5000, seed=42
├── src/
│   └── ch1_architecture/
│       ├── __init__.py
│       ├── config.py           # ConfigLoader: reads YAML, returns dataclass
│       ├── tokenizer.py        # BPETokenizer: train(), encode(), decode()
│       ├── data.py             # CorpusDataset, DataLoader factory
│       ├── model/
│       │   ├── __init__.py
│       │   ├── embeddings.py   # TokenEmbedding, PositionalEmbedding (sinusoidal)
│       │   ├── attention.py    # MultiHeadSelfAttention (no nn.MultiheadAttention)
│       │   ├── mlp.py          # PositionwiseMLP (two linear + GELU)
│       │   ├── norm.py         # LayerNorm (hand-written, not nn.LayerNorm)
│       │   ├── block.py        # TransformerBlock = norm→attention→norm→mlp
│       │   └── gpt.py          # GPTModel: embed → N×block → head
│       ├── kv_cache.py         # KVCache: per-layer key/value tensor store
│       ├── quantize.py         # QuantizedLinear: fp32→fp16→int8→4-bit (absmax)
│       ├── train.py            # __main__: training loop, W&B logging
│       ├── generate.py         # autoregressive generation, cached + uncached
│       └── benchmark.py        # tokens/sec measurement, perplexity computation
└── tests/
    ├── test_tokenizer.py       # round-trip encode/decode
    ├── test_shapes.py          # shape assertions for every module
    ├── test_overfit.py         # overfit-a-tiny-batch: loss < 0.1 in 100 steps
    ├── test_kv_cache.py        # logit identity: cached == uncached (atol=1e-5)
    └── test_quantize.py        # quantized inference runs; output shape correct
```

**Component responsibilities:**

| Component | Responsibility | Key constraint |
|---|---|---|
| `config.py` | Parse YAML into a frozen dataclass; expose `load_config(path)` | No defaults in code; every value from YAML |
| `tokenizer.py` | Minimal BPE: character-level merge loop; `train(corpus)`, `encode(str)→List[int]`, `decode(List[int])→str` | Pure Python; no `tokenizers` library |
| `attention.py` | Scaled dot-product attention with causal mask; accepts optional `KVCache` | Hand-written `einsum` or explicit matmul; no `F.scaled_dot_product_attention` |
| `kv_cache.py` | Stores past K/V tensors per layer; `update(layer_idx, k, v)→(k_full, v_full)` | Stateful; reset between sequences |
| `quantize.py` | `quantize_model(model, mode)` where mode ∈ {fp16, int8, int4}; absmax quantization for int8/int4 | In-place weight replacement; no `bitsandbytes` |
| `train.py` | AdamW optimizer, cosine LR schedule, gradient clipping; W&B `init`/`log`/`finish` | Reads config; never hardcodes hyperparams |
| `benchmark.py` | Measures wall-clock tokens/sec over N=100 generation steps; computes perplexity on a held-out slice | Deterministic; seeded |

### 2.2 Chapter 2 — Adaptation Package (`ch2_adaptation/`)

```
ch2_adaptation/
├── configs/
│   ├── smoke.yaml      # CPU, max_steps=5, batch=1, lora_r=4, no 4-bit (CPU compat)
│   └── full.yaml       # GPU, max_steps=1000, batch=4, lora_r=16, bnb_4bit=true
├── data/               # gitignored; populated by data_gen.py
│   ├── train.jsonl     # generated synthetic pairs
│   ├── val.jsonl       # 10% split of synthetic
│   └── provenance.json # model used, prompt, date, cost, record count
├── src/
│   └── ch2_adaptation/
│       ├── __init__.py
│       ├── config.py           # ConfigLoader (shared pattern from ch1)
│       ├── schema.py           # Pydantic model mirroring eval/schema.json
│       ├── data_gen.py         # __main__: calls frontier API, writes train.jsonl
│       ├── data_loader.py      # HF datasets wrapper; tokenize + format prompt
│       ├── model.py            # load_base_model(): BitsAndBytesConfig + AutoModelForCausalLM
│       ├── lora.py             # apply_lora(): LoraConfig, get_peft_model()
│       ├── finetune.py         # __main__: SFTTrainer setup, train(), save_adapter()
│       ├── evaluate.py         # score_model(): runs model on eval set, returns metrics dict
│       └── baseline.py         # __main__: scores base model + frontier API few-shot
└── tests/
    ├── test_schema.py          # Pydantic validation: valid/invalid examples
    ├── test_data_loader.py     # tokenization output shape; prompt format
    ├── test_smoke_finetune.py  # smoke config runs end-to-end on CPU
    └── test_evaluate.py        # harness returns correct metric keys
```

**Component responsibilities:**

| Component | Responsibility | Key constraint |
|---|---|---|
| `schema.py` | Pydantic `ReviewOutput` model with fields: `severity: Literal["critical","major","minor","info"]`, `category: str`, `line: int`, `issue: str`, `suggested_fix: str` | Single source of truth; imports `eval/schema.json` to validate at startup |
| `data_gen.py` | Calls frontier API with a fixed prompt template; writes JSONL; validates each record against schema; writes `provenance.json` | Budget guard: aborts if estimated cost > $1.50; never reads `eval/holdout/` |
| `model.py` | Loads `Qwen/Qwen2.5-Coder-1.5B-Instruct` with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)` in full config; loads in fp32 in smoke config | Exact model tag is locked here |
| `lora.py` | `LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj","k_proj","o_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")` | r=4 in smoke config |
| `finetune.py` | `SFTTrainer` with `DataCollatorForSeq2Seq`; saves adapter to `outputs/adapter/`; logs to W&B | Never reads `eval/holdout/` |
| `evaluate.py` | Loads model + adapter; runs inference on a dataset; validates each output against schema; returns `{schema_validity_rate, bug_catch_rate, n_samples}` | Shared with ch3 via `eval/harness.py` |
| `baseline.py` | Scores base model (zero-shot) and frontier API (3-shot) on the same stub set; writes to `eval/results/baselines.json` | Runs before any fine-tuning |

**Base model selection (locked):**
- **Model tag:** `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- **Rationale:** Instruct variant provides instruction-following capability needed for structured JSON output; 1.5B fits in Colab free tier with 4-bit quantization; mature QLoRA recipes exist.
- **Smoke config override:** Load in fp32, no bitsandbytes (CPU compatibility); use `lora_r=4`.

### 2.3 Chapter 3 — Operation Package (`ch3_operation/`)

```
ch3_operation/
├── configs/
│   ├── smoke.yaml      # model_path=tests/fixtures/tiny.gguf (committed test fixture)
│   └── full.yaml       # model_path loaded from env MODEL_PATH
├── api/
│   └── ch3_operation/
│       ├── __init__.py
│       ├── config.py           # ConfigLoader
│       ├── model_backend.py    # LlamaCppBackend: wraps llama-cpp-python
│       ├── schema.py           # Pydantic ReviewOutput (same fields as ch2)
│       ├── validator.py        # validate_output(): schema check + repair strategy
│       ├── main.py             # FastAPI app: POST /review, GET /health, GET /metrics
│       ├── metrics.py          # PrometheusMetrics: latency histogram, counter, gauge
│       └── middleware.py       # RequestTimingMiddleware
├── monitoring/
│   └── ch3_operation/
│       ├── drift.py            # RollingQualityMonitor: schema_validity over window=100
│       └── exporter.py         # expose /metrics in Prometheus text format
├── docker/
│   ├── Dockerfile              # multi-stage; final stage: python:3.11-slim
│   ├── .dockerignore
│   └── entrypoint.sh
└── tests/
    ├── fixtures/
    │   └── tiny.gguf           # minimal valid GGUF for smoke tests (committed)
    ├── test_api_smoke.py       # /health returns 200; /review returns valid schema
    ├── test_validator.py       # malformed output → 422; valid output → 200
    ├── test_metrics.py         # /metrics contains expected metric names
    └── test_docker.py          # docker build succeeds (skipped in CI unless DOCKER=1)
```

**Component responsibilities:**

| Component | Responsibility | Key constraint |
|---|---|---|
| `model_backend.py` | Wraps `llama_cpp.Llama`; exposes `generate(prompt: str, max_tokens: int) → str`; loads model once at startup | CPU-only; `n_gpu_layers=0` always |
| `validator.py` | `validate_output(raw: str) → ReviewOutput`: attempt JSON parse → Pydantic validation → if fail, attempt single repair (extract JSON from markdown fence) → if still fail, raise `ValidationError` | Never returns invalid output to caller |
| `main.py` | `POST /review` accepts `{"code": str, "context": str}`; calls backend; validates; records metrics; returns `ReviewOutput` or `{"error": str, "code": 422}` | Startup loads model; no per-request model load |
| `metrics.py` | `request_latency_seconds` (Histogram, buckets=[0.1,0.5,1,2,5,10,30]), `requests_total` (Counter, labels=[status]), `schema_validity_rate` (Gauge, rolling window) | Prometheus client library: `prometheus-client` |
| `drift.py` | Maintains a deque of length 100; computes `valid_count/100` after each request; updates the gauge | Thread-safe deque |

### 2.4 Shared Eval Harness (`eval/`)

```
eval/
├── schema.json             # canonical JSON Schema (Draft 7)
├── harness.py              # score_outputs(outputs, holdout_path) → EvalResult
├── metrics.py              # compute_schema_validity(), compute_bug_catch_rate()
├── holdout/                # SACRED — leakage guard blocks all non-eval access
│   ├── manifest.json       # provenance: sources, dedup method, freeze date
│   └── holdout.jsonl       # frozen eval set (synthetic-clean + mined-real)
└── results/
    ├── baselines.json      # base model + API few-shot scores
    ├── finetuned.json      # fine-tuned model scores
    └── plots/              # speedup_curve.png, perplexity_tradeoff.png, eval_table.png
```

**`eval/schema.json` (complete, locked):**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ReviewOutput",
  "type": "object",
  "required": ["severity", "category", "line", "issue", "suggested_fix"],
  "additionalProperties": false,
  "properties": {
    "severity": {
      "type": "string",
      "enum": ["critical", "major", "minor", "info"]
    },
    "category": {
      "type": "string",
      "minLength": 1,
      "maxLength": 64
    },
    "line": {
      "type": "integer",
      "minimum": 1
    },
    "issue": {
      "type": "string",
      "minLength": 1,
      "maxLength": 512
    },
    "suggested_fix": {
      "type": "string",
      "minLength": 1,
      "maxLength": 1024
    }
  }
}
```

**`harness.py` interface:**
```python
@dataclass
class EvalResult:
    schema_validity_rate: float   # fraction of outputs passing schema.json
    bug_catch_rate: float         # fraction of holdout bugs identified (line ± 2)
    n_samples: int
    n_valid: int
    n_caught: int
    per_sample: list[dict]        # {id, valid, caught, raw_output}

def score_outputs(
    outputs: list[str],
    holdout_path: Path,
    schema_path: Path = Path("eval/schema.json"),
) -> EvalResult: ...
```

### 2.5 Claude Code Scaffold (`.claude/`)

```
.claude/
├── settings.json               # MCP server wiring, tool permissions, hook registration
├── commands/
│   ├── specify.md              # /specify <name>
│   ├── clarify.md              # /clarify
│   ├── plan.md                 # /plan <spec>
│   ├── tasks.md                # /tasks <spec>
│   ├── implement.md            # /implement <spec> <task>
│   ├── smoke.md                # /smoke <pkg>
│   ├── eval.md                 # /eval
│   └── checkpoint.md           # /checkpoint
├── skills/
│   ├── from-scratch-guard.md
│   ├── qlora-recipe.md
│   ├── eval-table.md
│   ├── gguf-export.md
│   └── spec-review.md
├── agents/
│   ├── spec-reviewer.md
│   ├── code-reviewer.md
│   ├── data-curator.md
│   ├── experiment-analyst.md
│   └── explorer.md
└── hooks/
    ├── gpu_budget_guard.py     # PreToolUse on Bash
    ├── leakage_guard.py        # PreToolUse on Read/Write/Edit
    ├── auto_format_smoke.py    # PostToolUse after file edits
    ├── commit_hygiene.py       # PreToolUse on git commit
    └── session_greeter.py      # SessionStart
```

**`settings.json` structure (locked):**
```json
{
  "mcpServers": {
    "huggingface": {
      "url": "https://huggingface.co/mcp",
      "env": { "HF_TOKEN": "${HF_TOKEN}" },
      "enabledTools": [
        "search_models", "search_datasets", "search_papers",
        "get_model", "push_to_hub", "deploy_space", "get_space_status"
      ]
    },
    "wandb": {
      "command": "npx",
      "args": ["-y", "wandb-mcp-server"],
      "env": { "WANDB_API_KEY": "${WANDB_API_KEY}" }
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "${GITHUB_TOKEN}" }
    }
  },
  "hooks": {
    "PreToolUse": [
      ".claude/hooks/gpu_budget_guard.py",
      ".claude/hooks/leakage_guard.py",
      ".claude/hooks/commit_hygiene.py"
    ],
    "PostToolUse": [".claude/hooks/auto_format_smoke.py"],
    "SessionStart": [".claude/hooks/session_greeter.py"]
  },
  "permissions": {
    "allow": ["Bash", "Read", "Write", "Edit", "Glob", "Grep"],
    "deny": []
  }
}
```

---

## 3. Data Flow

### 3.1 Chapter 1 Data Flow

```
[raw text corpus]
        │
        ▼
  tokenizer.py::BPETokenizer.train(corpus)
        │  produces vocab + merge rules
        ▼
  tokenizer.py::BPETokenizer.encode(text) → List[int]
        │
        ▼
  data.py::CorpusDataset
  (sliding window, seq_len from config)
        │
        ▼
  train.py::training loop
  ┌─────────────────────────────────────────┐
  │  for batch in DataLoader:               │
  │    logits = GPTModel(input_ids)         │
  │    loss = cross_entropy(logits, labels) │
  │    loss.backward()                      │
  │    optimizer.step()                     │
  │    wandb.log({loss, perplexity, step})  │
  └─────────────────────────────────────────┘
        │  saves checkpoint: outputs/ch1/model.pt
        ▼
  generate.py::generate(model, prompt, use_cache=False)
  generate.py::generate(model, prompt, use_cache=True)
        │
        ▼
  benchmark.py
  ┌──────────────────────────────────────────────────────┐
  │  for precision in [fp32, fp16, int8, int4]:          │
  │    q_model = quantize_model(model, precision)        │
  │    tps = measure_tokens_per_sec(q_model, n=100)      │
  │    ppl = compute_perplexity(q_model, test_corpus)    │
  │    results[precision] = {tps, ppl}                   │
  └──────────────────────────────────────────────────────┘
        │  writes eval/results/ch1_benchmark.json
        │  writes eval/results/plots/speedup_curve.png
        │  writes eval/results/plots/perplexity_tradeoff.png
```

### 3.2 Chapter 2 Data Flow

```
PHASE A — DATA GENERATION (one-time, ~$1 API spend)

  [Java/Spring buggy code snippets]  ←── mined from GitHub via GitHub MCP
        │
        ▼
  data_gen.py
  ┌──────────────────────────────────────────────────────────────┐
  │  for snippet in buggy_snippets:                              │
  │    response = frontier_api.complete(PROMPT_TEMPLATE+snippet) │
  │    record = parse_json(response)                             │
  │    if schema_valid(record): write to data/train.jsonl        │
  │  write data/provenance.json                                  │
  └──────────────────────────────────────────────────────────────┘
        │  data/train.jsonl (~N schema-valid pairs)
        │  data/val.jsonl   (10% split)
        │  data/provenance.json

PHASE B — BASELINE MEASUREMENT (before any fine-tuning)

  eval/holdout/holdout.jsonl (frozen)
        │
        ├──► baseline.py::score_base_model()
        │         loads Qwen2.5-Coder-1.5B-Instruct (fp32 smoke / 4-bit full)
        │         runs zero-shot inference
        │         → eval/results/baselines.json [base_model entry]
        │
        └──► baseline.py::score_frontier_api()
                  3-shot prompt to frontier API
                  → eval/results/baselines.json [frontier_api entry]

PHASE C — FINE-TUNING (GPU box only)

  data/train.jsonl
        │
        ▼
  data_loader.py::format_prompt(record) → instruction-tuning format
        │
        ▼
  finetune.py::SFTTrainer
  ┌──────────────────────────────────────────────────────────────┐
  │  base model (4-bit NF4) + LoRA adapter (r=16)               │
  │  SFTTrainer.train()                                          │
  │  wandb.log({train_loss, eval_loss, lr, step})               │
  │  save_pretrained("outputs/adapter/")                         │
  └──────────────────────────────────────────────────────────────┘
        │  outputs/adapter/ (gitignored, pushed to HF Hub)

PHASE D — EVALUATION

  eval/holdout/holdout.jsonl (frozen)
        │
        ▼
  evaluate.py::score_model(adapter_path, holdout_path)
        │  uses eval/harness.py
        │  → eval/results/finetuned.json
        │
        ▼
  eval/harness.py::compare_results(baselines.json, finetuned.json)
        │  → eval/results/plots/eval_table.png
        │  → README.md table (updated by /eval command)
```

### 3.3 Chapter 3 Data Flow

```
PHASE A — MODEL PREPARATION (one-time, after C5)

  outputs/adapter/  (LoRA adapter, from Ch2)
        │
        ▼
  scripts/merge_adapter.py
        │  peft.merge_and_unload() → outputs/merged/
        ▼
  scripts/export_gguf.py
        │  llama.cpp convert_hf_to_gguf.py → outputs/model.gguf
        ▼
  scripts/quantize_gguf.py
        │  llama.cpp quantize → outputs/model-Q4_K_M.gguf
        │  validate: eval/harness.py on sample → schema_validity ≥ adapter baseline
        │  push to HF Hub: Qwen2.5-Coder-1.5B-JavaReviewer-Q4_K_M.gguf

PHASE B — SERVING (long-lived process)

  HTTP client
        │  POST /review  {"code": "...", "context": "..."}
        ▼
  main.py::FastAPI
        │
        ├──► middleware.py::RequestTimingMiddleware
        │         records request start time
        │
        ▼
  model_backend.py::LlamaCppBackend.generate(prompt)
        │  llama-cpp-python inference (CPU, n_gpu_layers=0)
        │  returns raw string
        ▼
  validator.py::validate_output(raw_string)
        │
        ├── attempt 1: json.loads(raw) → Pydantic validation
        ├── attempt 2: extract from ```json ... ``` fence → Pydantic validation
        └── fail: raise ValidationError → 422 response
        │
        ▼  (on success)
  metrics.py::record_request(latency_ms, valid=True/False)
        │  updates Prometheus histogram + counter + drift gauge
        ▼
  HTTP response: ReviewOutput JSON  or  {"error": "...", "code": 422}

PHASE C — METRICS SCRAPE

  Prometheus scraper (or manual curl)
        │  GET /metrics
        ▼
  exporter.py::generate_latest()
        │  returns Prometheus text format:
        │    review_request_latency_seconds{quantile="0.5"} ...
        │    review_request_latency_seconds{quantile="0.99"} ...
        │    review_requests_total{status="success"} ...
        │    review_requests_total{status="error"} ...
        │    review_schema_validity_rate ...
```

### 3.4 CI Data Flow

```
git push / pull request
        │
        ▼
  .github/workflows/ci.yml
        │
        ├──► ruff check .
        ├──► black --check .
        ├──► pytest ch1_architecture/tests/ --config configs/smoke.yaml
        ├──► pytest ch2_adaptation/tests/   --config configs/smoke.yaml
        ├──► pytest ch3_operation/tests/    (uses tiny.gguf fixture)
        └──► [NEVER] full training runs
```

---

## 4. Service Boundaries

### 4.1 Boundary Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│  DEVELOPMENT BOUNDARY (local machine / Claude Code session)             │
│                                                                         │
│  ch1_architecture  ←→  filesystem only (no network)                    │
│  ch2_adaptation/data_gen.py  ←→  frontier API (one-time, ~$1)          │
│  ch2_adaptation/baseline.py  ←→  frontier API (one-time, eval only)    │
│  .claude/ hooks/commands  ←→  MCP servers (HF, W&B, GitHub)            │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  GPU TRAINING BOUNDARY (Colab / Kaggle / cluster — manual launch only) │
│                                                                         │
│  ch2_adaptation/finetune.py  ←→  HF Hub (model download)               │
│                              ←→  W&B (metrics upload)                  │
│                              ←→  HF Hub (adapter upload after train)   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  SERVING BOUNDARY (HF Spaces CPU / cloud free tier)                    │
│                                                                         │
│  ch3_operation/api  ←→  HTTP clients (POST /review)                    │
│                     ←→  Prometheus scraper (GET /metrics)              │
│                     ←→  HF Hub (model download at container start)     │
│  NO outbound calls during inference                                     │
└─────────────────────────────────