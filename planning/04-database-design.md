# database-design.md

> **Status:** v1.0 — Derived from `planning/02-architecture.md` and `planning/03-system-design.md`. All entity names, field types, and constraints are consistent with the locked architecture. This document specifies the persistent data model, ownership boundaries, and access patterns for every module in the monorepo.

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Storage Topology](#2-storage-topology)
3. [Entity Catalog](#3-entity-catalog)
4. [Relational Schema](#4-relational-schema)
5. [Data Ownership Map](#5-data-ownership-map)
6. [Access Patterns](#6-access-patterns)
7. [Multi-Tenancy](#7-multi-tenancy)
8. [Integrity Constraints and Invariants](#8-integrity-constraints-and-invariants)
9. [Migration and Evolution Strategy](#9-migration-and-evolution-strategy)
10. [Rationale Log](#10-rationale-log)

---

## 1. Design Philosophy

This system has **no shared operational database**. The three chapters are technically independent packages that communicate only through files. The "database" is therefore a set of well-defined file-based stores, each with a clear owner, schema, and access contract.

Four storage tiers are used:

| Tier | Technology | Mutable? | Shared across chapters? |
|------|-----------|----------|------------------------|
| **Artifact store** | Filesystem / HF Hub (binary files: `.pt`, `.gguf`, adapter dirs) | Write-once per run | Via HF Hub only |
| **Record store** | JSONL files (append-only during generation; read-only after freeze) | Append during gen; frozen after | `eval/holdout/` is read by ch2 + ch3 eval |
| **Result store** | JSON files (atomic write; versioned by run) | One writer per file | Read by any chapter for comparison |
| **Operational store** | In-process memory (Prometheus metrics, rolling deque) | Live; ephemeral | No — ch3 only |

No relational database engine (PostgreSQL, SQLite, etc.) is deployed. The rationale is in [§10](#10-rationale-log). The schema defined here is the logical model that the file formats implement; it is enforced by Pydantic validators and JSON Schema at runtime rather than by a database engine.

---

## 2. Storage Topology

```
monorepo root/
│
├── eval/                          ← SHARED READ (ch2 eval, ch3 eval)
│   ├── schema.json                  canonical JSON Schema (read-only, committed)
│   ├── holdout/                     SACRED — leakage-guarded
│   │   ├── manifest.json            provenance record (read-only, committed)
│   │   └── holdout.jsonl            frozen eval set (read-only, committed)
│   └── results/                     SHARED WRITE (one writer per file)
│       ├── ch1_benchmark.json         written by ch1/benchmark.py
│       ├── baselines.json             written by ch2/baseline.py
│       ├── finetuned.json             written by ch2/evaluate.py
│       └── plots/                     written by ch1/benchmark.py, ch2/evaluate.py
│
├── ch2_adaptation/
│   └── data/                      ← OWNED BY ch2/data_gen.py
│       ├── train.jsonl               synthetic training pairs
│       ├── val.jsonl                 10% validation split
│       └── provenance.json           generation audit record
│
├── outputs/                       ← ARTIFACT STORE (write-once per run)
│   ├── ch1/
│   │   └── model.pt                  GPT checkpoint
│   ├── adapter/                      LoRA adapter (pushed to HF Hub)
│   ├── merged/                       merged full model weights
│   └── model-Q4_K_M.gguf             quantized GGUF (pushed to HF Hub)
│
└── ch3_operation/                 ← OPERATIONAL STORE (in-process, ephemeral)
    └── [Prometheus metrics in RAM]
```

---

## 3. Entity Catalog

This section defines every logical entity in the system — what it represents, where it lives, and who owns it.

---

### 3.1 `ReviewRecord`

**What it represents:** A single (buggy Java code snippet, structured review) pair. The fundamental unit of training data and evaluation.

**Lives in:** `ch2_adaptation/data/train.jsonl`, `ch2_adaptation/data/val.jsonl`, `eval/holdout/holdout.jsonl`

**Owner:** `ch2_adaptation/data_gen.py` (training/val); the holdout curator (frozen at project init)

**Fields:**

| Field | Type | Constraints | Source |
|-------|------|-------------|--------|
| `id` | `string` | UUID v4; globally unique across all JSONL files | Generated at write time |
| `code` | `string` | Non-empty; raw Java source snippet | GitHub mining / manual curation |
| `context` | `string` | May be empty string; describes surrounding file/class context | GitHub mining / manual curation |
| `severity` | `string` | `enum["critical","major","minor","info"]` | Frontier API / human label |
| `category` | `string` | `minLength=1, maxLength=64` | Frontier API / human label |
| `line` | `integer` | `minimum=1` | Frontier API / human label |
| `issue` | `string` | `minLength=1, maxLength=512` | Frontier API / human label |
| `suggested_fix` | `string` | `minLength=1, maxLength=1024` | Frontier API / human label |
| `split` | `string` | `enum["train","val","holdout"]` | Set at write time |
| `source` | `string` | `enum["synthetic","mined"]` | Set at write time |
| `created_at` | `string` | ISO 8601 UTC timestamp | Set at write time |

**Notes:**
- The five fields `severity`, `category`, `line`, `issue`, `suggested_fix` are exactly the `ReviewOutput` schema. The `ReviewRecord` is the full training pair; `ReviewOutput` is the label subset.
- `id` is written into every JSONL line so that `per_sample` entries in `EvalResult` can be traced back to their source record.

---

### 3.2 `ReviewOutput`

**What it represents:** The structured output a model is expected to produce for a given code snippet. This is both the training label and the inference output schema.

**Lives in:** Embedded within `ReviewRecord`; also the shape of every model response at inference time.

**Owner:** `eval/schema.json` (canonical definition); `ch2_adaptation/schema.py` and `ch3_operation/schema.py` (runtime enforcement)

**Fields:** Subset of `ReviewRecord` — `severity`, `category`, `line`, `issue`, `suggested_fix`. See §3.1 for types and constraints.

**Relationship to `ReviewRecord`:** `ReviewOutput` is a projection of `ReviewRecord`. Every `ReviewRecord` contains exactly one embedded `ReviewOutput`. There is no separate storage for `ReviewOutput` instances; they are always read as part of a `ReviewRecord` or produced transiently during inference.

---

### 3.3 `DataProvenance`

**What it represents:** An audit record of a single data generation run — what model was called, what prompt was used, when, how much it cost, and how many records were produced.

**Lives in:** `ch2_adaptation/data/provenance.json` (one file per generation run; overwritten on re-run)

**Owner:** `ch2_adaptation/data_gen.py`

**Fields:**

| Field | Type | Constraints |
|-------|------|-------------|
| `run_id` | `string` | UUID v4 |
| `generated_at` | `string` | ISO 8601 UTC |
| `frontier_model` | `string` | e.g. `"claude-3-5-sonnet-20241022"` |
| `prompt_template_hash` | `string` | SHA-256 of `PROMPT_TEMPLATE` string |
| `n_requested` | `integer` | Number of snippets submitted |
| `n_valid` | `integer` | Number passing schema validation |
| `n_skipped` | `integer` | `n_requested - n_valid` |
| `estimated_cost_usd` | `float` | Pre-run estimate |
| `actual_cost_usd` | `float` | Post-run actual (from API response metadata) |
| `train_path` | `string` | Relative path to `train.jsonl` |
| `val_path` | `string` | Relative path to `val.jsonl` |
| `split_seed` | `integer` | Seed used for train/val split |

---

### 3.4 `HoldoutManifest`

**What it represents:** Provenance record for the frozen evaluation set. Committed once; never modified.

**Lives in:** `eval/holdout/manifest.json`

**Owner:** Project maintainer (committed at project initialization; the leakage guard prevents any automated write)

**Fields:**

| Field | Type | Constraints |
|-------|------|-------------|
| `freeze_date` | `string` | ISO 8601 UTC |
| `n_synthetic` | `integer` | Count of synthetic records in holdout |
| `n_mined` | `integer` | Count of mined-real records in holdout |
| `n_total` | `integer` | `n_synthetic + n_mined` |
| `dedup_method` | `string` | Description of deduplication strategy applied |
| `sources` | `array[string]` | GitHub repos or datasets mined from |
| `schema_version` | `string` | Hash or version tag of `eval/schema.json` at freeze time |
| `curator` | `string` | Identity of the person/process that froze the set |

---

### 3.5 `BenchmarkResult`

**What it represents:** The output of `ch1_architecture/benchmark.py` — tokens/sec and perplexity for each quantization mode.

**Lives in:** `eval/results/ch1_benchmark.json`

**Owner:** `ch1_architecture/benchmark.py`

**Fields:**

| Field | Type | Constraints |
|-------|------|-------------|
| `run_id` | `string` | UUID v4 |
| `recorded_at` | `string` | ISO 8601 UTC |
| `checkpoint_path` | `string` | Path to `outputs/ch1/model.pt` |
| `config_path` | `string` | Path to YAML config used |
| `seed` | `integer` | Seed used for measurement |
| `results` | `object` | Keyed by mode: `{"fp32": {...}, "fp16": {...}, "int8": {...}, "int4": {...}}` |
| `results[mode].tokens_per_sec` | `float` | Wall-clock measurement |
| `results[mode].perplexity` | `float` | Exp(mean cross-entropy) |
| `results[mode].n_steps` | `integer` | Always 100 (warm-up excluded) |

---

### 3.6 `BaselineResult`

**What it represents:** Evaluation scores for the base model (zero-shot) and frontier API (3-shot) before any fine-tuning.

**Lives in:** `eval/results/baselines.json`

**Owner:** `ch2_adaptation/baseline.py`

**Fields:**

| Field | Type | Constraints |
|-------|------|-------------|
| `run_id` | `string` | UUID v4 |
| `recorded_at` | `string` | ISO 8601 UTC |
| `holdout_manifest_run_id` | `string` | FK → `HoldoutManifest` (by convention; not enforced by engine) |
| `base_model` | `EvalResult` | See §3.8 |
| `frontier_api` | `EvalResult` | See §3.8 |
| `frontier_model_tag` | `string` | e.g. `"claude-3-5-sonnet-20241022"` |

---

### 3.7 `FinetunedResult`

**What it represents:** Evaluation scores for the fine-tuned model after QLoRA training.

**Lives in:** `eval/results/finetuned.json`

**Owner:** `ch2_adaptation/evaluate.py`

**Fields:**

| Field | Type | Constraints |
|-------|------|-------------|
| `run_id` | `string` | UUID v4 |
| `recorded_at` | `string` | ISO 8601 UTC |
| `adapter_path` | `string` | Path or HF Hub repo ID of the adapter |
| `base_model_tag` | `string` | `"Qwen/Qwen2.5-Coder-1.5B-Instruct"` |
| `config_path` | `string` | Path to YAML config used for training |
| `eval_result` | `EvalResult` | See §3.8 |

---

### 3.8 `EvalResult`

**What it represents:** The structured output of `eval/harness.py::score_outputs`. Embedded in `BaselineResult` and `FinetunedResult`; also returned in-memory by `ch2_adaptation/evaluate.py` and `ch3_operation/api/evaluate.py`.

**Lives in:** Embedded in `baselines.json` and `finetuned.json`; never stored independently.

**Fields:**

| Field | Type | Constraints |
|-------|------|-------------|
| `schema_validity_rate` | `float` | `[0.0, 1.0]` |
| `bug_catch_rate` | `float` | `[0.0, 1.0]` |
| `n_samples` | `integer` | `>= 0` |
| `n_valid` | `integer` | `<= n_samples` |
| `n_caught` | `integer` | `<= n_valid` |
| `per_sample` | `array[PerSampleResult]` | Length == `n_samples` |

**`PerSampleResult` (embedded array element):**

| Field | Type | Constraints |
|-------|------|-------------|
| `id` | `string` | FK → `ReviewRecord.id` |
| `valid` | `boolean` | Whether output passed schema validation |
| `caught` | `boolean` | Whether bug was identified (line ± 2) |
| `raw_output` | `string` | Raw model output string, unmodified |

---

### 3.9 `GPTCheckpoint`

**What it represents:** A serialized `GPTModel` state dict plus the config and tokenizer needed to reload it.

**Lives in:** `outputs/ch1/model.pt` (PyTorch checkpoint file)

**Owner:** `ch1_architecture/train.py`

**Logical fields (stored as a Python dict inside the `.pt` file):**

| Key | Type | Description |
|-----|------|-------------|
| `model_state_dict` | `dict` | `nn.Module.state_dict()` |
| `optimizer_state_dict` | `dict` | `AdamW.state_dict()` |
| `config` | `dict` | `asdict(GPTConfig)` |
| `step` | `integer` | Training step at save time |
| `train_loss` | `float` | Loss at save step |
| `vocab` | `dict` | `BPETokenizer.vocab` |
| `merges` | `list` | `BPETokenizer.merges` |

---

### 3.10 `RequestLog` (ephemeral, in-process)

**What it represents:** A single inference request handled by the ch3 FastAPI service. Never persisted to disk; exists only in the rolling deque inside `RollingQualityMonitor` and in Prometheus metric accumulators.

**Lives in:** `ch3_operation` process memory only

**Logical fields:**

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | `string` | UUID v4; generated per request for log correlation |
| `received_at` | `float` | `time.perf_counter()` timestamp |
| `latency_s` | `float` | Wall-clock seconds from request receipt to response send |
| `valid` | `boolean` | Whether `validator.py` accepted the model output |
| `status` | `string` | `enum["success","error"]` |

Only `valid` is retained in the deque (as a boolean). All other fields are used transiently within the request lifecycle and then discarded.

---

## 4. Relational Schema

Although no relational engine is deployed, the logical relationships between entities are defined here in relational terms. This makes the foreign-key conventions explicit and enables a future migration to SQLite or PostgreSQL without redesign.

### 4.1 Entity-Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   HoldoutManifest ──────────────────────────────────────────────────┐  │
│   (1 row, committed)                                                 │  │
│                                                                      │  │
│   ReviewRecord ─────────────────────────────────────────────────────┤  │
│   (N rows per split)                                                 │  │
│   PK: id (UUID)                                                      │  │
│   FK: split → {"train","val","holdout"}                              │  │
│   Embeds: ReviewOutput (not a separate table)                        │  │
│                                                                      │  │
│   DataProvenance ────────────────────────────────────────────────┐  │  │
│   (1 row per data_gen run)                                        │  │  │
│   PK: run_id (UUID)                                               │  │  │
│   References: train.jsonl, val.jsonl (by path, not FK)            │  │  │
│                                                                   │  │  │
│   BaselineResult ─────────────────────────────────────────────┐  │  │  │
│   (1 row per baseline run)                                     │  │  │  │
│   PK: run_id (UUID)                                            │  │  │  │
│   Soft-FK: holdout_manifest_run_id → HoldoutManifest.run_id   │  │  │  │
│   Embeds: EvalResult × 2 (base_model, frontier_api)           │  │  │  │
│                                                                │  │  │  │
│   FinetunedResult ─────────────────────────────────────────┐  │  │  │  │
│   (1 row per evaluate run)                                 │  │  │  │  │
│   PK: run_id (UUID)                                        │  │  │  │  │
│   Embeds: EvalResult × 1                                   │  │  │  │  │
│                                                            │  │  │  │  │
│   EvalResult (embedded, not a standalone table)            │  │  │  │  │
│   Contains: PerSampleResult[]                              │  │  │  │  │
│   PerSampleResult.id → ReviewRecord.id (soft FK)          │  │  │  │  │
│                                                            │  │  │  │  │
│   BenchmarkResult ──────────────────────────────────────┐ │  │  │  │  │
│   (1 row per benchmark run)                             │ │  │  │  │  │
│   PK: run_id (UUID)                                     │ │  │  │  │  │
│                                                         │ │  │  │  │  │
└─────────────────────────────────────────────────────────┴─┴──┴──┴──┴──┘
```

### 4.2 DDL (Logical, SQLite-compatible)

The following DDL is the authoritative logical schema. It is **not executed** in the current file-based implementation but serves as the migration target if a database engine is ever introduced (see §9).

```sql
-- ============================================================
-- CORE REVIEW DATA
-- ============================================================

CREATE TABLE review_record (
    id                  TEXT        NOT NULL PRIMARY KEY,   -- UUID v4
    code                TEXT        NOT NULL CHECK(length(code) > 0),
    context             TEXT        NOT NULL DEFAULT '',
    severity            TEXT        NOT NULL CHECK(severity IN ('critical','major','minor','info')),
    category            TEXT        NOT NULL CHECK(length(category) BETWEEN 1 AND 64),
    line                INTEGER     NOT NULL CHECK(line >= 1),
    issue               TEXT        NOT NULL CHECK(length(issue) BETWEEN 1 AND 512),
    suggested_fix       TEXT        NOT NULL CHECK(length(suggested_fix) BETWEEN 1 AND 1024),
    split               TEXT        NOT NULL CHECK(split IN ('train','val','holdout')),
    source              TEXT        NOT NULL CHECK(source IN ('synthetic','mined')),
    created_at          TEXT        NOT NULL   -- ISO 8601 UTC
);

-- No UPDATE or DELETE permitted on this table after initial insert.
-- Enforced at application layer (data_gen.py, holdout curator).
-- A trigger is defined below to make this explicit.

CREATE TRIGGER review_record_no_update
    BEFORE UPDATE ON review_record
BEGIN
    SELECT RAISE(ABORT, 'review_record is append-only: UPDATE is forbidden');
END;

CREATE TRIGGER review_record_no_delete
    BEFORE DELETE ON review_record
BEGIN
    SELECT RAISE(ABORT, 'review_record is append-only: DELETE is forbidden');
END;

-- Indexes for eval harness access patterns
CREATE INDEX idx_review_record_split  ON review_record(split);
CREATE INDEX idx_review_record_source ON review_record(source);

-- ============================================================
-- HOLDOUT MANIFEST
-- ============================================================

CREATE TABLE holdout_manifest (
    run_id              TEXT        NOT NULL PRIMARY KEY,   -- UUID v4
    freeze_date         TEXT        NOT NULL,               -- ISO 8601 UTC
    n_synthetic         INTEGER     NOT NULL CHECK(n_synthetic >= 0),
    n_mined             INTEGER     NOT NULL CHECK(n_mined >= 0),
    n_total             INTEGER     NOT NULL
                            GENERATED ALWAYS AS (n_synthetic + n_mined) STORED,
    dedup_method        TEXT        NOT NULL,
    sources             TEXT        NOT NULL,               -- JSON array serialized as TEXT
    schema_version      TEXT        NOT NULL,
    curator             TEXT        NOT NULL
);

-- Exactly one row is permitted.
CREATE TRIGGER holdout_manifest_singleton
    BEFORE INSERT ON holdout_manifest
    WHEN (SELECT COUNT(*) FROM holdout_manifest) >= 1
BEGIN
    SELECT RAISE(ABORT, 'holdout_manifest must have exactly one row');
END;

-- ============================================================
-- DATA PROVENANCE
-- ============================================================

CREATE TABLE data_provenance (
    run_id                  TEXT    NOT NULL PRIMARY KEY,   -- UUID v4
    generated_at            TEXT    NOT NULL,               -- ISO 8601 UTC
    frontier_model          TEXT    NOT NULL,
    prompt_template_hash    TEXT    NOT NULL,               -- SHA-256
    n_requested             INTEGER NOT NULL CHECK(n_requested >= 0),
    n_valid                 INTEGER NOT NULL CHECK(n_valid >= 0),
    n_skipped               INTEGER NOT NULL
                                GENERATED ALWAYS AS (n_requested - n_valid) STORED,
    estimated_cost_usd      REAL    NOT NULL CHECK(estimated_cost_usd >= 0),
    actual_cost_usd         REAL    CHECK(actual_cost_usd >= 0),   -- NULL until run completes
    train_path              TEXT    NOT NULL,
    val_path                TEXT    NOT NULL,
    split_seed              INTEGER NOT NULL
);

-- ============================================================
-- BENCHMARK RESULTS
-- ============================================================

CREATE TABLE benchmark_result (
    run_id              TEXT    NOT NULL PRIMARY KEY,   -- UUID v4
    recorded_at         TEXT    NOT NULL,
    checkpoint_path     TEXT    NOT NULL,
    config_path         TEXT    NOT NULL,
    seed                INTEGER NOT NULL
);

CREATE TABLE benchmark_mode_result (
    id                  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    benchmark_run_id    TEXT    NOT NULL REFERENCES benchmark_result(run_id),
    mode                TEXT    NOT NULL CHECK(mode IN ('fp32','fp16','int8','int4')),
    tokens_per_sec      REAL    NOT NULL CHECK(tokens_per_sec > 0),
    perplexity          REAL    NOT NULL CHECK(perplexity > 0),
    n_steps             INTEGER NOT NULL DEFAULT 100,
    UNIQUE(benchmark_run_id, mode)
);

-- ============================================================
-- EVAL RESULTS
-- ============================================================

CREATE TABLE eval_run (
    run_id                  TEXT    NOT NULL PRIMARY KEY,   -- UUID v4
    recorded_at             TEXT    NOT NULL,
    run_type                TEXT    NOT NULL
                                CHECK(run_type IN ('baseline_base','baseline_frontier','finetuned')),
    -- For baseline runs:
    frontier_model_tag      TEXT,                           -- NULL for finetuned
    holdout_manifest_run_id TEXT    REFERENCES holdout_manifest(run_id),
    -- For finetuned runs:
    adapter_path            TEXT,                           -- NULL for baseline
    base_model_tag          TEXT,
    config_path             TEXT,
    -- Aggregate metrics (denormalized for fast dashboard reads)
    schema_validity_rate    REAL    NOT NULL CHECK(schema_validity_rate BETWEEN 0 AND 1),
    bug_catch_rate          REAL    NOT NULL CHECK(bug_catch_rate BETWEEN 0 AND 1),
    n_samples               INTEGER NOT NULL CHECK(n_samples >= 0),
    n_valid                 INTEGER NOT NULL CHECK(n_valid >= 0),
    n_caught                INTEGER NOT NULL CHECK(n_caught >= 0)
);

CREATE TABLE per_sample_result (
    id                  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    eval_run_id         TEXT    NOT NULL REFERENCES eval_run(run_id),
    review_record_id    TEXT    NOT NULL REFERENCES review_record(id),
    valid               INTEGER NOT NULL CHECK(valid IN (0,1)),   -- boolean
    caught              INTEGER NOT NULL CHECK(caught IN (0,1)),  -- boolean
    raw_output          TEXT    NOT NULL
);

CREATE INDEX idx_per_sample_eval_run ON per_sample_result(eval_run_id);
CREATE INDEX idx_per_sample_record   ON per_sample_result(review_record_id);

-- ============================================================
-- GPT CHECKPOINT REGISTRY
-- (Metadata only; binary .pt files live on disk / HF Hub)
-- ============================================================

CREATE TABLE gpt_checkpoint (
    id                  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    saved_at            TEXT    NOT NULL,               -- ISO 8601 UTC
    step                INTEGER NOT NULL,
    train_loss          REAL    NOT NULL,
    config_path         TEXT    NOT NULL,
    file_path           TEXT    NOT NULL UNIQUE,        -- relative path to .pt file
    is_final            INTEGER NOT NULL DEFAULT 0 CHECK(is_final IN (0,1))
);
```

---

## 5. Data Ownership Map

Ownership means: **one module may write; all others may only read**. Writes from non-owning modules are a bug, not a design choice.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Entity / File                    │ Owner (writer)           │ Readers        │
├──────────────────────────────────────────────────────────────────────────────┤
│  eval/schema.json                 │ Project maintainer       │ ch2/schema.py  │
│                                   │ (committed; never auto-  │ ch3/schema.py  │
│                                   │  written)                │ eval/harness.py│
├──────────────────────────────────────────────────────────────────────────────┤
│  eval/holdout/holdout.jsonl       │ Project maintainer       │ ch2/evaluate.py│
│  eval/holdout/manifest.json       │ (committed; leakage      │ ch3/evaluate.py│
│                                   │  guard blocks all writes)│ eval/harness.py│
├──────────────────────────────────────────────────────────────────────────────┤
│  ch2_adaptation/data/train.jsonl  │ ch2/data_gen.py          │ ch2/data_      │
│  ch2_adaptation/data/val.jsonl    │                          │ loader.py      │
│                                   │                          │ ch2/finetune.py│
├──────────────────────────────────────────────────────────────────────────────┤
│  ch2_adaptation/data/             │ ch2/data_gen.py          │ ch2/baseline.py│
│    provenance.json                │                          │ (audit only)   │
├──────────────────────────────────────────────────────────────────────────────┤
│  eval/results/ch1_benchmark.json  │ ch1/benchmark.py         │ README updater │
│  eval/results/plots/              │ ch1/benchmark.py         │ (read-only)    │
│                                   │ ch2/evaluate.py          │                │
├──────────────────────────────────────────────────────────────────────────────┤
│  eval/results/baselines.json      │ ch2/baseline.py          │ ch2/evaluate.py│
│                                   │                          │ README updater │
├──────────────────────────────────────────────────────────────────────────────┤
│  eval/results/finetuned.json      │ ch2/evaluate.py          │ README updater │
├──────────────────────────────────────────────────────────────────────────────┤
│  outputs/ch1/model.pt             │ ch1/train.py             │ ch1/benchmark  │
│                                   │                          │ ch1/generate   │
├──────────────────────────────────────────────────────────────────────────────┤
│  outputs/adapter/                 │ ch2/finetune.py          │ ch2/evaluate.py│
│                                   │                          │ ch3/merge script│
├──────────────────────────────────────────────────────────────────────────────┤
│  outputs/merged/                  │ ch3/merge_adapter.py     │ ch3/export_    │
│                                   │                          │ gguf.py        │
├──────────────────────────────────────────────────────────────────────────────┤
│  outputs/model-Q4_K_M.gguf        │ ch3/quantize_gguf.py     │ ch3/model_     │
│                                   │                          │ backend.py     │
├──────────────────────────────────────────────────────────────────────────────┤
│  Prometheus metrics (in-process)  │ ch3/metrics.py           │ ch3/main.py    │
│  Rolling deque (in-process)       │ ch3/drift.py             │ ch3/exporter.py│
└──────────────────────────────────────────────────────────────────────────────┘
```

### 5.1 Write-Once Semantics