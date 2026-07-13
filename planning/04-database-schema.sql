```sql
-- =============================================================================
-- DATABASE SCHEMA
-- llm-engineering-architecture-to-edge
-- =============================================================================
--
-- Design rationale:
--   This schema persists all durable state produced by the three-chapter
--   monorepo: training runs, evaluation results, generated data provenance,
--   serving request logs, and drift metrics.  Runtime state (KV-cache,
--   in-flight requests) is never stored here.
--
-- Normalisation target: 3NF throughout.  Denormalised summary columns are
--   added only where they are written once and never updated (e.g. final
--   benchmark aggregates).
--
-- Identifier convention:
--   Every table has a surrogate BIGSERIAL primary key named <table>_id.
--   Natural-key uniqueness is enforced separately via UNIQUE constraints so
--   that foreign keys are always integer joins.
--
-- Timestamp convention:
--   All timestamps are TIMESTAMPTZ (UTC).  created_at is set by DEFAULT
--   now() and is never updated.  updated_at is maintained by a trigger
--   (see trigger section at the bottom).
--
-- Migration structure (intended):
--   Migrations live in db/migrations/ and are numbered sequentially:
--     0001_initial_schema.sql   -- this file (all CREATE TABLE statements)
--     0002_add_indexes.sql      -- all CREATE INDEX statements below, separated
--                               -- for environments that build indexes CONCURRENTLY
--     0003_seed_enum_values.sql -- INSERT into lookup tables (severity, mode, etc.)
--   Each migration file is idempotent (uses IF NOT EXISTS / IF EXISTS guards).
--   A migrations table (defined last in this file) tracks applied migrations.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- EXTENSIONS
-- ---------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- gen_random_uuid() for idempotency keys


-- =============================================================================
-- SECTION 1 — SHARED / LOOKUP TABLES
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1.1  severity_level
--      Canonical enum for ReviewOutput.severity.
--      Matches eval/schema.json enum exactly.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS severity_level (
    severity_level_id   SMALLINT        PRIMARY KEY,
    code                VARCHAR(16)     NOT NULL,
    display_order       SMALLINT        NOT NULL,   -- 1=critical … 4=info

    CONSTRAINT uq_severity_level_code UNIQUE (code),
    CONSTRAINT ck_severity_level_code CHECK (code IN ('critical','major','minor','info'))
);

COMMENT ON TABLE  severity_level                IS 'Lookup: ReviewOutput severity enum values.';
COMMENT ON COLUMN severity_level.code           IS 'Matches eval/schema.json enum.';
COMMENT ON COLUMN severity_level.display_order  IS '1=critical, 2=major, 3=minor, 4=info.';


-- ---------------------------------------------------------------------------
-- 1.2  quantization_mode
--      Canonical enum for ch1 quantization modes.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS quantization_mode (
    quantization_mode_id    SMALLINT        PRIMARY KEY,
    code                    VARCHAR(8)      NOT NULL,

    CONSTRAINT uq_quantization_mode_code UNIQUE (code),
    CONSTRAINT ck_quantization_mode_code CHECK (code IN ('fp32','fp16','int8','int4'))
);

COMMENT ON TABLE quantization_mode IS 'Lookup: ch1 quantize.py precision modes.';


-- ---------------------------------------------------------------------------
-- 1.3  chapter
--      Identifies which monorepo chapter produced a record.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chapter (
    chapter_id      SMALLINT        PRIMARY KEY,
    code            VARCHAR(4)      NOT NULL,   -- 'ch1' | 'ch2' | 'ch3'
    description     TEXT            NOT NULL,

    CONSTRAINT uq_chapter_code UNIQUE (code)
);

COMMENT ON TABLE chapter IS 'Lookup: monorepo chapter identifier.';


-- =============================================================================
-- SECTION 2 — CHAPTER 1: TRAINING RUNS & BENCHMARKS
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 2.1  ch1_run
--      One row per training run (python train.py --config <path>).
--      Corresponds to a single W&B run.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ch1_run (
    ch1_run_id          BIGSERIAL       PRIMARY KEY,
    run_name            VARCHAR(128)    NOT NULL,
    config_path         VARCHAR(256)    NOT NULL,   -- e.g. 'configs/smoke.yaml'
    vocab_size          INTEGER         NOT NULL,
    d_model             INTEGER         NOT NULL,
    n_heads             INTEGER         NOT NULL,
    n_layers            INTEGER         NOT NULL,
    seq_len             INTEGER         NOT NULL,
    batch_size          INTEGER         NOT NULL,
    max_steps           INTEGER         NOT NULL,
    seed                INTEGER         NOT NULL,
    device              VARCHAR(16)     NOT NULL,   -- 'cpu' | 'cuda'
    wandb_run_id        VARCHAR(64)     NULL,       -- W&B run ID, set after init
    checkpoint_path     VARCHAR(512)    NULL,       -- outputs/ch1/model.pt
    final_train_loss    DOUBLE PRECISION NULL,
    final_perplexity    DOUBLE PRECISION NULL,
    completed_at        TIMESTAMPTZ     NULL,       -- NULL until training finishes
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_ch1_run_name UNIQUE (run_name),
    CONSTRAINT ck_ch1_run_d_model_heads CHECK (d_model % n_heads = 0),
    CONSTRAINT ck_ch1_run_device CHECK (device IN ('cpu','cuda'))
);

COMMENT ON TABLE  ch1_run                   IS 'One row per ch1 training run.';
COMMENT ON COLUMN ch1_run.wandb_run_id      IS 'Populated after wandb.init(); NULL for smoke runs that skip W&B.';
COMMENT ON COLUMN ch1_run.checkpoint_path   IS 'Relative path from repo root to saved .pt file.';


-- ---------------------------------------------------------------------------
-- 2.2  ch1_training_step
--      Per-step metrics logged during training.
--      Mirrors wandb.log({step, train_loss, perplexity, lr, tokens_per_sec}).
--      Logged every 10 steps; high-volume table — partitioned by ch1_run_id
--      in large deployments (partition DDL omitted here; add in migration 0004).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ch1_training_step (
    ch1_training_step_id    BIGSERIAL       PRIMARY KEY,
    ch1_run_id              BIGINT          NOT NULL
                                REFERENCES ch1_run (ch1_run_id)
                                ON DELETE CASCADE,
    step                    INTEGER         NOT NULL,
    train_loss              DOUBLE PRECISION NOT NULL,
    perplexity              DOUBLE PRECISION NOT NULL,
    learning_rate           DOUBLE PRECISION NOT NULL,
    tokens_per_sec          DOUBLE PRECISION NOT NULL,
    logged_at               TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_ch1_training_step UNIQUE (ch1_run_id, step)
);

COMMENT ON TABLE ch1_training_step IS 'Per-step training metrics; logged every 10 steps.';


-- ---------------------------------------------------------------------------
-- 2.3  ch1_benchmark_result
--      Aggregate benchmark output from benchmark.py.
--      One row per (run, quantization_mode).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ch1_benchmark_result (
    ch1_benchmark_result_id     BIGSERIAL       PRIMARY KEY,
    ch1_run_id                  BIGINT          NOT NULL
                                    REFERENCES ch1_run (ch1_run_id)
                                    ON DELETE CASCADE,
    quantization_mode_id        SMALLINT        NOT NULL
                                    REFERENCES quantization_mode (quantization_mode_id),
    tokens_per_sec              DOUBLE PRECISION NOT NULL,
    perplexity                  DOUBLE PRECISION NOT NULL,
    n_warmup_steps              INTEGER         NOT NULL DEFAULT 10,
    n_measured_steps            INTEGER         NOT NULL DEFAULT 100,
    result_json_path            VARCHAR(512)    NULL,   -- eval/results/ch1_benchmark.json
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_ch1_benchmark_result UNIQUE (ch1_run_id, quantization_mode_id)
);

COMMENT ON TABLE  ch1_benchmark_result              IS 'Benchmark output from benchmark.py; one row per run × precision.';
COMMENT ON COLUMN ch1_benchmark_result.n_warmup_steps IS 'Warm-up steps discarded before measurement.';


-- =============================================================================
-- SECTION 3 — CHAPTER 2: DATA GENERATION, FINE-TUNING & EVALUATION
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 3.1  data_generation_run
--      Records a single execution of data_gen.py.
--      Provenance mirrors data/provenance.json.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS data_generation_run (
    data_generation_run_id  BIGSERIAL       PRIMARY KEY,
    frontier_model          VARCHAR(128)    NOT NULL,   -- e.g. 'claude-3-5-sonnet-20241022'
    prompt_template_hash    CHAR(64)        NOT NULL,   -- SHA-256 of PROMPT_TEMPLATE
    estimated_cost_usd      NUMERIC(8,4)    NOT NULL,
    actual_cost_usd         NUMERIC(8,4)    NULL,       -- filled after completion
    n_snippets_attempted    INTEGER         NOT NULL,
    n_records_valid         INTEGER         NOT NULL,
    n_records_skipped       INTEGER         NOT NULL,
    train_jsonl_path        VARCHAR(512)    NOT NULL,
    val_jsonl_path          VARCHAR(512)    NOT NULL,
    provenance_json_path    VARCHAR(512)    NOT NULL,
    budget_limit_usd        NUMERIC(8,4)    NOT NULL DEFAULT 1.50,
    completed_at            TIMESTAMPTZ     NULL,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT ck_data_gen_cost CHECK (estimated_cost_usd <= budget_limit_usd),
    CONSTRAINT ck_data_gen_counts CHECK (
        n_records_valid + n_records_skipped <= n_snippets_attempted
    )
);

COMMENT ON TABLE  data_generation_run                   IS 'Provenance for one data_gen.py execution; mirrors data/provenance.json.';
COMMENT ON COLUMN data_generation_run.prompt_template_hash IS 'SHA-256 of the locked PROMPT_TEMPLATE string for reproducibility.';
COMMENT ON COLUMN data_generation_run.budget_limit_usd  IS 'Budget guard threshold at time of run (default $1.50).';


-- ---------------------------------------------------------------------------
-- 3.2  generated_sample
--      Individual schema-valid records written to data/train.jsonl or
--      data/val.jsonl.  Stores the raw model output alongside the parsed
--      fields so the original response is never lost.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS generated_sample (
    generated_sample_id         BIGSERIAL       PRIMARY KEY,
    data_generation_run_id      BIGINT          NOT NULL
                                    REFERENCES data_generation_run (data_generation_run_id)
                                    ON DELETE CASCADE,
    split                       VARCHAR(8)      NOT NULL,   -- 'train' | 'val'
    source_snippet_hash         CHAR(64)        NOT NULL,   -- SHA-256 of input Java snippet
    raw_frontier_response       TEXT            NOT NULL,
    severity_level_id           SMALLINT        NOT NULL
                                    REFERENCES severity_level (severity_level_id),
    category                    VARCHAR(64)     NOT NULL,
    line                        INTEGER         NOT NULL,
    issue                       VARCHAR(512)    NOT NULL,
    suggested_fix               VARCHAR(1024)   NOT NULL,
    schema_valid                BOOLEAN         NOT NULL DEFAULT TRUE,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT ck_generated_sample_split CHECK (split IN ('train','val')),
    CONSTRAINT ck_generated_sample_line  CHECK (line >= 1)
);

COMMENT ON TABLE  generated_sample                      IS 'Individual schema-valid records from data_gen.py.';
COMMENT ON COLUMN generated_sample.source_snippet_hash  IS 'SHA-256 of the Java snippet sent to the frontier API; enables dedup.';
COMMENT ON COLUMN generated_sample.schema_valid         IS 'Always TRUE here; invalid records are not inserted (they are logged in data_generation_skip).';


-- ---------------------------------------------------------------------------
-- 3.3  data_generation_skip
--      Records snippets that failed schema validation during data_gen.py.
--      Kept separate to avoid polluting generated_sample with invalid rows.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS data_generation_skip (
    data_generation_skip_id     BIGSERIAL       PRIMARY KEY,
    data_generation_run_id      BIGINT          NOT NULL
                                    REFERENCES data_generation_run (data_generation_run_id)
                                    ON DELETE CASCADE,
    source_snippet_hash         CHAR(64)        NOT NULL,
    raw_frontier_response       TEXT            NOT NULL,
    validation_error            TEXT            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT now()
);

COMMENT ON TABLE data_generation_skip IS 'Snippets rejected by schema validation during data_gen.py.';


-- ---------------------------------------------------------------------------
-- 3.4  ch2_finetune_run
--      One row per finetune.py execution.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ch2_finetune_run (
    ch2_finetune_run_id     BIGSERIAL       PRIMARY KEY,
    data_generation_run_id  BIGINT          NOT NULL
                                REFERENCES data_generation_run (data_generation_run_id),
    config_path             VARCHAR(256)    NOT NULL,
    model_tag               VARCHAR(128)    NOT NULL,   -- 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
    lora_r                  SMALLINT        NOT NULL,
    lora_alpha              SMALLINT        NOT NULL,
    lora_dropout            NUMERIC(5,4)    NOT NULL,
    target_modules          TEXT[]          NOT NULL,   -- ['q_proj','v_proj','k_proj','o_proj']
    max_steps               INTEGER         NOT NULL,
    batch_size              SMALLINT        NOT NULL,
    learning_rate           DOUBLE PRECISION NOT NULL,
    use_4bit                BOOLEAN         NOT NULL,
    seed                    INTEGER         NOT NULL,
    wandb_run_id            VARCHAR(64)     NULL,
    adapter_path            VARCHAR(512)    NULL,       -- outputs/adapter/
    hf_hub_repo             VARCHAR(256)    NULL,       -- HF Hub repo after push
    n_trainable_params      BIGINT          NULL,       -- from print_trainable_parameters()
    n_total_params          BIGINT          NULL,
    final_train_loss        DOUBLE PRECISION NULL,
    final_eval_loss         DOUBLE PRECISION NULL,
    completed_at            TIMESTAMPTZ     NULL,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT ck_ch2_finetune_lora_r     CHECK (lora_r > 0),
    CONSTRAINT ck_ch2_finetune_lora_alpha CHECK (lora_alpha > 0),
    CONSTRAINT ck_ch2_finetune_lr         CHECK (learning_rate > 0)
);

COMMENT ON TABLE  ch2_finetune_run                  IS 'One row per finetune.py execution.';
COMMENT ON COLUMN ch2_finetune_run.target_modules   IS 'LoRA target projection names; stored as text array.';
COMMENT ON COLUMN ch2_finetune_run.hf_hub_repo      IS 'Populated after adapter is pushed to HF Hub.';


-- ---------------------------------------------------------------------------
-- 3.5  ch2_finetune_step
--      Per-step metrics from SFTTrainer (train_loss, eval_loss, lr).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ch2_finetune_step (
    ch2_finetune_step_id    BIGSERIAL       PRIMARY KEY,
    ch2_finetune_run_id     BIGINT          NOT NULL
                                REFERENCES ch2_finetune_run (ch2_finetune_run_id)
                                ON DELETE CASCADE,
    step                    INTEGER         NOT NULL,
    train_loss              DOUBLE PRECISION NULL,
    eval_loss               DOUBLE PRECISION NULL,
    learning_rate           DOUBLE PRECISION NULL,
    logged_at               TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_ch2_finetune_step UNIQUE (ch2_finetune_run_id, step)
);

COMMENT ON TABLE ch2_finetune_step IS 'Per-step SFTTrainer metrics; mirrors W&B logs.';


-- =============================================================================
-- SECTION 4 — SHARED EVALUATION
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 4.1  holdout_manifest
--      Mirrors eval/holdout/manifest.json.  One row per holdout freeze.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS holdout_manifest (
    holdout_manifest_id     BIGSERIAL       PRIMARY KEY,
    freeze_date             DATE            NOT NULL,
    holdout_jsonl_path      VARCHAR(512)    NOT NULL,
    manifest_json_path      VARCHAR(512)    NOT NULL,
    n_samples               INTEGER         NOT NULL,
    dedup_method            VARCHAR(128)    NOT NULL,
    sources                 TEXT[]          NOT NULL,   -- e.g. ['synthetic-clean','mined-real']
    content_hash            CHAR(64)        NOT NULL,   -- SHA-256 of holdout.jsonl
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_holdout_manifest_hash UNIQUE (content_hash)
);

COMMENT ON TABLE  holdout_manifest              IS 'Mirrors eval/holdout/manifest.json; one row per frozen holdout version.';
COMMENT ON COLUMN holdout_manifest.content_hash IS 'SHA-256 of holdout.jsonl; used to detect accidental mutation.';


-- ---------------------------------------------------------------------------
-- 4.2  eval_run
--      One row per execution of eval/harness.py::score_outputs().
--      Links to the holdout version used and the model/adapter being scored.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_run (
    eval_run_id             BIGSERIAL       PRIMARY KEY,
    holdout_manifest_id     BIGINT          NOT NULL
                                REFERENCES holdout_manifest (holdout_manifest_id),
    chapter_id              SMALLINT        NOT NULL
                                REFERENCES chapter (chapter_id),
    -- Exactly one of the following run FK columns is non-NULL:
    ch1_run_id              BIGINT          NULL
                                REFERENCES ch1_run (ch1_run_id),
    ch2_finetune_run_id     BIGINT          NULL
                                REFERENCES ch2_finetune_run (ch2_finetune_run_id),
    ch3_serve_model_id      BIGINT          NULL,   -- FK added after ch3_serve_model is defined below
    -- Baseline variant (NULL when scoring a fine-tuned model)
    baseline_variant        VARCHAR(32)     NULL,   -- 'base_model_zero_shot' | 'frontier_api_3shot'
    schema_validity_rate    DOUBLE PRECISION NOT NULL,
    bug_catch_rate          DOUBLE PRECISION NOT NULL,
    n_samples               INTEGER         NOT NULL,
    n_valid                 INTEGER         NOT NULL,
    n_caught                INTEGER         NOT NULL,
    result_json_path        VARCHAR(512)    NULL,   -- eval/results/*.json
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT ck_eval_run_exactly_one_source CHECK (
        (
            (ch1_run_id IS NOT NULL)::INT +
            (ch2_finetune_run_id IS NOT NULL)::INT +
            (ch3_serve_model_id IS NOT NULL)::INT +
            (baseline_variant IS NOT NULL)::INT
        ) = 1
    ),
    CONSTRAINT ck_eval_run_rates CHECK (
        schema_validity_rate BETWEEN 0.0 AND 1.0 AND
        bug_catch_rate       BETWEEN 0.0 AND 1.0
    ),
    CONSTRAINT ck_eval_run_counts CHECK (
        n_valid  <= n_samples AND
        n_caught <= n_samples
    ),
    CONSTRAINT ck_eval_run_baseline_variant CHECK (
        baseline_variant IS NULL OR
        baseline_variant IN ('base_model_zero_shot','frontier_api_3shot')
    )
);

COMMENT ON TABLE  eval_run                      IS 'One row per harness.py score_outputs() call.';
COMMENT ON COLUMN eval_run.baseline_variant     IS 'Non-NULL only for baseline.py runs; identifies which baseline strategy was used.';
COMMENT ON COLUMN eval_run.ch3_serve_model_id   IS 'FK to ch3_serve_model; populated after that table is created (see ALTER TABLE below).';


-- ---------------------------------------------------------------------------
-- 4.3  eval_sample_result
--      Per-sample detail from EvalResult.per_sample.
--      High-volume; references eval_run.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_sample_result (
    eval_sample_result_id   BIGSERIAL       PRIMARY KEY,
    eval_run_id             BIGINT          NOT NULL
                                REFERENCES eval_run (eval_run_id)
                                ON DELETE CASCADE,
    sample_id               VARCHAR(128)    NOT NULL,   -- id from holdout.jsonl
    valid                   BOOLEAN         NOT NULL,
    caught                  BOOLEAN         NOT NULL,
    raw_output              TEXT            NOT NULL,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_eval_sample_result UNIQUE (eval_run_id, sample_id)
);

COMMENT ON TABLE eval_sample_result IS 'Per-sample detail rows from EvalResult.per_sample.';


-- =============================================================================
-- SECTION 5 — CHAPTER 3: SERVING
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 5.1  ch3_serve_model
--      Tracks a deployed GGUF model artifact.
--      One row per distinct model file pushed to HF Hub or loaded locally.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ch3_serve_model (
    ch3_serve_model_id      BIGSERIAL       PRIMARY KEY,
    ch2_finetune_run_id     BIGINT          NULL
                                REFERENCES ch2_finetune_run (ch2_finetune_run_id),
    gguf_filename           VARCHAR(256)    NOT NULL,   -- e.g. 'model-Q4_K_M.gguf'
    gguf_quantization       VARCHAR(32)     NOT NULL,   -- e.g. 'Q4_K_M'
    hf_hub_repo             VARCHAR(256)    NULL,
    hf_hub_revision         VARCHAR(64)     NULL,       -- git SHA on HF Hub
    file_size_bytes         BIGINT          NULL,
    content_hash            CHAR(64)        NULL,       -- SHA-256 of .gguf file
    schema_validity_rate_at_export DOUBLE PRECISION NULL, -- harness check after export
    exported_at             TIMESTAMPTZ     NULL,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_ch3_serve_model_hash UNIQUE (content_hash),
    CONSTRAINT ck_ch3_serve_model_validity CHECK (
        schema_validity_rate_at_export IS NULL OR
        schema_validity_rate_at_export BETWEEN 0.0 AND 1.0
    )
);

COMMENT ON TABLE  ch3_serve_model                               IS 'Deployed GGUF model artifact; one row per distinct file.';
COMMENT ON COLUMN ch3_serve_model.schema_validity_rate_at_export IS 'Harness score immediately after GGUF export; must be >= adapter baseline.';


-- Add the deferred FK from eval_run back to ch3_serve_model:
ALTER TABLE eval_run
    ADD CONSTRAINT fk_eval_run_ch3_serve_model
    FOREIGN KEY (ch3_serve_model_id)
    REFERENCES ch3_serve_model (ch3_serve_model_id);


-- ---------------------------------------------------------------------------
-- 5.2  serve_request
--      One row per POST /review request handled by ch3 FastAPI service.
--      This is the highest-volume table; partition by day in production
--      (partition DDL in migration 0005).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS serve_request (
    serve_request_id        BIGSERIAL       PRIMARY KEY,
    ch3_serve_model_id      BIGINT          NOT NULL
                                REFERENCES ch3_serve_model (ch3_serve_model_id),
    idempotency_key         UUID            NOT NULL DEFAULT gen_random_uuid(),
    code_hash               CHAR(64)        NOT NULL,   -- SHA-256 of request.code
    context_hash            CHAR(64)        NULL,       -- SHA-256 of request.context (nullable)
    raw_model_output        TEXT            NULL,       -- NULL on backend error
    -- Parsed ReviewOutput fields (NULL when schema_valid = FALSE):
    severity_level_id       SMALLINT        NULL
                                REFERENCES severity_level (severity_level_id),
    category                VARCHAR(64)     NULL,
    line                    INTEGER         NULL,
    issue                   VARCHAR(512)    NULL,
    suggested_fix           VARCHAR(1024)   NULL,
    -- Outcome
    schema_valid            BOOLEAN         NOT NULL,
    repair_attempted        BOOLEAN         NOT NULL DEFAULT FALSE,
    http_status             SMALLINT        NOT NULL,   -- 200 | 422 | 500
    latency_ms              DOUBLE PRECISION NOT NULL,
    requested_at            TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_serve_request_idempotency UNIQUE (idempotency_key),
    CONSTRAINT ck_serve_request_http_status CHECK (http_status IN (200, 422, 500)),
    CONSTRAINT ck_serve_request_line        CHECK (line IS NULL OR line >= 1),
    CONSTRAINT ck_serve_request_latency     CHECK (latency_ms >= 0),
    CONSTRAINT ck_serve_request_valid_fields CHECK (
        -- If schema_valid, all parsed fields must be present
        (schema_valid = FALSE) OR (
            severity_level_id IS NOT NULL AND
            category          IS NOT NULL AND
            line              IS NOT NULL AND
            issue             IS NOT NULL AND
            suggested_fix     IS NOT NULL
        )
    )
);

COMMENT ON TABLE  serve_request                     IS 'One row per POST /review request; highest-volume table.';
COMMENT ON COLUMN serve_request.code_hash           IS 'SHA-256 of request body code field; raw code is not stored (PII/IP concern).';
COMMENT ON COLUMN serve_request.repair_attempted    IS 'TRUE when validator.py fell through to the markdown-fence extraction path.';
COMMENT ON COLUMN serve_request.idempotency_key     IS 'Client-supplied or server-generated UUID; used for exactly-once semantics.';


-- ---------------------------------------------------------------------------
-- 5.3  drift_snapshot
--      Periodic snapshots of RollingQualityMonitor state.
--      Written by exporter.py or a background task every N requests.
--      Enables offline trend analysis independent of Prometheus retention.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS drift_snapshot (
    drift_snapshot_id       BIGSERIAL       PRIMARY KEY,
    ch3_serve_model_id      BIGINT          NOT NULL
                                REFERENCES ch3_serve_model (ch3_serve_model_id),
    window_size             INTEGER         NOT NULL,   -- config.metrics_window (default 100)
    n_valid_in_window       INTEGER         NOT NULL,
    schema_validity_rate    DOUBLE PRECISION NOT NULL,
    total_requests_at_snap  BIGINT          NOT NULL,   -- cumulative counter value
    snapped_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT ck_drift_snapshot_rate    CHECK (schema_validity_rate BETWEEN 0.0 AND 1.0),
    CONSTRAINT ck_drift_snapshot_window  CHECK (n_valid_in_window BETWEEN 0 AND window_size)
);

COMMENT ON TABLE  drift_snapshot                        IS 'Periodic snapshots of RollingQualityMonitor; supplements Prometheus retention.';
COMMENT ON COLUMN drift_snapshot.total_requests_at_snap IS 'Value of requests_total counter at snapshot time; enables rate computation.';


-- =============================================================================
-- SECTION 6 — CLAUDE CODE SCAFFOLD AUDIT
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 6.1  hook_event
--      Audit log written by PreToolUse / PostToolUse hooks.
--      Enables post-hoc analysis of guard activations (leakage, GPU budget).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS hook_event (
    hook_event_id           BIGSERIAL       PRIMARY KEY,
    hook_name               VARCHAR(64)     NOT NULL,   -- e.g. 'leakage_guard'
    hook_phase              VARCHAR(16)     NOT NULL,   -- 'PreToolUse' | 'PostToolUse' | 'SessionStart'
    tool_name               VARCHAR(64)     NULL,       -- 'Read' | 'Write' | 'Bash' | etc.
    tool_input_summary      TEXT            NULL,       -- truncated, no secrets
    outcome                 VARCHAR(8)      NOT NULL,   -- 'allow' | 'block'
    block_reason            TEXT            NULL,
    session_id              VARCHAR(128)    NULL,       -- Claude Code session identifier
    occurred_at             TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT ck_hook_event_phase   CHECK (hook_phase IN ('PreToolUse','PostToolUse','SessionStart')),
    CONSTRAINT ck_hook_event_outcome CHECK (outcome IN ('allow','block'))
);

COMMENT ON TABLE  hook_event                    IS 'Audit log for Claude Code hook activations.';
COMMENT ON COLUMN hook_event.tool_input_summary IS 'Truncated summary only; never stores raw file contents or secrets.';


-- =============================================================================
-- SECTION 7 — MIGRATION TRACKING
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 7.1  schema_migration
--      Tracks which migration files have been applied.
--      Convention: migration runner inserts a row before executing each file
--      and sets applied_at after success; failed migrations leave applied_at NULL.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_migration (
    schema_migration_id     BIGSERIAL       PRIMARY KEY,
    migration_number        SMALLINT        NOT NULL,
    filename                VARCHAR(256)    NOT NULL,
    checksum                CHAR(64)        NOT NULL,   -- SHA-256 of migration file
    applied_at              TIMESTAMPTZ     NULL,       -- NULL = in-