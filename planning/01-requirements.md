# requirements.md

> **Document status:** v1.0 — derived from `project.md` Draft v1
> **Traceability:** Every requirement is numbered for reference by downstream feature specs, task lists, and acceptance tests. Source sections from `project.md` are cited inline as `[§N]`.

---

## 1. Business Goals

| ID | Goal | Success Indicator |
|----|------|-------------------|
| BG-1 | Demonstrate end-to-end understanding of large language model internals — from raw tensor operations through fine-tuning to production edge deployment — as a portfolio artifact for engineering roles. | Completed README with architecture diagram, eval table, speedup curves, and a live deployment link. |
| BG-2 | Prove the most in-demand GenAI skill: converting a generalist open model into a cheap, private specialist that outperforms few-shot prompting on a narrow task. | Head-to-head eval table (fine-tuned vs. base vs. frontier API few-shot) on a held-out set, published in the README. |
| BG-3 | Demonstrate production/MLOps competence by operationalising a trained model — not merely training it. | A containerised, monitored, publicly accessible inference endpoint serving the fine-tuned model. |
| BG-4 | Complete the entire project within a solo-developer, weekend-only schedule of approximately six months and a total LLM-API spend of ≈ $1. | Project delivered within budget; API cost documented in the metrics scorecard. |
| BG-5 | Produce a reusable Claude Code SDD scaffold (commands, skills, agents, hooks, MCP wiring) that can be ported to sibling portfolio projects. | The `.claude/` directory is self-contained and documented; exercised at least once per component type. |

---

## 2. Stakeholders

| ID | Stakeholder | Interest |
|----|-------------|----------|
| SH-1 | **Solo developer (project owner)** | Builds, maintains, and presents the project; primary decision-maker on all trade-offs. |
| SH-2 | **Prospective employers / technical interviewers** | Evaluate depth of understanding across architecture, fine-tuning, and MLOps; consume the README, eval table, live demo, and interview-defense notes. |
| SH-3 | **Sibling-project consumers (Career Copilot, SWE-agent)** | Reuse the Java/Spring code-review model and the Claude Code SDD scaffold. |
| SH-4 | **Open-source / portfolio audience** | Reproduce results using documented smoke configs and public model artifacts on Hugging Face Hub. |

---

## 3. Users / Personas

| ID | Persona | Description | Primary Touchpoints |
|----|---------|-------------|---------------------|
| U-1 | **Developer-author (SH-1)** | Solo engineer running the project; interacts via Claude Code slash commands, CLI entrypoints, and Jupyter/Colab notebooks on GPU. | CLAUDE.md, slash commands, smoke configs, CI, GPU cluster. |
| U-2 | **Technical interviewer** | Senior engineer or hiring manager reviewing the portfolio; reads the README, inspects code quality, and may run the live demo. | README, HF Spaces demo, eval table, interview-defense notes (X3). |
| U-3 | **API consumer (downstream project)** | A sibling project (Career Copilot, SWE-agent) calling the FastAPI endpoint to obtain structured Java code-review JSON. | FastAPI `/review` endpoint, `eval/schema.json`. |
| U-4 | **Reproducibility auditor** | A developer attempting to reproduce results from scratch using only the public repository and documented instructions. | `make`/`uv` targets, smoke configs, Hub model artifacts, `specs/` directory. |

---

## 4. Functional Requirements

### 4.1 Phase 0 — Foundation

| ID | Requirement | Acceptance Criteria | Source |
|----|-------------|---------------------|--------|
| FR-1 | The repository SHALL be structured as a single monorepo containing three independent packages (`ch1_architecture/`, `ch2_adaptation/`, `ch3_operation/`), a shared `eval/` directory, and a shared `specs/` directory, conforming exactly to the layout specified in `project.md §5`. | Directory tree matches the specified layout; `pytest` discovers tests in all three packages from the repo root. | §1 D1, §5 |
| FR-2 | The repository SHALL include a `uv`-managed Python environment with `ruff`, `black`, and `pytest` configured and passing on an empty codebase from day one. | `uv run ruff check .`, `uv run black --check .`, and `uv run pytest -q` all exit 0 on a clean checkout. | §4, §5 F1 |
| FR-3 | Every training and evaluation entrypoint in all three packages SHALL accept a `--config <path>` argument and support at minimum a `smoke` profile (CPU-only, completes in seconds) and a `full` profile (GPU). | Running any entrypoint with `--config configs/smoke.yaml` completes without error on a CPU-only machine in under 120 seconds. | §1 D6, §5 F2 |
| FR-4 | A shared configuration loader SHALL be implemented and used by all three packages to parse YAML config files; no magic numbers SHALL appear in source code. | All numeric hyperparameters (batch size, learning rate, max steps, etc.) are read from config files; `grep`-based audit finds no bare numeric literals for hyperparameters in `src/` files. | §4, §5 F2 |
| FR-5 | A shared eval harness SHALL be implemented at `eval/` that can score any model's outputs against `eval/schema.json` and produce a structured result; the `eval/holdout/` directory SHALL be reserved and empty at project start. | `eval/harness.py` (or equivalent) accepts a list of model outputs and a holdout set path, validates each output against `eval/schema.json`, and returns per-item pass/fail plus aggregate metrics. | §5 F3 |
| FR-6 | `eval/schema.json` SHALL define the canonical JSON contract for Chapter 2/3 outputs: an object containing exactly the fields `severity`, `category`, `line`, `issue`, and `suggested_fix`, with types and allowed values specified. | Any output that omits a required field or uses a disallowed value fails schema validation; a conforming output passes. | §1 D2, §5 F3 |
| FR-7 | The Claude Code scaffold SHALL be fully configured: `CLAUDE.md` (project constitution), all slash commands (`/specify`, `/clarify`, `/plan`, `/tasks`, `/implement`, `/smoke`, `/eval`, `/checkpoint`), all skills (`from-scratch-guard`, `qlora-recipe`, `eval-table`, `gguf-export`, `spec-review`), all subagents (`spec-reviewer`, `code-reviewer`, `data-curator`, `experiment-analyst`, `explorer`), all hooks (GPU-budget guard, leakage guard, auto-format & smoke, commit hygiene, session greeter), and MCP server wiring (Hugging Face, Weights & Biases, GitHub). | Each component is present in `.claude/`; each hook and command is exercised at least once and produces the documented behaviour; `specs/STATUS.md` is created and updated. | §6, §5 F4 |
| FR-8 | `specs/STATUS.md` SHALL track every spec's state as one of `{draft, planned, building, done}` and SHALL be updated at every `/checkpoint` invocation. | After each `/checkpoint` run, `STATUS.md` reflects the current state of all specs; no spec transitions state without a corresponding commit. | §6.3, §7 |

### 4.2 Chapter 1 — Architecture

| ID | Requirement | Acceptance Criteria | Source |
|----|-------------|---------------------|--------|
| FR-9 | A tokenizer and data pipeline SHALL be implemented in `ch1_architecture/src/` using pure Python and PyTorch only (no `transformers` library). The tokenizer SHALL support at minimum a hand-rolled or minimal BPE algorithm. | A round-trip encode→decode test passes: `decode(encode(text)) == text` for all printable ASCII inputs in the test corpus. | §2 Ch1, §4, §8 A1 |
| FR-10 | Core transformer modules SHALL be implemented from scratch in pure PyTorch using only tensors and autograd: multi-head self-attention, position-wise MLP, layer normalisation, positional embeddings, and a full transformer block. The `nn.Transformer` module and the `transformers` library SHALL NOT be used anywhere in `ch1_architecture/src/`. | (a) `import transformers` does not appear in any file under `ch1_architecture/src/`. (b) `nn.Transformer` does not appear. (c) Tensor shape tests pass for each module at documented input dimensions. (d) An overfit-a-tiny-batch test: loss decreases to near zero on a batch of ≤ 8 sequences within 100 steps. | §2 Ch1, §4, §8 A2 |
| FR-11 | A custom training loop SHALL be implemented (no `Trainer` abstraction) that trains the Chapter 1 model on a small corpus, logs loss and perplexity to Weights & Biases, and produces coherent (non-random) text samples. | (a) Smoke config completes on CPU. (b) W&B run is created with `loss` and `perplexity` logged. (c) Generated text samples are logged and visually non-random (assessed by developer). | §2 Ch1, §4, §8 A3 |
| FR-12 | A KV-cache SHALL be implemented for the Chapter 1 model's autoregressive inference path. | (a) A correctness test confirms that cached and non-cached inference produce identical logits (within floating-point tolerance) for the same input. (b) Tokens/sec is measured before and after KV-cache and the improvement is recorded in the metrics scorecard. | §2 Ch1, §8 A4 |
| FR-13 | INT8 and 4-bit quantization SHALL be implemented for the Chapter 1 model's inference path. | (a) Quantized inference produces outputs without runtime errors. (b) A speedup curve (tokens/sec vs. precision: fp32 → fp16 → INT8 → 4-bit) is generated and saved to `eval/` or `ch1_architecture/`. (c) A perplexity-vs-quantization tradeoff plot is generated and saved alongside the speedup curve. (d) Both plots are referenced in the README. | §2 Ch1, §8 A5 |

### 4.3 Chapter 2 — Adaptation

| ID | Requirement | Acceptance Criteria | Source |
|----|-------------|---------------------|--------|
| FR-14 | The exact Hugging Face model tag for the base model (Qwen2.5-Coder-1.5B, base or instruct variant) SHALL be fixed and documented in `ch2_adaptation/configs/` before any training or data generation begins. | `configs/full.yaml` contains a `model.name_or_path` field with a fully-qualified HF model tag (e.g., `Qwen/Qwen2.5-Coder-1.5B`); the same tag is referenced in `specs/C1-*.md`. | §1 D4, §8 C1 |
| FR-15 | Baseline performance of the unfine-tuned base model AND a frontier-API few-shot approach SHALL be measured on a stub evaluation set and recorded before any fine-tuning begins. | Baseline scores (schema-validity rate, bug-catch rate) for both the base model and the frontier-API few-shot approach are recorded in `eval/results/` and referenced in `specs/STATUS.md` before spec C4 moves to `building`. | §8 C1 |
| FR-16 | An independent held-out evaluation set SHALL be assembled containing both synthetic-clean and mined-real examples (real Java diffs and lint findings), deduplicated against the training set, and frozen under `eval/holdout/`. This set SHALL NOT be read by any training, data-generation, or prompt-tuning code path. | (a) `eval/holdout/` contains the frozen set with a provenance manifest. (b) The leakage-guard hook blocks any read/write to `eval/holdout/` from training or data-generation code. (c) A deduplication report is present in `eval/holdout/`. | §1 D3, §8 C2 |
| FR-17 | A synthetic training dataset of (buggy Java/Spring code → structured review JSON) pairs SHALL be generated using a frontier API at a cost of approximately $1, cleaned, validated against `eval/schema.json`, and stored with a provenance document. | (a) Every record in the training set passes `eval/schema.json` validation. (b) A provenance document records the frontier model used, prompt template, generation date, and total API cost. (c) The dataset is gitignored and stored on HF Hub or locally. | §1 D3, §8 C3 |
| FR-18 | A QLoRA fine-tune of the chosen base model SHALL be implemented using Hugging Face `transformers`, `peft`, `trl`, and `bitsandbytes` (4-bit NF4). The smoke config SHALL run end-to-end on CPU; the full config SHALL target the GPU box. | (a) `uv run python -m ch2_adaptation.finetune --config configs/smoke.yaml` completes without error on CPU. (b) The full GPU run produces a saved LoRA adapter. (c) W&B logs training loss, eval loss, and learning-rate schedule. | §1 D4, §4, §8 C4 |
| FR-19 | A head-to-head evaluation table SHALL be produced comparing three systems on the frozen holdout set: (1) fine-tuned model, (2) base model (zero-shot or few-shot), (3) frontier-API few-shot. The table SHALL report at minimum schema-validity rate and bug-catch rate for each system. | (a) The table is present in `eval/results/` and embedded in the README. (b) All three systems are scored on the identical holdout set using the shared eval harness. (c) Results are reproducible by re-running `/eval`. | §2 Ch2, §8 C5, §9 |

### 4.4 Chapter 3 — Operation

| ID | Requirement | Acceptance Criteria | Source |
|----|-------------|---------------------|--------|
| FR-20 | A FastAPI application SHALL be implemented that serves the base model (as an early thin end-to-end slice, before the fine-tuned model is available) via a GGUF/llama.cpp or Ollama backend on CPU. | `uvicorn ch3_operation.api.main:app` starts without error; a POST request to the review endpoint returns a response; `pytest ch3_operation/tests` passes. | §8 O0 |
| FR-21 | The fine-tuned LoRA adapter SHALL be merged into the base model weights, converted to GGUF format, and quantized using `llama.cpp` tooling. The resulting GGUF model SHALL pass the eval contract (schema-validity rate no lower than the pre-merge adapter). | (a) A GGUF file is produced and stored outside git (locally or on HF Hub). (b) Running the eval harness against the GGUF-served model produces a schema-validity rate ≥ the adapter's rate on the same sample. | §2 Ch3, §8 O1 |
| FR-22 | The FastAPI endpoint SHALL validate every model output against `eval/schema.json` before returning it to the caller. Outputs that fail validation SHALL be either repaired (if a repair strategy is defined) or rejected with a structured error response; they SHALL NOT be returned as-is. | (a) A test that sends a request designed to elicit malformed output confirms the endpoint returns a 422 or structured error, not the raw malformed output. (b) A test that sends a normal request confirms a schema-valid response is returned. | §2 Ch3, §8 O2 |
| FR-23 | The serving layer SHALL expose Prometheus-compatible metrics including: request latency p50 and p99 (milliseconds), throughput (requests/second), and at least one quality/drift signal (e.g., schema-validity rate over a rolling window). | (a) A `/metrics` endpoint (or equivalent) returns the three metric families. (b) Latency and throughput values are recorded in the metrics scorecard. (c) The quality/drift signal updates with each request. | §2 Ch3, §8 O3, §9 |
| FR-24 | The application SHALL be containerised in a reproducible Docker image that serves the GGUF model on CPU. | `docker build` succeeds; `docker run -p 8000:8000 <image>` starts the server; the review endpoint responds correctly to a test request. | §8 O4 |
| FR-25 | The containerised application SHALL be deployed to Hugging Face Spaces (CPU tier) and at least one cloud free tier (Railway, Render, or Fly.io). Both deployments SHALL be publicly accessible. | (a) A public HF Spaces URL is present in the README and responds to requests. (b) A public cloud-tier URL is present in the README and responds to requests. | §2 Ch3, §8 O5 |
| FR-26 | *(Stretch)* The quantized model SHOULD be made runnable client-side in a web browser via WebLLM/MLC and WebGPU, with a linked demo. | A public browser-based demo URL is present in the README; the model runs inference client-side without a server round-trip. | §1 D5, §8 O6 |

### 4.5 Cross-cutting / Portfolio

| ID | Requirement | Acceptance Criteria | Source |
|----|-------------|---------------------|--------|
| FR-27 | The repository README SHALL contain: (a) a single architecture diagram covering all three chapters, (b) the Chapter 2 head-to-head eval table, (c) the Chapter 1 speedup curve and perplexity-vs-quantization plot, (d) the live HF Spaces link, and (e) a brief description of each chapter. | All five elements are present in `README.md` and render correctly on GitHub. | §8 X1 |
| FR-28 | Smoke-config results SHALL be fully reproducible from a clean checkout using documented `make` or `uv` targets. Full-run instructions (GPU box, Colab/Kaggle) SHALL be documented in the README or a `REPRODUCING.md`. | A developer following only the documented instructions can reproduce smoke results without additional guidance. | §8 X2 |
| FR-29 | Interview-defense notes SHALL be written for each chapter, covering: (a) why naive attention is slow and how KV-cache addresses it, (b) why a small fine-tuned specialist can outperform few-shot prompting on a narrow task, (c) the quantization tradeoff (speed vs. quality). | `specs/` or a `docs/` directory contains a written explanation for each of the three topics; each explanation is technically precise and self-contained. | §8 X3 |

---

## 5. Non-Functional Requirements

### 5.1 Performance

| ID | Requirement | Source |
|----|-------------|--------|
| NFR-1 | Every smoke-config entrypoint SHALL complete end-to-end on a CPU-only machine (no GPU, no accelerator) in under 120 seconds. | §1 D6, FR-3 |
| NFR-2 | The Chapter 1 KV-cache implementation SHALL produce a measurable improvement in tokens/sec relative to the non-cached baseline; the improvement SHALL be recorded in the metrics scorecard. | §8 A4, §9 |
| NFR-3 | The deployed FastAPI endpoint (Chapter 3) SHALL sustain a throughput of at least 1 request/second on the HF Spaces CPU tier under single-user load, with p99 latency recorded and reported. | §9 |

### 5.2 Correctness & Reproducibility

| ID | Requirement | Source |
|----|-------------|--------|
| NFR-4 | All randomness (model initialisation, data shuffling, sampling) SHALL be controlled by explicit seeds set in config files. Running the same config twice SHALL produce bit-identical results for deterministic operations. | §4, §6.1 |
| NFR-5 | The KV-cache implementation SHALL produce logits identical (within IEEE 754 floating-point tolerance, e.g., max absolute difference < 1e-5) to the non-cached implementation for the same input sequence. | FR-12 |
| NFR-6 | No training, data-generation, or prompt-tuning code path SHALL read from or write to `eval/holdout/`. This constraint SHALL be enforced by the leakage-guard hook in addition to code review. | §1 D3, §6.4, FR-16 |
| NFR-7 | All reported metrics SHALL be computed on the frozen holdout set using the shared eval harness. Cherry-picking, post-hoc threshold tuning, or selective reporting of results is prohibited. | §6.1, §10 |

### 5.3 Code Quality

| ID | Requirement | Source |
|----|-------------|--------|
| NFR-8 | All Python source files SHALL pass `ruff check` and `black --check` with zero errors before any commit. The auto-format hook SHALL enforce this automatically. | §4, §6.4 |
| NFR-9 | All Python functions SHALL include type hints on parameters and return values. | §6.1 |
| NFR-10 | No Python function in `src/` SHALL exceed approximately 40 lines of code (excluding docstrings and blank lines). Functions that would exceed this limit SHALL be decomposed. | §6.1 |
| NFR-11 | Every new module SHALL have at least one corresponding `pytest` test. Tests SHALL be co-located in the package's `tests/` directory. | §6.1 |
| NFR-12 | No magic numbers (bare numeric literals used as hyperparameters, thresholds, or configuration values) SHALL appear in `src/` files. All such values SHALL be read from config files. | §4, FR-4 |

### 5.4 Security & Data Hygiene

| ID | Requirement | Source |
|----|-------------|--------|
| NFR-13 | Model weights, training datasets, `.gguf` files, `.env` files, and any file exceeding a defined size threshold SHALL be listed in `.gitignore` and SHALL never be committed to the repository. The commit-hygiene hook SHALL enforce this. | §5, §6.4 |
| NFR-14 | API tokens, secrets, and credentials SHALL be stored exclusively in environment variables and SHALL never appear in any file tracked by git. | §6.6 |
| NFR-15 | The leakage-guard hook SHALL be active in all sessions and SHALL deny any tool call (Read, Write, Edit, Bash) that targets a path under `eval/holdout/` from a training or data-generation context. | §6.4, NFR-6 |

### 5.5 Observability

| ID | Requirement | Source |
|----|-------------|--------|
| NFR-16 | All training and fine-tuning runs SHALL log loss, perplexity, and learning-rate schedule to Weights & Biases. W&B run IDs SHALL be recorded in `specs/STATUS.md` for traceability. | §4, §6.1 |
| NFR-17 | The Chapter 3 serving layer SHALL expose latency (p50, p99), throughput (req/s), and a quality/drift signal as Prometheus-compatible metrics. | FR-23, §9 |

### 5.6 Maintainability & Portability

| ID | Requirement | Source |
|----|-------------|--------|
| NFR-18 | The `.claude/` scaffold (commands, skills, agents, hooks, MCP wiring) SHALL be self-contained and documented such that it can be copied to a sibling project repository with only environment-variable changes. | §1 D7, §6.7, BG-5 |
| NFR-19 | The Chapter 3 Docker image SHALL be fully self-contained: it SHALL NOT require the host to have any ML framework installed beyond Docker itself. | FR-24 |
| NFR-20 | The `eval/schema.json` contract SHALL be the single source of truth for output structure. Any change to the schema SHALL require updating all downstream validators, tests, and documentation in the same commit. | FR-6, FR-22 |

### 5.7 CI / Automation

| ID | Requirement | Source |
|----|-------------|--------|
| NFR-21 | The GitHub Actions CI pipeline SHALL run on every push and pull request, executing: lint (`ruff`, `black`), all `pytest` tests, and all three packages' smoke configs. It SHALL NOT run full training runs. | §4, §5 |
| NFR-22 | The GPU-budget guard hook SHALL block any Bash command that matches patterns indicating a full/long training run (e.g., large `max_steps`, `--config*full*`, `accelerate launch`) unless the environment variable `ALLOW_FULL_RUN=1` is explicitly set in the shell. | §6.4 |

---

## 6. Constraints

| ID | Constraint | Source |
|----|------------|--------|
| CON-1 | **Solo developer.** All design, implementation, testing, and documentation is performed by one person. No external contributors are assumed. | §1 preamble |
| CON-2 | **Weekend-only schedule, approximately six months.** The project must be completable within this time envelope; scope cut-lines are defined in §10 of `project.md`. | §1 preamble |
| CON-3 | **Total LLM API budget ≈ $1.** Frontier API calls are used exclusively for synthetic training-data generation (Chapter 2). All other LLM usage (Claude Code, Copilot) is covered by existing subscriptions. | §1 preamble |
| CON-4 | **No GPU in CI or Claude Code sessions.** All code executed during a Claude Code session or in GitHub Actions CI MUST run on CPU only. Full training/fine-tune runs are launched manually on the GPU box (Colab, Kaggle, or college cluster) outside any session. | §1 D6, §6.1 |
| CON-5 | **Chapter 1 core: pure PyTorch only.** `nn.Transformer`, the `transformers` library, and any other high-level transformer abstraction are prohibited in `ch1_architecture/src/`. | §1 D2, §4 |
| CON-6 | **Eval holdout set is sacred.** No training, fine-tuning, prompt-tuning, hyperparameter search, or data-cleaning code may read from `eval/holdout/`. Violation invalidates the headline metric. | §1 D3, §6.1 |
| CON-7 | **Structured output is a contract.** All Chapter 2 and Chapter 3 model outputs must validate against `eval/schema.json`. The schema may not be relaxed to make a model appear to perform better. | §1 D2, §6.1 |
| CON-8 | **No secrets or large files in git.** Weights, datasets, `.gguf` files, `.env` files, and files over the defined size threshold must never be committed. | §5, §6.4 |
| CON-9 | **Spec-driven development discipline.** No implementation work may begin on a feature without an accepted spec and an accepted plan. The workflow is `/specify → /clarify → /plan → /tasks → /implement`. | §6.1, §7 |
| CON-10 | **One task = one commit.** Each `/implement` invocation covers exactly one task from the task list and produces exactly one commit. | §7 |
| CON-11 | **Base model: Qwen2.5-Coder-1.5B (default), 3B (stretch).** No other base model may be substituted without updating the spec and re-running baselines. | §1 D4 |
| CON-12 | **Edge target: GGUF, CPU-only.** The primary deployment target is a quantized GGUF model served via llama.cpp/Ollama on CPU. GPU-only serving is not an acceptable primary deployment. | §1 D5 |

---

## 7. Technologies

### 7.1 Explicitly Specified

| Layer | Technology | Version / Notes |
|-------|------------|-----------------|
| **Package / env management** | `uv` | All environments managed via `uv`; no `pip install` directly. |
| **Linting / formatting** | `ruff`, `black` | Enforced in CI and via post-edit hook. |
| **Testing** | `pytest` | All packages; smoke configs run in CI. |
| **Ch1 — model** | PyTorch (tensors + autograd only) | No `nn.Transformer`; no `transformers`. |
| **Ch1 — tokenizer** | Hand-rolled or minimal BPE | Pure Python + PyTorch. |
| **Ch1 — experiment tracking** | Weights & Biases (`wandb`) | Loss, perplexity, generated samples. |
| **Ch2 — base model** | Qwen2.5-Coder-1.5B (HF Hub) | Exact tag fixed in spec C1. |
| **Ch2 — fine-tuning** | Hugging Face `transformers`, `peft` (QLoRA), `trl` | Standard HF stack. |
| **Ch2 — quantization (training)** | `bitsandbytes` (4-bit NF4) | QLoRA 4-bit base. |
| **Ch2 — datasets** | Hugging Face `datasets` | Training data loading. |
| **Ch2 — experiment tracking** | Weights & Biases (`wandb`) | Training and eval runs. |
| **Ch3 — inference runtime** | `llama.cpp` / Ollama | GGUF, CPU-only. |
| **Ch3 — API framework** | FastAPI | REST endpoint. |
| **Ch3 — output validation** | Pydantic | Schema validation against `eval/schema.json`. |
| **Ch3 — metrics** | Prometheus-compatible metrics library | Latency, throughput, drift. |
| **Ch3 — containerisation** | Docker | Reproducible image. |
| **Ch3 — deployment (primary)** | Hugging Face Spaces (CPU tier) | Public demo. |
| **Ch3 — deployment (secondary)** | Railway, Render, or Fly.io (free tier) | Public endpoint. |
| **Ch3 — stretch deployment** | WebLLM / MLC, WebGPU | In-browser inference. |
| **Config management** | YAML (Hydra-style) | All packages; smoke and full profiles. |
| **MCP — model/dataset/Hub** | Hugging Face official MCP server | Model search, Hub push, Spaces deploy. |
| **MCP — experiment tracking** | Weights & Biases official MCP server (`wandb/wandb-mcp-server`) | Run queries, result reports. |
| **MCP — repo / eval data** | GitHub official MCP server | PR/diff mining for eval set; repo ops. |
| **Claude Code scaffold** | Native Claude Code primitives | CLAUDE.md, commands, skills, agents, hooks. |

### 7.2 Implied / Inferred

| Layer | Technology | Rationale |
|-------|------------|-----------|
| **Version control** | Git + GitHub | CI via GitHub Actions; MCP server; public repo. |
| **CI** | GitHub Actions | Specified in `project.md §4` and §5 layout. |
| **Python version** | Python ≥ 3.10 | Required by `peft`, `trl`, and type-hint syntax used. |
| **Notebook environment** | Google Colab / Kaggle Notebooks | Free GPU for full training runs. |
| **GPU cluster** | College GPU cluster | Full fine-tune runs. |
| **Frontier API (data gen)** | Unspecified frontier model API (e.g., Anthropic, OpenAI) | ≈ $1 budget for synthetic data generation. |
| **Merge tooling** | `peft` merge utilities or `llama.cpp` convert scripts | Adapter merge before GGUF export. |
| **GGUF conversion** | `llama.cpp` `convert_hf_to_gguf.py` (or equivalent