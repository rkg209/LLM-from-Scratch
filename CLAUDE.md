# LLM Engineering: From Architecture to Edge

## What this repo is

A monorepo with three independent packages that share `eval/` and `specs/`:

- `ch1_architecture/` — a GPT-style LM written from scratch in **pure PyTorch** (+ KV-cache, + quantization).
- `ch2_adaptation/` — **QLoRA fine-tune** of `Qwen/Qwen2.5-Coder-1.5B-Instruct` into a Java code reviewer that emits structured JSON.
- `ch3_operation/` — **GGUF + FastAPI + monitoring + deploy** (CPU edge).

The chapters have no runtime dependency on each other; they communicate only through files (`eval/`, `outputs/`, HF Hub). See `README.md` for the diagram and `planning/` for the locked design documents.

## Non-negotiables

1. **NEVER start a full training/fine-tune run in a session.** Run the `smoke` config only (CPU, seconds). Full runs are launched manually on the GPU box. The GPU-budget hook enforces this.
2. `ch1_architecture/src/` is **pure PyTorch** — no `transformers`, no `nn.Transformer`, no `nn.MultiheadAttention`, no `nn.LayerNorm`, no `F.scaled_dot_product_attention`. Hand-written or it doesn't count.
3. **The holdout set is sacred.** No training, data-generation, or prompt-tuning code may read or write `eval/holdout/`. Leakage invalidates the headline metric. The leakage hook enforces this.
4. **Structured output is a contract.** Every Ch2/Ch3 model output must validate against `eval/schema.json`. The schema is never relaxed to make a model look better.
5. **Seeded, config-driven, no magic numbers.** Every hyperparameter comes from a YAML config. Every entrypoint takes `--config`.
6. **No secrets, weights, datasets, or `.gguf` files in git.** The commit-hygiene hook enforces this.

## Build & test

```bash
# the default check — run this before any commit
uv run ruff check . && uv run black --check . && uv run pytest -q

# smoke runs (the only "run" commands to use in a session)
uv run python -m ch1_architecture.train    --config ch1_architecture/configs/smoke.yaml
uv run python -m ch2_adaptation.finetune   --config ch2_adaptation/configs/smoke.yaml
uv run python -m ch3_operation.serve       --config ch3_operation/configs/smoke.yaml
```

Heavy dependencies are optional extras — `uv sync --extra ch1` (torch), `--extra ch2` (transformers/peft/trl), `--extra ch3` (fastapi/llama-cpp). The base env plus `--extra dev` is enough for lint and the shared eval tests.

## Workflow

Specs live in `specs/`; their state is tracked in `specs/STATUS.md`.

```
/specify → /clarify → [human accepts] → /plan → /tasks → loop(/implement) → /checkpoint
```

**Never implement without an accepted spec and plan.** One task = one commit. For ML specs, the full run happens outside the session on the GPU box; its numbers come back via `/eval`.

## Conventions enforced in review

Type hints on every function. Functions under ~40 lines. Config-driven, no magic numbers. W&B logging on every train/eval run. A test for every new module. **Honest metrics** — report failures, never cherry-pick.
