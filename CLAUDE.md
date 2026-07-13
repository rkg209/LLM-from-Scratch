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
7. **NEVER put a `Co-Authored-By:` trailer in a commit message.** No `Co-Authored-By: Claude ...`, no co-author trailer of any kind. It breaks pushing to GitHub for this repo. Commit messages end at the last line of the body.

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

## `progress_report.md` — append after every meaningful change

`progress_report.md` is the project's development log. **Append an entry to it after every meaningful change** — a completed task, a resolved bug, a design decision, a reversed decision. Do this as part of the work, not as a separate chore at the end; a log written from memory a week later is fiction.

Each entry answers three questions:

- **What** changed — the concrete change.
- **Why** — the reasoning. This is the part that exists nowhere else: git shows what changed, the code shows what it is, but neither records why it was done this way and not the obvious other way.
- **How** — the approach taken.

And, whenever something went wrong: **the problem, what was tried that did not work, and what finally did.** The failed attempts are the most valuable content in the file — record them, do not tidy them away.

Rules: newest entries go at the **bottom**; **never rewrite an earlier entry.** If a past decision turns out to be wrong, say so in a *new* entry. A log that has been cleaned up has stopped being evidence.

## Conventions enforced in review

Type hints on every function. Functions under ~40 lines. Config-driven, no magic numbers. W&B logging on every train/eval run. A test for every new module. **Honest metrics** — report failures, never cherry-pick.
