# F1 — Repo + tooling skeleton

| | |
|---|---|
| **State** | done |
| **Depends on** | — |
| **Requirements** | FR-1, FR-2, NFR-8, NFR-13, NFR-21 |
| **W&B run** | — |

## Problem

Nothing can be built, reviewed, or trusted until the repository has a shape and a gate. Without a lint/format/test gate from day one, style drift and broken tests accumulate silently and the "green CI" signal becomes meaningless. Without a `.gitignore` that understands this project, the first training run will try to commit a 3 GB checkpoint.

## Scope

The monorepo skeleton: the three package directories, the shared `eval/` and `specs/` directories, a `uv`-managed environment with `ruff`/`black`/`pytest` configured, a `.gitignore` that blocks weights and secrets, a README skeleton, and a GitHub Actions workflow that runs lint plus tests on every push.

## Acceptance criteria

1. The directory tree matches `planning/02-architecture.md` §2 — `ch1_architecture/`, `ch2_adaptation/`, `ch3_operation/`, `eval/`, `specs/`, `scripts/`, `.github/workflows/`.
2. `uv sync --extra dev` builds the environment from a clean checkout without error.
3. `uv run ruff check .` exits 0.
4. `uv run black --check .` exits 0.
5. `uv run pytest -q` exits 0 and discovers tests in all three packages from the repo root.
6. Heavy ML dependencies are **optional extras** (`ch1`, `ch2`, `ch3`) — the base environment installs without `torch`, so CI and the eval tests stay fast.
7. `.gitignore` covers `outputs/`, `*.pt`, `*.gguf`, `ch2_adaptation/data/*.jsonl`, `.env`, `wandb/`, `.venv/`.
8. `.github/workflows/ci.yml` runs lint, format-check, and tests on push and pull request, and never runs a full training job.

## Out of scope

- The config loader and smoke/full profiles → **F2**.
- The eval schema and harness → **F3**.
- The `.claude/` scaffold → **F4**.
- Any model code.

## Clarifications

- **Q:** Python version? → **A:** Pinned to 3.12. 3.13 is ahead of parts of the ML stack (bitsandbytes, llama-cpp-python wheels). *(2026-07-13)*
- **Q:** One package per `pyproject.toml`, or one at the root? → **A:** One at the root with three importable packages and optional extras. A solo monorepo does not need workspace indirection. *(2026-07-13)*

## Technical plan

**Files** — `pyproject.toml` (deps, extras, ruff/black/pytest config), `.gitignore`, `.env.example`, `README.md`, `.github/workflows/ci.yml`, the three package trees.

**Approach** — a single root `pyproject.toml`; packages resolved from `ch*/src` via `tool.setuptools`/hatch package-dirs. Base deps are `pyyaml` and `jsonschema` only. Extras: `ch1` (torch, wandb, matplotlib), `ch2` (transformers, peft, trl, bitsandbytes, datasets, pydantic), `ch3` (fastapi, uvicorn, llama-cpp-python, prometheus-client, pydantic), `dev` (ruff, black, pytest).

## Tasks

- [x] **T1** — root `pyproject.toml` with extras + ruff/black/pytest config · verify: `uv sync --extra dev`
- [x] **T2** — `.gitignore`, `.env.example`, `README.md` skeleton · verify: `git status` shows no artifacts
- [x] **T3** — three package trees + `eval/` + `specs/` · verify: `uv run pytest -q` collects from root
- [x] **T4** — `.github/workflows/ci.yml` · verify: lint + tests run on push
