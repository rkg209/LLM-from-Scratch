# X2 — Reproducibility

| | |
|---|---|
| **State** | draft |
| **Depends on** | F2 |
| **Requirements** | FR-28, NFR-4, BG-4 |
| **W&B run** | — |

## Problem

An unreproducible result is an anecdote. The claim this project makes — a small fine-tuned specialist beats few-shot prompting on structured output, and here is the table — is only worth anything if a skeptic can check it, and the skeptic will not email for help. They will follow the instructions once, hit a missing step, and conclude the project is decoration.

There are two audiences with different budgets, and both must be served: someone who will spend **five minutes** (they run the smoke configs and see the code path work end-to-end) and someone who will spend **a GPU afternoon** (they re-run the full fine-tune and reproduce the table). The five-minute path must not require a GPU, an API key, or a W&B account.

## Scope

`REPRODUCING.md` plus the `make`/`uv` targets that make the smoke path a single command, and documented instructions for the full GPU runs.

## Acceptance criteria

1. **From a clean checkout, a developer following only the written instructions reproduces the smoke results with no additional guidance** (FR-28) — the real test is handing it to someone and watching them not get stuck.
2. The smoke path requires **no GPU, no API key, and no W&B account**. W&B falls back to offline mode; the frontier API is stubbed.
3. `make smoke` (or the documented `uv run` equivalents) runs all three packages' smoke configs and exits 0.
4. Full-run instructions cover the GPU box (Colab / Kaggle / cluster): environment, expected wall-clock, expected cost, and the exact commands — including `ALLOW_FULL_RUN=1`, with an explanation of why that guard exists.
5. **Re-running the same config with the same seed produces identical results** for deterministic operations (NFR-4). Where full determinism is not achievable (CUDA nondeterminism), the limitation is stated rather than quietly ignored.
6. Every published number names the config file and the W&B run that produced it.
7. Model artifacts are pulled from HF Hub by ID, pinned to a revision — not "the latest adapter", which is a moving target.

## Out of scope

- Reproducing on Windows, or on a GPU that is not CUDA. Documented as untested rather than claimed.
- Bit-exact reproduction of the full GPU run across different hardware — physically not on offer; say so.

## Clarifications

*(filled by `/clarify`)*

## Technical plan

*(filled by `/plan`)*

## Tasks

*(filled by `/tasks`)*
