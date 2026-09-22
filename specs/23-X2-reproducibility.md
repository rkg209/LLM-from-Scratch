# X2 — Reproducibility

| | |
|---|---|
| **State** | done |
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

None open — `make smoke`/`REPRODUCING.md` naming was already literal in the spec text, and
X2 has no dependency on the diagram-tooling question X1 raised.

## Technical plan

See `.claude/plans/Extension.md`, section "X2 — Reproducibility" (built first, per that
plan's D-9: X2 has no blocked dependency and README's Reproducing section links to its
output). Key decisions: `Makefile` at the repo root wrapping existing `uv run` commands
verbatim (D-2); `REPRODUCING.md` at the root, not under `docs/`, since X2 names it that way
and a reproducer looks at the root (D-5); determinism verified by a real test rather than
asserted in prose (D-7); provenance served by one table in `REPRODUCING.md` (D-8).

One deviation from the plan's literal D-2 table: `make setup` syncs all three chapters'
extras (`--extra dev --extra ch1 --extra ch2 --extra ch3`), not just `--extra dev` — see
`progress_report.md`'s X1–X3 entry for why AC-2/AC-3 require it.

## Tasks

All four tasks from the plan are committed:

1. ✅ `Makefile` — `setup`, `check`, `smoke-ch1/ch2/ch3`, `smoke`.
2. ✅ `REPRODUCING.md` — five-minute path, smoke-path scope, GPU-afternoon runbook,
   `ALLOW_FULL_RUN` explanation, determinism section, provenance table, pinned-artifacts
   note.
3. ✅ `eval/tests/test_determinism.py` — `seed_everything` direct check plus a ch1 smoke
   training loop run twice from the same seed, asserted bit-identical.
4. ✅ `README.md#reproducing` now links `REPRODUCING.md`; `.github/workflows/ci.yml` runs
   `make setup && make check && make smoke`.

**Verified this session:** `make setup && make smoke` exits 0; the same with
`GEMINI_API_KEY`/`WANDB_API_KEY`/`HF_TOKEN` unset also exits 0 (AC-2); the determinism
test passes; repo-wide `ruff check . && black --check . && pytest -q` is green (361
passed, 3 skipped). **Not verified, and cannot be from a session:** AC-1 — the actual test
of `REPRODUCING.md` is handing it to a person who has not seen this project and watching
them not get stuck. That remains an open human step, named here rather than claimed.
