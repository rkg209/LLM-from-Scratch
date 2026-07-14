# C1 — Task + schema + base model fixed, baselines measured

| | |
|---|---|
| **State** | draft |
| **Depends on** | F3 |
| **Requirements** | FR-14, FR-15, CON-11, NFR-7 |
| **W&B run** | — |

## Problem

Two things must be nailed down before a single training token is spent, and both are cheap now and expensive later.

**The base model tag.** Changing it after data generation means re-running every baseline and every comparison (CON-11). The prompt format, the chat template, and the LoRA target modules all follow from it.

**The baselines.** This is the one that gets skipped, and skipping it quietly ruins the project. A baseline measured *after* the fine-tune exists is a baseline measured by someone who knows what number they need it to be. Even with no bad intent, the prompt gets tweaked, the parse gets a little more forgiving, and the comparison stops being a comparison. Measure the base model and the frontier API **first**, freeze the numbers, and the eval table means something.

## Scope

Fix `Qwen/Qwen2.5-Coder-1.5B-Instruct` in config; write the `ReviewOutput` Pydantic model mirroring `eval/schema.json`; write `baseline.py`; measure base-model zero-shot and frontier-API 3-shot on a stub set and record them.

## Acceptance criteria

1. `ch2_adaptation/configs/full.yaml` contains `model.name_or_path: Qwen/Qwen2.5-Coder-1.5B-Instruct`, and this spec is the only place the choice is made (FR-14).
2. `ch2_adaptation/schema.py::ReviewOutput` is a Pydantic model with `extra="forbid"`, whose fields and constraints match `eval/schema.json` exactly. **On import it structurally compares itself against `eval/schema.json` and raises if they have diverged** (NFR-20) — the two can never silently drift apart.
3. `baseline.py` scores the base model zero-shot **and** a frontier API 3-shot on the same stub set, through `eval/harness.py` — not through ad-hoc scoring code.
4. Both scores (schema-validity rate, bug-catch rate) are written to `eval/results/baselines.json` with the frontier model tag and an ISO timestamp, **atomically** (write to `.tmp`, rename).
5. `baselines.json` exists and is committed **before spec C4 moves to `building`** (FR-15). This ordering is the point of the spec.
6. The smoke path runs on CPU in under 120 seconds: fp32, no bitsandbytes, a handful of stub examples.
7. The frontier API spend for this spec is recorded and counts against the ≈$1 total budget (CON-3).

## Out of scope

- The frozen holdout set → **C2**. C1's baselines run against a small **stub** set precisely because the real holdout does not exist yet — and once it does, the baselines are re-run against it under **C5**.
- Generating training data → **C3**. Fine-tuning → **C4**.

## Clarifications

*(filled by `/clarify` — open question: which frontier model and API for the 3-shot baseline, and is it the same one used for data generation in C3? Using the same model for both makes the distillation story cleaner but the comparison more circular.)*

## Technical plan

*(filled by `/plan`)*

## Tasks

*(filled by `/tasks`)*
