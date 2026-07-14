# C5 — Head-to-head eval table

| | |
|---|---|
| **State** | draft |
| **Depends on** | C4 |
| **Requirements** | FR-19, NFR-7, BG-2 |
| **W&B run** | — |

## Problem

**This is the headline result of the entire project.** Everything in Chapter 2 exists to produce this one table, and a hiring manager who reads nothing else will read this.

Its value is entirely in its methodology. Three systems, one frozen holdout set, one harness, numbers reported exactly as they came out. The moment any of that bends — a different sample for one system, a re-run kept because it looked better, a failing case quietly dropped — the table becomes marketing, and an interviewer who probes for thirty seconds will find out.

The most likely outcome is worth naming in advance: the fine-tuned model **wins decisively on schema-validity** (a small model trained on one output format beats a generalist prompted for it) and probably **loses to the frontier API on bug-catch rate**. That is a good, honest, defensible result — a specialist that is 100× cheaper, private, and structurally reliable, trading some raw capability. It should be reported as exactly that, not massaged into a clean sweep.

## Scope

`evaluate.py`: score the fine-tuned adapter on the frozen holdout; re-score the base model and frontier API on the *same* set; produce the comparison table and plots; publish to the README.

## Acceptance criteria

1. All three systems — fine-tuned, base zero-shot, frontier-API 3-shot — are scored on the **identical** frozen `eval/holdout/holdout.jsonl` using **`eval/harness.py`** (FR-19b). No system is scored by any other code path.
2. The table reports **schema-validity rate** and **bug-catch rate** with **`n`** for each system (FR-19).
3. Results are written to `eval/results/finetuned.json` and `eval/results/baselines.json`, and the table is embedded in `README.md` (FR-19a).
4. `per_sample` results are retained, so every aggregate number can be traced back to the records that produced it.
5. `/eval` reproduces the table from the saved artifacts; the W&B run ID for the adapter is recorded in `specs/STATUS.md` (FR-19c, NFR-16).
6. **Failures are reported.** Every holdout record the model got wrong is counted in the denominator. No sample is excluded for any reason (NFR-7).
7. **If the fine-tuned model loses on a metric, the README says so, in the table and in words**, and makes the cost/latency/privacy argument instead of hiding the loss.
8. `eval/schema.json` is byte-identical to its state when the baselines were measured. A schema that changed mid-comparison invalidates the comparison (CON-7).

## Out of scope

- Improving the model in response to what the table shows. Tuning against the holdout is exactly the leakage this project is built to avoid — a disappointing table is a finding to publish and analyze, not a bug to fix by iterating on the eval set.
- Serving → **O0/O1**.

## Clarifications

*(filled by `/clarify`)*

## Technical plan

*(filled by `/plan`)*

## Tasks

*(filled by `/tasks`)*
