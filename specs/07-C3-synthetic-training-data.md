# C3 — Synthetic training data

| | |
|---|---|
| **State** | draft |
| **Depends on** | C1 |
| **Requirements** | FR-17, CON-3, CON-6, CON-7 |
| **W&B run** | — |

## Problem

A 1.5B base model does not emit schema-conformant JSON code reviews on request. It needs to be shown, a few thousand times, what one looks like. Nobody is going to hand-write a few thousand, and no public dataset of (buggy Java/Spring snippet → structured review JSON) exists in the shape this schema demands.

So the data is distilled from a frontier model for about a dollar. The honest framing matters and is load-bearing for the whole project: this is **distillation of a large model's capability into a small specialist**, not a small model spontaneously outperforming a large one. Every number in the eval table rests on that framing being stated plainly, and on the provenance being documented well enough that a skeptic can check it.

## Scope

`data_gen.py`: source buggy Java/Spring snippets, call the frontier API with a fixed prompt template, validate each response against the schema, drop failures, split train/val, and write a full provenance record.

## Acceptance criteria

1. Every record in `data/train.jsonl` and `data/val.jsonl` validates against `eval/schema.json`. Records that fail are **dropped and logged, never hand-repaired** (FR-17a) — repaired records teach the model the repairer's distribution, not the target.
2. `data/provenance.json` records `run_id`, `generated_at`, `frontier_model` (exact tag), `prompt_template_hash` (SHA-256), `n_requested`, `n_valid`, `n_skipped`, `estimated_cost_usd`, `actual_cost_usd`, `split_seed` (FR-17b).
3. **The budget guard aborts before spending** if the pre-run estimate exceeds $1.50 (CON-3). `PRICE_PER_1K_TOKENS` is read from the environment, never hardcoded.
4. Actual cost is read from API response metadata — the estimate is not recorded as the actual.
5. Train/val split is 90/10 with a **seeded** split; the seed is in the provenance.
6. Records are deduplicated by normalized code hash within the training set.
7. `data_gen.py` never reads `eval/holdout/` — not to dedupe, not to sanity-check (CON-6). Deduplication against the holdout is C2's job, done at curation time.
8. The generated `.jsonl` files are **gitignored**; only `provenance.json` is committed.
9. The smoke path generates a handful of records against a stubbed/mocked API — CI must never spend money.

## Out of scope

- The holdout set → **C2**. The fine-tune → **C4**.
- Human review of every generated label. The schema validator plus the independent eval set are the quality mechanism; a hand-audited training set is not affordable at this budget, and the write-up says so.

## Clarifications

*(filled by `/clarify` — open questions: where the buggy Java snippets come from (mined via GitHub MCP, or synthesized by injecting bugs into clean code); target record count at the ≈$1 budget; which frontier model.)*

## Technical plan

*(filled by `/plan`)*

## Tasks

*(filled by `/tasks`)*
