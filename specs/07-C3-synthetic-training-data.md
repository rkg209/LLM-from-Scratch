# C3 — Synthetic training data

| | |
|---|---|
| **State** | done (2026-09-17, `d78179f`) |
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

Resolved during `/plan` (full reasoning: `.claude/plans/07-C3-synthetic-training-data.md`):

- **Snippet source = synthetic bug-injection into a curator-collected pool of clean Java/Spring
  code**, not live GitHub-MCP mining. C2 already mines real snippets for the holdout; reusing
  that same live pool for ~300 training records doesn't scale within a session and risks
  source-pool overlap that dedup would then have to clean up after the fact. Synthetic injection
  is disjoint from the holdout by construction and scales to hundreds of records without
  additional mining sessions.
- **Generator model = Gemini 2.5 Flash**, the same frontier model C1 benchmarks as the baseline.
  This is deliberately circular in the narrow sense C1's own spec flagged as an open fork; the
  honest framing is recorded as-is: the project's claim is "distillation — the specialist
  approaches its teacher at a fraction of the size/cost/latency," not "beats a bigger model."
  Reusing `frontier.GeminiClient` verbatim also means $0 real cost on the free tier and no new
  API key.
- **Target size ≈300 records**, generated from a 150-entry seed pool of clean snippets with up
  to 2 bug-injection variants each (`variants_per_snippet=2` in `datagen_full.yaml`), split
  90/10 → 270 train / 30 val after any budget/dedup losses.

## Amendments (recorded here per plan, not applied silently)

- **(2026-09-16) Line-label check added (user decision).** Beyond schema validation, a record is
  dropped (never corrected) if its `line` is not within `eval.metrics.LINE_TOLERANCE` of a line
  the injection actually changed; the count is `n_dropped_line_mismatch` in provenance.
  Also fixed pre-run: the Gemini client forced the 5-field review schema (no `code`, so
  0 records), and generation now checkpoints and resumes. See progress_report.md.

- **AC-3** names `PRICE_PER_1K_TOKENS` as the environment variable read for cost estimation.
  C1 already shipped `PRICE_PER_1K_INPUT_USD`/`PRICE_PER_1K_OUTPUT_USD` instead
  (`ch2_adaptation/src/ch2_adaptation/frontier.py::estimate_cost_usd`), and C3 reuses that
  function unmodified rather than inventing a second cost-estimation path. **AC-3 is amended**
  to the two-variable name actually shipped. The $1.50 budget cap itself is not hardcoded in
  logic — it is `DataGenConfig.max_budget_usd`, read from `datagen_full.yaml`, so the cap can
  change without a code edit (CLAUDE.md's "every hyperparameter from config" rule).
- **The hash-only holdout index is not named `eval/holdout_hashes.txt`** as C2's plan originally
  proposed. `leakage_guard.py`'s path check is a bare substring match on the `eval/holdout`
  prefix with no distinction between the frozen directory and a same-prefixed file next to it,
  and writes have no override — so that exact filename can never be written from a session, by
  design. C3's `data_gen.py::HOLDOUT_HASHES_PATH` instead points at `eval/frozen_hashes.txt`, a
  plan amendment shared with C2 (see C2's own progress-report entries for the discovery).
- **A pre-existing `commit_hygiene.py` bug was found and fixed while confirming AC-8's
  "gitignored `.jsonl`" guarantee.** The hook's generated-data check used a recursive
  `startswith("ch2_adaptation/data/")` match, broader than `.gitignore`'s own single-level glob
  (`ch2_adaptation/data/*.jsonl`) — it would have wrongly blocked re-staging the legitimately
  committed `ch2_adaptation/data/seed_snippets/clean_pool.jsonl` as if it were generated output.
  Narrowed to match `.gitignore` exactly; verified both directions (clean_pool no longer
  flagged, a stray `train.jsonl` still denied) through the actual hook script.

## Technical plan

See `.claude/plans/07-C3-synthetic-training-data.md` for full detail. Summary:

**New files** — `ch2_adaptation/src/ch2_adaptation/data_gen.py`;
`ch2_adaptation/data/seed_snippets/{clean_pool.jsonl,README.md}` (150 hand-templated clean
Java/Spring methods, source material — committed, not generated output);
`ch2_adaptation/configs/{datagen_smoke.yaml,datagen_full.yaml}`;
`ch2_adaptation/tests/{test_data_gen.py,test_data_gen_main.py,test_data_gen_config.py}`.

**Modified** — `config.py` (`DataGenConfig`, `load_data_gen_config`, mirroring `BaselineConfig`'s
smoke/full split: a smoke run may never write the committed `data/provenance.json`);
`.claude/hooks/commit_hygiene.py` (the false-positive fix above).

**Key design points:**

- One frontier call injects and labels a bug together
  (`BUG_INJECTION_PROMPT_TEMPLATE`, distinct from `prompts.py`'s locked review-only prompt) —
  a two-call design (inject, then separately describe) risks the description drifting from what
  was actually inserted.
- `inject_and_label` never hand-repairs a malformed response (AC-1): a response that fails to
  parse, is missing `code`, or fails `ReviewOutput` validation is dropped and counted in
  `n_skipped`, never patched.
- Dedup is two-layered and shares one primitive with C2: `eval.dedup.code_hash` (normalize,
  hash) backs both `dedup_against_holdout` (against the hash-only index, never
  `eval/holdout/` itself — CON-6/AC-7) and `dedup_within_training_set` (AC-6).
- `_check_budget` runs strictly before any network call and is skipped entirely for a smoke run
  (the stub client costs nothing); a real run whose pre-flight estimate exceeds
  `max_budget_usd` aborts before spending anything (AC-3).
- `_load_holdout_hashes` tolerates a missing hash-index file only for a smoke run; a real run
  with a missing/misconfigured index raises rather than silently deduping against an empty set
  (a gap the code-reviewer caught and which was fixed before this task shipped).
- `build_provenance` mirrors `DataProvenance` per `planning/04-database-design.md` §3.3
  field-for-field (AC-2).

## Tasks

All six tasks from `.claude/plans/07-C3-synthetic-training-data.md` are committed:

1. ✅ `DataGenConfig` + smoke/full configs + tests.
2. ✅ 150-entry clean seed snippet pool + README, confirmed unique and disjoint from the
   stub/holdout sets.
3. ✅ `data_gen.py` core (`inject_and_label`, `dedup_against_holdout`,
   `dedup_within_training_set`, `split_train_val`, `build_provenance`) + fake-client tests.
4. ✅ `main()` wiring; smoke run (`datagen_smoke.yaml`) green in well under a second, no
   `GEMINI_API_KEY`, writes only to gitignored `outputs/`.
5. ✅ `.gitignore` confirmed correct as-is; `commit_hygiene.py`'s false-positive fixed.
6. ✅ This spec/STATUS update.

**Remaining before this spec can move to `done`:** the full generation run
(`GEMINI_API_KEY=... PRICE_PER_1K_INPUT_USD=... PRICE_PER_1K_OUTPUT_USD=... uv run python -m
ch2_adaptation.data_gen --config ch2_adaptation/configs/datagen_full.yaml`) is manual-only per
CLAUDE.md non-negotiable #1 and the GPU-budget hook, and it additionally needs C2's
`eval/frozen_hashes.txt` to exist first (C2 is currently paused after its own task 3, blocked on
`GITHUB_TOKEN` for the GitHub-MCP mining step — see `progress_report.md`). `data/train.jsonl`,
`data/val.jsonl`, and the committed `data/provenance.json` do not exist yet. `specs/STATUS.md`
records C3 as `building`, not `done`, until that run happens and its output is committed.

**Done (2026-09-17).** Superseding the paragraph above, which predates C2's freeze. The real run
completed: `provenance.json` committed in `d78179f` with 300 requested, 226 valid (203 train / 23
val), 74 skipped (1 schema failure, 13 line-label drops, 60 dedup removals), $0.00.
Acceptance criteria: AC-1 all 226 validate, and failures are dropped, never repaired. AC-2 every
field present, plus the additive `n_dropped_line_mismatch`. AC-3 guard ran before any call
(estimate $0.00 because free-tier prices are 0). AC-4 actual cost is summed from response usage
metadata. AC-5 seed 42 recorded. AC-6 0 internal duplicates. AC-7 only the hash index is read,
0 collisions. AC-8 the `.jsonl` files are gitignored (verified). AC-9 the smoke stub now yields
real records (it yielded 0 before `0b0669b`).
**Target missed, stated plainly:** 226, not ~300. Dedup removed 60, because each snippet is sent
twice and both variants sometimes get the same bug. Four pre-run and mid-run bugs were found and
fixed (`0b0669b`, `fdc9a95`, `0fc1183`, `24cee2c`); see progress_report.md. Known data
limitations: free-text categories (70 distinct), severity skew (0 info, 8 minor), near-duplicate
bugs across templated seed methods.
