# C5 — Head-to-head eval table

| | |
|---|---|
| **State** | building |
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

None open — the table format, the two metrics, and the honesty rules were already locked
verbatim in the `eval-table` skill before this spec was planned. C5's job is almost entirely
reuse: wiring already-built pieces (`eval/harness.py::score_outputs`, `baseline.py`, `prompts.py`)
at the frozen holdout instead of the stub set, plus one new module (`evaluate.py`) to produce the
fine-tuned score and render the table.

## Amendments (recorded here per plan, not applied silently)

- **AC-3's "written to `eval/results/baselines.json`" is amended to a separate
  `eval/results/baselines_holdout.json`.** Re-scoring the base model and frontier API on the
  frozen holdout (`baseline_holdout.yaml`, unmodified `baseline.py`) produces a genuinely
  different measurement than C1's original `baselines.json` (scored on the stub set) — the two
  are not interchangeable, and overwriting the original would destroy a separately-published,
  already-complete spec's artifact. `eval_full.yaml`'s adapter run reads
  `baselines_holdout.json`'s `schema_sha256` for AC-8's drift check, not `baselines.json`'s.
  Tradeoff acknowledged: two "baseline" files now live in `eval/results/`; a future README/skill
  note should say which one feeds the head-to-head table (`baselines_holdout.json`, always).
- **`EvaluateConfig` added**, mirroring `BaselineConfig`'s smoke/full split. `is_smoke` gates on
  `use_4bit` (matching `FinetuneConfig`'s convention, since this reloads the identical base model
  + adapter that training produced) rather than a model-tag check directly — a documented,
  slightly indirect proxy, consistent with precedent elsewhere in this file.
- **`model.py` gained two shared helpers** (`generate_completions`, `load_base_model`), extracted
  from `make_hf_generator` (C1) and `load_base_model_for_training` (C4) respectively, so
  `evaluate.py`'s adapter generator doesn't duplicate the chat-template/generate/decode loop or
  the fp32-CPU/4-bit-GPU loading branch a second time. Confirmed a pure, behavior-preserving
  refactor — existing C1/C4 tests still pass unchanged.

## Technical plan

See `.claude/plans/09-C5-head-to-head-eval-table.md` for full detail. Summary:

**New files** — `ch2_adaptation/src/ch2_adaptation/evaluate.py`;
`ch2_adaptation/configs/{eval_smoke.yaml,eval_full.yaml,baseline_holdout.yaml}`;
`ch2_adaptation/tests/{test_evaluate.py,test_evaluate_config.py,test_evaluate_main.py}`.

**Modified** — `baseline.py` (test coverage added for the `schema_sha256` field that was already
present, added ahead of time in C1 T5); `config.py` (`EvaluateConfig`); `model.py` (the two shared
helpers above); `README.md` (the `EVAL_TABLE_START`/`END` marker pair around the Chapter 2
results table).

**Key design points:**

- `score_adapter_on_holdout(prompts, generate, holdout_path)` takes an injected `generate`
  callable rather than the plan's originally-stated `(adapter_path, holdout_path, config)`
  signature — mirrors `baseline.py::score_system`'s established shape, keeping the scoring logic
  testable with a fake (no torch, no adapter load). It forwards directly to `score_system` — the
  only scoring path (AC-1), never a bespoke comparison.
- `verify_schema_unchanged` (AC-8) is checked before any generation or scoring happens in a real
  run, against `baseline_holdout.json`'s recorded hash.
- `render_table`'s output matches the `eval-table` skill's example table byte-for-byte (a test
  diffs it directly); `update_readme_results_section` is idempotent and hardened against
  malformed marker pairs (duplicate or reversed markers raise rather than silently splicing
  wrong).
- `make_adapter_generator` loads the base model, attaches the adapter via
  `peft.PeftModel.from_pretrained(..., is_trainable=False)` (explicit, not relying on the
  library default), and scores zero-shot — the fine-tuned model needs no few-shot examples.
- Never reads `eval/holdout/` directly anywhere in `evaluate.py` (CON-6) — only ever through
  `config.holdout_path`, gated by the existing leakage guard when it actually points at the real
  holdout.

## Tasks

All five code tasks from `.claude/plans/09-C5-head-to-head-eval-table.md` are committed:

1. ✅ `baseline.py`'s `schema_sha256` field — already present from C1, test coverage added.
2. ✅ `evaluate.py::score_adapter_on_holdout` + `verify_schema_unchanged`, fake-driven tests.
3. ✅ `render_table` + `update_readme_results_section`, matching the skill's example
   byte-for-byte; README markers added.
4. ✅ `EvaluateConfig` + the three new configs; a real `use_4bit` config bug caught and fixed by
   the config-loading tests before it could reach a real run.
5. ✅ `evaluate.py::main()` CLI wiring — verified live end to end: training a smoke adapter then
   running the smoke eval config completed in ~4 seconds, loading the real base model, attaching
   the real adapter, scoring, and writing results.
6. ✅ This spec/STATUS update.

**Remaining before this spec can move to `done`:** the full run genuinely cannot happen until
C1's baselines, C2's frozen holdout, and C4's full adapter all exist simultaneously — none of
which is true yet (C1's AC-5 is still open, C2 is paused on `GITHUB_TOKEN`, C4's full GPU run is
manual-only and hasn't happened). Once they do, the runbook is: run `baseline_holdout.yaml`
(writes `baselines_holdout.json`), then `eval_full.yaml` (reads that file's `schema_sha256`,
writes `finetuned.json`), then render and publish the table to `README.md`. **AC-7's honest-loss
prose is written by a human** after real numbers exist — no code auto-generates that judgment,
and none should. The W&B run ID (AC-5) gets recorded in `specs/STATUS.md` after the real runs,
same deferred-completion shape as every other cross-dependent spec in this project.
