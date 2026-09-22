# Plan: C5 — Head-to-head eval table

## Context

This is the project's headline artifact (BG-2): three systems — fine-tuned adapter, base model
zero-shot, frontier API 3-shot — scored on the identical frozen holdout by the identical harness,
reported exactly as measured. `specs/09-C5-head-to-head-eval-table.md` has no open
`/clarify`-worthy forks (unlike C3): the table format, the two metrics, and the honesty rules are
already locked verbatim in the `eval-table` skill. C5's job is almost entirely **reuse**: wiring
already-built pieces (`eval/harness.py::score_outputs`, `baseline.py`, `prompts.py`) at the
frozen holdout instead of the stub set, plus one new module to produce the fine-tuned score and
render the table.

**Real dependency chain, not just the spec's stated `Depends on: C4`:** a genuine full run of
this spec needs C1's `eval/results/baselines.json` (still open — C1's AC-5), C2's frozen
`eval/holdout/holdout.jsonl` (staged only per C2's plan, not yet moved in), and C4's full adapter
(manual GPU run, not yet done) to all exist simultaneously. None of that blocks writing and
testing `evaluate.py`'s code now — every other C-spec in this project follows the same "build
and test the code against fixtures now, run the real thing by hand later" shape, and this plan
follows it too.

**Key reuse decision:** the spec says "re-score the base model and frontier API on the *same*
set." Rather than writing new scoring code, `baseline.py` (from C1) already takes a
`stub_set_path` in its config — `BaselineConfig.stub_set_path`. Re-scoring against the holdout is
just a **new config**, `baseline_holdout.yaml`, pointing `stub_set_path` at
`eval/holdout/holdout.jsonl` and run with `EVAL_CONTEXT=1`. `baseline.py` itself needs zero
changes for this. (The field name `stub_set_path` stays as-is rather than being renamed —
renaming a shipped, tested field for a one-word accuracy improvement isn't worth the churn; a
comment at the new config's point of use is enough.)

## Files

**Modified — `ch2_adaptation/src/ch2_adaptation/baseline.py`**
Add a `schema_sha256` field to `build_baselines_doc`'s output (SHA-256 of `eval/schema.json` at
scoring time) — needed so C5's AC-8 ("`eval/schema.json` is byte-identical to its state when the
baselines were measured") has something concrete to check against. Small, additive change; existing
`baselines.json` consumers are unaffected since it's a new field, not a renamed one.

**New — `ch2_adaptation/src/ch2_adaptation/evaluate.py`**
- `score_adapter_on_holdout(adapter_path, holdout_path, config) -> EvalResult` — loads the base
  model + LoRA adapter (`peft.PeftModel.from_pretrained`), generates on each holdout record using
  `prompts.format_zero_shot` (the fine-tuned model needs no few-shot examples — that's the point
  of fine-tuning), scores via `eval.harness.score_outputs` (the **only** scoring path, per AC-1 —
  not a bespoke comparison).
- `verify_schema_unchanged(recorded_sha256: str) -> None` — recomputes the current
  `eval/schema.json` hash and raises if it differs from the one recorded when baselines were
  measured (AC-8). Reused by both the baseline-re-run config and this module.
- `render_table(finetuned: EvalResult, base: EvalResult, frontier: EvalResult) -> str` — the
  exact markdown shape from the `eval-table` skill: `| System | Schema-validity | Bug-catch | n |`
  with one row per system, `n` always stated (AC-2, AC-6 — every sample counts, nothing excluded).
- `update_readme_results_section(table_md: str, readme_path) -> None` — replaces content between
  two marker comments (`<!-- EVAL_TABLE_START -->` / `<!-- EVAL_TABLE_END -->`, added to
  `README.md` once) so re-running `/eval` regenerates the section idempotently (AC-3, AC-5).
- `main()` — requires `EVAL_CONTEXT=1` to read the holdout at all (the leakage guard enforces
  this regardless); writes `eval/results/finetuned.json` via `write_json_atomic`.

**New — `ch2_adaptation/configs/{eval_smoke.yaml,eval_full.yaml,baseline_holdout.yaml}`**
- `eval_smoke.yaml` — CI-safe: points at `eval/stub/stub_eval_smoke.jsonl` (never the holdout),
  uses `SMOKE_MODEL_TAG`, no `EVAL_CONTEXT` needed. Proves the load → generate → score → write
  path in seconds, same convention as every other smoke config in this project.
- `eval_full.yaml` — the real run: `LOCKED_MODEL_TAG` + the trained adapter path +
  `eval/holdout/holdout.jsonl`. Requires `EVAL_CONTEXT=1`; run by hand once C1/C2/C4 all have
  real output.
- `baseline_holdout.yaml` — re-runs `baseline.py` (unmodified) against
  `eval/holdout/holdout.jsonl` instead of the C1 stub set, for the base-model and frontier-API
  rows.

**New — tests:** `ch2_adaptation/tests/test_evaluate.py` —
`score_adapter_on_holdout` driven by a fake generator (same pattern as `test_baseline.py`),
`render_table` output matches the skill's exact format byte-for-byte against a fixture,
`verify_schema_unchanged` raises on a deliberately mismatched hash and passes on a matching one.
`ch2_adaptation/tests/test_baseline.py` gets one addition covering the new `schema_sha256` field.

**Modified — `README.md`** — add the `<!-- EVAL_TABLE_START/END -->` marker pair (empty/placeholder
table) so `update_readme_results_section` has something to target from the first run onward.

## Task sequence

1. `baseline.py`'s `schema_sha256` addition + test. Verify:
   `uv run pytest ch2_adaptation/tests/test_baseline.py -q`.
2. `evaluate.py::score_adapter_on_holdout` + `verify_schema_unchanged`, driven by fakes, no
   torch/network. Verify: `uv run pytest ch2_adaptation/tests/test_evaluate.py -q`.
3. `render_table` + `update_readme_results_section` + README markers, tested against a fixed
   fixture matching the `eval-table` skill's example table verbatim.
4. The three new configs; smoke path exercised in-session
   (`uv run python -m ch2_adaptation.evaluate --config ch2_adaptation/configs/eval_smoke.yaml`).
5. `main()` CLI wiring; document the full-run runbook (`EVAL_CONTEXT=1 uv run python -m
   ch2_adaptation.evaluate --config .../eval_full.yaml`, preceded by the `baseline_holdout.yaml`
   re-run) as a "remaining before `done`" note, since it genuinely cannot run until C1/C2/C4 all
   land — same deferred shape as every other cross-dependent spec here.
6. Spec/STATUS updates — note explicitly that **AC-7's honest-loss prose is written by a human**
   after real numbers exist ("the fine-tuned model wins/loses on X, here's the cost/latency/
   privacy argument") — no code auto-generates that judgment, and none should. Move C5 to
   `planned`/`building` in `specs/STATUS.md`.

## Verification

- `uv run ruff check . && uv run black --check . && uv run pytest -q` after every task.
- Smoke run confirmed CI-safe (no `EVAL_CONTEXT`, no real holdout, no real adapter).
- `render_table`'s output diffed against the skill's literal example table for exact column
  order and formatting.
- Full end-to-end check (by hand, once C1/C2/C4 are all real): run `baseline_holdout.yaml`, then
  `eval_full.yaml`, confirm `eval/results/{baselines.json,finetuned.json}` both carry the same
  `schema_sha256`, confirm the rendered README table's `n` matches the holdout's actual size, and
  confirm every holdout record appears in some system's `per_sample` — nothing silently excluded.
