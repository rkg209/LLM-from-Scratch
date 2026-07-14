# `eval/stub/` — the C1 baseline stub set

24 hand-written, single-bug Java snippets used to measure the base-model and frontier-API
baselines in spec C1, **before** the real holdout set exists.

## What this is, and what it is not

- **This is not the holdout set.** `eval/holdout/` (built in spec C2) is the frozen,
  independently-sourced set that the fine-tuned model is finally judged against. It is off
  limits to every training and data-generation path.
- **This is a stub.** Small, hand-written, and *not* held to the same provenance bar as the
  holdout — its only job is to give C1's `baseline.py` something to score against so the
  base-model and frontier numbers can be measured and frozen first (CON-11). When the holdout
  lands, C5 re-runs both baselines against it for the numbers that actually get published.

## Format

One JSON object per line: `{id, code, line, category, severity}`.

- `code` — a short Java snippet containing exactly one bug.
- `line` — the 1-indexed line, within `code`, where the bug is.
- `category` / `severity` — the ground-truth labels a correct review would report.

`stub_eval_smoke.jsonl` is a strict 3-record subset of `stub_eval.jsonl`, used by
`ch2_adaptation/configs/baseline_smoke.yaml` so the smoke path finishes in seconds.

## Leakage rule

The three few-shot examples in `ch2_adaptation/src/ch2_adaptation/prompts.py` must never appear
here. Drawing few-shot examples from the same set used to score a system is leakage — the exact
failure mode this spec exists to prevent — and `ch2_adaptation/tests/test_stub_eval_set.py`
asserts the two are disjoint.
