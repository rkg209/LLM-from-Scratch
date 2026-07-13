---
name: eval-table
description: Run the shared eval harness and regenerate the head-to-head comparison table and plots in the fixed project format. Use when asked to produce, refresh, or publish results — the Chapter 2 eval table, the Chapter 1 speedup/quantization curves, or the README results section. Invoked by the /eval command.
---

# Producing the results

The eval table is the headline artifact of the whole project (BG-2). Its credibility rests entirely on the methodology being boring and identical across systems.

## The one rule

**All systems are scored on the same frozen holdout set, by the same harness, in the same run.** `eval/harness.py::score_outputs` is the only thing that produces a number that goes in the table. A metric computed by hand, in a notebook, or from a different sample is not comparable and does not get published.

Reading `eval/holdout/` requires `EVAL_CONTEXT=1` on the command — the leakage guard blocks it otherwise. That friction is deliberate.

## Chapter 2 — the head-to-head table

Three systems, scored against `eval/holdout/holdout.jsonl`:

| System | Source of numbers |
|---|---|
| Fine-tuned (QLoRA adapter) | `eval/results/finetuned.json` ← `ch2_adaptation/evaluate.py` |
| Base model, zero-shot | `eval/results/baselines.json` → `base_model` ← `ch2_adaptation/baseline.py` |
| Frontier API, 3-shot | `eval/results/baselines.json` → `frontier_api` ← `ch2_adaptation/baseline.py` |

Two metrics, both from `EvalResult`:

- **schema-validity rate** — fraction of outputs that pass `eval/schema.json`. This is where a small specialist usually wins outright.
- **bug-catch rate** — fraction of holdout bugs identified, where the reported `line` is within ±2 of ground truth.

Render into `README.md` in exactly this shape, with `n` stated so the reader can judge the noise:

```markdown
| System | Schema-validity | Bug-catch | n |
|---|---|---|---|
| Fine-tuned (QLoRA, Qwen2.5-Coder-1.5B) | 0.94 | 0.61 | 120 |
| Base model (zero-shot)                 | 0.38 | 0.44 | 120 |
| Frontier API (3-shot)                  | 0.97 | 0.72 | 120 |
```

## Chapter 1 — the curves

From `eval/results/ch1_benchmark.json`, produced by `ch1_architecture/benchmark.py`:

- `speedup_curve.png` — tokens/sec across fp32 → fp16 → int8 → int4, plus the no-cache vs KV-cache pair. Warm-up steps excluded.
- `perplexity_tradeoff.png` — perplexity against the same precision axis, so the cost of each speedup is visible next to it.

Both go in `eval/results/plots/` and are linked from the README.

## Honesty rules

These exist because the temptation is real and the project's whole differentiator is that it did not give in to it (NFR-7):

- **Report the run you got.** Do not re-run with different seeds until a favorable number appears. If you re-run, report every run.
- **Do not drop failing samples.** A model that emits garbage on 12 of 120 inputs has a schema-validity rate of 0.90, not 1.00 on "the ones that parsed."
- **Do not relax the schema.** Loosening `eval/schema.json` to make the base model look better invalidates every prior number (CON-7).
- **If the fine-tune loses, say so in the README, in the table, in plain words** — and then make the cost/latency/privacy argument, which is true regardless.
- Record the W&B run ID next to every published number in `specs/STATUS.md` (NFR-16). A number nobody can trace back to a run is a number nobody should trust.
