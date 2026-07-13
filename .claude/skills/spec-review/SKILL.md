---
name: spec-review
description: Audit a spec for ambiguity, untestable acceptance criteria, scope overlap, and missing dependencies before it is planned or implemented. Use right after a spec is written or edited, when a spec is about to move from draft to planned, or when the /clarify command runs.
---

# Auditing a spec

A spec's job is to be unambiguous enough that the implementation is mechanical and the acceptance check is objective. Most specs fail at the second part while looking fine.

## Delegate the read

Dispatch the **`spec-reviewer`** subagent to audit the spec cold, in its own context. It has not read the conversation that produced the spec, so it cannot fill gaps with assumptions the way you will. Ask it for findings, not a rewrite.

## What to look for

**Untestable acceptance criteria.** The most common defect. A criterion must name the check that decides it.

| Untestable | Testable |
|---|---|
| "the tokenizer works correctly" | "`decode(encode(t)) == t` for every string in the test corpus" |
| "the model produces good reviews" | "≥ 90% of outputs on the holdout validate against `eval/schema.json`" |
| "the KV-cache is faster" | "tokens/sec with cache > tokens/sec without, same seed and prompt, warm-up excluded" |
| "serving is reliable" | "p99 latency recorded and reported; throughput ≥ 1 req/s on HF Spaces CPU" |

**Criteria with no requirement behind them.** Every acceptance criterion should trace to an FR or NFR in `planning/01-requirements.md`. One that traces to nothing is either scope creep or a missing requirement — decide which.

**Requirements with no criterion in front of them.** Read the spec's FR list and check each is actually covered. This is where specs quietly shed their hardest obligation.

**Scope overlap.** Does another spec already own this? `eval/harness.py` belongs to F3, not to whichever spec first needs it. Cross-check `specs/STATUS.md`.

**Missing dependencies.** Does this spec need a frozen holdout (C2), a fixed model tag (C1), or the shared config loader (F2) to exist first? An unlisted dependency becomes a mid-implementation surprise.

**Assumed decisions.** Anywhere the spec says "we'll use X" without X appearing in `planning/` or `CLAUDE.md`, someone made a decision in their head. Surface it as an open question rather than ratifying it.

**Constraint violations.** Does the spec quietly require a GPU in a session (CON-4), a `transformers` import in ch1 (CON-5), a peek at the holdout (CON-6), or a schema relaxation (CON-7)? These are hard stops.

## Output

A short list of concrete findings, each with the fix. Then:

- Genuine decisions for the user → ask them with `AskUserQuestion` and record the answers in the spec's **Clarifications** section.
- Everything else → fix it in the spec directly.

A spec does not move from `draft` to `planned` while an open question remains.
