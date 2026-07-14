# C2 — Independent eval set

| | |
|---|---|
| **State** | draft |
| **Depends on** | C1 |
| **Requirements** | FR-16, CON-6, NFR-6, NFR-15 |
| **W&B run** | — |

## Problem

The obvious objection to this entire project is one sentence long: *"You generated training data with a frontier model and then showed your model imitates that frontier model. So what?"*

If the eval set were also frontier-generated, that objection would be fatal, and it would be correct. The answer is this spec: an evaluation set that is **independent of the training distribution and partly real** — mined from actual Java diffs and lint findings, not synthesized by the same model that taught the student. A win on that set is a win on the real task, not on the teacher's idiosyncrasies.

Once frozen, this set is the most valuable thing in the repository, and the fastest way to destroy it is to look at it. Every glance at the holdout while debugging a training failure leaks a little information into the next design decision. That is why the leakage guard denies reads by default rather than trusting discipline.

## Scope

Assemble the holdout: synthetic-clean examples plus mined-real examples (real Java/Spring diffs and lint findings, via the GitHub MCP server); dedupe against everything that will be trained on; freeze under `eval/holdout/` with a provenance manifest.

## Acceptance criteria

1. `eval/holdout/holdout.jsonl` exists, is committed, and every record validates against `eval/schema.json` (FR-16a).
2. `eval/holdout/manifest.json` records `freeze_date`, `n_synthetic`, `n_mined`, `n_total`, `dedup_method`, `sources` (the repos mined), `schema_version`, and `curator` — per `planning/04-database-design.md` §3.4.
3. The set contains a **non-zero count of mined-real records**. A wholly synthetic holdout does not answer the circularity objection and does not satisfy this spec.
4. A deduplication report in `eval/holdout/` shows the method used and the number of overlapping records removed. **Zero records in the holdout share a normalized code hash with any training record** (FR-16c).
5. The leakage-guard hook denies a Write to `eval/holdout/` and denies a Read without `EVAL_CONTEXT=1` — exercised and confirmed (FR-16b, NFR-15).
6. The set is large enough that the eval table's differences exceed sampling noise; `n` is stated with every published metric.
7. Mined code is compatible with redistribution (permissively licensed) and the licence of each source repo is noted in the manifest.

## Out of scope

- Scoring anything against this set → **C5**.
- Growing or rebalancing the set after freezing. It is frozen. If it is wrong, that is a new spec with a new freeze date, and every previously published number is re-run and re-reported.

## Clarifications

*(filled by `/clarify` — open questions: target size of the holdout; the synthetic/mined ratio; how ground-truth labels for the mined records are established, since real diffs do not come with `severity` and `category` fields.)*

## Technical plan

*(filled by `/plan`)*

## Tasks

*(filled by `/tasks`)*
