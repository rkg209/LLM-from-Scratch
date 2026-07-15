# C2 — Independent eval set

| | |
|---|---|
| **State** | building (paused after task 3 of 8 — tasks 4-7 need `GITHUB_TOKEN`, see progress_report.md) |
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

Resolved during `/plan` (full reasoning: `.claude/plans/06-C2-independent-eval-set.md`):

- **Mined-record labels** come from the frontier API (Gemini 2.5 Flash, reusing C1's
  `frontier.GeminiClient`), not lint-tool output or pure hand-labeling — same schema pipeline as
  the synthetic records, small added cost against the ~$1 project budget.
- **Target size: 40 total records, 30 synthetic / 10 mined** (nonzero mined count satisfies
  AC-3; matches the project's existing stub-set scale).
- **Cross-spec dedup:** C2 is built before C3, so `train.jsonl` won't exist at freeze time. C2
  publishes a hash-only index outside `eval/holdout/` (see amendment below) that C3's
  `data_gen.py` reads to self-exclude any colliding generated record. AC-4's own dedup, run at
  freeze time, checks the 40 assembled records against each other and against
  `eval/stub/{stub_eval.jsonl,stub_eval_smoke.jsonl}`.

## Amendments (recorded here per plan, not applied silently)

- **Staging directory is not named `eval/holdout_staging/`** as originally planned.
  `leakage_guard.py`'s path check is a bare substring match on the `eval/holdout` prefix, with no
  distinction between the frozen directory and a same-prefixed staging directory next to it —
  and writes have no `EVAL_CONTEXT=1` override, ever, by design. The staging directory actually
  used is `eval/staging/`.
- **The committed hash-only index is not named `eval/holdout_hashes.txt`**, for the same reason.
  It is `eval/frozen_hashes.txt` (top-level, committed, outside `eval/holdout/`) — the name C3's
  `data_gen.py` already reads.

## Technical plan

See `.claude/plans/06-C2-independent-eval-set.md` for full detail. Summary:

**New files** — `ch2_adaptation/src/ch2_adaptation/holdout_curator.py` (core:
`assemble_records`, `validate_records`, `dedup_records`, `build_manifest`; CLI subcommands
`label-mined`/`freeze` not yet built — blocked, see below); `eval/dedup.py` (shared
`normalize_code`/`code_hash` primitive, also used by C3); `eval/staging/synthetic.jsonl` (30
hand-authored synthetic-clean records, committed); `eval/tests/test_dedup.py`,
`ch2_adaptation/tests/test_holdout_curator.py`.

**Key design points:**

- `eval/dedup.py`'s comment-stripping is literal-aware (tracks string/char-literal state) rather
  than a naive regex — a naive `//`/`/* */` regex would truncate any snippet whose string content
  contains a comment delimiter (e.g. a URL), causing false-positive dedup collisions. Caught in
  code review before this shipped.
- `dedup_records` raises a record-id-bearing error on a record missing its `code` field, rather
  than a bare `KeyError` — another review-caught gap.
- The 30 synthetic records span all four severities (8 critical / 12 major / 7 minor / 3 info)
  and 30 distinct categories, validated against `eval/schema.json` before committing.

## Tasks

1. ✅ `eval/dedup.py` + tests.
2. ✅ `holdout_curator.py` core (`assemble_records`/`validate_records`/`dedup_records`/
   `build_manifest`) + tests.
3. ✅ 30 synthetic-clean records authored and committed.
4. ⏸ Mine 10 real records via the GitHub MCP server — **blocked**: the `github` MCP server
   (`.mcp.json`) needs `GITHUB_TOKEN`, which is unset in this environment. Asked the user
   directly; decision was to defer tasks 4-7 and continue with C3/C4/C5, which don't depend on
   GitHub access.
5. ⏸ `label-mined` CLI path — blocked on task 4 (needs the mined snippets to label).
6. ⏸ `freeze` CLI path — blocked on tasks 4-5 (needs all 40 records to assemble).
7. ⏸ Exercise the leakage guard (AC-5) — not actually blocked on GitHub access, but grouped with
   4-6 in the user's deferral decision; can be picked up independently once resumed.
8. ⏸ This spec/STATUS update — intentionally partial (this entry): describes real progress
   through task 3, not a fabricated end state. Full closure (including the manual move-into-
   `eval/holdout/` + commit step, run by the user outside any session, per this plan's own
   constraint 1) waits for `GITHUB_TOKEN` and tasks 4-7.

**Remaining before this spec can move to `done`:** set `GITHUB_TOKEN`, resume tasks 4-7, then the
manual freeze (move staged output into `eval/holdout/`, commit outside a session) — exactly
parallel in shape to C1's still-open AC-5.
