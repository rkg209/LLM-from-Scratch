# C2 — Independent eval set

| | |
|---|---|
| **State** | building (6 of 8 tasks done; task 4 — mining — paused on `GITHUB_TOKEN`, see progress_report.md) |
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
`assemble_records`, `validate_records`, `dedup_records`, `build_manifest`, `DedupReport.to_markdown`;
CLI: `label-mined` and `freeze` subcommands, both built and tested — only the actual GitHub
mining that feeds `label-mined` is blocked, not the code around it); `eval/dedup.py` (shared
`normalize_code`/`code_hash` primitive, also used by C3); `eval/staging/{synthetic.jsonl,README.md}`
(30 hand-authored synthetic-clean records, committed, plus the manual-freeze runbook);
`eval/tests/test_dedup.py`, `ch2_adaptation/tests/test_holdout_curator.py`.

**Key design points:**

- `eval/dedup.py`'s comment-stripping is literal-aware (tracks string/char-literal state) rather
  than a naive regex — a naive `//`/`/* */` regex would truncate any snippet whose string content
  contains a comment delimiter (e.g. a URL), causing false-positive dedup collisions. Caught in
  code review before this shipped.
- `dedup_records` raises a record-id-bearing error on a record missing its `code` field, rather
  than a bare `KeyError` — another review-caught gap.
- `label_mined_records` reuses the exact locked zero-shot review prompt (`prompts.format_zero_shot`)
  C1/C4/C5 all use — no bespoke prompt for this spec — and reuses `BaselineConfig` for frontier
  settings rather than inventing a new config schema, per the plan.
- `_run_freeze` dedupes against `eval/stub/{stub_eval.jsonl,stub_eval_smoke.jsonl}` — the only
  training-adjacent data that exists at freeze time (AC-4) — and writes the committed hash-only
  index alongside the staged output.
- The 30 synthetic records span all four severities (8 critical / 12 major / 7 minor / 3 info)
  and 30 distinct categories, validated against `eval/schema.json` before committing. Running the
  real `freeze` step against all 30 (not just fixtures) caught a genuine content bug: one record
  was a byte-identical duplicate (after normalization) of a record in `stub_eval_smoke.jsonl`,
  which would have silently dropped the count to 29 at freeze time. Fixed by rewriting that one
  record to a different scenario; verified all 30 are now pairwise-unique and disjoint from both
  stub files.
- AC-5 (leakage guard denies a Write unconditionally, denies a Read without `EVAL_CONTEXT=1`,
  allows a Read with it) was exercised directly against the hook script with crafted payloads —
  all four scenarios (write/no-context, read/no-context, read/context, write/context) behaved
  exactly as the hook's source claims.

## Tasks

1. ✅ `eval/dedup.py` + tests.
2. ✅ `holdout_curator.py` core (`assemble_records`/`validate_records`/`dedup_records`/
   `build_manifest`) + tests.
3. ✅ 30 synthetic-clean records authored and committed (one duplicate/one off-by-one fixed
   during task 6's real-data exercise, see below).
4. ⏸ Mine 10 real records via the GitHub MCP server — **the only genuinely blocked task**: the
   `github` MCP server (`.mcp.json`) needs `GITHUB_TOKEN`; attempts to connect it during this
   session returned `HTTP 400` even once a token was supplied, consistent with
   `api.githubcopilot.com/mcp/` being picky about token type/account access (see
   `progress_report.md`). Tasks 5-7 were re-scoped to proceed without this, once it became clear
   only the actual mined *content* needs live GitHub access — the code around it doesn't.
5. ✅ `label-mined` CLI path — `label_mined_records` + subcommand, verified end-to-end with the
   stub client (real content still pending task 4).
6. ✅ `freeze` CLI path — `_run_freeze` + subcommand, verified end-to-end against the real
   30-record synthetic set (a fake mined record stood in for the still-missing real 10); caught
   and fixed the duplicate-content bug above.
7. ✅ Exercised the leakage guard (AC-5) directly against crafted `PreToolUse` payloads; all four
   scenarios documented in `progress_report.md`.
8. ✅ This spec/STATUS update.

**Remaining before this spec can move to `done`:** task 4 (mine 10 real records — needs a working
`GITHUB_TOKEN`/MCP connection, or an alternative like `gh` CLI), then run `label-mined` and
`freeze` for real against that content, then the manual freeze (move staged output into the
frozen directory, commit outside a session) — exactly parallel in shape to C1's still-open AC-5.
