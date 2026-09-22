# Plan: C2 — Independent eval set

## Context

The project's headline metric (the eval table in C5) is only honest if it is measured on
something the fine-tuned model's training pipeline never touched and never resembled too
closely. `specs/06-C2-independent-eval-set.md` is the spec for assembling that set: 30
hand-authored synthetic-clean records plus 10 mined-real Java/Spring records (via the GitHub
MCP server), frozen under `eval/holdout/` with a provenance manifest. The spec was `draft`
with three open clarifications; these were resolved with the user before this plan was written:

- **Mined-record labels** come from the frontier API (Gemini 2.5 Flash, reusing C1's
  `frontier.GeminiClient`), not from lint-tool output or pure hand-labeling — same schema
  pipeline as the synthetic records, small added cost against the ~$1 project budget.
- **Target size**: 40 total records, 30 synthetic / 10 mined (nonzero mined count satisfies AC-3;
  matches the project's existing stub-set scale).
- **Cross-spec dedup**: C2 is built before C3 (synthetic training data), so `train.jsonl` won't
  exist yet when the holdout freezes. C2 publishes a hash-only index,
  `eval/holdout_hashes.txt` (normalized code hashes, no code content, **outside**
  `eval/holdout/` so the leakage guard's path-substring check doesn't block reads of it), which
  C3's `data_gen.py` will read later to self-exclude any colliding generated record. AC-4's own
  dedup, run at freeze time, checks the assembled 40 records against each other and against
  `eval/stub/{stub_eval.jsonl,stub_eval_smoke.jsonl}` (the only training-adjacent data that
  exists at this point).

**Two hard constraints shape every task below, confirmed live against the existing hooks:**

1. `.claude/hooks/leakage_guard.py` denies **every** Write/Edit/Bash-write targeting
   `eval/holdout/` unconditionally — there is no `EVAL_CONTEXT=1` escape hatch for writes, ever.
   This means the actual freeze (moving files into `eval/holdout/` and committing them) **cannot
   happen from inside this Claude Code session at all**, by design. It must be run by the user in
   their own terminal, exactly like C1's deferred full-baseline run (AC-5 in
   `specs/05-C1-task-schema-and-base-model.md`). All in-session work therefore targets a staging
   directory, `eval/holdout_staging/`, and produces a short "run this by hand" runbook.
2. GitHub access in this repo is MCP-only (`.mcp.json` → `github` server); there is no
   `PyGithub`/`requests`-based client and none should be added. Mining real snippets is
   necessarily an interactive step done with the GitHub MCP tool during a curation session, not a
   deterministic script — the deterministic code only handles staging-file I/O, labeling,
   validation, dedup, and manifest assembly.

C2 formally depends only on C1, and C1's tasks are all committed (C1's own AC-5 — the full
baseline run — is still open, but that doesn't block C2's code/data work, only C4's eventual
`building` transition per FR-15).

## Files

**New — `ch2_adaptation/src/ch2_adaptation/holdout_curator.py`**
CLI module (`python -m ch2_adaptation.holdout_curator <subcommand>`), mirroring `baseline.py`'s
shape (pure functions driven by injected fakes in tests; heavy imports local to `main()`):

- `label_mined_records(raw: list[dict], client: FrontierClient) -> list[dict]` — for each staged
  mined snippet, calls the client with the **same locked `PROMPT_TEMPLATE`** from
  `ch2_adaptation/prompts.py` used by C1/C4/C5 (no bespoke prompt), validates the response against
  `ch2_adaptation.schema.ReviewOutput`, and attaches the five label fields to the record. Reuses
  `frontier.GeminiClient` / `frontier.StubFrontierClient` and `frontier.estimate_cost_usd`
  unmodified — no new frontier-client code.
- `assemble_records(synthetic: list[dict], mined: list[dict]) -> list[dict]` — stamps `id`
  (uuid4), `split="holdout"`, `source` (`"synthetic"`/`"mined"`), `created_at` onto each raw
  record, producing full `ReviewRecord` dicts per `planning/04-database-design.md` §3.1.
- `validate_records(records) -> None` — validates every record's label subset against
  `eval.harness.load_schema()` (reuse, don't hand-roll a second validator).
- `dedup_records(records, external_hash_pools: dict[str, list[str]]) -> DedupReport` — computes
  the normalized code hash for each record (new shared helper, see below) and drops any internal
  duplicate or any collision against `eval/stub/stub_eval.jsonl` / `stub_eval_smoke.jsonl`;
  returns a report object with method description, checked/removed counts, and which pairs
  collided.
- `build_manifest(records, dedup_report, sources, curator) -> dict` — builds the
  `HoldoutManifest` dict exactly per `planning/04-database-design.md` §3.4 (`freeze_date`,
  `n_synthetic`, `n_mined`, `n_total`, `dedup_method`, `sources`, `schema_version` — SHA-256 of
  `eval/schema.json` — `curator`).
- `main()` — CLI with subcommands `label-mined` (calls the frontier API, budget-guarded like
  C1's baseline) and `freeze` (runs assemble → validate → dedup → manifest → writes
  `eval/holdout_staging/{holdout.jsonl,manifest.json,dedup_report.md}` and the top-level
  `eval/holdout_hashes.txt`). Never writes under `eval/holdout/` itself — the hook would block it
  even if it tried.

**New — `eval/dedup.py`** (shared by C2 now, C3 later — single source of truth for FR-16c / C3
AC-6's "normalized code hash"):
- `normalize_code(code: str) -> str` — case-fold, strip comments, collapse whitespace.
- `code_hash(code: str) -> str` — SHA-256 hex digest of `normalize_code(code)`.
Lives in `eval/` (base deps only — no torch/genai needed for hashing) so both `ch2_adaptation`
and the future `data_gen.py` import the identical function; avoids C3 re-implementing dedup
differently than C2 did.

**New — staging & output data:**
- `eval/holdout_staging/synthetic.jsonl` — 30 hand-authored `ReviewRecord`s (bug injected into a
  known-clean Java/Spring snippet, ground truth written by the curator directly — same authorship
  mode as C1's `eval/stub/stub_eval.jsonl`).
- `eval/holdout_staging/mined_raw.jsonl` — 10 records from GitHub MCP mining: `code`, `context`,
  `source_repo`, `license`, `commit_url` (no labels yet).
- `eval/holdout_staging/mined_labeled.jsonl` — output of `label-mined`.
- `eval/holdout_staging/{holdout.jsonl,manifest.json,dedup_report.md}` — output of `freeze`;
  staged, not yet committed to `eval/holdout/`.
- `eval/holdout_hashes.txt` — top-level (not under `eval/holdout/`), one normalized hash per
  line, sorted. Readable by C3 without tripping the leakage guard.
- `eval/holdout_staging/README.md` — the manual runbook: how to review the staged output, then
  move it into `eval/holdout/` and commit, **outside this session** (the hook blocks it inside).

**New — tests:**
- `ch2_adaptation/tests/test_holdout_curator.py` — `assemble_records`, `validate_records`,
  `dedup_records`, `build_manifest` driven by small fixtures and a fake frontier client (mirrors
  `test_baseline.py`'s fake-`Generator` pattern) — no network, no torch.
- `eval/tests/test_dedup.py` — `normalize_code`/`code_hash` behavior (whitespace/comment
  insensitivity, case sensitivity choice, collision/no-collision cases).

**Modified:**
- `ch2_adaptation/configs/` — no new config needed; `label-mined` reads `GEMINI_API_KEY` /
  `PRICE_PER_1K_INPUT_USD` / `PRICE_PER_1K_OUTPUT_USD` the same way `baseline.py` does (env, not
  hardcoded), so it stays consistent with C1's cost-tracking convention.
- `.gitignore` — add `eval/holdout_staging/` (staging artifacts are working files, not the
  frozen product; only the final files moved into `eval/holdout/` are committed, plus
  `eval/holdout_hashes.txt` which **is** committed since C3 needs it).

## Task sequence

1. **`eval/dedup.py` + tests** — the shared normalized-hash primitive, pure code, no dependencies
   on anything mined or labeled yet. Verify: `uv run pytest eval/tests/test_dedup.py -q`.
2. **`holdout_curator.py` core** (`assemble_records`, `validate_records`, `dedup_records`,
   `build_manifest`) + tests, driven entirely by fixtures/fakes. Verify:
   `uv run pytest ch2_adaptation/tests/test_holdout_curator.py -q`.
3. **Author the 30 synthetic-clean records** into `eval/holdout_staging/synthetic.jsonl` by hand
   (curation content, not code) — one known bug per snippet, realistic Java/Spring surrounding
   context, spanning all four severities and a reasonable spread of categories.
4. **Mine 10 real records via the GitHub MCP server** (interactive tool calls in-session):
   search permissively-licensed (MIT/Apache-2.0/BSD) Java/Spring repos for bugfix commits or
   lint-flagged diffs; record the pre-fix snippet, surrounding context, source repo, license, and
   commit URL into `eval/holdout_staging/mined_raw.jsonl`.
5. **`label-mined` CLI path** (`holdout_curator.py::main`, frontier-API-backed like C1's
   `baseline.py::main`) + `frontier.py` reuse — labels the 10 mined snippets, curator spot-checks
   the output before accepting. Verify:
   `uv run python -m ch2_adaptation.holdout_curator label-mined --config ...` runs end-to-end
   against the 10 staged records, cost logged.
6. **`freeze` CLI path** — assembles all 40, validates, dedupes (writes the dedup report),
   builds the manifest, writes staged `holdout.jsonl`/`manifest.json`/`dedup_report.md` plus the
   committed `eval/holdout_hashes.txt`; write `eval/holdout_staging/README.md` documenting the
   manual move-and-commit step. Verify: run `freeze`, inspect staged output, confirm
   `manifest.json` matches §3.4 field-for-field, confirm `holdout_hashes.txt` has 40 (or fewer,
   post-dedup) lines and no code content.
7. **Exercise the leakage guard** (AC-5) — from a shell, pipe a crafted `PreToolUse` payload for
   a `Write` to `eval/holdout/holdout.jsonl` into `.claude/hooks/leakage_guard.py` and confirm
   `deny`; repeat for a `Read` of `eval/holdout/manifest.json` with and without
   `EVAL_CONTEXT=1` exported in the shell (not inlined in the command string — confirmed during
   planning that inline prefixes don't reach the hook's own `os.environ`). Document the exact
   commands and observed output in `progress_report.md`.
8. **Spec + STATUS updates** — record the three resolved clarifications in
   `specs/06-C2-independent-eval-set.md`'s Clarifications section (mirroring how C1 recorded its
   own), note the staged-vs-frozen split as a plan amendment, move `C2` to `planned`/`building` in
   `specs/STATUS.md` as appropriate, and leave an explicit "remaining before `done`" note — the
   manual move-into-`eval/holdout/` + commit step, run by the user outside this session, exactly
   parallel to C1's still-open AC-5.

Each numbered item above is one commit, per the project's one-task-one-commit convention.

## Verification

- `uv run ruff check . && uv run black --check . && uv run pytest -q` after every task.
- After task 6, manually inspect `eval/holdout_staging/holdout.jsonl` (40 lines, each validating
  against `eval/schema.json` via `eval.harness.load_schema()`) and `manifest.json` (all eight
  §3.4 fields present, `n_total == n_synthetic + n_mined == 40`).
- Task 7's hook exercise is the AC-5 verification — both the unconditional write-deny and the
  context-gated read-deny must be observed and logged, not merely asserted.
- Final end-to-end check (documented for the user, run outside the session): move the staged
  files into `eval/holdout/`, commit, then confirm `eval/holdout_hashes.txt` still matches (no
  drift between staged and frozen content) and that `eval/harness.py` can load the frozen set
  under `EVAL_CONTEXT=1`.
