# Progress report

The development log for this project, appended to as work happens. Every entry answers three questions — **what** changed, **why** it changed, and **how** it was done — and records the problems hit along the way and how each was resolved.

Read top to bottom, this is the story of how the project was built. Newest entries go at the **bottom**.

**Maintenance rule:** append an entry after every meaningful change (see `CLAUDE.md`). Never rewrite history here — if an earlier decision turned out to be wrong, say so in a *new* entry. A log that has been tidied up is a log that has stopped being evidence.

---

## Session 1 — 2026-07-13 — Bootstrap: SDD scaffold + Phase-0 foundation

Starting point: a directory containing only planning documents — `llm-engineering-architecture-to-edge.md` plus five documents in `planning/`. No code, no git repo, no Claude Code configuration. The design was fully locked (repo layout, JSON schema, base model tag, module signatures, hook behaviors); nothing had been built.

Goal for the session: turn those documents into a working spec-driven-development environment, plus the Phase-0 foundation (specs F1–F3) with green tests and CI.

---

### Entry 001 — Repository hygiene

**What.** `git init` on branch `main`; `.gitignore`; `.env.example`; a `README.md` skeleton with the chapter table and placeholders for the eval table, plots, and live demo link.

**Why.** Nothing else can be tracked, reviewed, or committed until the repo exists. The `.gitignore` had to come *first*, before any model code: the first training run will happily try to commit a multi-gigabyte checkpoint, and a weight file that reaches GitHub is permanently expensive to remove.

**How.** `.gitignore` blocks `outputs/`, `*.pt`, `*.gguf`, `*.safetensors`, `ch2_adaptation/data/*.jsonl`, `.env`, `wandb/`, `.venv/`. One deliberate exception: `!ch3_operation/tests/fixtures/tiny.gguf`, the CI fixture that spec O0 will add. `.env.example` documents every token the project needs (`HF_TOKEN`, `WANDB_API_KEY`, `GITHUB_TOKEN`, `ANTHROPIC_API_KEY`) plus the two escape-hatch variables (`ALLOW_FULL_RUN`, `EVAL_CONTEXT`), with a note that they are set per-command and never exported in a shell profile.

---

### Entry 002 — `CLAUDE.md`, the project constitution

**What.** A short always-loaded file: what the repo is, six non-negotiables, the build/test commands, the SDD workflow, and the review conventions.

**Why.** `CLAUDE.md` is read on every turn, so it costs tokens on every turn. It earns that cost only if it carries the rules that are *expensive to get wrong* and nothing else. The six non-negotiables are exactly the rules whose violation is silent and destructive: a full run in-session (burns the free-GPU budget), a `transformers` import in Chapter 1 (destroys the chapter's entire claim), a peek at the holdout (invalidates the headline metric), a relaxed schema (invalidates the comparison), an unseeded run (unreproducible), a committed weight file (permanent).

**How.** Condensed from §6.1 of the root design doc. Kept to roughly 50 lines — the map, the rules, the commands, and a pointer to `specs/`.

---

### Entry 003 — The Claude Code scaffold (spec F4)

**What.** Five lifecycle hooks, eight slash commands, five skills, five subagents, `.claude/settings.json`, and `.mcp.json`.

**Why.** Three of the project's rules cannot survive on good intentions, because each is broken by an action that feels locally reasonable in the moment: starting a full run (*it's right there and it would answer the question*), peeking at the holdout (*it's just a file, and I want to know why the model is failing*), committing a checkpoint (*it's the artifact I just spent an hour producing*). Code review catches these sometimes. **Hooks catch them every time.** That is the difference between a rule and a guardrail, and it is why the scaffold sits at the foundation rather than being a nicety bolted on later.

**How.**

| Hook | Event | Behavior |
|---|---|---|
| `gpu_budget_guard.py` | PreToolUse · Bash | Denies full-run patterns (`configs/full.yaml`, `accelerate launch`, `torchrun`, `deepspeed`, large `max_steps`, `load_in_4bit`) unless `ALLOW_FULL_RUN=1`. Smoke configs always pass. |
| `leakage_guard.py` | PreToolUse · Bash/Read/Write/Edit/Glob/Grep | Denies **all** writes to `eval/holdout/` unconditionally; denies reads unless `EVAL_CONTEXT=1`. Catches `cat eval/holdout/...` in Bash too. |
| `auto_format.py` | PostToolUse · Write/Edit | `ruff check --fix` + `black` on the touched file. |
| `commit_hygiene.py` | PreToolUse · Bash (`git commit`) | Inspects what is actually **staged** — denies `.gguf`/`.pt`/`.env`/holdout/generated-data files, and anything over 5 MB. |
| `session_greeter.py` | SessionStart | Prints the rules and the first not-`done` spec from `specs/STATUS.md`. |

All five share `_hooklib.py`, whose `run()` swallows any exception: **a guardrail that crashes the tool it guards is worse than no guardrail**, so a broken hook can never wedge a session.

Commands implement the SDD flow (`/specify → /clarify → /plan → /tasks → /implement`, plus `/smoke`, `/eval`, `/checkpoint`). Skills encode the know-how that would otherwise be reinvented each time (`from-scratch-guard`, `qlora-recipe`, `eval-table`, `gguf-export`, `spec-review`). Subagents keep heavy reading out of the main context (`spec-reviewer`, `code-reviewer`, `data-curator`, `experiment-analyst`, `explorer`).

**Problem — the planning docs specified a config format that does not load.** `planning/02-architecture.md` §2.5 put MCP servers inside `.claude/settings.json`, registered hooks as bare file paths, and used flat `skills/<name>.md` files. None of that is how Claude Code actually reads configuration.

**Resolution.** Intent preserved, form corrected: MCP servers moved to `.mcp.json`; hooks registered as `{matcher, hooks: [{type: "command", command}]}` objects with scripts that read a JSON payload on stdin and emit a permission decision; skills written as `skills/<name>/SKILL.md` with YAML frontmatter. The deviation is recorded in spec F4's Clarifications section so the next reader does not have to rediscover it.

---

### Entry 004 — The spec backlog (24 specs)

**What.** `specs/_template.md`, `specs/STATUS.md` (the live backlog table), and 24 spec files: **F1–F4** (foundation), **A1–A5** (Chapter 1), **C1–C5** (Chapter 2), **O0–O6** (Chapter 3), **X1–X3** (cross-cutting).

**Why.** The backlog existed only as a table in a design document. As files, each spec becomes a place to record clarifications, a plan, and a task list — and `STATUS.md` becomes a single source of truth that the session greeter can read, so no session ever starts by guessing where the project left off.

**How.** Each spec carries Problem / Scope / Acceptance criteria / Out-of-scope / Dependencies, traced to the FR and NFR numbers in `planning/01-requirements.md`. **Technical plan and Tasks are deliberately left empty** — they belong to `/plan` and `/tasks`, written when the spec is actually reached, not pre-committed now.

Every acceptance criterion had to be made *mechanically checkable*. "The tokenizer works correctly" is not a criterion; "`decode(encode(t)) == t` for every string in the test corpus" is.

Three specs carry open questions that could not be resolved from the planning documents, flagged for `/clarify`:

- **C1** — which frontier model backs the 3-shot baseline, and whether it is the same one that generates C3's training data. The same model makes the distillation story cleaner but the comparison more circular.
- **C2** — how ground-truth labels get established for mined real diffs, which do not come with `severity` and `category` fields.
- **O0** — where the `tiny.gguf` CI fixture comes from; it must be genuinely tiny and permissively licensed, and it is the one exception to the no-weights-in-git rule.

---

### Entry 005 — Phase-0 foundation code (specs F1, F2, F3)

**What.** `pyproject.toml`; the shared config loader; `eval/schema.json`, `eval/harness.py`, `eval/metrics.py`; three packages with smoke/full configs, frozen config dataclasses, and entrypoints; GitHub Actions CI.

**Why.** F3 (the eval harness) has to exist *before* Chapter 2 begins, for a reason that is easy to get backwards: **the scoring rules must be fixed while nobody yet knows which system they will favor.** A harness written after the results are in is a harness that was tuned, however unconsciously, to produce them.

F2 (the config convention) is what makes the smoke-only rule enforceable — there is always a legitimate thing to run instead of the full job.

**How.**

- **`pyproject.toml`** — `uv`, Python pinned to 3.12 (3.13 is ahead of parts of the ML stack). Base dependencies are only `pyyaml` and `jsonschema`; the heavy stacks are **optional extras** (`ch1`, `ch2`, `ch3`, `dev`) so that CI and the shared eval tests run in seconds without installing torch.
- **`eval/config.py`** — the shared loader. `load_config(path, cls)` reads YAML into a frozen dataclass and raises `KeyError` on a missing key **or an unknown one**. A Python-side default is a hyperparameter that never appears in the config file; a typo'd key is a setting the user believes they set. Both produce a run nobody can reproduce from the config that supposedly describes it.
- **`eval/schema.json`** — the locked Draft-07 contract, verbatim from the design docs.
- **`eval/harness.py`** — `score_outputs()` returns schema-validity rate, bug-catch rate (line within ±2), and `per_sample` records so every published number traces back to the record that produced it. Every sample counts in the denominator; unparseable output is a **failure, not an exclusion**.
- **Entrypoints are honest scaffold stubs.** `train.py`, `finetune.py`, and `serve.py` load and validate config, seed the RNGs, print what they *would* do, and exit — each printing `SCAFFOLD: no training loop yet — that is spec A3`. They exist so `/smoke` and CI have something real to run from day one; a config that does not load is a bug worth catching before there is a model to blame.

**Design conflict resolved.** FR-4 required a *shared* config loader while `planning/02-architecture.md` gave each package its own `config.py`. Both now hold: the generic loader lives in `eval/config.py`, and each package's `config.py` defines its own frozen dataclass (`GPTConfig`, `FinetuneConfig`, `ServeConfig`) on top of it, enforcing its own invariants (`d_model % n_heads == 0`; the locked base-model tag; `n_gpu_layers = 0`).

---

### Entry 006 — Verification, and three things that broke

**What.** Ran the full gate and exercised every hook against a real payload. Final state: **62 tests passing, ruff and black clean, all three smoke entrypoints running on CPU in under a second, all five hooks denying exactly what they should.**

**Why.** FR-7 requires each hook to be exercised at least once. A guardrail nobody has watched fire is a guardrail nobody knows works — and these five are load-bearing.

#### Problem 1 — pytest could not collect the tests

Four packages each had a `tests/test_config.py`. With no `__init__.py` in the test directories, pytest's default importer cannot tell four same-named modules apart:

```
import file mismatch: imported module 'test_config' has this __file__ attribute:
  eval/tests/test_config.py
which is not the same as the test file we want to collect:
  ch1_architecture/tests/test_config.py
```

**First attempt (failed):** switched to `--import-mode=importlib`. This produced a *different* and more confusing error — `ModuleNotFoundError: No module named 'ch3_operation.config'` — even though the same import worked fine from a plain `python -c`. The cause: in importlib mode pytest synthesizes dummy parent modules from the test file's path, so collecting `ch3_operation/tests/test_config.py` inserts a **fake `ch3_operation` module** into `sys.modules`, shadowing the real installed package and hiding its `config` submodule.

**Second attempt (failed):** a root `conftest.py` that prepended the three `src` directories to `sys.path`. This did not help, because the problem was never `sys.path` ordering — it was the fake module already sitting in `sys.modules`.

**Resolution:** renamed the test files to unique basenames (`test_gpt_config.py`, `test_finetune_config.py`, `test_serve_config.py`, leaving `eval/tests/test_config.py`), reverted to pytest's default import mode, and deleted both the `conftest.py` and the `addopts` line. Simpler than either failed fix. A comment in `pyproject.toml` records the constraint so the next person does not re-add a colliding filename.

*Discovered while diagnosing:* the editable-install `.pth` lists the repo root **before** the three `src` directories. The bare `ch1_architecture/` etc. directories at the root therefore shadow the real packages as namespace packages under some importers. Python's namespace rules make plain imports resolve correctly anyway, but it is a sharp edge worth knowing about.

#### Problem 2 — lint failures on first run

Ten ruff errors and three black reformats. Most were auto-fixable (import sorting, `Callable` from `collections.abc`). Two needed real changes: `UP047` wanted PEP 695 generic syntax, so `load_config` became `def load_config[T](path, cls: type[T]) -> T` and the `TypeVar` was dropped; three lines exceeded the 100-character limit and were wrapped.

**Resolution:** `ruff check --fix` + `black`, then the two manual fixes. The `auto_format` hook now prevents this class of problem from recurring — it formats on every edit, so a lint failure is never something discovered at commit time.

#### Problem 3 — a bug in the verification, not the code

While testing `commit_hygiene`, the "clean tree should be allowed" case came back **deny**. The hook was right and the test was wrong: the cleanup step used `git restore --staged`, which **fails in a repo with no commits** (there is no `HEAD` to restore from), so the test `.gguf` stayed in the index and the hook correctly objected to it.

**Resolution:** cleaned the index with `git rm --cached` and re-ran; the allow path passed. Worth recording because the failure looked exactly like a hook bug and was not — the hook was doing its job on a genuinely dirty index.

#### Hook verification results

| Hook | Case | Result |
|---|---|---|
| GPU-budget | `--config configs/full.yaml` | **deny** ✓ |
| GPU-budget | same, with `ALLOW_FULL_RUN=1` | allow ✓ |
| GPU-budget | `--config configs/smoke.yaml` | allow ✓ |
| GPU-budget | `accelerate launch train.py` | **deny** ✓ |
| Leakage | Write to `eval/holdout/` | **deny** ✓ |
| Leakage | Write to `eval/holdout/` *with* `EVAL_CONTEXT=1` | **deny** ✓ (frozen is frozen) |
| Leakage | Read `eval/holdout/` without `EVAL_CONTEXT` | **deny** ✓ |
| Leakage | Read `eval/holdout/` with `EVAL_CONTEXT=1` | allow ✓ |
| Leakage | `cat eval/holdout/...` in Bash | **deny** ✓ |
| Commit hygiene | commit with a staged `.gguf` | **deny** ✓ (names the file) |
| Commit hygiene | clean commit | allow ✓ |
| Auto-format | unformatted file | reformatted, unused imports stripped ✓ |
| Session greeter | new session | prints rules + "next up: A1" ✓ |

**Left uncommitted deliberately.** The first commit belonged to the developer, not to the tooling.

---

### Entry 007 — Known friction, accepted for now

**What.** The leakage guard's path matching also blocks reading `eval/holdout/README.md` — a documentation file with no evaluation data in it.

**Why accepted.** The rule is path-based, which is what makes it hard to argue around. Narrowing it to exempt `*.md` is a one-line change, but a guard with exceptions is a guard whose exceptions grow. Left strict; revisit only if it becomes genuinely annoying in practice.

---

## Session 2 — 2026-07-14

### Entry 008 — Ban `Co-Authored-By:` trailers in commit messages

**What.** A seventh non-negotiable in `CLAUDE.md`: never put a `Co-Authored-By:` trailer in a commit message. Reinforced in the `/checkpoint` command and **enforced** by the `commit_hygiene` hook.

**Why.** The trailer breaks pushing this repo to GitHub. The failure surfaces at *push* time — by which point the bad commit already exists in history and has to be amended or rebased away. Blocking it at commit time means the bad commit is never created.

**How.** Three layers, deliberately:

1. `CLAUDE.md` non-negotiable #7 — the rule, stated where it is read every turn.
2. `/checkpoint` — repeated at the exact step where commits get made.
3. `commit_hygiene.py` — a `CO_AUTHORED_BY` regex (case-insensitive, tolerant of spacing) checked against the commit command; a match is **denied** before `git` runs.

The instruction alone would have been enough most of the time. Most of the time is not the standard the rest of this project's guardrails are held to.

**History check.** Audited the existing log for the trailer: one commit exists (`46d8589 "Specs Written"`, authored by the developer), and it carries **no** `Co-Authored-By` line. Nothing to strip; no history rewrite needed.

**Verified.** A message containing `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` is denied; the lowercase/spaced variant `co-authored-by :` is denied; a clean message passes. Ruff and black clean on the modified hook.

---

### Entry 009 — This file

**What.** Created `progress_report.md` and added a `CLAUDE.md` instruction requiring it to be appended to after every meaningful change.

**Why.** The repo records *what the code is*. Git records *what changed*. Neither records **why** — the decision that was almost made and rejected, the fix that failed before the one that worked, the constraint discovered halfway through that reshaped the design. That reasoning is the most expensive thing produced here and the first thing forgotten, and on a six-month weekend project it will be forgotten between sessions.

It also has a second audience. This project is a portfolio artifact; an interviewer who reads a log showing three attempts at the pytest import problem, and *why* the third worked, learns something no polished README can convey.

**How.** Entries 001–008 were reconstructed from this session's work, including the failures — the importlib attempt that made things worse, the `conftest.py` that fixed the wrong problem, the verification bug that framed a working hook as broken. Those are the entries with the most value in them, so they are recorded in full rather than summarized into a clean narrative that never happened.

Going forward: an entry per meaningful change, appended at the bottom, answering what / why / how, plus any problem hit and how it was resolved.

---

### Entry 010 — F1 gap closure: `scripts/` was missing despite F1 being marked `done`

**What.** Created `scripts/README.md` as a directory-reservation placeholder. `scripts/` is named in `planning/02-architecture.md` §3.3 and in `specs/O1-merge-and-gguf-quantize.md`, which will land `merge_adapter.py`, `export_gguf.py`, and `quantize_gguf.py` there. With this, F1 now passes all 8 of its acceptance criteria against the live tree, not just 7.

**Why.** `specs/STATUS.md` marked F1 `done` with all tasks checked off, but a live re-check of its 8 acceptance criteria found AC-1 (tree matches the architecture doc) failing — `scripts/` did not exist. F1 is the sole dependency listed by F2, F3, and F4, so a silently-failing AC in the foundation spec would have surfaced later as improvisation mid-O1, at a worse time to notice it. The failure worth recording isn't the missing folder — it's that `done` was set without mechanically re-checking every AC against the tree.

**How.** Reused the pattern already established by `eval/holdout/README.md` and `eval/results/README.md`: a placeholder README that reserves the directory *and* documents its contract, instead of a content-free `.gitkeep`. The README states the directory's purpose (standalone helpers, not an importable package), tables the three planned scripts with their owning spec (O1), and notes explicitly that none of them exist yet so the empty directory doesn't read as a bug. No stub Python files were added — writing them now would be implementing O1 without an accepted plan. Also added a dated note under `specs/F1-repo-tooling-skeleton.md`'s Clarifications recording the miss, per the project rule that specs are annotated, not silently patched.

---

## 2026-07-15 — Spec filenames carry the build order

**What.** Renamed all 24 spec files from `ID-slug.md` to `NN-ID-slug.md`, where `NN` is the global
build order: `01-F1-…` → `04-F4-…` (foundation), `05-C1-…` → `09-C5-…` (Chapter 2),
`10-O0-…` → `16-O6-…` (Chapter 3), `17-A1-…` → `21-A5-…` (Chapter 1), `22-X1-…` → `24-X3-…`
(cross-cutting). Spec **IDs are unchanged** — `C1` is still `C1`. `specs/STATUS.md` gained a `#`
column, a track legend, and a progress summary, and its sections were reordered to match the
filenames. `session_greeter.py`'s row regex and the `specs/$1-*.md` globs in `/implement`,
`/specify`, `/clarify` and `/plan` were updated to match.

**Why.** `ls specs/` sorted alphabetically into `A, C, F, O, X`, which is neither the build order
nor any order at all — the actual sequence lived only in prose inside `STATUS.md` ("the critical
path is F1→F3 → C1→C5 → O0→O5"). A reader of the directory had no way to know that `F1` comes
before `A1`, or that Chapter 1 is a parallel track rather than step one. The letters were *not*
arbitrary (F/A/C/O/X are tracks), so the fix was to surface the order the letters already implied,
not to flatten them away — a flat `01..24` would have lost the track grouping and forced every
`C5`/`A5` reference in `README.md` and the planning docs to be rewritten.

**How.** `git mv` for all 24 files so history follows the rename. Kept the IDs verbatim — including
the `O0` oddity — precisely so that nothing outside `specs/` had to change: `README.md`,
`planning/`, and the historical `progress_report.md` entries all cite specs by ID, and those
citations are still correct. The prefix is decoration for humans reading the directory; the ID
remains the key.

**The problem that nearly shipped.** The first pass renamed the files and stopped. Two things broke
silently, neither caught by lint or tests:

1. The four slash commands read a spec via the glob `specs/$1-*.md`. With an `05-` prefix,
   `/implement C1` matched *nothing* — the command would have reported "no such spec" for every
   spec in the repo. Fixed by widening the glob to `specs/*$1-*.md`.
2. `session_greeter.py` matched a backlog row with `^\|\s*([A-Z]\d+)\s*\|` — ID in column one. The
   new `#` column shifted the ID to column two, so *no* row matched, `_next_spec()` returned
   `None`, and every session would have opened with "specs/STATUS.md not found or empty — start
   with /specify." A green `pytest` proved nothing here; the hook has no test. It was caught only
   by piping `{}` into the hook and reading the JSON it emitted.

The lesson worth keeping: the specs directory is not inert documentation — it is an *interface*
that hooks and slash commands parse. Renaming its files is a schema change, and the only honest
verification is to execute the things that read it.

**Also — a priority flip, stated out loud.** `STATUS.md` claimed the critical path was
F→C→O with Chapter 1 in parallel, but its *tables* listed Chapter 1 before Chapter 2, and the
greeter picks the first not-`done` row in file order. So every session had been opening with
"next up: A1" — the parallel track — contradicting the file's own stated critical path. Reordering
the sections to match the build order makes the greeter say **C1**, which is what the critical path
actually calls for. This is a real change in what gets built next, made deliberately: it is the
file being made consistent with itself, not a new decision about priorities. If Chapter 1 is in
fact wanted first, the fix is to renumber the A track to `05`–`09` and re-sort, not to revert
the naming.

**Status snapshot.** 4 of 24 specs done (F1–F4), all foundation. Chapters 1–3 are unstarted:
`train.py`, `finetune.py` and `serve.py` are 42–48-line config-loading stubs with no model code.

---

## Session 2 — 2026-07-15 — C1: schema, baselines, and the drift guard

Goal: implement spec C1 (task-schema-and-base-model) — freeze the base model tag (already landed
by F2), write the Pydantic mirror of `eval/schema.json`, the locked review prompt, a hand-written
stub eval set, and `baseline.py`, which measures the base model zero-shot and a frontier API
3-shot on that stub set and records both to `eval/results/baselines.json` before any fine-tuning
starts.

---

### Entry — C1 plan finalized; open `/clarify` question resolved

**What.** Filled `specs/05-C1-task-schema-and-base-model.md`'s Technical plan section from
`.claude/plans/05-C1-task-schema-and-base-model.md`, resolved the spec's open clarification, and
wrote the six-task checklist. Moved C1 to `building` in `specs/STATUS.md`.

**Why.** The `/tasks` skill refuses to write a task list against a spec with no Technical plan —
and the plan document (written in a prior session) already contained one, plus the answer to the
spec's open question (which frontier API for the 3-shot baseline). Rather than re-deriving that
from scratch, it was transcribed into the spec so the normal `/tasks` → `/implement` loop could
run unmodified.

**How.** Frontier model = **Gemini 2.5 Flash**, free tier (`GEMINI_API_KEY`) — no Claude API key
is available in this environment, and Gemini's free tier is rate-limited rather than
credit-limited, so "Frontier API, 3-shot" stays an honest row label; native JSON mode also avoids
the markdown-fence failure mode C1's baseline scoring deliberately does not repair. Recorded three
spec amendments surfaced by the plan rather than applied silently: AC-1's `model.name_or_path` →
the shipped flat `model_tag` key; AC-6's implied real-fp32 smoke run → a named tiny stand-in tag
(fp32 Qwen-1.5B cannot honor the under-120s smoke budget); AC-7 as an explicit carve-out to CON-3
(the API budget line was written for training-data generation, and a 3-shot eval call is a
different use that needed to be named, not assumed-covered). Flagged for C3: if C3's synthetic-data
generator also ends up being Gemini, the fine-tune is distilled from the same model it is
benchmarked against — survivable, but C3's spec must decide that on purpose.

---

### Entry — C1 T1: `ReviewOutput` schema + import-time drift guard

**What.** `ch2_adaptation/src/ch2_adaptation/schema.py::ReviewOutput` — a Pydantic model
(`extra="forbid"`) with the same five fields and constraints as `eval/schema.json`. On import it
calls `_assert_matches_eval_schema()`, which compares the two on every load-bearing key (required
fields, `additionalProperties`, the severity enum, and each field's length/minimum constraint) and
raises `SchemaDriftError` if they disagree. Promoted `pydantic` from the `ch2`/`ch3` optional
extras into the base `dependencies`, since this module must import in the base+dev CI environment
alongside the rest of the eval contract. Ten tests in `test_review_schema.py`, including three
that mutate a deep copy of the real schema and assert the guard actually raises on each mutated
key — not merely that the guard runs.

**Why.** The design doc (`planning/03-system-design.md:371`) specifies the guard as a bare
`assert`. Deliberately not used here: `assert` is stripped under `python -O`, which would silently
disable the one check whose entire purpose is to never be silent. A dedicated
`SchemaDriftError(RuntimeError)` can't be optimized away. Also: a naive
`ReviewOutput.model_json_schema() == json.load(open("eval/schema.json"))` does not work — Pydantic
emits a `title` key per property and no top-level `$schema` key that the reference JSON has — so
the comparison had to be narrowed to exactly the keys that constrain scoring, the same set
`eval/tests/test_schema.py` already pins.

**How.** Wrote the model and guard, then a code-reviewer pass on the diff. It flagged one real
issue: a `_SEVERITY_VALUES` tuple was defined but never referenced — the `Literal[...]` on the
`severity` field repeated the same four strings independently, so if the enum ever changed, one
copy could drift from the other with nothing catching it (the guard only compares the `Literal`
against `eval/schema.json`, never against that unused tuple). Deleted it rather than wiring it in,
since a Python `Literal` type can't be parameterized from a runtime tuple without extra
machinery this file has no other use for. Full gate (`ruff`, `black`, `pytest -q`, 72 tests) green
after the fix. Committed as two commits: one for the plan-fill/task-list (not itself a numbered
task), one for T1's code — kept separate so the commit history doesn't attribute planning prose to
the schema module's diff.

---

### Entry — C1 T2: stub eval set + locked review prompt

**What.** `eval/stub/stub_eval.jsonl` — 24 hand-written, single-bug Java snippets
(`{id, code, line, category, severity}`) plus `eval/stub/stub_eval_smoke.jsonl`, a 3-record
strict subset, and a README explaining the stub set is not the C2 holdout. `ch2_adaptation/prompts.py`
— the locked `SYSTEM_PROMPT` (inlines `eval/schema.json`, explicitly forbids markdown fences) and
three hand-written few-shot examples behind `format_zero_shot()` / `format_three_shot()`, which C4
and C5 must reuse verbatim so every system is scored against the same prompt.

**Why.** The stub set exists so C1's baselines can be measured before the real holdout (built in
C2) exists at all — the whole point of the spec is not skipping that step. It's generated (not
committed as generated code) via a one-off script kept in the scratchpad rather than the repo,
since the deliverable is the data file, not a generator for it.

**How — the code-review catch.** Wrote the 24 records by hand-computing each bug's line number
inside its snippet, then dispatched `code-reviewer` on the diff before committing. It found two
real defects that the existing tests (unique ids, line-in-range) could not catch because they only
check *shape*, not *correctness against the actual bug*:

1. `stub-017`'s `line` pointed at `flags.add(flag)` (the call site), 3 lines past
   `eval/metrics.py`'s `LINE_TOLERANCE = 2` from the real defect — the `static List<String> flags`
   field declaration. A model that correctly named the field would have scored as *wrong*. Fixed
   by pointing `line` at the declaration.
2. `stub-009`'s bug (`sum += q * Integer.MAX_VALUE`, which overflows on every non-zero input,
   guaranteeing a wrong result on virtually every call) was labeled `"severity": "minor"`. Relabeled
   `"major"`.

It also flagged something neither test could see: two of the three few-shot examples shared an
*exact* bug category with stub-set records (`off-by-one`, `resource-leak` — the same string used
by `stub-003` and by `stub-001`/`stub-024`/`stub-019`'s pattern). The disjointness test only checks
that example *code* doesn't literally appear in the stub set, so same-category priming passed it
silently while still biasing the three-shot baseline toward the categories most represented in the
stub set. Replaced those two examples with categories absent from the stub set entirely
(`equals-override-signature-mismatch`, `path-traversal`) rather than trying to extend the test to
catch semantic-pattern overlap, which would need a real bug-similarity metric to do honestly.

**The lesson worth keeping (again).** A test that checks a ground-truth record's *shape* (right
keys, in-range numbers, no exact-string duplication) proves nothing about whether the record is
*correct*. For hand-written eval data, a review pass that actually reads the code and checks the
label against it is not optional — it's the only thing standing between a plausible-looking record
and a silently wrong ground truth that the metrics trust completely.

---

### Entry — C1 T3: `BaselineConfig` + `baseline_smoke.yaml` / `baseline_full.yaml`

**What.** `ch2_adaptation/config.py` gained `SMOKE_MODEL_TAG` (the one allowlisted tiny stand-in,
`hf-internal-testing/tiny-random-Qwen2ForCausalLM`), `BaselineConfig`, and `load_baseline_config()`.
`__post_init__` enforces CON-11 for the baseline path the same way `FinetuneConfig` already does
for the fine-tune path: the full profile must use `LOCKED_MODEL_TAG`; the smoke profile must use
`SMOKE_MODEL_TAG` and must not write to `eval/results/baselines.json`. Two new configs,
`baseline_full.yaml` and `baseline_smoke.yaml`, and `test_baseline_config.py` exercising both
guards via `dataclasses.replace`.

**Why.** This is the mechanism behind spec C1's Risk 5 — "the worst outcome this spec has" is a
smoke run's tiny-random-model score silently overwriting the committed, published baseline numbers
that C4 is gated on. Encoding the guard in the config's `__post_init__` means it fires the moment a
`BaselineConfig` is constructed, not only when `baseline.py`'s `main()` happens to remember to
check.

**How — the code-review catch.** `code-reviewer` found the `results_path` guard compared strings
verbatim (`self.results_path == "eval/results/baselines.json"`), so a smoke config written as
`./eval/results/baselines.json` — a different string, the same file — would have bypassed the
check entirely while still clobbering the real file. Fixed by comparing `Path(...).resolve()` on
both sides instead of the raw strings. The reviewer also noted `max_new_tokens`/`temperature`/
`n_few_shot` have no range validation yet; left unvalidated deliberately, since T3 is config wiring
and those fields aren't consumed by any code until T5.

---

### Entry — C1 T4: `write_json_atomic` primitive + `baseline.py` core scoring

**What.** Extracted `write_json_atomic(payload, path)` in `eval/harness.py` as a public
primitive — `write_result` is now a one-line wrapper over it — so C1's `baselines.json` and any
future result file share one atomic-write implementation instead of each writing its own
tmp-then-rename. Added `ch2_adaptation/baseline.py`: `score_system(prompts, generate, stub_path)`
takes an injected `Generator` callable so every test drives it with a fake instead of a real
model; `build_baselines_doc(base, frontier, cfg, usage)` assembles the eventual `baselines.json`
payload, including `stub_set_sha256`/`schema_sha256` so a committed baseline file is re-checkable
later if either input silently changed underneath it (X2). No `main()` yet — that's T5, once the
real generators exist.

**Why.** `score_system` scores exclusively through `eval.harness.score_outputs` rather than any
ad-hoc parsing, because AC-3/NFR-7 require every system (base, frontier, eventually the fine-tuned
model) to be scored by the same code — a metric computed a different way for one system is not
comparable to the others. The injected-generator design is what makes this testable in the
base+dev CI environment: no torch, no network, just a fake that returns canned strings.

**How — the code-review catch.** `code-reviewer` found `build_baselines_doc` hardcoded
`"dtype": "float32"` regardless of `cfg.use_4bit`. Harmless today (baseline configs always run
fp32), but exactly the kind of config-drift trap the project's "config-driven, no magic numbers"
rule exists to prevent: if a baseline config ever flips `use_4bit`, the recorded dtype in the
published `baselines.json` would silently lie about what actually ran. Fixed to derive it:
`"int4" if cfg.use_4bit else "float32"`.

---

### Entry — C1 T5: `model.py` + `frontier.py`, `baseline.py::main` wired, smoke run green

**What.** `ch2_adaptation/model.py::make_hf_generator` (loads the base model once, returns a
`prompts -> raw responses` closure) and `ch2_adaptation/frontier.py` (`GeminiClient` — native
JSON mode via `response_mime_type`/`response_schema`, `StubFrontierClient`, `estimate_cost_usd`).
Both keep `torch`/`transformers`/`google.genai` imports function-local. `baseline.py::main()` now
wires the whole pipeline end to end: load config → format zero-/three-shot prompts → score both
systems → write atomically → log to W&B if enabled. The actual command,
`uv run python -m ch2_adaptation.baseline --config ch2_adaptation/configs/baseline_smoke.yaml`,
was run by hand and finishes in **~4 seconds**, well under the 120s NFR-1 budget, writes only to
gitignored `outputs/baselines-smoke.json`, and leaves `eval/results/` untouched.

**Why.** This is the task that turns the spec's five prior tasks into something that actually
runs — everything before this was scaffolding around a pipeline nobody had executed yet. Running
it for real, not just unit-testing its pieces, is what surfaced the two problems below; neither
would have been caught by a green test suite alone.

**Problem 1 — the guessed `SMOKE_MODEL_TAG` doesn't exist.** `hf-internal-testing/tiny-random-Qwen2ForCausalLM`
(written into spec/config from a plausible-sounding naming convention, not verified against the
Hub) 401'd as `RepositoryNotFoundError` on first run. Searched the Hub for a real tiny Qwen2 test
model, found `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5` (a ~2.4M-param model TRL's own test
suite depends on), and — the lesson from T2 repeating itself — **verified it actually downloads
and has a chat template before writing it into the config**, rather than trusting the name again.

**Problem 2 — `apply_chat_template(..., return_tensors="pt")` didn't return what `model.generate()`
expected.** The installed `transformers` version returns a plain tensor by default; the code
unpacked it as `**encoded` assuming a dict, and hit `KeyError: 'shape'` deep inside
`model.generate()`. Fixed by passing `return_dict=True` explicitly and reading
`encoded["input_ids"].shape[-1]` for the prompt-length slice, rather than relying on
`apply_chat_template`'s default return shape.

**How — the code-review catch (three issues).**

1. Neither `frontier.py` nor `model.py` had a test yet. Added `test_frontier.py` — fully testable
   without `google-genai` or network (`estimate_cost_usd`'s env-var guard, `StubFrontierClient`,
   `make_client`'s provider dispatch, and that `GeminiClient.generate` raises before ever touching
   the network if `GEMINI_API_KEY` is unset) — and a `pytest.importorskip("torch")`-gated
   `test_model.py` that exercises `make_hf_generator` for real against `SMOKE_MODEL_TAG` whenever
   the `ch2` extra is installed, and skips cleanly in the base+dev CI environment.
2. `GeminiClient.generate` computed `cost_usd = estimate_cost_usd(...) if input_tokens else 0.0` —
   if the API ever returned `usage_metadata=None`, `estimate_cost_usd`'s missing-price
   `RuntimeError` would never fire, and a misconfigured environment would silently record an
   indistinguishable-from-honest `$0.00` on a real paid-tier-capable call. Fixed by validating the
   price env vars unconditionally at the top of `generate()`, before any network call, so a
   misconfigured environment fails loud regardless of what the API returns.
3. `main()` had grown to ~51 lines, past the ~40-line convention. Split into `_score_base(cfg,
   prompts)` / `_score_frontier(cfg, prompts)` helpers, leaving `main()` as orchestration and
   printing only.

**The lesson worth keeping (a third time).** Every hand-written identifier that names something
external — a model tag, a package name, a schema key — is a claim, not a fact, until it's actually
exercised. T2 caught this for ground-truth labels; this entry catches it for a model tag on the
Hub. The pattern is the same: write the plausible thing, then run it before trusting it.

---

### Entry — C1 T6: task checklist closed; AC-5 left for the human

**What.** All six of C1's tasks are now ticked and committed; the full gate (`ruff`, `black`,
`pytest -q`) is green at 98 tests. Added a "Remaining before this spec can move to `done`" section
to the spec recording that **AC-5 is still open**: `eval/results/baselines.json` does not exist
yet, because producing it requires the full baseline run — the real 1.5B model plus a real
`GEMINI_API_KEY` call — which cannot happen inside this session. `specs/STATUS.md` keeps C1 at
`building`, not `done`.

**Why.** This is by design, not an oversight: `.claude/hooks/gpu_budget_guard.py` blocks any
command matching `--config .../full.yaml`, and `baseline_full.yaml` matches it — correctly, since
that command downloads ~3GB and calls a paid-tier-capable API. Spec C1's own Risk 4 named this in
advance. FR-15 is explicit that `baselines.json` must exist and be committed *before* C4 moves to
`building`, so the spec cannot honestly claim `done` until a human runs the full command and
commits the result.

**A small friction hit while writing this very entry:** the commit-message draft for T6 quoted the
literal string `--config .../full.yaml` as an example, and the GPU-budget hook denied the `git
commit` itself — it matches on the whole command string, including quoted example text inside a
commit message, not just an actual invocation. Not a bug in the hook (a hook that can be talked out
of firing by nesting the trigger string inside a comment is not much of a guardrail); reworded the
message to describe the behavior without reproducing the literal flag pattern.

**Status snapshot.** C1's code is complete and merged: schema + drift guard (T1), stub eval set +
locked prompt (T2), `BaselineConfig` (T3), scoring core (T4), the real generators and a green
smoke run (T5). What's left is entirely outside the session's reach: run
`GEMINI_API_KEY=... PRICE_PER_1K_INPUT_USD=... PRICE_PER_1K_OUTPUT_USD=... uv run python -m
ch2_adaptation.baseline --config ch2_adaptation/configs/baseline_full.yaml`, inspect the result,
commit it, then flip C1 to `done`. Only after that may C2/C3 begin in earnest.

---

### Entry — C2 T1: `eval/dedup.py`, the shared normalized-code-hash primitive

**What.** Added `eval/dedup.py` (`normalize_code`, `code_hash`) and its tests, per task 1 of
`.claude/plans/06-C2-independent-eval-set.md`. This is the dedup identity both C2 (holdout
curation) and C3 (synthetic training data) will use — case-fold, strip comments, collapse
whitespace, then SHA-256. Lives in `eval/` (base deps only, no torch/genai) so both packages
import the identical function rather than each rolling their own.

**Why.** A single shared definition of "the same snippet" is the whole point of FR-16c: if C2 and
C3 defined dedup differently, a snippet could count as unique to one and a duplicate to the
other, and nobody would notice until the eval table looked wrong.

**What went wrong first.** The first cut of `_strip_comments` was a plain regex —
`re.compile(r"//.*?$|/\*.*?\*/", re.MULTILINE\|re.DOTALL)`. The dispatched `code-reviewer` caught
that this is wrong for real Java: a string literal like `"http://example.com"` gets truncated at
the `//` as if it were a line comment, and `"a /* b */ c"` inside a string gets treated as a block
comment. Two snippets differing only in string content (e.g. two different URLs) would then
normalize to the same string and **collide in `code_hash`** — a false-positive dedup that silently
drops a genuinely distinct record. Fixed by replacing the regex with a small character-scanning
pass that tracks string/char-literal state and only treats `//`/`/* */` as comment delimiters
outside of a literal. Added tests for exactly this boundary (URL-in-string, `*/`-looking sequence
inside a string, empty input) so the regression can’t come back silently.

**Status.** Task 1 of 8 for C2. Next: `holdout_curator.py` core (assemble/validate/dedup/manifest)
plus its fixture-driven tests.
