# F1 — Repo + tooling skeleton: gap-closure plan

## Context

`specs/F1-repo-tooling-skeleton.md` is marked **done** in `specs/STATUS.md`, and T1–T4 are all checked off. A live verification of the repo confirms most of that is true — but not all of it, and F1 is the foundation every other spec depends on (F2, F3, F4 all list it as their sole dependency). A foundation spec that claims `done` while an acceptance criterion silently fails is exactly the kind of rot the gate exists to prevent.

Verified against the current working tree:

| AC | Criterion | Status |
|---|---|---|
| 1 | Tree matches `planning/02-architecture.md` §2 | ❌ **`scripts/` missing** |
| 2 | `uv sync --extra dev` builds clean | ✅ (`uv.lock` present, `uv run` works) |
| 3 | `uv run ruff check .` exits 0 | ✅ "All checks passed!" |
| 4 | `uv run black --check .` exits 0 | ✅ 25 files unchanged |
| 5 | `uv run pytest -q` exits 0, collects all 3 packages | ✅ 62 passed in 0.09s |
| 6 | Heavy ML deps are optional extras; base has no torch | ✅ base = `pyyaml`, `jsonschema` only |
| 7 | `.gitignore` covers weights/data/secrets | ✅ all seven patterns present |
| 8 | `ci.yml` runs lint + format + tests, never a full train | ✅ CPU-only, smoke configs only |

**The entire remaining scope of F1 is one missing directory: `scripts/`.**

It is not cosmetic. `scripts/` is the home of Chapter 3's model-preparation helpers — `planning/02-architecture.md` §3.3 places `merge_adapter.py`, `export_gguf.py`, and `quantize_gguf.py` there, and `specs/O1-merge-and-gguf-quantize.md` names all three by path. Creating it in F1 (where the skeleton spec puts it) rather than improvising it in O1 keeps the skeleton honest and gives O1 a place to land.

**Intended outcome:** `scripts/` exists and is tracked, all 8 acceptance criteria pass against the live repo, and the discrepancy — a spec marked `done` while AC-1 failed — is recorded in `progress_report.md` rather than quietly patched.

## Approach

Git cannot track an empty directory, and a `.gitkeep` would leave a future reader guessing what the directory is for. The repo already solves this exact problem twice: `eval/holdout/README.md` and `eval/results/README.md` are placeholder READMEs that reserve a directory *and* document its contract (who writes into it, under which spec). Reuse that pattern rather than inventing a new one.

## Changes

### 1. Save this plan into the repo

Copy this file to `.claude/plans/F1-repo-tooling-skeleton.md` (the path requested; plan mode only permits writing the harness plan file, so this is the first executed step).

### 2. `scripts/README.md` — new file

Reserves the directory and states its contract, mirroring the table format of `eval/results/README.md`:

- One-line purpose: standalone helper scripts that are not part of any importable package — one-shot model-preparation steps run by hand between chapters, not imported at runtime.
- A table of the three planned scripts, each with its writer spec, matching the `eval/results/README.md` shape:

  | Script | Purpose | Spec |
  |---|---|---|
  | `merge_adapter.py` | `peft.merge_and_unload()` → `outputs/merged/` | O1 |
  | `export_gguf.py` | llama.cpp `convert_hf_to_gguf.py` → `outputs/model.gguf` | O1 |
  | `quantize_gguf.py` | llama.cpp quantize → `outputs/model-Q4_K_M.gguf` | O1 |

- A note that these are not yet written — O1 owns them — so nobody reads the empty directory as a bug.

No Python file is added. Writing a stub `merge_adapter.py` now would be implementing O1 without an accepted plan, which the workflow forbids.

### 3. `specs/F1-repo-tooling-skeleton.md` — annotate, do not rewrite

Add one line under **Clarifications** recording that AC-1 was found unmet after F1 was closed and was fixed on 2026-07-14. The spec's task list stays as-is; the point is that the record shows the miss, not that it looks clean.

### 4. `progress_report.md` — append an entry at the bottom

Per `CLAUDE.md`, newest at the bottom, never rewrite an earlier entry. The entry covers:

- **What** — created `scripts/` with a contract README; F1 now passes all 8 ACs.
- **Why** — F1 was marked `done` with AC-1 unverified. `scripts/` is named by path in O1 and in the architecture doc, so its absence would have surfaced as improvisation mid-O1. The deeper point, and the one worth recording: the `done` state was set without mechanically re-checking every acceptance criterion against the tree. That is the failure to learn from, not the missing folder.
- **How** — README-as-placeholder, reusing the `eval/holdout/` + `eval/results/` precedent instead of `.gitkeep`, so the directory documents its own purpose.

## Files

| File | Action |
|---|---|
| `.claude/plans/F1-repo-tooling-skeleton.md` | new — this plan |
| `scripts/README.md` | new — directory placeholder + contract table |
| `specs/F1-repo-tooling-skeleton.md` | edit — one clarification line |
| `progress_report.md` | edit — append entry at bottom |

Deliberately untouched: `pyproject.toml`, `.gitignore`, `.env.example`, `.github/workflows/ci.yml`, all three package trees. They already satisfy their criteria; changing them would be churn.

## Verification

Re-check all 8 acceptance criteria, not just the one that changed:

```bash
# AC-1 — the tree now matches planning/02-architecture.md §2
ls -d ch1_architecture ch2_adaptation ch3_operation eval specs scripts .github/workflows

# AC-2 — clean environment build
uv sync --extra dev

# AC-3, AC-4, AC-5 — the project gate from CLAUDE.md
uv run ruff check . && uv run black --check . && uv run pytest -q

# AC-6 — base env has no torch
uv run python -c "import importlib.util; print('torch present:', importlib.util.find_spec('torch') is not None)"

# AC-7 — an ignored artifact stays untracked
git check-ignore -v outputs/model.pt eval/../wandb/ .env

# AC-8 — no untracked artifacts, and CI is CPU-only
git status --short
```

Expected: `scripts/` listed, gate exits 0 (62+ tests pass), `torch present: False`, every `git check-ignore` path matches a `.gitignore` rule, `git status` shows only the four intended files.

`scripts/README.md` is documentation with no runtime surface, so there is no behavior to drive beyond the tree and gate checks above.

## Commit

One task, one commit (`CLAUDE.md` workflow). No `Co-Authored-By:` trailer of any kind — it breaks pushing to this repo.

```
F1: add missing scripts/ directory, closing AC-1

scripts/ is named by path in planning/02-architecture.md §3.3 and in
specs/O1; F1 was marked done without it. Reserved with a contract README,
matching the eval/holdout/ and eval/results/ precedent.
```

`specs/STATUS.md` needs no change — F1 stays `done`, and now actually is.
