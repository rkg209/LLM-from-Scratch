# F4 — Claude Code scaffold

| | |
|---|---|
| **State** | done |
| **Depends on** | F1 |
| **Requirements** | FR-7, FR-8, NFR-15, NFR-18, NFR-22, BG-5 |
| **W&B run** | — |

## Problem

Three of this project's rules cannot survive on good intentions, because each one is violated by an action that feels locally reasonable:

- Starting a full training run in a session (it is right there, and it would answer the question).
- Peeking at the holdout to see why the model is failing (it is just a file, and the curiosity is genuine).
- Committing a checkpoint (it is the artifact you just spent an hour producing).

Each is a single keystroke, and each silently destroys something the project depends on — the GPU budget, the headline metric, the repo. Code review catches these *sometimes*. **Hooks catch them every time.** That is the difference between a rule and a guardrail, and it is why this spec exists at the foundation rather than as a nicety.

The scaffold is also a portfolio artifact in its own right (BG-5): it is designed to be lifted wholesale into the sibling projects.

## Scope

`CLAUDE.md`; the five lifecycle hooks; the eight workflow slash commands; the five model-invoked skills; the five subagents; `.claude/settings.json` wiring; `.mcp.json` for the Hugging Face, W&B, and GitHub servers.

## Acceptance criteria

1. `CLAUDE.md` states the six non-negotiables, the build/test commands, and the SDD workflow.
2. **GPU-budget guard**: a Bash command matching a full-run pattern (`configs/full.yaml`, `accelerate launch`, `torchrun`, large `max_steps`) is **denied**; the same command with `ALLOW_FULL_RUN=1` set is allowed; a `--config smoke.yaml` command is always allowed (NFR-22).
3. **Leakage guard**: a Write or Edit targeting `eval/holdout/` is denied unconditionally; a Read of it is denied unless `EVAL_CONTEXT=1` (NFR-15).
4. **Commit hygiene**: a `git commit` with a `.gguf`, `.pt`, `.env`, or `eval/holdout/` file staged is denied, naming the offending file.
5. **Auto-format**: editing a Python file leaves it `ruff --fix`-ed and `black`-formatted (NFR-8).
6. **Session greeter**: a new session prints the smoke-only rule and the first not-`done` spec from `specs/STATUS.md`.
7. All eight commands (`/specify`, `/clarify`, `/plan`, `/tasks`, `/implement`, `/smoke`, `/eval`, `/checkpoint`), five skills, and five subagents load without error and are exercised at least once.
8. `.mcp.json` references tokens by environment variable only; **no secret appears in any tracked file** (NFR-14).
9. `specs/STATUS.md` exists and tracks every spec's state (FR-8).

## Out of scope

- Packaging the scaffold as a redistributable plugin for the sibling projects — worth doing once it has proven itself here, not before.
- GitHub Spec Kit. The native primitives cover the flow; layer Spec Kit later only if specs must be readable by someone who never opens Claude Code.

## Clarifications

- **Q:** `planning/02-architecture.md` §2.5 puts MCP servers inside `.claude/settings.json` and registers hooks as bare paths. → **A:** That shape does not load. MCP goes in `.mcp.json`; hooks need matcher/command objects and scripts that read a JSON payload on stdin; skills are `skills/<name>/SKILL.md`. Intent preserved, form corrected. *(2026-07-13)*
- **Q:** Should the auto-format hook also run the package's smoke test on every edit, as §6.4 suggests? → **A:** No — a test run per edit makes the loop crawl. It runs tests only under `HOOK_RUN_TESTS=1`; `/smoke` and CI cover the rest. *(2026-07-13)*

## Technical plan

**Files** — `CLAUDE.md`, `.claude/settings.json`, `.mcp.json`, `.env.example`, `.claude/hooks/{_hooklib,gpu_budget_guard,leakage_guard,auto_format,commit_hygiene,session_greeter}.py`, `.claude/commands/*.md` (8), `.claude/skills/*/SKILL.md` (5), `.claude/agents/*.md` (5).

**Approach** — each hook reads the payload on stdin and emits a `permissionDecision` (deny) or nothing (allow). `_hooklib.run()` swallows exceptions so a broken hook can never wedge a session — a guardrail that crashes the tool it guards is worse than no guardrail.

## Tasks

- [x] **T1** — `CLAUDE.md` constitution
- [x] **T2** — five hooks + `settings.json` wiring · verify: each hook denies its target payload
- [x] **T3** — eight slash commands
- [x] **T4** — five skills + five subagents
- [x] **T5** — `.mcp.json` + `.env.example` · verify: no secrets tracked
- [x] **T6** — `specs/STATUS.md` + `_template.md` + the 24 spec files
