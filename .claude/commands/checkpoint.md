---
description: Lint, test, commit the completed task, and update the backlog state.
argument-hint: [spec-id] (optional, e.g. A1)
---

Close out the work in the working tree.

1. **Check.** Run the full gate and do not proceed if it fails:
   ```
   uv run ruff check . && uv run black --check . && uv run pytest -q
   ```
   A red gate means the task is not done. Report the failure; do not commit around it.

2. **Review the diff.** `git status` and `git diff`. Confirm it contains only this one task's changes — no stray weights, no `.env`, no generated data, no unrelated refactor that drifted in. The commit-hygiene hook is the backstop, not the plan.

3. **Commit.** One task, one commit. Message: what changed and why, referencing the spec and task (`A1 T2: hand-rolled BPE merge loop`). If nothing is staged, stage the task's files deliberately — never `git add -A` on a dirty tree you have not read.

4. **Update `specs/STATUS.md`.** Tick the task in the spec file. Move the spec's state on only when *every* task in it is done and its acceptance criteria are all met — `building → done`. Record the W&B run ID if this task produced a run.

5. **Report.** State plainly what was committed, what state the spec is now in, and what the next task is.

If the user named a spec (`$1`), scope steps 3–4 to it.
