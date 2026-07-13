---
description: Implement exactly one task from a spec's task list. One task, one commit.
argument-hint: <spec-id> <task-id> (e.g. A1 T2)
---

Implement task **$2** of spec **$1** — that task and nothing else.

1. Read `specs/$1-*.md`. Find task **$2**. If the spec has no Technical plan or no task list, stop and point the user at `/plan` or `/tasks`.
2. Re-read the plan's Approach section for the part that covers this task, and read the existing code it builds on.
3. Implement it, following the conventions in `CLAUDE.md`: type hints everywhere, functions under ~40 lines, every hyperparameter from config, no magic numbers, seeds set explicitly.
4. Write the task's test in the package's `tests/` directory. The test must fail if the implementation is wrong, not merely execute it.
5. Verify with the task's stated verify command, then the full check:
   ```
   uv run ruff check . && uv run black --check . && uv run pytest -q
   ```
6. Dispatch the `code-reviewer` subagent on the diff. Fix what it finds, or say why you disagree.
7. Tick the task's checkbox in the spec.

**Scope discipline is the point of this command.** If you notice something else that needs doing — a bug in an earlier task, a missing config key, a refactor that would help — do not fix it here. Note it, finish this task, and raise it. Drift is how one commit becomes six.

Do not commit. `/checkpoint` commits.
