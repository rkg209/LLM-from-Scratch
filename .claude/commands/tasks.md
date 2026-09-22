---
description: Decompose an accepted plan into an ordered list of independently-committable tasks.
argument-hint: <spec-id> (e.g. A1, C3, O2)
---

Write the **Tasks** section of spec **$1**.

Refuse to proceed if the spec has no Technical plan — point the user at `/plan`.

Break the plan into an ordered checklist where each task:

- **Is one commit.** One coherent change with a message you could write in a sentence (CON-10).
- **Leaves the repo green.** After the task, `ruff check . && black --check . && pytest -q` passes. A task that needs the *next* task in order to import is two halves of one task, not two tasks.
- **Carries its own test.** The task that adds a module adds that module's test (NFR-11). Never a trailing "write the tests" task.
- **Names its files.** State which files it creates or edits.
- **Says how it is verified.** The exact command that proves it worked.

Order them so dependencies flow: config and schema first, then the units with no dependencies, then the units built on them, then the entrypoint that wires them together.

Write them into the spec as a checklist:

```
- [ ] **T1** — <what> · files: `<paths>` · verify: `<command>`
```

Then set the item's state to `building` in `specs/STATUS.md`.

Write no code.
