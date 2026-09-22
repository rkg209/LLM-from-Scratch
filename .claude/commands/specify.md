---
description: Write the spec for a backlog item — problem, scope, acceptance criteria, out-of-scope. No implementation.
argument-hint: <spec-id> (e.g. A1, C3, O2)
---

Author or revise the spec for backlog item **$1**.

1. Read `specs/STATUS.md` to find the item, and read its existing spec file `specs/*$1-*.md` if one exists.
2. Read the source material this spec derives from, and only that — do not read the whole `planning/` folder:
   - `llm-engineering-architecture-to-edge.md` §8 for the backlog entry (goal, acceptance signal, dependency).
   - `planning/01-requirements.md` for the FR/NFR numbers this spec must satisfy.
   - `planning/02-architecture.md` and `planning/03-system-design.md` for the components it touches.
3. Fill in the spec file following `specs/_template.md`:
   - **Problem** — what is missing and why it matters to the project's goals. Not a task list.
   - **Scope** — what this spec covers, in prose.
   - **Acceptance criteria** — numbered, each one *mechanically checkable*. "Works correctly" is not a criterion; "`decode(encode(text)) == text` for every string in the test corpus" is. Every criterion must trace to an FR/NFR.
   - **Out of scope** — the things a reader would reasonably assume are included but are not, and which spec owns them instead.
   - **Dependencies** — the specs that must be `done` first.
4. Leave **Clarifications**, **Technical plan**, and **Tasks** empty. Those belong to `/clarify`, `/plan`, and `/tasks`.
5. Set the item's state to `draft` in `specs/STATUS.md`.

Write no code. If the requirements are ambiguous, do not resolve the ambiguity yourself — record it as an open question and tell the user to run `/clarify`.
