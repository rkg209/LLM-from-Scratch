# <ID> — <Title>

| | |
|---|---|
| **State** | draft \| planned \| building \| done |
| **Depends on** | <spec IDs, or —> |
| **Requirements** | <FR-n, NFR-n from planning/01-requirements.md> |
| **W&B run** | <run ID, once there is one> |

## Problem

What is missing, and why it matters to the project's goals. Prose, not a task list. A reader who knows nothing about this spec should finish this section understanding why it exists.

## Scope

What this spec covers.

## Acceptance criteria

Numbered, each one **mechanically checkable** — name the command, assertion, or measurement that settles it. "Works correctly" is not a criterion.

1. …
2. …

## Out of scope

The things a reader would reasonably assume are in here but are not — and which spec owns them instead.

## Clarifications

*(filled by `/clarify` — one line per resolved question, with the date)*

- **Q:** … → **A:** … *(YYYY-MM-DD)*

## Technical plan

*(filled by `/plan`)*

**Files** — every file created or modified, one line of purpose each.

**Approach** — how it works, in prose. Function names and signatures. The invariants that must hold.

**Config** — new keys in `configs/smoke.yaml` and `configs/full.yaml`; the smoke values that keep it under 120 seconds on CPU.

**Tests** — the test that proves each acceptance criterion above.

**Risks** — what could quietly be wrong, and how a test catches it.

## Tasks

*(filled by `/tasks` — each one is a single commit that leaves the repo green)*

- [ ] **T1** — … · files: `…` · verify: `…`
