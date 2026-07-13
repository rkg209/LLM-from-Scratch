---
description: Turn an accepted spec into a technical plan — files, approach, config shape, tests. Still no code.
argument-hint: <spec-id> (e.g. A1, C3, O2)
---

Write the **Technical plan** section of spec **$1**.

Refuse to proceed if the spec's state in `specs/STATUS.md` is not at least `planned`, or if its Clarifications section still has open questions — say so and point the user at `/clarify`.

1. Read `specs/$1-*.md` in full, plus the sections of `planning/02-architecture.md` and `planning/03-system-design.md` that describe the modules this spec touches. Those documents already specify module names, function signatures, dataclass fields, and invariants — **use them; do not invent parallel designs.** Where the plan must deviate from them, say so explicitly and give the reason.
2. Read the code that already exists in the target package before proposing new code. Reuse the shared config loader (`eval/config.py`) and the shared harness (`eval/harness.py`) rather than reimplementing either.
3. Write the plan into the spec file:
   - **Files** — every file created or modified, with a one-line purpose each.
   - **Approach** — how it works, in prose. Name the functions and their signatures. Call out the invariants that must hold.
   - **Config** — what new keys appear in `configs/smoke.yaml` and `configs/full.yaml`, and the smoke values that keep the run under 120 seconds on CPU (NFR-1).
   - **Tests** — the test that proves each acceptance criterion. One criterion with no test is a hole in the plan.
   - **Risks** — what could quietly be wrong, and how a test would catch it.
4. Every acceptance criterion in the spec must be addressed somewhere in the plan. If one cannot be, the spec is wrong — go back to `/specify`.

Write no implementation code. The plan is prose and signatures.
