---
description: Interrogate a draft spec for ambiguity and record the answers before any planning happens.
argument-hint: <spec-id> (e.g. A1, C3, O2)
---

Close the open questions in spec **$1** before it can be planned.

1. Read `specs/$1-*.md`.
2. Run the `spec-review` skill (or dispatch the `spec-reviewer` subagent directly) to audit the spec cold, without your framing. It looks for: acceptance criteria that cannot be mechanically checked, scope that overlaps another spec, missing dependencies, and claims that assume a decision nobody has made.
3. Combine the auditor's findings with your own reading into a **short list of genuine decisions the user must make** — things where the answer changes what gets built. Do not ask about anything already locked in `planning/` or `CLAUDE.md`; look it up instead.
4. Ask them with `AskUserQuestion`, one batch, recommended option first.
5. Record every answer in the spec's **Clarifications** section as `Q → A`, with the date. This is the audit trail for why the spec says what it says.
6. Fold the answers back into Scope / Acceptance criteria / Out of scope so the spec stands alone.
7. Set the item's state to `planned` in `specs/STATUS.md` once no open questions remain.

Write no code.
