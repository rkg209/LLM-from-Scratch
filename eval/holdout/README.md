# eval/holdout/ — the frozen evaluation set

**Empty at project start.** The set is curated and frozen in spec **C2**.

## What goes here

- `holdout.jsonl` — the frozen evaluation records: synthetic-clean **plus mined-real** Java diffs and lint findings.
- `manifest.json` — provenance: freeze date, counts, sources, dedup method, schema version, curator.

## Why it is guarded

This directory is the answer to the only serious objection to Chapter 2: *"you generated your training data with a frontier model and then showed your model imitates that frontier model."* An independent, partly-real eval set is what makes the comparison a measurement rather than a tautology.

It is destroyed by being looked at. Every glance while debugging a training failure leaks a little information into the next design decision — the prompt gets tweaked, the parse gets more forgiving — and the metric stops being an honest one long before anybody notices.

So the rule is not "be careful." The rule is enforced:

- **Writes are denied unconditionally.** The set is frozen; if it is wrong, that is a new spec with a new freeze date and every published number re-run.
- **Reads are denied** unless `EVAL_CONTEXT=1` is set on the command — which `/eval` and the eval harness set for themselves, and training code never does.

`.claude/hooks/leakage_guard.py` enforces both (CON-6, NFR-6, NFR-15).
