# Holdout curation staging area

This directory holds the working files for building C2's frozen holdout set — never the
frozen set itself. `eval/holdout/` is where the finished, committed set lives, and it is
sacred (CLAUDE.md #3): `.claude/hooks/leakage_guard.py` denies **every** write to
`eval/holdout/` unconditionally, from any session, with no override. That is by design —
the actual freeze (moving files out of here and into `eval/holdout/`) has to happen by
hand, outside a Claude Code session, exactly like C1's deferred full-baseline run.

## Files in this directory

- `synthetic.jsonl` — 30 hand-authored synthetic-clean records (committed; this is
  curation content, not generated output).
- `mined_raw.jsonl` — 10 real Java/Spring snippets mined via the GitHub MCP server (not
  yet produced — mining is blocked on `GITHUB_TOKEN`; see `progress_report.md`'s C2 T4
  entry). No labels yet.
- `mined_labeled.jsonl` — output of `holdout_curator.py label-mined`: the 10 mined
  snippets with the five `ReviewOutput` fields attached by the frontier API.
- `holdout.jsonl`, `manifest.json`, `dedup_report.md` — output of
  `holdout_curator.py freeze`: the assembled, validated, deduped 40-record set plus its
  provenance manifest and dedup report. **Staged, not committed** — see the runbook below.

## The runbook: staging → frozen

1. Mine 10 real records (needs `GITHUB_TOKEN` set and the `github` MCP server
   connected):
   ```
   # interactive GitHub MCP tool calls, writing eval/staging/mined_raw.jsonl by hand
   ```
2. Label them via the frontier API:
   ```
   uv run python -m ch2_adaptation.holdout_curator label-mined \
     --config ch2_adaptation/configs/baseline_full.yaml
   ```
3. Freeze the full 40-record set:
   ```
   uv run python -m ch2_adaptation.holdout_curator freeze --curator "<your name>"
   ```
   Inspect the staged `holdout.jsonl` (should have 40, or fewer post-dedup, lines),
   `manifest.json`, and `dedup_report.md` before proceeding.
4. **Outside this session, in your own terminal** — the leakage guard blocks this from
   here by design:
   ```bash
   mkdir -p eval/holdout
   mv eval/staging/holdout.jsonl eval/holdout/holdout.jsonl
   mv eval/staging/manifest.json eval/holdout/manifest.json
   git add eval/holdout/holdout.jsonl eval/holdout/manifest.json eval/frozen_hashes.txt
   git commit -m "C2: freeze the 40-record independent eval set"
   ```
5. Confirm the freeze: `EVAL_CONTEXT=1 uv run python -c "from eval.harness import load_holdout; print(len(load_holdout('eval/holdout/holdout.jsonl')))"` should print the same count `manifest.json` records.

After this, the set is frozen. Growing or rebalancing it after freezing is out of scope
for this spec (`specs/06-C2-independent-eval-set.md`'s Out of scope section) — a change
needs a new spec with a new freeze date, and every previously published number gets
re-run.
