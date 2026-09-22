---
description: Run the shared eval harness against the holdout set and regenerate the comparison table and plots.
---

Invoke the `eval-table` skill to refresh the project's headline numbers.

This is the **only** context in which `eval/holdout/` may be read. The leakage guard requires `EVAL_CONTEXT=1` on the commands that touch it — set it per-command, never in the shell profile:

```
EVAL_CONTEXT=1 uv run python -m ch2_adaptation.evaluate --config ch2_adaptation/configs/full.yaml
```

The skill covers the details. The rules that do not bend:

- **All three systems are scored on the identical frozen holdout set** using the same `eval/harness.py`. A number produced any other way does not go in the table.
- **Report what the harness returns.** If the fine-tuned model loses to the baseline, the table says it lost. No re-running until a favorable seed appears, no quietly dropping the samples it failed (NFR-7).
- Numbers land in `eval/results/*.json`, plots in `eval/results/plots/`, and the table is mirrored into `README.md` by `ch2_adaptation.publish_table` (never by hand).
- Re-publishing the table from artifacts that already exist needs no GPU, no API key and no `EVAL_CONTEXT=1` — only the scoring runs that produce those artifacts do.
- Record the W&B run ID in `specs/STATUS.md` so every published number traces back to a run (NFR-16).
