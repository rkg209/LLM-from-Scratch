---
description: Run a package's smoke config on CPU and report pass/fail. The only run command to use in a session.
argument-hint: <package> — ch1 | ch2 | ch3 | all
---

Run the smoke profile for **$1** and report the result honestly.

| `$1` | Command |
|---|---|
| `ch1` | `uv run python -m ch1_architecture.train --config ch1_architecture/configs/smoke.yaml` |
| `ch2` | `uv run python -m ch2_adaptation.finetune --config ch2_adaptation/configs/smoke.yaml` |
| `ch3` | `uv run python -m ch3_operation.serve --config ch3_operation/configs/smoke.yaml` |
| `all` | all three, in order |

Then run that package's tests: `uv run pytest -q <package>/tests`.

Rules:

- **Smoke only.** Never substitute `full.yaml`. If a smoke run needs a GPU or takes more than ~120 seconds, that is a bug in the config (NFR-1), not a reason to reach for the full profile.
- If the package's heavy extra is not installed, say so and give the user the `uv sync --extra <ch1|ch2|ch3>` command rather than installing it silently.
- **Report what happened.** If it fails, show the actual error output and say it failed. Do not summarize a failure as a partial success.
