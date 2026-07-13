# eval/results/ — published numbers

One writer per file. Everything here is produced by the shared harness and is safe to read from anywhere.

| File | Written by | Spec |
|---|---|---|
| `ch1_benchmark.json` | `ch1_architecture/benchmark.py` | A5 |
| `baselines.json` | `ch2_adaptation/baseline.py` | C1 |
| `finetuned.json` | `ch2_adaptation/evaluate.py` | C5 |
| `plots/` | `benchmark.py`, `evaluate.py` | A5, C5 |

Every number here traces back to a config file and a W&B run ID recorded in `specs/STATUS.md`. A number that cannot be traced to a run does not go in the README.
