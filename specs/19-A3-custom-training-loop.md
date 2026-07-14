# A3 — Custom training loop

| | |
|---|---|
| **State** | draft |
| **Depends on** | A2 |
| **Requirements** | FR-11, NFR-1, NFR-4, NFR-16 |
| **W&B run** | — |

## Problem

`Trainer.train()` is one line and teaches nothing. The loop — forward, loss, backward, clip, step, schedule, log — is where the details that actually bite live: gradient clipping that stops a loss spike from destroying an hour of training, a cosine schedule that decides whether the model converges, the difference between logging loss and logging perplexity.

It is also the first point where the model produces something a human can judge. Coherent text out of a model you wrote from an empty file is the moment Chapter 1 becomes a portfolio piece rather than an exercise.

## Scope

`train.py`: AdamW, cosine LR decay, gradient clipping, checkpointing, W&B logging, and the generation samples that show the model is learning. Plus `generate.py` (uncached path only — the cached path is A4).

## Acceptance criteria

1. `uv run python -m ch1_architecture.train --config ch1_architecture/configs/smoke.yaml` completes on **CPU in under 120 seconds** (NFR-1, FR-11a).
2. Every hyperparameter — learning rate, max steps, clip norm, warmup, batch size — is read from the config. No numeric literal in `train.py` is a hyperparameter (NFR-12).
3. A W&B run is created and logs `train_loss`, `perplexity`, `lr`, and `tokens_per_sec` at a config-set interval (FR-11b, NFR-16). W&B runs in offline mode when no API key is present, so the smoke path never depends on the network.
4. Gradient clipping is applied at `max_norm` from config; the LR follows cosine decay from `lr_max` to `lr_max/10` over `max_steps`.
5. A checkpoint is written to `outputs/ch1/model.pt` containing the model state, optimizer state, config, step, and the tokenizer's vocab and merges — enough to reload and generate without the original corpus.
6. Re-running the same config with the same seed produces an identical loss curve (NFR-4).
7. After a full-config run, generated samples are recognizably English rather than random tokens (FR-11c — judged by the developer, and the samples are logged to W&B so the judgment is on the record).

## Out of scope

- KV-cache and the cached generation path → **A4**.
- Quantization and the benchmark → **A5**.
- Any claim about speed. This spec makes the model learn; it does not make it fast.

## Clarifications

*(filled by `/clarify`)*

## Technical plan

*(filled by `/plan`)*

## Tasks

*(filled by `/tasks`)*
