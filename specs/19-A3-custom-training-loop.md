# A3 — Custom training loop

| | |
|---|---|
| **State** | building (code + tests done, verified end-to-end on the smoke config in 1.1s CPU; AC-7's "recognizably English" judgment needs the full-config GPU run) |
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

- **Tokenizer caching.** The BPE merge loop runs once and is cached to `outputs/ch1/tokenizer.json`, keyed by a SHA-256 hash of the corpus text (a sibling `.hash` file). `train.py` retrains only when the corpus changed. Without this, the smoke run would spend its 120s budget retraining the tokenizer on every invocation instead of training the model.
- **Schedule.** A hand-written `lr_lambda` (linear warmup then cosine decay to `learning_rate * lr_min_ratio`) rather than composing `LambdaLR`/`CosineAnnealingLR` — warmup + `eta_min` together are clearer as one closed-form function than as two chained schedulers, and the closed form is what `test_train.py` checks against directly.
- **Sampling failures are not training failures.** An early, near-random model can sample a byte sequence that is not valid UTF-8 on its own (vocab is seeded with all 256 raw bytes). `training_loop` catches `UnicodeDecodeError` around the periodic sample decode and logs a placeholder string rather than crashing the run.
- **Checkpoint atomicity.** Write to a `.tmp` path then `os.replace` — an interrupted save cannot corrupt the previous checkpoint.

## Technical plan

See `.claude/plans/AfterOperation.md` §A3. `main()` stays under 40 lines by delegating to `build_tokenizer_and_loader`, `build_model`, `build_optimizer`, `build_scheduler`, `training_loop`, `save_checkpoint`. Step: forward → `F.cross_entropy` → backward → `clip_grad_norm_` → `optimizer.step()` → `scheduler.step()`. W&B is inline and config-gated (`wandb_mode: disabled` in smoke, so the smoke path never touches the network); every `log_every` steps logs `{step, train_loss, perplexity, lr, tokens_per_sec}`, every `sample_every` steps generates and logs a text sample. Checkpoint holds `{model_state, optimizer_state, config, step, tokenizer_vocab, tokenizer_merges}`. `generate.py`'s `_generate_uncached` returns per-step logits alongside sampled tokens — A4's identity test needs the logits, not just the tokens.

## Tasks

- **A3-T1** — config fields (`warmup_steps`, `lr_min_ratio`, `ckpt_every`, `sample_every`, `sample_prompt`, `max_new_tokens`, `temperature`, `top_k`, `checkpoint_path`, `wandb_project`, `wandb_mode`), both YAMLs, `test_gpt_config.py` update. **Done.**
- **A3-T2** — tokenizer caching (hash-keyed) + `build_tokenizer_and_loader`. **Done.**
- **A3-T3** — training loop: optimizer, scheduler, clipping, loss/perplexity, determinism test. **Done.**
- **A3-T4** — checkpointing (atomic write, round-trip test). **Done.**
- **A3-T5** — `generate.py` uncached path + top-k/temperature tests. **Done.**
- **A3-T6** — W&B wiring + periodic sample logging; smoke run timed at **1.1s** (well under the 120s budget), recorded in `progress_report.md`. **Done.**

Remaining: a full-config run on the GPU box (out of session) to produce a checkpoint whose samples are judgeable for AC-7, and its W&B run ID for `specs/STATUS.md`.
