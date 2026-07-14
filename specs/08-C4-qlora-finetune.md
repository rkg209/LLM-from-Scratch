# C4 — QLoRA fine-tune

| | |
|---|---|
| **State** | draft |
| **Depends on** | C2, C3 |
| **Requirements** | FR-18, CON-4, CON-6, NFR-1, NFR-16 |
| **W&B run** | — |

## Problem

This is the skill the project is built to demonstrate (BG-2): turning a generalist open model into a cheap, private specialist on one narrow task, on free compute, with a technique that fits in a Colab session.

The engineering difficulty is not the LoRA config — that is twelve lines and it is already locked in `planning/03-system-design.md`. It is the split between the two environments: **the code must be provable on a CPU in seconds and correct on a GPU for an hour**, and the person who wrote it will not be watching when the GPU run happens. Every bug that only appears at 4-bit, or only at batch size 4, costs a re-run of the free-tier GPU budget.

## Scope

`model.py` (base model loading, 4-bit for full / fp32 for smoke), `lora.py` (the LoRA config), `data_loader.py` (chat-template formatting with prompt-token masking), and `finetune.py` (SFTTrainer, W&B, adapter save).

## Acceptance criteria

1. `uv run python -m ch2_adaptation.finetune --config configs/smoke.yaml` runs **end-to-end on CPU in under 120 seconds** — fp32, no bitsandbytes, `max_steps=5`, `lora_r=4` (FR-18a, NFR-1).
2. The full config produces a saved LoRA adapter at `outputs/adapter/`, launched **manually on the GPU box** — never from a session (FR-18b, CON-4).
3. `print_trainable_parameters()` reports roughly **0.5%** of parameters trainable. Anywhere near 100% means LoRA did not attach and the run must be stopped.
4. **Loss is computed on the assistant turn only** — system and user tokens are masked to `-100`. Verified by a test asserting the label tensor is `-100` across the prompt span. Training on the prompt teaches the model to reproduce buggy Java.
5. W&B logs `train_loss`, `eval_loss`, and the LR schedule; the run ID is recorded in `specs/STATUS.md` (NFR-16).
6. The 4-bit config is exactly NF4 + double-quant + bf16 compute, per the `qlora-recipe` skill.
7. `finetune.py` never reads `eval/holdout/` (CON-6), including for eval-during-training — validation uses `data/val.jsonl`.
8. Seeds are set from config; the smoke run is reproducible (NFR-4).

## Out of scope

- Scoring the adapter → **C5**. Merging and GGUF export → **O1**.
- Hyperparameter sweeps. The recipe is locked; the free GPU budget does not fund a search, and a sweep against the holdout would be leakage anyway.
- Any claim that the fine-tune is better than the baselines. That claim belongs to C5, after it is measured.

## Clarifications

*(filled by `/clarify`)*

## Technical plan

*(filled by `/plan`)*

## Tasks

*(filled by `/tasks`)*
