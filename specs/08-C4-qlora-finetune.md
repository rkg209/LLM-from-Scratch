# C4 — QLoRA fine-tune

| | |
|---|---|
| **State** | building |
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

None open — unlike C2/C3, almost nothing here was an undecided fork. The LoRA config, 4-bit
setup, and prompt/masking format are locked verbatim in the `qlora-recipe` skill and
`planning/03-system-design.md` §1.2; C4's job was mechanical: write the modules the recipe
already specifies, against configs that already existed.

## Amendments (recorded here per plan, not applied silently)

- **`FinetuneConfig` gained the smoke/full model-tag split `BaselineConfig` already had.**
  `smoke.yaml` set `model_tag` to the real 1.5B model with no carve-out, which would try to
  download and load a ~3GB model on CPU — unable to reliably finish inside AC-1's 120-second
  budget. `__post_init__` now requires `model_tag == SMOKE_MODEL_TAG` for smoke,
  `LOCKED_MODEL_TAG` for full, mirroring `BaselineConfig` exactly.
- **`train_path`, `val_path`, `wandb_mode` added to `FinetuneConfig`.** Not originally specified
  as config fields — `data_loader.ReviewDataset` needs a JSONL path, and it has to come from
  config, not a hardcoded string (CLAUDE.md's "every hyperparameter from config" rule). A small
  committed `ch2_adaptation/data/finetune_smoke/{train,val}.jsonl` fixture (4 hand-authored
  records) lets the smoke run stay self-contained rather than depending on a prior `data_gen`
  smoke run's gitignored output.
- **`transformers.Trainer` used instead of `trl.SFTTrainer`.** AC-1/AC-4/AC-5 don't name a
  specific trainer class, but the plan's Files section did. The installed trl version (1.8.0)
  requires `train_dataset` to be a real `datasets.Dataset` (it accesses `.column_names`
  internally) — confirmed directly: constructing `SFTTrainer` against `ReviewDataset` raises
  `AttributeError: 'ReviewDataset' object has no attribute 'column_names'`. `ReviewDataset`
  already does its own tokenization and `-100` masking, so `trl.SFTTrainer`'s raw-text +
  `formatting_func` design isn't the right fit anyway; `transformers.Trainer` +
  `DataCollatorForSeq2Seq(..., label_pad_token_id=-100)` was verified to train successfully
  against the same objects and preserve the masking through padding.
- **AC-3's floor was measured, not assumed, and turned out to need a real fix.** The `qlora-recipe`
  skill's "roughly 0.5%" figure is for the full recipe (r=16 on the 1.5B model). Measured directly
  against `SMOKE_MODEL_TAG` + the smoke recipe (r=4): **0.018% trainable** — the same four target
  modules are a larger proportional share of a ~2.4M-parameter model but still tiny in absolute
  count. A floor calibrated only against the full figure would have failed every legitimate smoke
  run. `lora.py`'s floor is `0.001%`, with a regression test locking in the measured smoke-model
  margin.

## Technical plan

See `.claude/plans/08-C4-qlora-finetune.md` for full detail. Summary:

**New files** — `ch2_adaptation/src/ch2_adaptation/{lora.py,data_loader.py}`;
`ch2_adaptation/data/finetune_smoke/{train.jsonl,val.jsonl}`;
`ch2_adaptation/tests/{test_lora.py,test_data_loader.py,test_finetune.py}`.

**Modified** — `config.py` (`FinetuneConfig` smoke/full split, `train_path`/`val_path`/
`wandb_mode`); `smoke.yaml`/`full.yaml` (new fields, `SMOKE_MODEL_TAG`); `model.py`
(`load_base_model_for_training`, alongside the existing inference-only `make_hf_generator`);
`finetune.py` (real `main()`, replacing the F2 scaffold).

**Key design points:**

- `attach_lora` calls `peft.get_peft_model` then a hard trainable-fraction check (AC-3) —
  `peft.get_peft_model` freezes every non-LoRA parameter itself (verified empirically), so the
  guard's realistic trigger is a misconfigured `target_modules` covering too much of the model,
  not "forgot to freeze the base."
- `data_loader.ReviewDataset` tokenizes the prompt-only prefix and the full sequence separately
  to find the exact mask boundary (AC-4), then verifies `full_ids[:len(prefix_ids)] ==
  prefix_ids` before trusting it — BPE merge rules are context-sensitive right at the
  newline-before-JSON boundary this split sits on, so an unverified offset could silently mask
  the wrong span.
- `model.py::load_base_model_for_training`'s `bitsandbytes`/`BitsAndBytesConfig` import stays
  inside `if config.use_4bit`, confirmed by a test that patches `transformers.BitsAndBytesConfig`
  to raise if constructed and checks the smoke path never touches it.
- `finetune.py`'s `TrainingArguments` sets `eval_strategy="steps", eval_steps=config.max_steps`
  so validation is actually exercised once per run (AC-5's `eval_loss`), not just loaded and left
  unused — an early cut wired `eval_dataset` in without this and silently skipped evaluation
  entirely.

## Tasks

All five code tasks from `.claude/plans/08-C4-qlora-finetune.md` are committed:

1. ✅ `FinetuneConfig` smoke/full amendment + `smoke.yaml` fix + test.
2. ✅ `lora.py` (`build_lora_config`, `attach_lora`, the trainable-fraction guard) + tests,
   including a regression test against the real cached smoke model.
3. ✅ `data_loader.py` (`format_prompt`, `ReviewDataset`) + tests, including the AC-4 masking
   assertion and a prefix-drift guard.
4. ✅ `model.py::load_base_model_for_training` (smoke + full branches) + tests.
5. ✅ `finetune.py::main()` — real training loop; smoke run timed at ~2s, well under AC-1's
   120s budget; adapter saved and verified on disk.
6. ✅ This spec/STATUS update.

**Remaining before this spec can move to `done`:** AC-2's full GPU run is manual-only per CON-4
and the GPU-budget hook — it additionally needs C2's frozen holdout and C3's real `train.jsonl`/
`val.jsonl` to exist first (both still incomplete: C2 paused on `GITHUB_TOKEN`, C3's full
generation run not yet launched). The W&B run ID (AC-5) gets recorded in `specs/STATUS.md`
**after** that manual run — same deferred-completion shape as every other cross-dependent spec in
this project.
