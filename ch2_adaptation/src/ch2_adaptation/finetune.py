"""Chapter 2 fine-tuning entrypoint (spec C4): QLoRA over `Qwen/Qwen2.5-Coder-1.5B-Instruct`.

Wires `data_loader.ReviewDataset`, `model.load_base_model_for_training`, and
`lora.attach_lora` into a real training loop. Never reads `eval/holdout/` (CON-6) --
validation comes only from `config.val_path`; the leakage guard would block a holdout
read anyway if this ever tried one.

The full config is never run from a session. The GPU-budget hook enforces that, and it
is right to: this is the run that costs real free-tier GPU hours.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ch2_adaptation.config import FinetuneConfig, load_finetune_config
from eval.config import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune the Chapter 2 base model.")
    parser.add_argument("--config", type=Path, required=True, help="path to a YAML config")
    return parser.parse_args()


def _build_trainer(config: FinetuneConfig) -> tuple[object, object]:
    """Load model/tokenizer, attach LoRA, build datasets and the `Trainer`.

    Uses plain `transformers.Trainer`, not `trl.SFTTrainer`: the installed trl version
    (1.8) requires `train_dataset` to be a real `datasets.Dataset` (it accesses
    `.column_names` internally), which a hand-tokenized, pre-masked map-style dataset
    like `ReviewDataset` is not and does not need to be -- `ReviewDataset` already does
    its own tokenization and `-100` label masking, so `Trainer` plus
    `DataCollatorForSeq2Seq` (which only pads, using its own `label_pad_token_id=-100`
    default) is the correct, simpler fit. Verified directly: `trl.SFTTrainer` raises
    `AttributeError: 'ReviewDataset' object has no attribute 'column_names'`, while
    `transformers.Trainer` trains successfully against the same objects.
    """
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    from ch2_adaptation.data_loader import ReviewDataset
    from ch2_adaptation.lora import attach_lora
    from ch2_adaptation.model import load_base_model_for_training

    model, tokenizer = load_base_model_for_training(config)
    peft_model = attach_lora(model, config)

    train_dataset = ReviewDataset(config.train_path, tokenizer)
    val_dataset = ReviewDataset(config.val_path, tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer, model=peft_model, label_pad_token_id=-100)

    args = TrainingArguments(
        output_dir=config.output_dir,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        seed=config.seed,
        report_to=["wandb"] if config.wandb_mode != "disabled" else [],
        logging_strategy="steps",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=config.max_steps,
        save_strategy="no",
    )
    trainer = Trainer(
        model=peft_model,
        args=args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    return trainer, peft_model


def main() -> None:
    args = parse_args()
    config: FinetuneConfig = load_finetune_config(args.config)
    seed_everything(config.seed)

    profile = "smoke (CPU, fp32)" if config.is_smoke else "full (GPU, 4-bit NF4)"
    print(f"[C4] config loaded: {args.config}")
    print(f"[C4] profile={profile} seed={config.seed}")
    print(f"[C4] base model: {config.model_tag}")
    print(
        f"[C4] LoRA: r={config.lora_r} alpha={config.lora_alpha} "
        f"dropout={config.lora_dropout} targets={','.join(config.target_modules)}"
    )

    if config.wandb_mode != "disabled":
        import wandb

        wandb.init(project=config.wandb_project, mode=config.wandb_mode, config=config.__dict__)

    trainer, peft_model = _build_trainer(config)
    trainer.train()

    peft_model.save_pretrained(config.output_dir)
    print(f"[C4] trained {config.max_steps} steps, adapter saved to {config.output_dir}")

    if config.wandb_mode != "disabled":
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
