"""Chapter 2 fine-tuning entrypoint.

SCAFFOLD (spec F2). Loads and validates the config, seeds the RNGs, and reports the
recipe it would train with — it does not train. The real QLoRA loop is spec C4, which
replaces the body of `main()` below, and which may not begin until C2 (the frozen
holdout) and C3 (the training data) are done.

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


def main() -> None:
    args = parse_args()
    config: FinetuneConfig = load_finetune_config(args.config)
    seed_everything(config.seed)

    profile = "smoke (CPU, fp32)" if config.is_smoke else "full (GPU, 4-bit NF4)"
    print(f"[ch2] config loaded: {args.config}")
    print(f"[ch2] profile={profile} seed={config.seed}")
    print(f"[ch2] base model: {config.model_tag}")
    print(
        f"[ch2] LoRA: r={config.lora_r} alpha={config.lora_alpha} "
        f"dropout={config.lora_dropout} targets={','.join(config.target_modules)}"
    )
    print(
        f"[ch2] would train {config.max_steps} steps, batch={config.batch_size}, "
        f"lr={config.learning_rate} → {config.output_dir}"
    )
    print("[ch2] SCAFFOLD: no fine-tune yet — that is spec C4. Config path verified.")


if __name__ == "__main__":
    main()
