"""Chapter 2 config. Fields are fixed in planning/03-system-design.md §1.2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eval.config import load_config as _load_config

# Locked in spec C1 (CON-11). Swapping the base model means re-running every baseline and
# every published comparison — it is a spec change, not a config tweak.
LOCKED_MODEL_TAG = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


@dataclass(frozen=True)
class FinetuneConfig:
    model_tag: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    max_steps: int
    batch_size: int
    learning_rate: float
    use_4bit: bool
    seed: int
    output_dir: str
    wandb_project: str

    def __post_init__(self) -> None:
        if self.model_tag != LOCKED_MODEL_TAG:
            raise ValueError(
                f"base model is locked to {LOCKED_MODEL_TAG!r} (CON-11), got {self.model_tag!r}. "
                "Changing it invalidates the measured baselines — update spec C1 first."
            )
        if not self.target_modules:
            raise ValueError("target_modules must name at least one module for LoRA to attach to")

    @property
    def is_smoke(self) -> bool:
        """The CPU path: fp32, no bitsandbytes (which needs CUDA)."""
        return not self.use_4bit


def load_finetune_config(path: Path | str) -> FinetuneConfig:
    return _load_config(path, FinetuneConfig)
