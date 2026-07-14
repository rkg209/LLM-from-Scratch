"""Chapter 2 config. Fields are fixed in planning/03-system-design.md §1.2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eval.config import load_config as _load_config

# Locked in spec C1 (CON-11). Swapping the base model means re-running every baseline and
# every published comparison — it is a spec change, not a config tweak.
LOCKED_MODEL_TAG = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# The one allowlisted stand-in for smoke baseline runs. fp32 Qwen-1.5B is ~6GB resident and
# 40-120s of generation alone on CPU — it cannot honour the under-120s smoke budget (NFR-1).
# This tiny model proves the code path (load -> prompt -> generate -> parse -> score ->
# atomic write); the full config produces the real, published number.
SMOKE_MODEL_TAG = "hf-internal-testing/tiny-random-Qwen2ForCausalLM"

# eval/results/baselines.json is the committed, published number. A smoke run must never
# overwrite it with output from SMOKE_MODEL_TAG (spec C1, Risk 5).
_PUBLISHED_BASELINES_PATH = "eval/results/baselines.json"


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


@dataclass(frozen=True)
class BaselineConfig:
    base_model_tag: str
    frontier_provider: str
    frontier_model_tag: str
    stub_set_path: str
    results_path: str
    max_new_tokens: int
    temperature: float
    n_few_shot: int
    use_4bit: bool
    seed: int
    wandb_mode: str
    wandb_project: str

    def __post_init__(self) -> None:
        if self.is_smoke:
            if self.base_model_tag != SMOKE_MODEL_TAG:
                raise ValueError(
                    f"smoke baseline runs are only allowed to use {SMOKE_MODEL_TAG!r}, got "
                    f"{self.base_model_tag!r}. Smoke proves the code path, not the number."
                )
            if Path(self.results_path).resolve() == Path(_PUBLISHED_BASELINES_PATH).resolve():
                raise ValueError(
                    f"a smoke run must never write {_PUBLISHED_BASELINES_PATH!r} — that is the "
                    "committed, published baseline. Point results_path at a gitignored path "
                    "under outputs/ instead."
                )
        elif self.base_model_tag != LOCKED_MODEL_TAG:
            raise ValueError(
                f"base model is locked to {LOCKED_MODEL_TAG!r} (CON-11), got "
                f"{self.base_model_tag!r}. Changing it invalidates every measured baseline — "
                "update spec C1 first."
            )

    @property
    def is_smoke(self) -> bool:
        """The stub frontier client makes no network call and needs no API key."""
        return self.frontier_provider == "stub"


def load_baseline_config(path: Path | str) -> BaselineConfig:
    return _load_config(path, BaselineConfig)
