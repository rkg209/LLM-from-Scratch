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
SMOKE_MODEL_TAG = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

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
    train_path: str
    val_path: str
    output_dir: str
    wandb_mode: str
    wandb_project: str

    def __post_init__(self) -> None:
        if self.is_smoke:
            if self.model_tag != SMOKE_MODEL_TAG:
                raise ValueError(
                    f"smoke fine-tune runs are only allowed to use {SMOKE_MODEL_TAG!r}, got "
                    f"{self.model_tag!r}. A first-time load of the real 1.5B model is a ~3GB "
                    "download that cannot reliably finish inside the smoke time budget."
                )
        elif self.model_tag != LOCKED_MODEL_TAG:
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


# ch2_adaptation/data/provenance.json is the committed, published generation record (spec C3,
# planning/04 §3.3). A smoke run must never overwrite it with output from the stub client —
# same shape as _PUBLISHED_BASELINES_PATH above.
_PUBLISHED_PROVENANCE_PATH = "ch2_adaptation/data/provenance.json"


@dataclass(frozen=True)
class DataGenConfig:
    frontier_provider: str
    frontier_model_tag: str
    temperature: float
    max_new_tokens: int
    seed_pool_path: str
    train_path: str
    val_path: str
    provenance_path: str
    val_frac: float
    n_snippets: int
    variants_per_snippet: int
    max_budget_usd: float
    seed: int
    wandb_mode: str
    wandb_project: str
    raw_responses_path: str
    checkpoint_every: int

    def __post_init__(self) -> None:
        if self.checkpoint_every <= 0:
            raise ValueError(f"checkpoint_every must be positive, got {self.checkpoint_every!r}")
        if not 0.0 < self.val_frac < 1.0:
            raise ValueError(f"val_frac must be in (0, 1), got {self.val_frac!r}")
        if self.n_snippets <= 0:
            raise ValueError(f"n_snippets must be positive, got {self.n_snippets!r}")
        if self.variants_per_snippet <= 0:
            raise ValueError(
                f"variants_per_snippet must be positive, got {self.variants_per_snippet!r}"
            )
        if self.max_budget_usd <= 0:
            raise ValueError(f"max_budget_usd must be positive, got {self.max_budget_usd!r}")
        if (
            self.is_smoke
            and Path(self.provenance_path).resolve() == Path(_PUBLISHED_PROVENANCE_PATH).resolve()
        ):
            raise ValueError(
                f"a smoke run must never write {_PUBLISHED_PROVENANCE_PATH!r} — that is the "
                "committed, published provenance record. Point provenance_path at a gitignored "
                "path under outputs/ instead."
            )

    @property
    def is_smoke(self) -> bool:
        """The stub frontier client makes no network call and needs no API key."""
        return self.frontier_provider == "stub"


def load_data_gen_config(path: Path | str) -> DataGenConfig:
    return _load_config(path, DataGenConfig)


# eval/results/finetuned.json is the committed, published fine-tuned-model score (spec
# C5). A smoke run must never overwrite it with output from SMOKE_MODEL_TAG -- same shape
# as _PUBLISHED_BASELINES_PATH/_PUBLISHED_PROVENANCE_PATH above.
_PUBLISHED_FINETUNED_RESULTS_PATH = "eval/results/finetuned.json"


@dataclass(frozen=True)
class EvaluateConfig:
    model_tag: str
    adapter_path: str
    holdout_path: str
    results_path: str
    max_new_tokens: int
    use_4bit: bool
    seed: int
    wandb_mode: str
    wandb_project: str

    def __post_init__(self) -> None:
        if self.is_smoke:
            if self.model_tag != SMOKE_MODEL_TAG:
                raise ValueError(
                    f"smoke eval runs are only allowed to use {SMOKE_MODEL_TAG!r}, got "
                    f"{self.model_tag!r}. Smoke proves the code path, not the number."
                )
            if (
                Path(self.results_path).resolve()
                == Path(_PUBLISHED_FINETUNED_RESULTS_PATH).resolve()
            ):
                raise ValueError(
                    f"a smoke run must never write {_PUBLISHED_FINETUNED_RESULTS_PATH!r} — "
                    "that is the committed, published result. Point results_path at a "
                    "gitignored path under outputs/ instead."
                )
        elif self.model_tag != LOCKED_MODEL_TAG:
            raise ValueError(
                f"base model is locked to {LOCKED_MODEL_TAG!r} (CON-11), got "
                f"{self.model_tag!r}. Changing it invalidates every measured comparison — "
                "update spec C1 first."
            )

    @property
    def is_smoke(self) -> bool:
        """fp32 on CPU with the tiny stand-in; no 4-bit, no real adapter needed."""
        return not self.use_4bit


def load_evaluate_config(path: Path | str) -> EvaluateConfig:
    return _load_config(path, EvaluateConfig)


@dataclass(frozen=True)
class PublishTableConfig:
    """Inputs for re-rendering the published table from finished artifacts (C5 AC-5).

    Paths only: this entrypoint scores nothing, so there is no seed, no model tag and
    nothing to keep in step with a run.
    """

    finetuned_path: str
    baselines_path: str
    readme_path: str

    def __post_init__(self) -> None:
        for field in ("finetuned_path", "baselines_path", "readme_path"):
            if not getattr(self, field):
                raise ValueError(
                    f"{field} must be set — this entrypoint only moves numbers "
                    "between existing files, so every path is required."
                )


def load_publish_table_config(path: Path | str) -> PublishTableConfig:
    return _load_config(path, PublishTableConfig)
