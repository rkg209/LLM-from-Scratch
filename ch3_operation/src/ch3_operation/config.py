"""Chapter 3 config. Fields are fixed in planning/03-system-design.md §1.3."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from eval.config import load_config as _load_config

_ENV_VAR = re.compile(r"^\$\{(\w+)\}$")

# CPU-only is a hard project constraint (CON-12), not a tunable default. The "edge" claim
# is that this runs on a free CPU tier; a GPU layer count is not offered.
N_GPU_LAYERS = 0


@dataclass(frozen=True)
class ServeConfig:
    model_path: str
    n_ctx: int
    n_threads: int
    max_tokens: int
    temperature: float
    host: str
    port: int
    metrics_window: int
    max_request_bytes: int
    api_version: str

    def __post_init__(self) -> None:
        if self.metrics_window < 1:
            raise ValueError(f"metrics_window must be positive, got {self.metrics_window}")
        if self.n_threads < 1:
            raise ValueError(f"n_threads must be positive, got {self.n_threads}")
        if self.max_request_bytes < 1:
            raise ValueError(f"max_request_bytes must be positive, got {self.max_request_bytes}")

    def resolved_model_path(self) -> Path:
        """Resolve a `${VAR}` model_path against the environment.

        The full config points at MODEL_PATH so the container can pull the GGUF from HF Hub
        at startup rather than baking a multi-GB layer into the image.
        """
        if match := _ENV_VAR.match(self.model_path):
            name = match.group(1)
            value = os.environ.get(name)
            if not value:
                raise KeyError(
                    f"model_path is ${{{name}}} but {name} is not set in the environment"
                )
            return Path(value)
        return Path(self.model_path)


def load_serve_config(path: Path | str) -> ServeConfig:
    return _load_config(path, ServeConfig)


# Duplicated from ch2_adaptation.config rather than imported: chapters communicate only
# through files (planning/03-system-design.md §1.3), and ch3_operation must stay importable
# with only the `ch3` extra, never `ch2`. scripts/merge_adapter.py -- which is allowed to
# import both packages (D-4) -- is what actually enforces these tags line up with the real
# adapter's base model at merge time.
_CH2_LOCKED_MODEL_TAG = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
_CH2_SMOKE_MODEL_TAG = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

# The published, committed artifacts a smoke export/eval run must never overwrite.
_PUBLISHED_MERGED_DIR = "outputs/merged"
_PUBLISHED_GGUF_F16_PATH = "outputs/model-f16.gguf"
_PUBLISHED_GGUF_QUANT_PATH = "outputs/model.gguf"
_PUBLISHED_GGUF_RESULTS_PATH = "eval/results/gguf.json"

_QUANT_LADDER = ("Q4_K_M", "Q5_K_M", "Q8_0")


@dataclass(frozen=True)
class ExportConfig:
    """Drives `scripts/merge_adapter.py`, `export_gguf.py`, `quantize_gguf.py` (spec O1)."""

    base_model_tag: str
    adapter_path: str
    merged_dir: str
    gguf_f16_path: str
    gguf_quant_path: str
    quant_type: str
    llama_cpp_dir: str
    sanity_sample_path: str
    sanity_n: int

    def __post_init__(self) -> None:
        if self.quant_type not in _QUANT_LADDER:
            raise ValueError(
                f"quant_type must be one of {_QUANT_LADDER} (AC-4's fallback ladder), got "
                f"{self.quant_type!r}"
            )
        if self.sanity_n < 1:
            raise ValueError(f"sanity_n must be positive, got {self.sanity_n}")
        if self.base_model_tag not in (_CH2_LOCKED_MODEL_TAG, _CH2_SMOKE_MODEL_TAG):
            raise ValueError(
                f"base_model_tag must be the locked ch2 model {_CH2_LOCKED_MODEL_TAG!r} or "
                f"its smoke stand-in {_CH2_SMOKE_MODEL_TAG!r}, got {self.base_model_tag!r}"
            )
        if self.is_smoke:
            for label, value, published in (
                ("merged_dir", self.merged_dir, _PUBLISHED_MERGED_DIR),
                ("gguf_f16_path", self.gguf_f16_path, _PUBLISHED_GGUF_F16_PATH),
                ("gguf_quant_path", self.gguf_quant_path, _PUBLISHED_GGUF_QUANT_PATH),
            ):
                if Path(value).resolve() == Path(published).resolve():
                    raise ValueError(
                        f"a smoke export run must never write {published!r} -- that is the "
                        f"published path. Point {label} at a gitignored outputs/smoke/ path "
                        "instead."
                    )

    @property
    def is_smoke(self) -> bool:
        return self.base_model_tag == _CH2_SMOKE_MODEL_TAG


def load_export_config(path: Path | str) -> ExportConfig:
    return _load_config(path, ExportConfig)


@dataclass(frozen=True)
class EvalConfig:
    """Drives `ch3_operation.evaluate` (spec O1): score a served GGUF on the holdout."""

    model_path: str
    n_ctx: int
    n_threads: int
    max_tokens: int
    temperature: float
    holdout_path: str
    results_path: str
    compare_to: str
    n_samples: int
    seed: int

    def __post_init__(self) -> None:
        if self.n_ctx < 1:
            raise ValueError(f"n_ctx must be positive, got {self.n_ctx}")
        if self.n_threads < 1:
            raise ValueError(f"n_threads must be positive, got {self.n_threads}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.n_samples < 1:
            raise ValueError(f"n_samples must be positive, got {self.n_samples}")
        if (
            self.is_smoke
            and Path(self.results_path).resolve() == Path(_PUBLISHED_GGUF_RESULTS_PATH).resolve()
        ):
            raise ValueError(
                f"a smoke eval run must never write {_PUBLISHED_GGUF_RESULTS_PATH!r} -- that "
                "is the published, committed result. Point results_path at a gitignored path "
                "under outputs/ instead."
            )

    @property
    def is_smoke(self) -> bool:
        return self.model_path.endswith("tests/fixtures/tiny.gguf")

    def resolved_model_path(self) -> Path:
        """Resolve a `${VAR}` model_path against the environment, same convention as
        `ServeConfig.resolved_model_path` -- the full profile points at `MODEL_PATH` so a
        manual evaluate run reuses whatever `.gguf` `export_full.yaml` just produced."""
        if match := _ENV_VAR.match(self.model_path):
            name = match.group(1)
            value = os.environ.get(name)
            if not value:
                raise KeyError(
                    f"model_path is ${{{name}}} but {name} is not set in the environment"
                )
            return Path(value)
        return Path(self.model_path)


def load_eval_config(path: Path | str) -> EvalConfig:
    return _load_config(path, EvalConfig)
