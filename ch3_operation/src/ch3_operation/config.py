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

    def __post_init__(self) -> None:
        if self.metrics_window < 1:
            raise ValueError(f"metrics_window must be positive, got {self.metrics_window}")
        if self.n_threads < 1:
            raise ValueError(f"n_threads must be positive, got {self.n_threads}")

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
