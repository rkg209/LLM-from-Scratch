"""The shared config loader (FR-4).

Every package's entrypoint is config-driven, and every config value comes from a YAML
file — never from a Python-side default. A default in code is a hyperparameter nobody
can see in the config, cannot reproduce from it, and will misreport in a paper or a
README. So a missing key is a `KeyError`, loudly, at load time.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Read a YAML file into a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open() as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"config must be a YAML mapping, got {type(raw).__name__}: {path}")
    return raw


def load_config[T](path: Path | str, cls: type[T]) -> T:
    """Load `path` into a frozen dataclass of type `cls`.

    Raises KeyError if the YAML is missing any field of `cls`, or names a field `cls`
    does not have — a typo'd key is silently ignored otherwise, which is how a run ends
    up using a learning rate nobody set.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass")

    raw = load_yaml(path)
    expected = {field.name for field in dataclasses.fields(cls)}
    provided = set(raw)

    if missing := sorted(expected - provided):
        raise KeyError(f"{path}: config is missing required key(s): {', '.join(missing)}")
    if unknown := sorted(provided - expected):
        raise KeyError(
            f"{path}: config has unknown key(s) for {cls.__name__}: {', '.join(unknown)}"
        )

    return cls(**raw)


def seed_everything(seed: int) -> None:
    """Seed every RNG in play, so a config plus a seed fully determines a run (NFR-4).

    Torch is seeded only if it is installed — the base environment does not have it, and
    the shared harness must import without it.
    """
    import random

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
