"""The shared loader must make a missing or misspelled config key loud.

A Python-side default is a hyperparameter that does not appear in the config file, and a
typo'd key is a hyperparameter that was silently ignored. Both produce a run nobody can
reproduce from the config that supposedly describes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from eval.config import load_config, load_yaml, seed_everything


@dataclass(frozen=True)
class ToyConfig:
    name: str
    steps: int
    lr: float


def write_yaml(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


def test_loads_a_complete_config(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "c.yaml", "name: run\nsteps: 5\nlr: 0.001\n")
    config = load_config(path, ToyConfig)

    assert config == ToyConfig(name="run", steps=5, lr=0.001)


def test_config_is_frozen(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "c.yaml", "name: run\nsteps: 5\nlr: 0.001\n")
    config = load_config(path, ToyConfig)

    with pytest.raises(AttributeError):
        config.steps = 10  # type: ignore[misc]


def test_missing_key_raises_naming_the_key(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "c.yaml", "name: run\nsteps: 5\n")

    with pytest.raises(KeyError, match="lr"):
        load_config(path, ToyConfig)


def test_unknown_key_raises(tmp_path: Path) -> None:
    """A typo'd key that is silently dropped is a setting the user thinks they set."""
    path = write_yaml(tmp_path / "c.yaml", "name: run\nsteps: 5\nlr: 0.001\nlearing_rate: 0.1\n")

    with pytest.raises(KeyError, match="learing_rate"):
        load_config(path, ToyConfig)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml", ToyConfig)


def test_non_mapping_yaml_raises(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "c.yaml", "- a\n- b\n")

    with pytest.raises(TypeError):
        load_yaml(path)


def test_non_dataclass_target_raises(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "c.yaml", "name: run\n")

    with pytest.raises(TypeError):
        load_config(path, dict)  # type: ignore[type-var]


def test_seed_everything_is_deterministic() -> None:
    import random

    seed_everything(42)
    first = [random.random() for _ in range(3)]
    seed_everything(42)
    second = [random.random() for _ in range(3)]

    assert first == second
