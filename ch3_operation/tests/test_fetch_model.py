"""A13: the served GGUF is fetched at a pinned commit SHA and sha256-verified, never `main`."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from ch3_operation.config import load_export_config, load_serve_config
from ch3_operation.fetch_model import ModelPin, fetch, resolve_pin

from ch3_operation import fetch_model

CONFIGS = Path("ch3_operation/configs")
SHA = "a" * 40


def test_full_config_is_pinned_to_a_commit_sha() -> None:
    config = load_serve_config(CONFIGS / "full.yaml")
    pin = resolve_pin(config, env={})

    assert pin.url == (
        "https://huggingface.co/rkg209/qwen2.5-coder-1.5b-java-review-gguf/resolve/"
        "a2133f6a226bc081a7ac29036d2d7fb681e6bfd6/model-Q4_K_M.gguf"
    )
    assert pin.sha256 is not None


def test_export_full_config_pins_the_adapter() -> None:
    config = load_export_config(CONFIGS / "export_full.yaml")
    assert config.adapter_revision == "f8b1cf764d630fcad9a197750624e66b0266fd50"


@pytest.mark.parametrize("revision", ["main", "a2133f6", "A" * 40])
def test_a_branch_or_short_sha_is_not_a_pin(revision: str) -> None:
    config = load_serve_config(CONFIGS / "full.yaml")
    with pytest.raises(ValueError, match="40-hex"):
        replace(config, model_revision=revision)


def test_a_partial_pin_is_rejected() -> None:
    config = load_serve_config(CONFIGS / "full.yaml")
    with pytest.raises(ValueError, match="together"):
        replace(config, model_sha256=None)


def test_env_override_must_name_a_revision() -> None:
    config = load_serve_config(CONFIGS / "full.yaml")
    with pytest.raises(KeyError, match="MODEL_REVISION"):
        resolve_pin(config, env={"MODEL_REPO": "someone/other", "MODEL_FILE": "m.gguf"})


def test_env_override_replaces_the_whole_pin() -> None:
    config = load_serve_config(CONFIGS / "full.yaml")
    env = {"MODEL_REPO": "someone/other", "MODEL_FILE": "m.gguf", "MODEL_REVISION": SHA}

    assert resolve_pin(config, env) == ModelPin("someone/other", "m.gguf", SHA, None)


def test_smoke_config_has_nothing_to_fetch() -> None:
    config = load_serve_config(CONFIGS / "smoke.yaml")
    with pytest.raises(KeyError, match="no pinned model"):
        resolve_pin(config, env={})


def _fake_download(source: Path) -> Callable[[str, Path], None]:
    def retrieve(url: str, dest: Path) -> None:
        shutil.copyfile(source, dest)

    return retrieve


def test_fetch_keeps_a_file_whose_sha256_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "hub.gguf"
    source.write_bytes(b"gguf bytes")
    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", _fake_download(source))
    pin = ModelPin("r/r", "m.gguf", SHA, hashlib.sha256(b"gguf bytes").hexdigest())

    fetch(pin, tmp_path / "models" / "model.gguf")

    assert (tmp_path / "models" / "model.gguf").read_bytes() == b"gguf bytes"


def test_fetch_deletes_a_file_whose_sha256_does_not_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "hub.gguf"
    source.write_bytes(b"tampered")
    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", _fake_download(source))
    dest = tmp_path / "model.gguf"

    with pytest.raises(ValueError, match="sha256 mismatch"):
        fetch(ModelPin("r/r", "m.gguf", SHA, "0" * 64), dest)
    assert not dest.exists()
