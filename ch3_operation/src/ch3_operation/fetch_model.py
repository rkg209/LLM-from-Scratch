"""Fetch the served GGUF from HF Hub at a pinned revision (A13), then verify its sha256.

Run by `docker/entrypoint.sh` at container start when `MODEL_PATH` is absent. The pin comes
from the serving config, so the image serves exactly the model the published numbers were
measured on. `MODEL_REPO` in the environment points it at a different repo instead, and then
`MODEL_FILE` and `MODEL_REVISION` are required too -- an override may change the model, never
unpin it.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ch3_operation.config import ServeConfig, check_hub_pin, load_serve_config

_HUB_URL = "https://huggingface.co"
_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ModelPin:
    repo: str
    file: str
    revision: str
    sha256: str | None

    def __post_init__(self) -> None:
        check_hub_pin("model", self.repo, self.revision)

    @property
    def url(self) -> str:
        return f"{_HUB_URL}/{self.repo}/resolve/{self.revision}/{self.file}"


def resolve_pin(config: ServeConfig, env: Mapping[str, str]) -> ModelPin:
    """The environment's pin if it names a repo, else the config's; never an unpinned one."""
    if env.get("MODEL_REPO"):
        missing = [name for name in ("MODEL_FILE", "MODEL_REVISION") if not env.get(name)]
        if missing:
            raise KeyError(f"MODEL_REPO is set, so {' and '.join(missing)} must be set too")
        return ModelPin(
            repo=env["MODEL_REPO"],
            file=env["MODEL_FILE"],
            revision=env["MODEL_REVISION"],
            sha256=env.get("MODEL_SHA256") or None,
        )
    if config.model_repo is None or config.model_file is None or config.model_revision is None:
        raise KeyError("the config has no pinned model to fetch, and MODEL_REPO is not set")
    return ModelPin(
        repo=config.model_repo,
        file=config.model_file,
        revision=config.model_revision,
        sha256=config.model_sha256,
    )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(pin: ModelPin, dest: Path) -> None:
    """Download `pin` to `dest`; delete it and raise if the sha256 does not match."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch_model] downloading {pin.url} -> {dest}")
    urllib.request.urlretrieve(pin.url, dest)
    if pin.sha256 is None:
        print("[fetch_model] no sha256 given; file not verified")
        return
    actual = sha256_of(dest)
    if actual != pin.sha256:
        dest.unlink()
        raise ValueError(f"sha256 mismatch: expected {pin.sha256}, got {actual}")
    print("[fetch_model] sha256 verified")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="ServeConfig YAML holding the pin")
    args = parser.parse_args(argv)

    config = load_serve_config(args.config)
    try:
        fetch(resolve_pin(config, os.environ), config.resolved_model_path())
    except (KeyError, ValueError) as error:
        print(f"[fetch_model] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
