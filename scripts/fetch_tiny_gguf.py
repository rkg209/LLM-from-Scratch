"""Fetch the CI test fixture: `stories260K.gguf` from `ggml-org/models-moved` (O0).

Idempotent and sha256-verified: re-running when the fixture is already present and correct
is a no-op. This is the one sanctioned exception to CLAUDE.md non-negotiable #6 (no weights
in git) — `.gitignore` and `.claude/hooks/commit_hygiene.py` both carve out
`ch3_operation/tests/fixtures/*.gguf` under 5 MB for exactly this file.

Run once, by hand, after cloning:
    uv run python scripts/fetch_tiny_gguf.py
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/ggml-org/models-moved/resolve/main/tinyllamas/stories260K.gguf"
SHA256 = "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d"
DEST = Path("ch3_operation/tests/fixtures/tiny.gguf")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(dest: Path = DEST) -> Path:
    """Download `dest` if missing or corrupt; verify sha256 either way."""
    if dest.exists() and _sha256(dest) == SHA256:
        print(f"[fetch_tiny_gguf] {dest} already present and verified, skipping download")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch_tiny_gguf] downloading {URL}")
    urllib.request.urlretrieve(URL, dest)

    actual = _sha256(dest)
    if actual != SHA256:
        dest.unlink()
        raise ValueError(f"sha256 mismatch: expected {SHA256}, got {actual}")

    print(f"[fetch_tiny_gguf] verified sha256, wrote {dest} ({dest.stat().st_size} bytes)")
    return dest


def main() -> None:
    fetch()


if __name__ == "__main__":
    main()
