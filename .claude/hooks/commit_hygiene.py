#!/usr/bin/env python3
"""PreToolUse(Bash): stop a `git commit` that would put the wrong thing in git history.

Weights, datasets, GGUF files and .env are permanently expensive to remove once pushed
(CLAUDE.md #6), and a modified holdout must never be committed at all (#3). This checks
what is actually staged, not what the command line says.

It also blocks a `Co-Authored-By:` trailer in the message (#7) — that one breaks pushing
to GitHub for this repo, and by the time the push fails the commit already exists.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hooklib import REPO_ROOT, allow, deny, run  # noqa: E402

MAX_BYTES = 5 * 1024 * 1024

FORBIDDEN_SUFFIXES = (".pt", ".pth", ".bin", ".safetensors", ".gguf")
GIT_COMMIT = re.compile(r"\bgit\b[^|;&]*\bcommit\b")
CO_AUTHORED_BY = re.compile(r"co-authored-by\s*:", re.IGNORECASE)


def _staged_files() -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _reject(path: str) -> str | None:
    """Return why this staged path must not be committed, or None if it is fine."""
    normalized = path.replace("\\", "/")

    if normalized.startswith("eval/holdout/"):
        return "modifies the frozen holdout set"
    if normalized.endswith(".env") or "/.env" in normalized:
        return "is an .env file (secrets never go in git)"
    if normalized.endswith(FORBIDDEN_SUFFIXES) and "tests/fixtures/" not in normalized:
        return "is a model-weight or GGUF artifact (these live on HF Hub, not in git)"
    if normalized.startswith("ch2_adaptation/data/") and normalized.endswith(".jsonl"):
        return "is generated training data (gitignored by design; only provenance is committed)"

    full = REPO_ROOT / normalized
    if full.is_file() and full.stat().st_size > MAX_BYTES:
        size_mb = full.stat().st_size / 1024 / 1024
        return f"is {size_mb:.1f} MB, over the {MAX_BYTES // 1024 // 1024} MB limit"
    return None


def check(payload: dict[str, Any]) -> None:
    command = payload.get("tool_input", {}).get("command", "")
    if not GIT_COMMIT.search(command):
        allow()

    if CO_AUTHORED_BY.search(command):
        deny(
            "PreToolUse",
            "Commit hygiene: this commit message carries a `Co-Authored-By:` trailer, which "
            "breaks pushing this repo to GitHub (CLAUDE.md non-negotiable #7).\n"
            "Re-run the commit with the trailer removed. The message ends at the last line "
            "of the body — no co-author trailer of any kind, for Claude or anyone else.",
        )

    violations = [
        f"  - {path} — {reason}"
        for path in _staged_files()
        if (reason := _reject(path)) is not None
    ]
    if violations:
        deny(
            "PreToolUse",
            "Commit hygiene: these staged files must not be committed:\n"
            + "\n".join(violations)
            + "\n\nUnstage them (`git restore --staged <path>`) and confirm .gitignore "
            "covers them. Weights and datasets belong on HF Hub; the repo tracks the "
            "code and the provenance, not the artifacts.",
        )

    allow()


if __name__ == "__main__":
    run(check)
