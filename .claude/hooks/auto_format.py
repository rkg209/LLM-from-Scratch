#!/usr/bin/env python3
"""PostToolUse(Write|Edit): keep every touched Python file lint-clean and formatted.

`ruff check --fix` then `black`, on the one file that changed. NFR-8 requires both to
pass before any commit; doing it here means the commit is never the place you find out.

Set HOOK_RUN_TESTS=1 to also run the touched package's unit tests on every edit. It is
off by default: a test run per keystroke-sized edit makes the loop crawl, and `/smoke`
plus CI already cover it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hooklib import REPO_ROOT, add_context, allow, run  # noqa: E402

PACKAGES = ("ch1_architecture", "ch2_adaptation", "ch3_operation", "eval")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120, check=False
    )


def _owning_package(path: Path) -> str | None:
    try:
        top = path.relative_to(REPO_ROOT).parts[0]
    except ValueError:
        return None
    return top if top in PACKAGES else None


def check(payload: dict[str, Any]) -> None:
    raw_path = payload.get("tool_input", {}).get("file_path", "")
    if not raw_path.endswith(".py"):
        allow()

    path = Path(raw_path)
    if not path.exists():
        allow()

    _run(["uv", "run", "ruff", "check", "--fix", "--quiet", str(path)])
    _run(["uv", "run", "black", "--quiet", str(path)])

    # Report only what a human still has to fix — anything ruff could not auto-fix.
    remaining = _run(["uv", "run", "ruff", "check", "--quiet", str(path)])
    messages: list[str] = []
    if remaining.returncode != 0 and remaining.stdout.strip():
        messages.append(f"ruff still reports issues in {path.name}:\n{remaining.stdout.strip()}")

    if os.environ.get("HOOK_RUN_TESTS") == "1":
        package = _owning_package(path.resolve())
        if package:
            tests = _run(["uv", "run", "pytest", "-q", "-x", f"{package}/tests"])
            if tests.returncode != 0:
                messages.append(f"{package} tests are failing:\n{tests.stdout.strip()[-2000:]}")

    if messages:
        add_context("PostToolUse", "\n\n".join(messages))
    allow()


if __name__ == "__main__":
    run(check)
