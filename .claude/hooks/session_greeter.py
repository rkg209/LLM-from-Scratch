#!/usr/bin/env python3
"""SessionStart: open every session with the rules and the next piece of work.

Reads specs/STATUS.md and reports the first spec that is not done, so a session never
starts by guessing where the project left off.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hooklib import REPO_ROOT, add_context, allow, run  # noqa: E402

STATUS_PATH = REPO_ROOT / "specs" / "STATUS.md"

# A backlog row: | 05 | C1 | task-schema-and-base-model | draft | ... |
# The leading build-order column is required, which is what distinguishes a backlog row from the
# other tables in STATUS.md (the track legend and the progress summary).
ROW = re.compile(
    r"^\|\s*\d+\s*\|\s*([A-Z]\d+)\s*\|\s*([^|]+?)\s*\|\s*(draft|planned|building|done)\s*\|"
)


def _next_spec() -> tuple[str, str, str] | None:
    if not STATUS_PATH.exists():
        return None
    building, first_open = None, None
    for line in STATUS_PATH.read_text().splitlines():
        match = ROW.match(line.strip())
        if not match:
            continue
        spec_id, name, state = match.group(1), match.group(2), match.group(3)
        if state == "building" and building is None:
            building = (spec_id, name, state)
        if state != "done" and first_open is None:
            first_open = (spec_id, name, state)
    return building or first_open


def check(_payload: dict[str, Any]) -> None:
    lines = [
        "LLM Engineering: From Architecture to Edge — session rules",
        "  - Smoke configs only. Never launch a full training run from this session.",
        "  - ch1_architecture/src is pure PyTorch. eval/holdout/ is off-limits.",
        "  - No implementation without an accepted spec and plan"
        " (/specify → /plan → /tasks → /implement).",
    ]

    nxt = _next_spec()
    if nxt is None:
        lines.append("  - specs/STATUS.md not found or empty — start with /specify.")
    else:
        spec_id, name, state = nxt
        verb = "in progress" if state == "building" else "next up"
        lines.append(f"  - Backlog {verb}: {spec_id} ({name}) — currently `{state}`.")

    add_context("SessionStart", "\n".join(lines))
    allow()


if __name__ == "__main__":
    run(check)
