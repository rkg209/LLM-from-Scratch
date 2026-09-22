#!/usr/bin/env python3
"""PreToolUse(Read|Write|Edit|Bash): keep eval/holdout/ sacred.

The headline metric is only meaningful if the holdout set never touched the
training path (CLAUDE.md #3). Writes are denied unconditionally — the set is frozen.
Reads are denied unless the caller declares an eval context (EVAL_CONTEXT=1), which
`/eval` and the eval harness set for themselves.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hooklib import allow, deny, run  # noqa: E402

HOLDOUT = "eval/holdout"
WRITE_TOOLS = {"Write", "Edit", "NotebookEdit"}
READ_TOOLS = {"Read", "Glob", "Grep"}

# Bash commands that would mutate the holdout rather than just read it.
BASH_WRITE = re.compile(
    r"(rm|mv|cp|touch|tee|truncate|chmod|sed\s+-i)\b[^|;&]*eval/holdout"
    r"|>\s*\S*eval/holdout"
    r"|eval/holdout\S*\s*<",
    re.IGNORECASE,
)

WRITE_DENIAL = (
    "Leakage guard: eval/holdout/ is FROZEN. Nothing may write to it — not training "
    "code, not data generation, not a fix-up script. It was curated once and committed; "
    "changing it silently invalidates every number in the eval table "
    "(CLAUDE.md non-negotiable #3)."
)

READ_DENIAL = (
    "Leakage guard: eval/holdout/ may not be read from a training or data-generation "
    "context. If a model, a prompt, or a cleaning step ever sees the holdout, the "
    "headline metric is no longer an honest measurement (CLAUDE.md non-negotiable #3).\n"
    "Legitimate eval work runs with EVAL_CONTEXT=1 — that is what `/eval` and the "
    "harness set."
)


def _touches_holdout(text: str) -> bool:
    return HOLDOUT in text.replace("\\", "/")


def check(payload: dict[str, Any]) -> None:
    tool = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {})
    in_eval_context = os.environ.get("EVAL_CONTEXT") == "1"

    if tool in WRITE_TOOLS:
        path = tool_input.get("file_path", "") or tool_input.get("notebook_path", "")
        if _touches_holdout(path):
            deny("PreToolUse", WRITE_DENIAL)

    elif tool in READ_TOOLS:
        target = " ".join(str(tool_input.get(key, "")) for key in ("file_path", "path", "pattern"))
        if _touches_holdout(target) and not in_eval_context:
            deny("PreToolUse", READ_DENIAL)

    elif tool == "Bash":
        command = tool_input.get("command", "")
        if BASH_WRITE.search(command):
            deny("PreToolUse", WRITE_DENIAL)
        if _touches_holdout(command) and not in_eval_context:
            deny("PreToolUse", READ_DENIAL)

    allow()


if __name__ == "__main__":
    run(check)
