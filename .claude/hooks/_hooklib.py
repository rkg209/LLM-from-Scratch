"""Shared helpers for the project's Claude Code hooks.

Every hook reads a JSON payload on stdin and communicates its decision back to
Claude Code on stdout. PreToolUse hooks emit a permission decision; other events
emit plain context. Exit code 0 always — a crashed hook must never wedge the
session, so callers wrap their logic in `run()`.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def read_payload() -> dict[str, Any]:
    """Parse the hook payload from stdin; an unreadable payload is an empty one."""
    try:
        return json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        return {}


def deny(event: str, reason: str) -> None:
    """Block the tool call and tell Claude why."""
    _emit(
        {
            "hookSpecificOutput": {
                "hookEventName": event,
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }
    )


def allow() -> None:
    """Say nothing and let the normal permission flow proceed."""
    sys.exit(0)


def add_context(event: str, context: str) -> None:
    """Surface text to Claude without making a permission decision."""
    _emit({"hookSpecificOutput": {"hookEventName": event, "additionalContext": context}})


def _emit(payload: dict[str, Any]) -> None:
    json.dump(payload, sys.stdout)
    sys.exit(0)


def run(hook: Callable[[dict[str, Any]], None]) -> None:
    """Run a hook body, swallowing any crash so the session keeps working."""
    try:
        hook(read_payload())
    except Exception as exc:  # noqa: BLE001 - a broken hook must not block the user
        print(f"hook error (ignored): {exc}", file=sys.stderr)
    sys.exit(0)
