#!/usr/bin/env python3
"""PreToolUse(Bash): refuse to launch a full training run from a session.

The project's hardest rule (CLAUDE.md #1): full fine-tunes belong on the GPU box,
launched by hand. A full run started inside a session burns the free-GPU budget and
hangs the session for hours. Escape hatch: export ALLOW_FULL_RUN=1.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hooklib import allow, deny, run  # noqa: E402

# Each pattern is a way a full run sneaks in. Order is irrelevant; any match blocks.
FULL_RUN_PATTERNS: list[tuple[str, str]] = [
    (r"--config[= ]\S*full", "targets a full.yaml config"),
    (r"\bconfigs/full\.ya?ml\b", "targets a full.yaml config"),
    (r"\baccelerate\s+launch\b", "uses `accelerate launch`"),
    (r"\btorchrun\b", "uses `torchrun`"),
    (r"\bdeepspeed\b", "uses `deepspeed`"),
    (r"--max[_-]steps[= ]\s*([1-9]\d{2,})", "sets max_steps in the hundreds or more"),
    (r"--num[_-]train[_-]epochs[= ]\s*([1-9]\d*)", "sets num_train_epochs"),
    (r"\bload_in_4bit\b|\bbnb_4bit\b", "loads a 4-bit quantized base model (GPU path)"),
]

# A smoke config is always fine, even if another pattern trips.
SMOKE_OVERRIDE = re.compile(r"--config[= ]\S*smoke|configs/smoke\.ya?ml")


def check(payload: dict[str, Any]) -> None:
    command = payload.get("tool_input", {}).get("command", "")
    if not command:
        allow()

    if os.environ.get("ALLOW_FULL_RUN") == "1":
        allow()

    if SMOKE_OVERRIDE.search(command):
        allow()

    for pattern, why in FULL_RUN_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            deny(
                "PreToolUse",
                f"GPU-budget guard: this command {why}, which looks like a full training run.\n"
                f"Full runs are launched manually on the GPU box, never from a session "
                f"(CLAUDE.md non-negotiable #1).\n"
                f"Run the smoke config instead (`--config .../configs/smoke.yaml`). "
                f"If you genuinely mean to launch a full run here, the user must "
                f"export ALLOW_FULL_RUN=1 first.",
            )

    allow()


if __name__ == "__main__":
    run(check)
