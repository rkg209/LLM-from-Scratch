"""Ch3's own zero-shot review prompt.

Mirrors `ch2_adaptation.prompts.format_zero_shot` in wording, but chapters do not import
from each other (`planning/03-system-design.md §1.3`): the prompt is duplicated here, and
both sides bind to the same shared contract, `eval/schema.json`, instead of to each other.
`ReviewRequest` carries an optional caller `context` that ch2's training format does not, so
it is threaded into the prompt right before the code under review.
"""

from __future__ import annotations

import json

from eval.harness import load_schema

_SCHEMA_JSON = json.dumps(load_schema(), indent=2)

SYSTEM_PROMPT = f"""You are a meticulous Java code reviewer. You are given one Java method or \
class and must identify exactly one bug in it.

Respond with a single bare JSON object matching this schema exactly. Do not wrap it in a \
markdown code fence. Do not add any prose before or after it — the response must be valid JSON \
and nothing else.

{_SCHEMA_JSON}"""


def format_zero_shot(code: str, context: str = "") -> str:
    """System message, optional caller context, then the code under review."""
    parts = [SYSTEM_PROMPT]
    if context:
        parts.append(f"Context: {context}")
    parts.append(f"Java code:\n```java\n{code}\n```")
    return "\n\n".join(parts)
