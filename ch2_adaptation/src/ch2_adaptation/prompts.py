"""The locked review prompt (FR-14/FR-15 surface, NFR-7).

C4 inference and C5's head-to-head comparison must reuse `format_zero_shot` /
`format_three_shot` verbatim — a different prompt per system is not a comparison.

No fence-repair anywhere near this module or the code that consumes its output:
`eval/metrics.py` does a plain `json.loads` with no markdown-fence stripping, so the
system message spells out "no markdown fences" and leaves it there. A model that wraps
its JSON in ```json anyway has produced a real schema-validity failure, not a formatting
inconvenience to paper over.
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

# Hand-written, deliberately distinct from every record in eval/stub/ — few-shot examples
# drawn from the set used to score a system are leakage, and test_stub_eval_set.py asserts
# this list and the stub set never overlap.
FEW_SHOT_EXAMPLES: list[tuple[str, str]] = [
    (
        "public String describe(Optional<Item> item) {\n" "    return item.get().getLabel();\n" "}",
        json.dumps(
            {
                "severity": "critical",
                "category": "optional-get-without-check",
                "line": 2,
                "issue": "Optional.get() is called without isPresent()/isEmpty(), so an "
                "empty Optional throws NoSuchElementException.",
                "suggested_fix": "Guard with item.isPresent() first, or use "
                "item.map(Item::getLabel).orElseThrow(...) to make the empty case explicit.",
            }
        ),
    ),
    (
        "public boolean equals(MyRecord other) {\n" "    return this.id == other.id;\n" "}",
        json.dumps(
            {
                "severity": "major",
                "category": "equals-override-signature-mismatch",
                "line": 1,
                "issue": "This overloads equals(MyRecord) instead of overriding "
                "Object.equals(Object), so collections like HashSet call the default "
                "identity-based equals() instead of this method.",
                "suggested_fix": "Change the parameter type to Object, add @Override, and "
                "cast/instanceof-check inside the method body.",
            }
        ),
    ),
    (
        "public byte[] readUserFile(String filename) throws IOException {\n"
        "    Path path = Paths.get(UPLOAD_DIR, filename);\n"
        "    return Files.readAllBytes(path);\n"
        "}",
        json.dumps(
            {
                "severity": "critical",
                "category": "path-traversal",
                "line": 2,
                "issue": "filename is joined onto UPLOAD_DIR without sanitizing '..' "
                "segments, letting a caller read arbitrary files outside UPLOAD_DIR.",
                "suggested_fix": "Resolve the path, normalize it, and reject any result "
                "that is not still under UPLOAD_DIR before reading it.",
            }
        ),
    ),
]


def format_zero_shot(code: str) -> str:
    """The base-model prompt: system message, then the code under review, no examples."""
    return f"{SYSTEM_PROMPT}\n\nJava code:\n```java\n{code}\n```"


def format_three_shot(code: str) -> str:
    """The frontier-API prompt: system message, all three worked examples, then the code."""
    parts = [SYSTEM_PROMPT]
    for example_code, example_output in FEW_SHOT_EXAMPLES:
        parts.append(f"Java code:\n```java\n{example_code}\n```\nReview:\n{example_output}")
    parts.append(f"Java code:\n```java\n{code}\n```\nReview:")
    return "\n\n".join(parts)
