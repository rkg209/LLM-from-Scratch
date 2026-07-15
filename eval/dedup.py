"""The normalized-code-hash primitive shared by C2's holdout curator and C3's data
generator (FR-16c). One definition of "the same snippet" so both specs dedup identically.

Lives in `eval/` rather than `ch2_adaptation/` because it needs only the base
dependencies (no torch, no genai) — the harness, the holdout curator, and the future
`data_gen.py` all import it without pulling in heavy extras.
"""

from __future__ import annotations

import hashlib
import re

_WHITESPACE_PATTERN = re.compile(r"\s+")


def _strip_comments(code: str) -> str:
    """Remove `//` and `/* */` comments, without touching `//`/`/*` inside string or
    char literals (e.g. a URL like `"http://example.com"` is not a comment).

    A plain regex over the whole source conflates the two, silently truncating any
    snippet whose string content happens to contain a comment delimiter — that turns
    two genuinely different snippets into one hash, which is a false-positive dedup.
    """
    out: list[str] = []
    i, n = 0, len(code)
    in_string = in_char = False
    while i < n:
        ch = code[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(code[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if in_char:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(code[i + 1])
                i += 2
                continue
            if ch == "'":
                in_char = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "'":
            in_char = True
            out.append(ch)
            i += 1
            continue
        if code[i : i + 2] == "//":
            i = code.find("\n", i)
            if i == -1:
                break
            continue
        if code[i : i + 2] == "/*":
            end = code.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def normalize_code(code: str) -> str:
    """Case-fold, strip comments, collapse whitespace.

    The goal is "the same snippet up to formatting and case," not "the same bytes" —
    two records that differ only in a comment or indentation are still one snippet for
    dedup purposes.
    """
    without_comments = _strip_comments(code)
    collapsed = _WHITESPACE_PATTERN.sub(" ", without_comments).strip()
    return collapsed.casefold()


def code_hash(code: str) -> str:
    """SHA-256 hex digest of the normalized code — the identity used for all dedup."""
    return hashlib.sha256(normalize_code(code).encode("utf-8")).hexdigest()
