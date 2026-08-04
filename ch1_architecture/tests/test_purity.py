"""Chapter 1's whole claim, checked with one grep: nothing under `src/` imports the thing
it is supposed to hand-write. See `.claude/skills/from-scratch-guard/SKILL.md`.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC_ROOT = Path("ch1_architecture/src")

# Substring, not a full import grammar: catches `import X`, `from X import ...`, and
# `import X as y` alike without parsing the file.
BANNED_IMPORTS = [
    "tokenizers",
    "sentencepiece",
    "transformers",
    "bitsandbytes",
]

# Named attributes/calls that are themselves banned even though their owning module
# (`torch`, `torch.nn.functional`) is allowed.
BANNED_SYMBOLS = [
    "nn.Transformer",
    "nn.MultiheadAttention",
    "nn.LayerNorm",
    "F.scaled_dot_product_attention",
]

IMPORT_RE = re.compile(r"^\s*(import|from)\s+([\w.]+)", re.MULTILINE)


def _all_source_files() -> list[Path]:
    return sorted(SRC_ROOT.rglob("*.py"))


def test_no_banned_imports() -> None:
    offenders: list[str] = []
    for path in _all_source_files():
        text = path.read_text()
        for match in IMPORT_RE.finditer(text):
            module = match.group(2)
            for banned in BANNED_IMPORTS:
                if module == banned or module.startswith(f"{banned}."):
                    offenders.append(f"{path}: imports {module!r}")
    assert not offenders, "banned import(s) found:\n" + "\n".join(offenders)


def test_no_banned_symbols() -> None:
    offenders: list[str] = []
    for path in _all_source_files():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            # Backtick-quoted mentions (docstrings/comments naming the ban list) are
            # prose, not usage — only bare, uses of the symbol in actual code count.
            if "`" in line:
                continue
            for banned in BANNED_SYMBOLS:
                if banned in line:
                    offenders.append(f"{path}:{lineno}: uses {banned!r}")
    assert not offenders, "banned symbol(s) found:\n" + "\n".join(offenders)
