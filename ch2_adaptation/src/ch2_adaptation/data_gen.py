"""Synthetic (buggy Java/Spring snippet -> ReviewOutput label) training data (spec C3).

One frontier call injects and labels a bug together, rather than one call to inject and a
second to describe it -- a two-call design risks the description drifting from what was
actually inserted. Distinct from `prompts.py`'s locked review-only prompt, which stays
reserved for C4 inference / C5 comparison.
"""

from __future__ import annotations

import hashlib
import json
import random
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from ch2_adaptation.baseline import Usage
from ch2_adaptation.frontier import FrontierClient
from ch2_adaptation.schema import ReviewOutput
from eval.dedup import code_hash

_LABEL_FIELDS = ("severity", "category", "line", "issue", "suggested_fix")

BUG_INJECTION_PROMPT_TEMPLATE = """You are creating training data for a Java code-review model. \
Given a clean Java method, introduce exactly one realistic bug, then describe it.

Clean Java method:
```java
{code}
```

Context: {context}

Respond with a single bare JSON object with exactly these six fields, and nothing else -- no \
markdown fence, no prose before or after:
"code" (the method with your bug inserted, as a single string),
"severity" (one of "critical", "major", "minor", "info"),
"category" (a short bug category name),
"line" (the 1-indexed line inside your modified code where the bug is),
"issue" (what is wrong),
"suggested_fix" (how to fix it)."""


def _parse_injection_response(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def inject_and_label(
    snippets: list[str], client: FrontierClient
) -> tuple[list[dict[str, Any]], Usage]:
    """Inject and label a bug into each snippet with one call per snippet.

    A response that fails to parse, is missing `code`, or fails `ReviewOutput` validation
    is dropped -- never hand-repaired (AC-1). Returns the surviving records alongside the
    `Usage` the client reported, so a caller can build an honest provenance record; the
    caller derives `n_skipped` as `len(snippets) - len(records)`.
    """
    prompts = [BUG_INJECTION_PROMPT_TEMPLATE.format(code=code, context="") for code in snippets]
    outputs, usage = client.generate(prompts)

    records: list[dict[str, Any]] = []
    for raw in outputs:
        parsed = _parse_injection_response(raw)
        if parsed is None or "code" not in parsed or not isinstance(parsed["code"], str):
            continue

        label_subset = {key: parsed.get(key) for key in _LABEL_FIELDS}
        try:
            ReviewOutput.model_validate(label_subset)
        except ValidationError:
            continue

        records.append({"code": parsed["code"], **label_subset})

    return records, usage


def dedup_against_holdout(
    records: list[dict[str, Any]], holdout_hashes: set[str]
) -> list[dict[str, Any]]:
    """Drop any generated record whose normalized code hash appears in the frozen holdout.

    Never reads `eval/holdout/` itself (CON-6/AC-7) -- `holdout_hashes` is meant to come
    from the committed hash-only index C2 publishes outside that directory.
    """
    return [record for record in records if code_hash(record["code"]) not in holdout_hashes]


def dedup_within_training_set(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop internal duplicates, keeping the first occurrence of each normalized hash."""
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for record in records:
        digest = code_hash(record["code"])
        if digest in seen:
            continue
        seen.add(digest)
        kept.append(record)
    return kept


def split_train_val(
    records: list[dict[str, Any]], frac: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Seeded train/val split -- `frac` is the fraction held out for validation (AC-5)."""
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_val = round(len(shuffled) * frac)
    return shuffled[n_val:], shuffled[:n_val]


@dataclass(frozen=True)
class Provenance:
    """Mirrors `DataProvenance` per planning/04-database-design.md §3.3."""

    run_id: str
    generated_at: str
    frontier_model: str
    prompt_template_hash: str
    n_requested: int
    n_valid: int
    n_skipped: int
    estimated_cost_usd: float
    actual_cost_usd: float
    train_path: str
    val_path: str
    split_seed: int

    def to_json(self) -> dict[str, Any]:
        return self.__dict__.copy()


def build_provenance(
    frontier_model: str,
    n_requested: int,
    n_valid: int,
    estimated_cost_usd: float,
    actual_cost_usd: float,
    train_path: str,
    val_path: str,
    split_seed: int,
) -> Provenance:
    """Assemble the `DataProvenance` record for one generation run."""
    return Provenance(
        run_id=str(uuid.uuid4()),
        generated_at=datetime.now(UTC).isoformat(),
        frontier_model=frontier_model,
        prompt_template_hash=hashlib.sha256(
            BUG_INJECTION_PROMPT_TEMPLATE.encode("utf-8")
        ).hexdigest(),
        n_requested=n_requested,
        n_valid=n_valid,
        n_skipped=n_requested - n_valid,
        estimated_cost_usd=estimated_cost_usd,
        actual_cost_usd=actual_cost_usd,
        train_path=train_path,
        val_path=val_path,
        split_seed=split_seed,
    )
