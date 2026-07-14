"""The shared scoring harness (FR-5).

Every number that reaches the eval table comes through `score_outputs` — the fine-tuned
model's, the base model's, and the frontier API's alike. That is the whole point: three
systems scored by one piece of code, on one frozen set, so the comparison means something.
A metric computed anywhere else is not comparable and does not get published (NFR-7).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from eval.metrics import is_bug_caught, is_schema_valid

DEFAULT_SCHEMA_PATH = Path(__file__).parent / "schema.json"


@dataclass
class EvalResult:
    """The scored outcome of one system on one holdout set."""

    schema_validity_rate: float
    bug_catch_rate: float
    n_samples: int
    n_valid: int
    n_caught: int
    per_sample: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def load_holdout(path: Path | str) -> list[dict[str, Any]]:
    """Read a JSONL holdout set. One record per line."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"holdout set not found: {path}")
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_schema(path: Path | str = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    """Read the canonical JSON contract. This file is the single source of truth (NFR-20)."""
    with Path(path).open() as handle:
        return json.load(handle)


def score_outputs(
    outputs: list[str],
    holdout_path: Path | str,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
) -> EvalResult:
    """Score one system's raw outputs against the holdout set.

    `outputs[i]` must be the model's raw response to `holdout[i]` — the alignment is
    positional, and a mismatch is an error rather than something to quietly truncate.

    Every sample counts in the denominator. Unparseable output is a failure, not an
    exclusion, and a sample nobody scored is a sample nobody may drop.
    """
    holdout = load_holdout(holdout_path)
    schema = load_schema(schema_path)

    if len(outputs) != len(holdout):
        raise ValueError(
            f"outputs and holdout must align 1:1, got {len(outputs)} outputs "
            f"for {len(holdout)} holdout records"
        )

    per_sample: list[dict[str, Any]] = []
    for raw, truth in zip(outputs, holdout, strict=True):
        valid = is_schema_valid(raw, schema)
        per_sample.append(
            {
                "id": truth.get("id", ""),
                "valid": valid,
                "caught": valid and is_bug_caught(raw, truth, schema),
                "raw_output": raw,
            }
        )

    n_samples = len(per_sample)
    n_valid = sum(sample["valid"] for sample in per_sample)
    n_caught = sum(sample["caught"] for sample in per_sample)

    return EvalResult(
        schema_validity_rate=n_valid / n_samples if n_samples else 0.0,
        bug_catch_rate=n_caught / n_samples if n_samples else 0.0,
        n_samples=n_samples,
        n_valid=n_valid,
        n_caught=n_caught,
        per_sample=per_sample,
    )


def write_json_atomic(payload: dict[str, Any], path: Path | str) -> None:
    """Write JSON atomically: to a `.tmp` file, then rename over the target.

    A half-written results file is worse than none — the rename is what makes a reader
    see either the old complete file or the new complete one, never a partial write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2)
    tmp.replace(path)


def write_result(result: EvalResult, path: Path | str) -> None:
    """Write one system's scored result to JSON atomically."""
    write_json_atomic(result.to_json(), path)
