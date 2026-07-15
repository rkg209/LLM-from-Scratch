"""Assembles, validates, dedups, and manifests the C2 holdout set.

Staging only (CLAUDE.md #3): this module never writes under `eval/holdout/` itself —
`.claude/hooks/leakage_guard.py` denies that unconditionally. It writes to
`eval/holdout_staging/`; a human moves the reviewed output into `eval/holdout/` and
commits it, outside any session, exactly like C1's deferred full-baseline run.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

from eval.dedup import code_hash
from eval.harness import DEFAULT_SCHEMA_PATH, load_schema

_LABEL_FIELDS = ("severity", "category", "line", "issue", "suggested_fix")

DEDUP_METHOD = (
    "normalized code hash (eval.dedup.code_hash): case-fold, strip comments, "
    "collapse whitespace, then SHA-256"
)


@dataclass
class DedupReport:
    """The outcome of deduping one batch of holdout records.

    `records` is the surviving set, in original order — callers use it directly rather
    than re-deriving "what got kept" from `removed`.
    """

    method: str
    n_checked: int
    n_removed: int
    removed: list[dict[str, str]] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)


def assemble_records(
    synthetic: list[dict[str, Any]], mined: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Stamp `id`/`split`/`source`/`created_at` onto raw records (planning/04 §3.1).

    `synthetic` and `mined` are raw label dicts (code/context plus the five label
    fields); this is where they become full `ReviewRecord`s.
    """
    records: list[dict[str, Any]] = []
    for source_name, raw_records in (("synthetic", synthetic), ("mined", mined)):
        for raw in raw_records:
            record = dict(raw)
            record["id"] = str(uuid.uuid4())
            record["split"] = "holdout"
            record["source"] = source_name
            record["created_at"] = datetime.now(UTC).isoformat()
            records.append(record)
    return records


def validate_records(records: list[dict[str, Any]]) -> None:
    """Validate every record's label subset against `eval/schema.json`.

    Reuses the harness's schema loader rather than hand-rolling a second validator
    (NFR-7) — the same contract the harness scores against is the one that gates entry
    into the holdout.
    """
    schema = load_schema()
    validator = Draft7Validator(schema)
    errors: list[str] = []

    for record in records:
        label_subset = {key: record[key] for key in _LABEL_FIELDS if key in record}
        problems = sorted(validator.iter_errors(label_subset), key=lambda e: list(e.path))
        if problems:
            record_id = record.get("id", "<no id>")
            messages = "; ".join(
                f"{'.'.join(str(part) for part in problem.path) or '<root>'}: {problem.message}"
                for problem in problems
            )
            errors.append(f"{record_id}: {messages}")

    if errors:
        raise ValueError("holdout records failed schema validation:\n" + "\n".join(errors))


def dedup_records(
    records: list[dict[str, Any]], external_hash_pools: dict[str, list[str]]
) -> DedupReport:
    """Drop internal duplicates and any collision against an external pool.

    `external_hash_pools` maps a pool name (e.g. `"stub_eval"`) to the normalized
    hashes already claimed by that pool, so a colliding record's report entry says
    which pool it collided with, not just that it collided.
    """
    external_owner: dict[str, str] = {
        pool_hash: pool_name
        for pool_name, pool_hashes in external_hash_pools.items()
        for pool_hash in pool_hashes
    }
    seen_owner: dict[str, str] = {}
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, str]] = []

    for record in records:
        record_id = record.get("id", "<no id>")
        if "code" not in record:
            raise KeyError(f"record {record_id} has no 'code' field to hash for dedup")
        digest = code_hash(record["code"])
        if digest in external_owner:
            removed.append({"id": record_id, "reason": f"collides with {external_owner[digest]}"})
            continue
        if digest in seen_owner:
            removed.append({"id": record_id, "reason": f"duplicate of {seen_owner[digest]}"})
            continue
        seen_owner[digest] = record_id
        kept.append(record)

    return DedupReport(
        method=DEDUP_METHOD,
        n_checked=len(records),
        n_removed=len(removed),
        removed=removed,
        records=kept,
    )


def _sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_manifest(
    records: list[dict[str, Any]],
    dedup_report: DedupReport,
    sources: list[str],
    curator: str,
) -> dict[str, Any]:
    """Build the `HoldoutManifest` dict exactly per planning/04-database-design.md §3.4."""
    n_synthetic = sum(1 for record in records if record["source"] == "synthetic")
    n_mined = sum(1 for record in records if record["source"] == "mined")
    return {
        "freeze_date": datetime.now(UTC).isoformat(),
        "n_synthetic": n_synthetic,
        "n_mined": n_mined,
        "n_total": n_synthetic + n_mined,
        "dedup_method": dedup_report.method,
        "sources": sources,
        "schema_version": _sha256_file(DEFAULT_SCHEMA_PATH),
        "curator": curator,
    }
