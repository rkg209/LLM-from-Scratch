"""Assembles, validates, dedups, and manifests the C2 holdout set.

Staging only (CLAUDE.md #3): this module never writes under `eval/holdout/` itself —
`.claude/hooks/leakage_guard.py` denies that unconditionally, and its path check is a bare
substring match on `eval/holdout` with no exception for a same-prefixed staging directory
next to it. Staging therefore lives at `eval/staging/`, not `eval/holdout_staging/` as
originally planned (see progress_report.md's C2 T3 entry) -- a human moves the reviewed
output into `eval/holdout/` and commits it, outside any session, exactly like C1's deferred
full-baseline run.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jsonschema import Draft7Validator
from pydantic import ValidationError

from ch2_adaptation.baseline import Usage
from eval.dedup import code_hash
from eval.harness import DEFAULT_SCHEMA_PATH, load_schema

if TYPE_CHECKING:
    import argparse

    from ch2_adaptation.config import BaselineConfig
    from ch2_adaptation.frontier import FrontierClient

_LABEL_FIELDS = ("severity", "category", "line", "issue", "suggested_fix")


def _parse_review_response(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def label_mined_records(
    raw: list[dict[str, Any]], client: FrontierClient
) -> tuple[list[dict[str, Any]], Usage]:
    """Label each raw mined snippet via the frontier API, one call per snippet.

    Uses the same locked zero-shot review prompt C1's baseline, C4's fine-tune, and C5's
    comparison all use (`prompts.format_zero_shot`) -- no bespoke prompt for this spec,
    so the mined-record labels are produced by the identical review task, not a
    differently-worded one. A response that fails to parse or fails
    `ch2_adaptation.schema.ReviewOutput` validation is dropped -- never hand-repaired,
    same rule C3's `inject_and_label` follows.
    """
    from ch2_adaptation.prompts import format_zero_shot
    from ch2_adaptation.schema import ReviewOutput

    prompts = [format_zero_shot(record["code"]) for record in raw]
    outputs, usage = client.generate(prompts)
    if len(outputs) != len(raw):
        raise ValueError(
            f"frontier client returned {len(outputs)} responses for {len(raw)} mined "
            "snippets -- outputs must align 1:1 with the input, or labels would attach "
            "to the wrong record."
        )

    labeled: list[dict[str, Any]] = []
    for record, output in zip(raw, outputs, strict=True):
        parsed = _parse_review_response(output)
        if parsed is None:
            continue
        label_subset = {key: parsed.get(key) for key in _LABEL_FIELDS}
        try:
            ReviewOutput.model_validate(label_subset)
        except ValidationError:
            continue
        labeled_record = dict(record)
        labeled_record.update(label_subset)
        labeled.append(labeled_record)

    return labeled, usage


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


def _load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(records: list[dict[str, Any]], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def parse_args() -> argparse.Namespace:
    """CLI with subcommands: `label-mined` (frontier-API-backed, like C1's baseline.py)
    and `freeze` (assemble/validate/dedup/manifest the 40-record set)."""
    import argparse

    parser = argparse.ArgumentParser(description="Curate the C2 independent eval set.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    label_parser = subparsers.add_parser(
        "label-mined", help="label staged mined snippets via the frontier API"
    )
    label_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="a BaselineConfig-shaped YAML (reuses frontier_provider/frontier_model_tag/"
        "temperature/max_new_tokens -- no new config needed for this spec)",
    )
    label_parser.add_argument("--input", type=Path, default=Path("eval/staging/mined_raw.jsonl"))
    label_parser.add_argument(
        "--output", type=Path, default=Path("eval/staging/mined_labeled.jsonl")
    )

    return parser.parse_args()


def _log_label_mined_to_wandb(
    cfg: BaselineConfig, n_raw: int, n_labeled: int, usage: Usage
) -> None:
    if cfg.wandb_mode == "disabled":
        return

    import wandb

    run = wandb.init(project=cfg.wandb_project, mode=cfg.wandb_mode, config=cfg.__dict__)
    wandb.log(
        {
            "label_mined/n_raw": n_raw,
            "label_mined/n_labeled": n_labeled,
            "label_mined/cost_usd": usage.cost_usd,
            "label_mined/input_tokens": usage.input_tokens,
            "label_mined/output_tokens": usage.output_tokens,
        }
    )
    run.finish()


def _run_label_mined(args: argparse.Namespace) -> None:
    from ch2_adaptation.config import load_baseline_config
    from ch2_adaptation.frontier import make_client

    cfg = load_baseline_config(args.config)
    raw = _load_jsonl(args.input)
    print(
        f"[C2] labeling {len(raw)} mined snippets via "
        f"{cfg.frontier_provider}/{cfg.frontier_model_tag}"
    )

    client = make_client(
        cfg.frontier_provider, cfg.frontier_model_tag, cfg.temperature, cfg.max_new_tokens
    )
    labeled, usage = label_mined_records(raw, client)
    print(
        f"[C2] {len(labeled)}/{len(raw)} snippets labeled; "
        f"cost=${usage.cost_usd:.4f} ({usage.tier})"
    )

    _write_jsonl(labeled, args.output)
    print(f"[C2] wrote {args.output}")

    _log_label_mined_to_wandb(cfg, len(raw), len(labeled), usage)


def main() -> None:
    args = parse_args()
    if args.command == "label-mined":
        _run_label_mined(args)
    else:
        raise ValueError(f"unknown command: {args.command!r}")


if __name__ == "__main__":
    main()
