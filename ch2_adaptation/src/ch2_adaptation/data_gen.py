"""Synthetic (buggy Java/Spring snippet -> ReviewOutput label) training data (spec C3).

One frontier call injects and labels a bug together, rather than one call to inject and a
second to describe it -- a two-call design risks the description drifting from what was
actually inserted. Distinct from `prompts.py`'s locked review-only prompt, which stays
reserved for C4 inference / C5 comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError

from ch2_adaptation.baseline import Usage
from ch2_adaptation.config import DataGenConfig, load_data_gen_config
from ch2_adaptation.frontier import FrontierClient
from ch2_adaptation.schema import ReviewOutput
from eval.config import seed_everything
from eval.dedup import code_hash
from eval.metrics import LINE_TOLERANCE

_LABEL_FIELDS = ("severity", "category", "line", "issue", "suggested_fix")

# Rough, tokenizer-free estimate used only for the pre-flight budget guard (AC-3) -- good
# enough to abort before spending a cent; the post-hoc actual_cost_usd in the provenance
# record is what gets trusted, not this estimate.
_CHARS_PER_TOKEN_ESTIMATE = 4

# The hash-only index C2 publishes outside eval/holdout/ (see C2's plan and progress_report.md
# for why it isn't literally named eval/holdout_hashes.txt: that prefix trips
# leakage_guard.py's substring check even for a committed, non-holdout file). Read-only here --
# this module never touches eval/holdout/ itself (CON-6/AC-7).
HOLDOUT_HASHES_PATH = Path("eval/frozen_hashes.txt")

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


class InjectionOutput(ReviewOutput):
    """The one response shape the injection prompt asks for: the buggy method plus its label.

    Passed to the frontier client as its enforced JSON schema. Forcing plain `ReviewOutput`
    there (the client's default) strips `code`, and every response would be dropped.
    """

    code: str = Field(min_length=1)


def _parse_injection_response(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_injection(raw: str) -> dict[str, Any] | None:
    """One raw response -> record, or None. Never hand-repaired (AC-1)."""
    parsed = _parse_injection_response(raw)
    if parsed is None or "code" not in parsed or not isinstance(parsed["code"], str):
        return None
    label_subset = {key: parsed.get(key) for key in _LABEL_FIELDS}
    try:
        ReviewOutput.model_validate(label_subset)
    except ValidationError:
        return None
    return {"code": parsed["code"], **label_subset}


def parse_injections(snippets: list[str], outputs: list[str]) -> list[tuple[str, dict[str, Any]]]:
    """Pair each surviving record with the clean snippet it was injected into.

    Outputs align positionally with snippets; a length mismatch is an error, not something
    to truncate, or labels would attach to the wrong snippet.
    """
    if len(outputs) != len(snippets):
        raise ValueError(f"got {len(outputs)} responses for {len(snippets)} snippets")
    pairs: list[tuple[str, dict[str, Any]]] = []
    for clean, raw in zip(snippets, outputs, strict=True):
        record = _parse_injection(raw)
        if record is not None:
            pairs.append((clean, record))
    return pairs


def inject_and_label(
    snippets: list[str], client: FrontierClient
) -> tuple[list[dict[str, Any]], Usage]:
    """Inject and label a bug into each snippet with one call per snippet.

    A response that fails to parse, is missing `code`, or fails `ReviewOutput` validation
    is dropped -- never hand-repaired (AC-1). Returns the surviving records alongside the
    `Usage` the client reported, so a caller can build an honest provenance record; the
    caller derives `n_skipped` as `len(snippets) - len(records)`.
    """
    prompts = [build_injection_prompt(code) for code in snippets]
    outputs, usage = client.generate(prompts)
    return [record for _, record in parse_injections(snippets, outputs)], usage


def build_injection_prompt(code: str) -> str:
    return BUG_INJECTION_PROMPT_TEMPLATE.format(code=code, context="")


def changed_lines(clean: str, buggy: str) -> set[int]:
    """1-indexed lines of `buggy` that differ from `clean`, ignoring indentation.

    A pure deletion has no line of its own in `buggy`, so it marks the lines on either side
    of the gap.
    """
    before = [line.strip() for line in clean.split("\n")]
    after = [line.strip() for line in buggy.split("\n")]
    lines: set[int] = set()
    matcher = SequenceMatcher(a=before, b=after, autojunk=False)
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if j2 > j1:
            lines.update(range(j1 + 1, j2 + 1))
        else:
            lines.update({max(j1, 1), j1 + 1})
    return lines


def filter_line_mismatches(
    pairs: list[tuple[str, dict[str, Any]]], tolerance: int = LINE_TOLERANCE
) -> tuple[list[dict[str, Any]], int]:
    """Drop records whose `line` is not within `tolerance` of a line the injection changed.

    Bug-catch is scored on `line` alone (eval.metrics.is_bug_caught, same tolerance), so a
    record whose label points somewhere the bug is not teaches exactly the wrong thing. The
    frontier model mislabeled 3 of 10 real bugs this way during C2 curation. Dropped, never
    corrected (AC-1). A response that changed nothing is dropped too: no bug was injected.
    """
    kept = [
        record
        for clean, record in pairs
        if any(abs(record["line"] - n) <= tolerance for n in changed_lines(clean, record["code"]))
    ]
    return kept, len(pairs) - len(kept)


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
    n_dropped_line_mismatch: int = 0

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
    n_dropped_line_mismatch: int = 0,
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
        n_dropped_line_mismatch=n_dropped_line_mismatch,
    )


def _load_seed_pool(path: Path | str) -> list[dict[str, Any]]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _build_snippet_batch(pool: list[dict[str, Any]], cfg: DataGenConfig) -> list[str]:
    selected = pool[: cfg.n_snippets]
    return [record["code"] for record in selected for _ in range(cfg.variants_per_snippet)]


def _load_holdout_hashes(allow_missing: bool, path: Path | str = HOLDOUT_HASHES_PATH) -> set[str]:
    """Read the committed hash-only index -- never `eval/holdout/` itself (CON-6/AC-7).

    A missing file is tolerated only when `allow_missing` (a smoke run: C2's frozen holdout
    may not exist yet in a given session, and smoke never needs real dedup coverage to prove
    the code path). For a real (non-smoke) run, a missing index is an error, not an empty
    set -- silently deduping against nothing would let a holdout-colliding record land in
    committed training data with no trace beyond a stdout line nobody reads after the fact.
    """
    path = Path(path)
    if not path.exists():
        if allow_missing:
            print(f"[C3] {path} not found -- proceeding with an empty holdout-hash set (smoke)")
            return set()
        raise FileNotFoundError(
            f"{path} not found -- a real generation run must dedup against C2's frozen "
            "holdout hash index. Run C2's freeze step first, or point at the right path."
        )
    with path.open() as handle:
        return {line.strip() for line in handle if line.strip()}


def _estimate_preflight_cost_usd(snippets: list[str], cfg: DataGenConfig) -> float:
    from ch2_adaptation.frontier import estimate_cost_usd

    prompts = [build_injection_prompt(code) for code in snippets]
    input_tokens = sum(len(prompt) for prompt in prompts) // _CHARS_PER_TOKEN_ESTIMATE
    output_tokens = len(snippets) * cfg.max_new_tokens
    return estimate_cost_usd(input_tokens, output_tokens)


def _check_budget(snippets: list[str], cfg: DataGenConfig) -> float:
    """Abort before spending a cent if the pre-flight estimate exceeds the config's cap (AC-3).

    Skipped for a smoke run: the stub client makes no network call and costs nothing, so
    there is no real budget to guard.
    """
    if cfg.is_smoke:
        return 0.0
    estimate = _estimate_preflight_cost_usd(snippets, cfg)
    if estimate > cfg.max_budget_usd:
        raise RuntimeError(
            f"pre-flight cost estimate ${estimate:.2f} exceeds max_budget_usd="
            f"${cfg.max_budget_usd:.2f} -- aborting before spending anything (AC-3). Reduce "
            "n_snippets/variants_per_snippet, or raise max_budget_usd deliberately."
        )
    return estimate


def _write_jsonl_atomic(records: list[dict[str, Any]], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    tmp.replace(path)


def _load_checkpoint(path: Path, prompts: list[str]) -> tuple[list[str], list[Usage]]:
    """Outputs and per-chunk usage already saved for a prefix of `prompts`.

    Each line is one chunk. A chunk whose prompts no longer match (seed pool or config
    changed) is an error: resuming would pair old responses with new snippets. A truncated
    final line (process killed mid-write) is discarded; that chunk is simply regenerated.
    """
    if not path.exists():
        return [], []
    lines = path.read_text().splitlines()
    outputs: list[str] = []
    usages: list[Usage] = []
    for number, line in enumerate(lines, start=1):
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            if number == len(lines):
                break
            raise
        expected = [
            _sha256(p) for p in prompts[len(outputs) : len(outputs) + len(chunk["outputs"])]
        ]
        if chunk["start"] != len(outputs) or chunk["prompt_sha256"] != expected:
            raise ValueError(
                f"{path} does not match the current prompts (config or seed pool changed). "
                "Delete it to start the generation over."
            )
        outputs.extend(chunk["outputs"])
        usages.append(Usage(**chunk["usage"]))
    return outputs, usages


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_with_checkpoints(
    prompts: list[str], client: FrontierClient, path: Path | str, chunk_size: int
) -> tuple[list[str], Usage]:
    """Call the client `chunk_size` prompts at a time, appending each chunk to `path`.

    A crash (a non-retryable API error, an exhausted daily quota, Ctrl+C) loses at most the
    chunk in flight; re-running the same command resumes after the last saved chunk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    outputs, usages = _load_checkpoint(path, prompts)
    if outputs:
        print(f"[C3] resuming: {len(outputs)}/{len(prompts)} responses already saved in {path}")
    for start in range(len(outputs), len(prompts), chunk_size):
        chunk = prompts[start : start + chunk_size]
        chunk_outputs, chunk_usage = client.generate(chunk)
        record = {
            "start": start,
            "prompt_sha256": [_sha256(prompt) for prompt in chunk],
            "outputs": chunk_outputs,
            "usage": chunk_usage.__dict__,
        }
        with path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        outputs.extend(chunk_outputs)
        usages.append(chunk_usage)
        print(f"[C3] {len(outputs)}/{len(prompts)} responses saved")
    total = Usage(
        input_tokens=sum(u.input_tokens for u in usages),
        output_tokens=sum(u.output_tokens for u in usages),
        cost_usd=sum(u.cost_usd for u in usages),
        tier=usages[0].tier if usages else "free",
    )
    return outputs, total


# The stub "injects" by inserting this line right after the method signature, so smoke runs
# exercise the line check with a real, known change instead of producing zero records.
_STUB_INJECTED_LINE = "    Object stubInjected = null; // smoke-only injected line"
_STUB_INJECTED_LINE_NUMBER = 2


class StubInjectionClient:
    """Smoke-only injection client: deterministic, no network, no API key."""

    def generate(self, prompts: list[str]) -> tuple[list[str], Usage]:
        outputs = []
        for prompt in prompts:
            clean = prompt.split("```java\n", 1)[1].split("\n```", 1)[0].split("\n")
            buggy = clean[:1] + [_STUB_INJECTED_LINE] + clean[1:]
            label = {
                "code": "\n".join(buggy),
                "severity": "minor",
                "category": "stub",
                "line": _STUB_INJECTED_LINE_NUMBER,
                "issue": "stub issue",
                "suggested_fix": "stub fix",
            }
            outputs.append(json.dumps(label))
        return outputs, Usage(input_tokens=0, output_tokens=0, cost_usd=0.0, tier="free")


def _make_injection_client(cfg: DataGenConfig) -> FrontierClient:
    if cfg.is_smoke:
        return StubInjectionClient()
    from ch2_adaptation.frontier import make_client

    return make_client(
        cfg.frontier_provider,
        cfg.frontier_model_tag,
        cfg.temperature,
        cfg.max_new_tokens,
        response_model=InjectionOutput,
    )


def _generate_records(
    cfg: DataGenConfig, snippets: list[str]
) -> tuple[list[dict[str, Any]], Usage, int]:
    """Generate, schema-filter, line-check, and dedup. Returns records, usage, line drops."""
    prompts = [build_injection_prompt(code) for code in snippets]
    client = _make_injection_client(cfg)
    outputs, usage = generate_with_checkpoints(
        prompts, client, cfg.raw_responses_path, cfg.checkpoint_every
    )
    pairs = parse_injections(snippets, outputs)
    print(f"[C3] {len(pairs)}/{len(snippets)} responses passed schema validation")
    records, n_line_mismatch = filter_line_mismatches(pairs)
    print(f"[C3] {n_line_mismatch} dropped: line label not within ±{LINE_TOLERANCE} of the change")

    holdout_hashes = _load_holdout_hashes(allow_missing=cfg.is_smoke)
    records = dedup_against_holdout(records, holdout_hashes)
    records = dedup_within_training_set(records)
    print(f"[C3] {len(records)} records survive dedup against the holdout and each other")
    return records, usage, n_line_mismatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic (buggy Java/Spring snippet -> label) training data."
    )
    parser.add_argument("--config", type=Path, required=True, help="path to a YAML config")
    return parser.parse_args()


def main() -> None:
    from eval.harness import write_json_atomic

    args = parse_args()
    cfg = load_data_gen_config(args.config)
    seed_everything(cfg.seed)
    print(f"[C3] config loaded: {args.config}")

    pool = _load_seed_pool(cfg.seed_pool_path)
    snippets = _build_snippet_batch(pool, cfg)
    estimated_cost_usd = _check_budget(snippets, cfg)
    print(f"[C3] requesting {len(snippets)} injections, estimated cost ${estimated_cost_usd:.4f}")

    records, usage, n_line_mismatch = _generate_records(cfg, snippets)

    train, val = split_train_val(records, frac=cfg.val_frac, seed=cfg.seed)
    _write_jsonl_atomic(train, cfg.train_path)
    _write_jsonl_atomic(val, cfg.val_path)

    provenance = build_provenance(
        frontier_model=cfg.frontier_model_tag,
        n_requested=len(snippets),
        n_valid=len(records),
        estimated_cost_usd=estimated_cost_usd,
        actual_cost_usd=usage.cost_usd,
        train_path=cfg.train_path,
        val_path=cfg.val_path,
        split_seed=cfg.seed,
        n_dropped_line_mismatch=n_line_mismatch,
    )
    write_json_atomic(provenance.to_json(), cfg.provenance_path)
    print(
        f"[C3] wrote {len(train)} train / {len(val)} val records; provenance at "
        f"{cfg.provenance_path}"
    )


if __name__ == "__main__":
    main()
