"""The stub set is what C1's baseline numbers are measured against — its own integrity
(unique ids, valid line numbers, no leakage from the few-shot examples) has to hold before
anything scored against it means anything."""

from __future__ import annotations

from pathlib import Path

from ch2_adaptation.prompts import FEW_SHOT_EXAMPLES

from eval.harness import load_holdout

STUB_PATH = Path("eval/stub/stub_eval.jsonl")
STUB_SMOKE_PATH = Path("eval/stub/stub_eval_smoke.jsonl")

REQUIRED_KEYS = {"id", "code", "line", "category", "severity"}


def test_every_record_has_required_keys() -> None:
    records = load_holdout(STUB_PATH)
    for record in records:
        assert REQUIRED_KEYS.issubset(record.keys()), record


def test_ids_are_unique() -> None:
    records = load_holdout(STUB_PATH)
    ids = [record["id"] for record in records]
    assert len(ids) == len(set(ids))


def test_line_numbers_are_valid() -> None:
    records = load_holdout(STUB_PATH)
    for record in records:
        assert record["line"] >= 1, record
        n_lines = len(record["code"].split("\n"))
        assert record["line"] <= n_lines, record


def test_smoke_set_is_a_strict_subset_of_full_set() -> None:
    full = load_holdout(STUB_PATH)
    smoke = load_holdout(STUB_SMOKE_PATH)
    full_by_id = {record["id"]: record for record in full}

    assert 0 < len(smoke) < len(full)
    for record in smoke:
        assert record["id"] in full_by_id
        assert record == full_by_id[record["id"]]


def test_no_few_shot_example_code_appears_in_the_stub_set() -> None:
    records = load_holdout(STUB_PATH)
    stub_code_blocks = [record["code"] for record in records]

    for example_code, _example_output in FEW_SHOT_EXAMPLES:
        for stub_code in stub_code_blocks:
            assert example_code not in stub_code
            assert stub_code not in example_code
