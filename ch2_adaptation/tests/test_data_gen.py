"""inject_and_label, dedup_against_holdout, dedup_within_training_set, split_train_val,
build_provenance -- driven entirely by a fake client, no network, no torch (spec C3, task 3)."""

from __future__ import annotations

import json

from ch2_adaptation.baseline import Usage
from ch2_adaptation.data_gen import (
    BUG_INJECTION_PROMPT_TEMPLATE,
    build_provenance,
    dedup_against_holdout,
    dedup_within_training_set,
    inject_and_label,
    split_train_val,
)

from eval.dedup import code_hash


def injection(code: str = "return 1;", **overrides: object) -> str:
    payload = {
        "code": code,
        "severity": "major",
        "category": "npe-risk",
        "line": 1,
        "issue": "Possible null dereference.",
        "suggested_fix": "Guard with Objects.requireNonNull.",
    }
    payload.update(overrides)
    return json.dumps(payload)


class FakeClient:
    def __init__(self, outputs: list[str], usage: Usage | None = None) -> None:
        self.outputs = outputs
        self.usage = usage or Usage(0, 0, 0.0, "free")
        self.seen_prompts: list[str] = []

    def generate(self, prompts: list[str]) -> tuple[list[str], Usage]:
        self.seen_prompts = prompts
        return self.outputs, self.usage


def test_inject_and_label_returns_valid_records() -> None:
    client = FakeClient([injection(code="a"), injection(code="b")])
    records, usage = inject_and_label(["snippet a", "snippet b"], client)

    assert len(records) == 2
    assert {r["code"] for r in records} == {"a", "b"}
    assert usage.cost_usd == 0.0


def test_inject_and_label_sends_one_prompt_per_snippet() -> None:
    client = FakeClient([injection(), injection()])
    inject_and_label(["snippet one", "snippet two"], client)

    assert len(client.seen_prompts) == 2
    assert "snippet one" in client.seen_prompts[0]


def test_inject_and_label_drops_unparseable_output() -> None:
    client = FakeClient(["not json", injection(code="b")])
    records, _ = inject_and_label(["a", "b"], client)
    assert len(records) == 1
    assert records[0]["code"] == "b"


def test_inject_and_label_drops_response_missing_code() -> None:
    bad = json.dumps(
        {
            "severity": "major",
            "category": "npe-risk",
            "line": 1,
            "issue": "x",
            "suggested_fix": "y",
        }
    )
    client = FakeClient([bad])
    records, _ = inject_and_label(["a"], client)
    assert records == []


def test_inject_and_label_drops_schema_invalid_label() -> None:
    client = FakeClient([injection(severity="catastrophic")])
    records, _ = inject_and_label(["a"], client)
    assert records == []


def test_inject_and_label_never_hand_repairs_a_bad_response() -> None:
    # A response missing "issue" must be dropped outright, not filled in with a default.
    payload = json.dumps(
        {
            "code": "x",
            "severity": "major",
            "category": "npe-risk",
            "line": 1,
            "suggested_fix": "y",
        }
    )
    client = FakeClient([payload])
    records, _ = inject_and_label(["a"], client)
    assert records == []


def test_dedup_against_holdout_drops_colliding_records() -> None:
    records = [{"code": "keep me"}, {"code": "in holdout"}]
    holdout_hashes = {code_hash("in holdout")}

    kept = dedup_against_holdout(records, holdout_hashes)

    assert kept == [{"code": "keep me"}]


def test_dedup_against_holdout_keeps_all_when_no_overlap() -> None:
    records = [{"code": "a"}, {"code": "b"}]
    kept = dedup_against_holdout(records, holdout_hashes=set())
    assert kept == records


def test_dedup_within_training_set_drops_internal_duplicates() -> None:
    records = [{"code": "same", "id": 1}, {"code": "same", "id": 2}, {"code": "diff", "id": 3}]
    kept = dedup_within_training_set(records)
    assert len(kept) == 2
    assert kept[0]["id"] == 1


def test_split_train_val_respects_fraction_and_is_seeded() -> None:
    records = [{"code": str(i)} for i in range(100)]

    train_a, val_a = split_train_val(records, frac=0.10, seed=42)
    train_b, val_b = split_train_val(records, frac=0.10, seed=42)

    assert len(val_a) == 10
    assert len(train_a) == 90
    assert train_a == train_b
    assert val_a == val_b


def test_split_train_val_different_seeds_can_differ() -> None:
    records = [{"code": str(i)} for i in range(100)]
    _, val_a = split_train_val(records, frac=0.10, seed=1)
    _, val_b = split_train_val(records, frac=0.10, seed=2)
    assert val_a != val_b


def test_split_train_val_partitions_every_record_exactly_once() -> None:
    records = [{"code": str(i)} for i in range(37)]
    train, val = split_train_val(records, frac=0.10, seed=7)

    assert len(train) + len(val) == 37
    assert {r["code"] for r in train} | {r["code"] for r in val} == {r["code"] for r in records}
    assert {r["code"] for r in train} & {r["code"] for r in val} == set()


def test_build_provenance_matches_database_design_field_list() -> None:
    provenance = build_provenance(
        frontier_model="gemini-2.5-flash",
        n_requested=10,
        n_valid=8,
        estimated_cost_usd=0.05,
        actual_cost_usd=0.04,
        train_path="ch2_adaptation/data/train.jsonl",
        val_path="ch2_adaptation/data/val.jsonl",
        split_seed=42,
    )
    doc = provenance.to_json()

    assert doc["n_requested"] == 10
    assert doc["n_valid"] == 8
    assert doc["n_skipped"] == 2
    assert doc["frontier_model"] == "gemini-2.5-flash"
    assert doc["split_seed"] == 42
    assert len(doc["prompt_template_hash"]) == 64
    assert doc["train_path"] == "ch2_adaptation/data/train.jsonl"
    assert doc["run_id"]
    assert doc["generated_at"]


def test_prompt_template_hash_is_stable_for_the_locked_template() -> None:
    import hashlib

    expected = hashlib.sha256(BUG_INJECTION_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
    provenance = build_provenance(
        frontier_model="gemini-2.5-flash",
        n_requested=1,
        n_valid=1,
        estimated_cost_usd=0.0,
        actual_cost_usd=0.0,
        train_path="t",
        val_path="v",
        split_seed=1,
    )
    assert provenance.prompt_template_hash == expected
