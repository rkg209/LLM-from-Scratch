"""inject_and_label, dedup_against_holdout, dedup_within_training_set, split_train_val,
build_provenance -- driven entirely by a fake client, no network, no torch (spec C3, task 3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ch2_adaptation.baseline import Usage
from ch2_adaptation.data_gen import (
    BUG_INJECTION_PROMPT_TEMPLATE,
    InjectionOutput,
    StubInjectionClient,
    build_injection_prompt,
    build_provenance,
    changed_lines,
    dedup_against_holdout,
    dedup_within_training_set,
    filter_line_mismatches,
    generate_with_checkpoints,
    inject_and_label,
    parse_injections,
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


# --- line-label check, response model, checkpointing, smoke stub -------------------------

CLEAN = "void f() {\n    int a = 1;\n    use(a);\n    log();\n    done();\n    end();\n}"


def test_changed_lines_finds_a_replaced_line_and_ignores_reindentation() -> None:
    buggy = "void f() {\n  int a = 0;\n  use(a);\n  log();\n  done();\n  end();\n}"
    assert changed_lines(CLEAN, buggy) == {2}


def test_changed_lines_marks_both_sides_of_a_pure_deletion() -> None:
    buggy = "void f() {\n    int a = 1;\n    log();\n    done();\n    end();\n}"
    assert changed_lines(CLEAN, buggy) == {2, 3}  # the lines either side of the removed use(a)


def test_filter_line_mismatches_keeps_labels_within_tolerance_and_drops_the_rest() -> None:
    buggy = CLEAN.replace("int a = 1;", "int a = 0;")  # change on line 2
    near = json.loads(injection(code=buggy, line=4))
    far = json.loads(injection(code=buggy, line=7))
    kept, n_dropped = filter_line_mismatches([(CLEAN, near), (CLEAN, far)])
    assert kept == [near]
    assert n_dropped == 1


def test_filter_line_mismatches_drops_a_response_that_changed_nothing() -> None:
    unchanged = json.loads(injection(code=CLEAN, line=2))
    assert filter_line_mismatches([(CLEAN, unchanged)]) == ([], 1)


def test_parse_injections_keeps_each_record_paired_with_its_snippet() -> None:
    pairs = parse_injections(["clean a", "clean b"], ["not json", injection(code="b2")])
    assert [(clean, record["code"]) for clean, record in pairs] == [("clean b", "b2")]


def test_parse_injections_rejects_misaligned_outputs() -> None:
    with pytest.raises(ValueError, match="responses"):
        parse_injections(["a", "b"], [injection()])


def test_injection_output_requires_code_on_top_of_the_review_fields() -> None:
    assert set(InjectionOutput.model_fields) == {
        "code",
        "severity",
        "category",
        "line",
        "issue",
        "suggested_fix",
    }


class CountingClient:
    """Echoes a numbered response per prompt; can fail after N calls to simulate a crash."""

    def __init__(self, fail_after_calls: int | None = None) -> None:
        self.calls = 0
        self.prompts_seen: list[str] = []
        self.fail_after_calls = fail_after_calls

    def generate(self, prompts: list[str]) -> tuple[list[str], Usage]:
        if self.fail_after_calls is not None and self.calls >= self.fail_after_calls:
            raise RuntimeError("simulated quota exhaustion")
        self.calls += 1
        self.prompts_seen.extend(prompts)
        return [f"out:{p}" for p in prompts], Usage(len(prompts), 1, 0.0, "free")


def test_generate_with_checkpoints_resumes_after_a_crash_without_repeating_calls(
    tmp_path: Path,
) -> None:
    prompts = [f"p{i}" for i in range(5)]
    cache = tmp_path / "raw.jsonl"
    with pytest.raises(RuntimeError, match="quota"):
        generate_with_checkpoints(prompts, CountingClient(fail_after_calls=1), cache, 2)

    resumed = CountingClient()
    outputs, usage = generate_with_checkpoints(prompts, resumed, cache, 2)

    assert outputs == [f"out:p{i}" for i in range(5)]
    assert resumed.prompts_seen == ["p2", "p3", "p4"]  # the saved chunk was not re-requested
    assert usage.input_tokens == 5  # usage from before the crash is still counted


def test_generate_with_checkpoints_refuses_a_cache_from_different_prompts(tmp_path: Path) -> None:
    cache = tmp_path / "raw.jsonl"
    generate_with_checkpoints(["a", "b"], CountingClient(), cache, 2)
    with pytest.raises(ValueError, match="does not match"):
        generate_with_checkpoints(["x", "y"], CountingClient(), cache, 2)


def test_generate_with_checkpoints_discards_a_truncated_final_line(tmp_path: Path) -> None:
    cache = tmp_path / "raw.jsonl"
    generate_with_checkpoints(["a", "b"], CountingClient(), cache, 1)
    lines = cache.read_text().splitlines()
    cache.write_text(lines[0] + "\n" + lines[1][:10])  # killed mid-write

    resumed = CountingClient()
    outputs, _ = generate_with_checkpoints(["a", "b"], resumed, cache, 1)
    assert outputs == ["out:a", "out:b"]
    assert resumed.prompts_seen == ["b"]


def test_stub_injection_client_output_survives_parsing_and_the_line_check() -> None:
    snippets = [CLEAN, "int g() {\n    return 1;\n}"]
    outputs, _ = StubInjectionClient().generate([build_injection_prompt(s) for s in snippets])
    kept, n_dropped = filter_line_mismatches(parse_injections(snippets, outputs))
    assert len(kept) == 2
    assert n_dropped == 0


def test_build_provenance_records_line_mismatch_drops() -> None:
    provenance = build_provenance(
        frontier_model="m",
        n_requested=3,
        n_valid=1,
        estimated_cost_usd=0.0,
        actual_cost_usd=0.0,
        train_path="t",
        val_path="v",
        split_seed=1,
        n_dropped_line_mismatch=2,
    )
    assert provenance.to_json()["n_dropped_line_mismatch"] == 2


def test_changed_lines_ignores_blank_lines_added_or_removed() -> None:
    """Live: 2 responses returned the clean method plus a trailing blank line, and a label
    within ±2 of that blank line passed the check as if a bug had been injected."""
    assert changed_lines(CLEAN, CLEAN + "\n") == set()
    assert changed_lines(CLEAN, CLEAN.replace("\n    log();", "\n\n    log();")) == set()
    unchanged_plus_blank = json.loads(injection(code=CLEAN + "\n", line=7))
    assert filter_line_mismatches([(CLEAN, unchanged_plus_blank)]) == ([], 1)


def test_filter_line_mismatches_drops_a_comment_only_change() -> None:
    """Live: a response appended only `// Bug: ...` to the untouched clean method."""
    commented = json.loads(injection(code=CLEAN + "\n// Bug: missing validation", line=7))
    assert filter_line_mismatches([(CLEAN, commented)]) == ([], 1)
