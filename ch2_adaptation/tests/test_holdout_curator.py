"""assemble_records, validate_records, dedup_records, build_manifest, label_mined_records
— driven entirely by fixtures, no network, no torch (spec C2, tasks 2 and 5)."""

from __future__ import annotations

import argparse
import json

import pytest
from ch2_adaptation.baseline import Usage
from ch2_adaptation.holdout_curator import (
    DedupReport,
    assemble_records,
    build_manifest,
    dedup_records,
    label_mined_records,
    validate_records,
)


def label(code: str = "return 1;", **overrides: object) -> dict[str, object]:
    record = {
        "code": code,
        "context": "",
        "severity": "major",
        "category": "npe-risk",
        "line": 1,
        "issue": "Possible null dereference.",
        "suggested_fix": "Guard with Objects.requireNonNull.",
    }
    record.update(overrides)
    return record


def test_assemble_records_stamps_id_split_source_created_at() -> None:
    synthetic = [label(code="a"), label(code="b")]
    mined = [label(code="c", source_repo="org/repo", license="MIT")]

    records = assemble_records(synthetic, mined)

    assert len(records) == 3
    assert {r["source"] for r in records[:2]} == {"synthetic"}
    assert records[2]["source"] == "mined"
    assert all(r["split"] == "holdout" for r in records)
    ids = {r["id"] for r in records}
    assert len(ids) == 3
    for record in records:
        assert record["created_at"]


def test_assemble_records_preserves_source_fields() -> None:
    mined = [label(code="c", source_repo="org/repo", license="MIT", commit_url="https://x")]
    records = assemble_records([], mined)
    assert records[0]["source_repo"] == "org/repo"
    assert records[0]["license"] == "MIT"


def test_validate_records_passes_for_conforming_records() -> None:
    records = assemble_records([label()], [])
    validate_records(records)  # must not raise


def test_validate_records_raises_on_missing_required_field() -> None:
    bad = label()
    del bad["severity"]
    records = assemble_records([bad], [])
    with pytest.raises(ValueError, match="severity"):
        validate_records(records)


def test_validate_records_raises_on_bad_enum_value() -> None:
    bad = label(severity="catastrophic")
    records = assemble_records([bad], [])
    with pytest.raises(ValueError, match="catastrophic"):
        validate_records(records)


def test_validate_records_raises_on_wrong_type() -> None:
    bad = label(line="one")
    records = assemble_records([bad], [])
    with pytest.raises(ValueError, match="line"):
        validate_records(records)


def test_validate_records_raises_on_string_too_long() -> None:
    bad = label(category="x" * 65)
    records = assemble_records([bad], [])
    with pytest.raises(ValueError, match="category"):
        validate_records(records)


def test_validate_records_raises_on_line_below_minimum() -> None:
    bad = label(line=0)
    records = assemble_records([bad], [])
    with pytest.raises(ValueError, match="line"):
        validate_records(records)


def test_validate_records_passes_for_empty_list() -> None:
    validate_records([])  # must not raise


def test_dedup_records_drops_internal_duplicate() -> None:
    records = assemble_records([label(code="same"), label(code="same")], [])

    report = dedup_records(records, external_hash_pools={})

    assert isinstance(report, DedupReport)
    assert report.n_checked == 2
    assert report.n_removed == 1
    assert len(report.records) == 1


def test_dedup_records_drops_external_collision() -> None:
    from eval.dedup import code_hash

    records = assemble_records([label(code="stub code")], [])
    pools = {"stub_eval": [code_hash("stub code")]}

    report = dedup_records(records, external_hash_pools=pools)

    assert report.n_removed == 1
    assert report.records == []
    assert "stub_eval" in report.removed[0]["reason"]


def test_dedup_records_keeps_distinct_records() -> None:
    records = assemble_records([label(code="a"), label(code="b")], [])
    report = dedup_records(records, external_hash_pools={})
    assert report.n_removed == 0
    assert len(report.records) == 2


def test_dedup_records_handles_empty_input() -> None:
    report = dedup_records([], external_hash_pools={})
    assert report.n_checked == 0
    assert report.n_removed == 0
    assert report.records == []


def test_dedup_records_raises_clear_error_when_code_missing() -> None:
    records = assemble_records([label(code="a")], [])
    del records[0]["code"]
    with pytest.raises(KeyError, match=records[0]["id"]):
        dedup_records(records, external_hash_pools={})


def test_build_manifest_matches_database_design_field_list() -> None:
    records = assemble_records([label(code="a"), label(code="b")], [label(code="c")])
    report = dedup_records(records, external_hash_pools={})

    manifest = build_manifest(
        report.records, report, sources=["org/repo1", "org/repo2"], curator="rahul"
    )

    assert manifest["n_synthetic"] == 2
    assert manifest["n_mined"] == 1
    assert manifest["n_total"] == 3
    assert manifest["dedup_method"] == report.method
    assert manifest["sources"] == ["org/repo1", "org/repo2"]
    assert manifest["curator"] == "rahul"
    assert len(manifest["schema_version"]) == 64
    assert manifest["freeze_date"]


def test_build_manifest_handles_zero_records() -> None:
    report = dedup_records([], external_hash_pools={})
    manifest = build_manifest([], report, sources=[], curator="rahul")
    assert manifest["n_synthetic"] == 0
    assert manifest["n_mined"] == 0
    assert manifest["n_total"] == 0


def mined(code: str = "return 1;", **overrides: object) -> dict[str, object]:
    record = {
        "code": code,
        "context": "",
        "source_repo": "org/repo",
        "license": "MIT",
        "commit_url": "https://github.com/org/repo/commit/abc",
    }
    record.update(overrides)
    return record


def review_response(**overrides: object) -> str:
    payload = {
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


def test_label_mined_records_attaches_the_five_label_fields() -> None:
    client = FakeClient([review_response(), review_response(severity="critical")])
    raw = [mined(code="a"), mined(code="b")]

    labeled, usage = label_mined_records(raw, client)

    assert len(labeled) == 2
    assert labeled[0]["severity"] == "major"
    assert labeled[1]["severity"] == "critical"
    assert labeled[0]["source_repo"] == "org/repo"  # provenance fields preserved
    assert usage.cost_usd == 0.0


def test_label_mined_records_uses_the_locked_zero_shot_prompt() -> None:
    from ch2_adaptation.prompts import format_zero_shot

    client = FakeClient([review_response()])
    raw = [mined(code="specific snippet")]

    label_mined_records(raw, client)

    assert client.seen_prompts == [format_zero_shot("specific snippet")]


def test_label_mined_records_drops_unparseable_response() -> None:
    client = FakeClient(["not json", review_response()])
    raw = [mined(code="a"), mined(code="b")]

    labeled, _ = label_mined_records(raw, client)

    assert len(labeled) == 1
    assert labeled[0]["code"] == "b"


def test_label_mined_records_drops_schema_invalid_response() -> None:
    client = FakeClient([review_response(severity="catastrophic")])
    labeled, _ = label_mined_records([mined()], client)
    assert labeled == []


def test_label_mined_records_never_hand_repairs_a_bad_response() -> None:
    bad = json.dumps(
        {"severity": "major", "category": "npe-risk", "line": 1, "issue": "x"}
    )  # missing suggested_fix
    client = FakeClient([bad])
    labeled, _ = label_mined_records([mined()], client)
    assert labeled == []


def test_label_mined_records_raises_clearly_on_a_response_count_mismatch() -> None:
    client = FakeClient([review_response()])  # 1 response for 2 snippets
    raw = [mined(code="a"), mined(code="b")]

    with pytest.raises(ValueError, match="2 mined snippets"):
        label_mined_records(raw, client)


def test_dedup_report_to_markdown_lists_removed_entries() -> None:
    report = DedupReport(
        method="test method",
        n_checked=5,
        n_removed=1,
        removed=[{"id": "abc-123", "reason": "duplicate of xyz-789"}],
        records=[],
    )

    markdown = report.to_markdown()

    assert "test method" in markdown
    assert "**Checked:** 5" in markdown
    assert "**Removed:** 1" in markdown
    assert "abc-123" in markdown
    assert "duplicate of xyz-789" in markdown


def test_dedup_report_to_markdown_states_none_removed() -> None:
    report = DedupReport(method="test method", n_checked=3, n_removed=0)
    markdown = report.to_markdown()
    assert "No records removed." in markdown


@pytest.fixture
def freeze_workspace(tmp_path):
    synthetic_path = tmp_path / "synthetic.jsonl"
    mined_path = tmp_path / "mined_labeled.jsonl"

    synthetic_records = [label(code=f"synthetic {i}") for i in range(3)]
    mined_records = [
        mined(
            code="mined 0",
            severity="critical",
            category="npe-risk",
            line=1,
            issue="Possible null dereference.",
            suggested_fix="Guard with Objects.requireNonNull.",
        )
    ]

    with synthetic_path.open("w") as handle:
        for record in synthetic_records:
            handle.write(json.dumps(record) + "\n")
    with mined_path.open("w") as handle:
        for record in mined_records:
            handle.write(json.dumps(record) + "\n")

    return synthetic_path, mined_path


def test_run_freeze_writes_holdout_manifest_and_dedup_report(
    freeze_workspace, tmp_path, monkeypatch
):
    from ch2_adaptation.holdout_curator import _run_freeze

    monkeypatch.setattr(
        "ch2_adaptation.holdout_curator._STUB_POOLS",
        {},  # no external pools -- keep this test independent of real stub-set content
    )
    synthetic_path, mined_path = freeze_workspace
    output_dir = tmp_path / "out"
    hashes_path = tmp_path / "frozen_hashes.txt"

    args = argparse.Namespace(
        synthetic=synthetic_path,
        mined_labeled=mined_path,
        output_dir=output_dir,
        hashes_path=hashes_path,
        curator="test-curator",
    )
    _run_freeze(args)

    holdout = [json.loads(line) for line in (output_dir / "holdout.jsonl").read_text().splitlines()]
    assert len(holdout) == 4  # 3 synthetic + 1 mined, no internal duplicates

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["n_synthetic"] == 3
    assert manifest["n_mined"] == 1
    assert manifest["n_total"] == 4
    assert manifest["curator"] == "test-curator"
    assert manifest["sources"] == ["org/repo"]

    dedup_report_md = (output_dir / "dedup_report.md").read_text()
    assert "Removed:** 0" in dedup_report_md

    hashes = hashes_path.read_text().splitlines()
    assert len(hashes) == 4


def test_run_freeze_drops_a_record_colliding_with_an_external_pool(
    freeze_workspace, tmp_path, monkeypatch
):
    from ch2_adaptation.holdout_curator import _run_freeze

    synthetic_path, mined_path = freeze_workspace
    stub_path = tmp_path / "stub.jsonl"
    stub_path.write_text(json.dumps({"code": "synthetic 0"}) + "\n")
    monkeypatch.setattr("ch2_adaptation.holdout_curator._STUB_POOLS", {"stub_eval": str(stub_path)})

    args = argparse.Namespace(
        synthetic=synthetic_path,
        mined_labeled=mined_path,
        output_dir=tmp_path / "out",
        hashes_path=tmp_path / "frozen_hashes.txt",
        curator="test-curator",
    )
    _run_freeze(args)

    holdout = [
        json.loads(line) for line in (tmp_path / "out" / "holdout.jsonl").read_text().splitlines()
    ]
    assert len(holdout) == 3  # one synthetic record dropped for colliding with the stub set
