"""assemble_records, validate_records, dedup_records, build_manifest — driven entirely by
fixtures, no network, no torch (spec C2, task 2)."""

from __future__ import annotations

import pytest
from ch2_adaptation.holdout_curator import (
    DedupReport,
    assemble_records,
    build_manifest,
    dedup_records,
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
