"""publish_table: re-render the published table from finished artifacts (spec C5, AC-5).

No torch, no network, no holdout -- this entrypoint only moves numbers between files, and
these tests pin the two ways that can go wrong: publishing rows that were scored on
different inputs, and rebuilding a rate as 0 from a malformed file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ch2_adaptation.publish_table import _as_result, main, verify_same_holdout

RESULT_FIELDS = {
    "schema_validity_rate": 0.5,
    "bug_catch_rate": 0.25,
    "n_samples": 4,
    "n_valid": 2,
    "n_caught": 1,
    "per_sample": [],
}


def _write(path: Path, doc: dict) -> Path:
    path.write_text(json.dumps(doc))
    return path


@pytest.fixture
def artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    finetuned = _write(
        tmp_path / "finetuned.json",
        {"holdout_sha256": "hhh", "schema_sha256": "sss", **RESULT_FIELDS},
    )
    baselines = _write(
        tmp_path / "baselines_holdout.json",
        {
            "stub_set_sha256": "hhh",
            "schema_sha256": "sss",
            "base_model": {**RESULT_FIELDS, "schema_validity_rate": 0.0, "n_valid": 0},
            "frontier_api": {**RESULT_FIELDS, "schema_validity_rate": 1.0, "n_valid": 4},
        },
    )
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- EVAL_TABLE_START -->\nold\n<!-- EVAL_TABLE_END -->\noutro\n")
    return finetuned, baselines, readme


def _config(tmp_path: Path, artifacts: tuple[Path, Path, Path]) -> Path:
    finetuned, baselines, readme = artifacts
    path = tmp_path / "publish.yaml"
    path.write_text(
        f"finetuned_path: {finetuned}\nbaselines_path: {baselines}\nreadme_path: {readme}\n"
    )
    return path


def test_main_renders_all_three_rows_into_the_readme(
    tmp_path: Path, artifacts: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, readme = artifacts
    config = _config(tmp_path, artifacts)
    monkeypatch.setattr("sys.argv", ["publish_table", "--config", str(config)])

    main()

    published = readme.read_text()
    assert "old" not in published
    assert "| Fine-tuned (QLoRA, Qwen2.5-Coder-1.5B) | 0.50 | 0.25 | 4 |" in published
    assert "| Base model (zero-shot) | 0.00 | 0.25 | 4 |" in published
    assert "| Frontier API (3-shot) | 1.00 | 0.25 | 4 |" in published
    assert published.startswith("intro") and published.endswith("outro\n")


def test_main_is_idempotent(
    tmp_path: Path, artifacts: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-publishing must not append a second table or drift the surrounding prose --
    this runs after every eval, so a non-idempotent version corrupts the README slowly."""
    _, _, readme = artifacts
    config = _config(tmp_path, artifacts)
    monkeypatch.setattr("sys.argv", ["publish_table", "--config", str(config)])

    main()
    once = readme.read_text()
    main()

    assert readme.read_text() == once


def test_mismatched_holdout_refuses_to_publish() -> None:
    with pytest.raises(ValueError, match="holdout_sha256 differs"):
        verify_same_holdout(
            {"holdout_sha256": "aaa", "schema_sha256": "sss"},
            {"stub_set_sha256": "bbb", "schema_sha256": "sss"},
        )


def test_mismatched_schema_refuses_to_publish() -> None:
    with pytest.raises(ValueError, match="schema_sha256 differs"):
        verify_same_holdout(
            {"holdout_sha256": "hhh", "schema_sha256": "aaa"},
            {"stub_set_sha256": "hhh", "schema_sha256": "bbb"},
        )


def test_missing_hashes_are_skipped_not_guessed() -> None:
    """finetuned.json from Rudra job 401990 predates the hash fields. Absence must not be
    read as a mismatch (which would block publishing a valid result) nor as a match."""
    verify_same_holdout({**RESULT_FIELDS}, {"stub_set_sha256": "hhh", "schema_sha256": "sss"})


def test_as_result_raises_on_a_missing_rate_rather_than_defaulting_to_zero() -> None:
    """A defaulted 0.0 would publish a wrong table that looks entirely normal."""
    with pytest.raises(KeyError):
        _as_result({k: v for k, v in RESULT_FIELDS.items() if k != "bug_catch_rate"})
