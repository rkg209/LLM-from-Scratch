"""The harness must score a known mix of outputs to exactly the expected rates.

This fixture is deliberately hand-built so the right answer is arithmetic, not opinion:
4 samples, 3 schema-valid, 2 of those on the right line.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.harness import EvalResult, load_holdout, score_outputs, write_result

HOLDOUT = [
    {"id": "a", "code": "...", "line": 10},
    {"id": "b", "code": "...", "line": 20},
    {"id": "c", "code": "...", "line": 30},
    {"id": "d", "code": "...", "line": 40},
]


def review(line: int, severity: str = "major") -> str:
    return json.dumps(
        {
            "severity": severity,
            "category": "npe-risk",
            "line": line,
            "issue": "Possible null dereference.",
            "suggested_fix": "Guard with Objects.requireNonNull.",
        }
    )


OUTPUTS = [
    review(10),  # valid, exact line          → valid, caught
    review(22),  # valid, line off by 2       → valid, caught (within tolerance)
    review(99),  # valid, wrong line entirely → valid, not caught
    "I think there might be a bug around line 40?",  # not JSON → invalid, not caught
]


@pytest.fixture
def holdout_path(tmp_path: Path) -> Path:
    path = tmp_path / "holdout.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in HOLDOUT))
    return path


def test_scores_are_exactly_as_counted(holdout_path: Path) -> None:
    result = score_outputs(OUTPUTS, holdout_path)

    assert result.n_samples == 4
    assert result.n_valid == 3
    assert result.n_caught == 2
    assert result.schema_validity_rate == 0.75
    assert result.bug_catch_rate == 0.5


def test_unparseable_output_counts_as_a_failure_not_an_exclusion(holdout_path: Path) -> None:
    """The denominator is every sample. Dropping the ones that failed is how metrics lie."""
    result = score_outputs(OUTPUTS, holdout_path)
    assert result.n_samples == len(HOLDOUT)
    assert result.per_sample[3]["valid"] is False
    assert result.per_sample[3]["caught"] is False


def test_line_tolerance_is_two(holdout_path: Path) -> None:
    inside = score_outputs([review(12), review(20), review(30), review(40)], holdout_path)
    outside = score_outputs([review(13), review(20), review(30), review(40)], holdout_path)

    assert inside.n_caught == 4  # 12 is within ±2 of 10
    assert outside.n_caught == 3  # 13 is not


def test_invalid_output_never_counts_as_caught(holdout_path: Path) -> None:
    """A review nobody can parse has caught nothing, however right its line number is."""
    outputs = ['{"line": 10, "severity": "nonsense"}'] + [review(20), review(30), review(40)]
    result = score_outputs(outputs, holdout_path)

    assert result.per_sample[0]["valid"] is False
    assert result.per_sample[0]["caught"] is False
    assert result.n_caught <= result.n_valid


def test_per_sample_traces_back_to_the_record(holdout_path: Path) -> None:
    result = score_outputs(OUTPUTS, holdout_path)
    assert [sample["id"] for sample in result.per_sample] == ["a", "b", "c", "d"]
    assert result.per_sample[0]["raw_output"] == OUTPUTS[0]


def test_misaligned_outputs_raise(holdout_path: Path) -> None:
    with pytest.raises(ValueError, match="align 1:1"):
        score_outputs(OUTPUTS[:2], holdout_path)


def test_missing_holdout_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        score_outputs([], tmp_path / "nope.jsonl")


def test_load_holdout_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "h.jsonl"
    path.write_text('{"id": "a", "line": 1}\n\n{"id": "b", "line": 2}\n')
    assert len(load_holdout(path)) == 2


def test_write_result_is_atomic_and_round_trips(tmp_path: Path) -> None:
    result = EvalResult(0.75, 0.5, 4, 3, 2, [{"id": "a", "valid": True}])
    path = tmp_path / "results" / "finetuned.json"
    write_result(result, path)

    assert not path.with_suffix(".json.tmp").exists()
    assert json.loads(path.read_text())["schema_validity_rate"] == 0.75
