"""score_adapter_on_holdout and verify_schema_unchanged, driven by fakes -- no torch, no
network (spec C5, task 2)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from ch2_adaptation.evaluate import score_adapter_on_holdout, verify_schema_unchanged

from eval.harness import DEFAULT_SCHEMA_PATH

HOLDOUT = [
    {"id": "a", "code": "...", "line": 10},
    {"id": "b", "code": "...", "line": 20},
]


@pytest.fixture
def holdout_path(tmp_path: Path) -> Path:
    path = tmp_path / "holdout.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in HOLDOUT))
    return path


def test_score_adapter_on_holdout_delegates_exactly_to_score_system(
    holdout_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """score_adapter_on_holdout has no logic of its own besides forwarding to
    score_system (already exhaustively tested in test_baseline.py) -- what's worth
    verifying here is that the forwarding itself is exact: same args, same return value,
    not a re-test of score_system's own rate arithmetic."""
    calls: list[tuple[object, ...]] = []
    sentinel = object()

    def fake_score_system(prompts, generate, holdout_path_arg):  # noqa: ANN001
        calls.append((prompts, generate, holdout_path_arg))
        return sentinel

    monkeypatch.setattr("ch2_adaptation.evaluate.score_system", fake_score_system)

    prompts = ["p1", "p2"]
    fake_generate = lambda batch: []  # noqa: E731

    result = score_adapter_on_holdout(prompts, fake_generate, holdout_path)

    assert result is sentinel
    assert calls == [(prompts, fake_generate, holdout_path)]


def test_verify_schema_unchanged_passes_for_a_matching_hash() -> None:
    real_sha256 = hashlib.sha256(Path(DEFAULT_SCHEMA_PATH).read_bytes()).hexdigest()
    verify_schema_unchanged(real_sha256)  # must not raise


def test_verify_schema_unchanged_raises_on_a_mismatched_hash() -> None:
    with pytest.raises(ValueError, match="has changed"):
        verify_schema_unchanged("0" * 64)


def test_verify_schema_unchanged_checks_a_given_path(tmp_path: Path) -> None:
    schema_copy = tmp_path / "schema.json"
    schema_copy.write_text('{"different": true}')
    real_sha256 = hashlib.sha256(Path(DEFAULT_SCHEMA_PATH).read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="has changed"):
        verify_schema_unchanged(real_sha256, schema_path=schema_copy)


def test_verify_schema_unchanged_raises_a_clear_error_for_a_missing_schema_path(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "no_such_schema.json"
    with pytest.raises(FileNotFoundError, match="cannot verify AC-8"):
        verify_schema_unchanged("irrelevant", schema_path=missing)
