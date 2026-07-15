"""score_adapter_on_holdout and verify_schema_unchanged, driven by fakes -- no torch, no
network (spec C5, task 2)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from ch2_adaptation.evaluate import (
    render_table,
    score_adapter_on_holdout,
    update_readme_results_section,
    verify_schema_unchanged,
)

from eval.harness import DEFAULT_SCHEMA_PATH, EvalResult

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


# The eval-table skill's own literal example table, byte-for-byte -- render_table's
# output must match this shape exactly, not just "look similar."
_SKILL_EXAMPLE_TABLE = (
    "| System | Schema-validity | Bug-catch | n |\n"
    "|---|---|---|---|\n"
    "| Fine-tuned (QLoRA, Qwen2.5-Coder-1.5B) | 0.94 | 0.61 | 120 |\n"
    "| Base model (zero-shot) | 0.38 | 0.44 | 120 |\n"
    "| Frontier API (3-shot) | 0.97 | 0.72 | 120 |"
)


def test_render_table_matches_the_eval_table_skills_example_exactly() -> None:
    finetuned = EvalResult(0.94, 0.61, 120, 113, 73, [])
    base = EvalResult(0.38, 0.44, 120, 46, 53, [])
    frontier = EvalResult(0.97, 0.72, 120, 116, 86, [])

    table = render_table(finetuned, base, frontier)

    assert table == _SKILL_EXAMPLE_TABLE


def test_render_table_states_n_for_every_row() -> None:
    result = EvalResult(1.0, 1.0, 7, 7, 7, [])
    table = render_table(result, result, result)

    assert table.count("| 7 |") == 3


def test_update_readme_results_section_replaces_between_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Title\n\nBefore.\n\n<!-- EVAL_TABLE_START -->\nold table\n"
        "<!-- EVAL_TABLE_END -->\n\nAfter.\n"
    )

    update_readme_results_section("new table", readme)

    content = readme.read_text()
    assert "old table" not in content
    assert "new table" in content
    assert content.startswith("# Title\n\nBefore.\n\n<!-- EVAL_TABLE_START -->\nnew table\n")
    assert content.endswith("<!-- EVAL_TABLE_END -->\n\nAfter.\n")


def test_update_readme_results_section_is_idempotent(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("<!-- EVAL_TABLE_START -->\nfirst\n<!-- EVAL_TABLE_END -->\n")

    update_readme_results_section("second", readme)
    update_readme_results_section("third", readme)

    content = readme.read_text()
    assert content.count("<!-- EVAL_TABLE_START -->") == 1
    assert "second" not in content
    assert "third" in content


def test_update_readme_results_section_raises_if_markers_missing(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# No markers here\n")

    with pytest.raises(ValueError, match="marker pair"):
        update_readme_results_section("table", readme)


def test_update_readme_results_section_raises_on_duplicate_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "<!-- EVAL_TABLE_START -->\na\n<!-- EVAL_TABLE_END -->\n"
        "<!-- EVAL_TABLE_START -->\nb\n<!-- EVAL_TABLE_END -->\n"
    )

    with pytest.raises(ValueError, match="exactly one"):
        update_readme_results_section("table", readme)


def test_update_readme_results_section_raises_if_markers_are_reversed(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("<!-- EVAL_TABLE_END -->\n<!-- EVAL_TABLE_START -->\n")

    with pytest.raises(ValueError, match="malformed"):
        update_readme_results_section("table", readme)
