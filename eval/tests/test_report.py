"""`eval.report.update_readme_section` -- generic marker-pair replacement."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.report import update_readme_section


def test_replaces_content_between_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("before\n<!-- FOO_START -->\nold\n<!-- FOO_END -->\nafter")

    update_readme_section(readme, "FOO", "new content")

    assert readme.read_text() == "before\n<!-- FOO_START -->\nnew content\n<!-- FOO_END -->\nafter"


def test_is_idempotent_across_repeated_runs(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("<!-- FOO_START -->\nold\n<!-- FOO_END -->")

    update_readme_section(readme, "FOO", "v1")
    update_readme_section(readme, "FOO", "v2")

    assert readme.read_text() == "<!-- FOO_START -->\nv2\n<!-- FOO_END -->"


def test_raises_if_markers_are_missing(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("no markers here")

    with pytest.raises(ValueError, match="marker pair"):
        update_readme_section(readme, "FOO", "content")


def test_raises_if_markers_are_out_of_order(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("<!-- FOO_END -->\n<!-- FOO_START -->")

    with pytest.raises(ValueError, match="malformed"):
        update_readme_section(readme, "FOO", "content")
