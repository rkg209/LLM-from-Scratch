"""O3: `update_readme_section` -- write markdown between a pair of marker comments.

Generalized from `ch2_adaptation.evaluate.update_readme_results_section`, which does exactly
this for one hardcoded marker pair (`<!-- EVAL_TABLE_START/END -->`). This version takes the
marker as a parameter so `ch3_operation`'s serving scorecard
(`<!-- SERVING_METRICS_START/END -->`) can reuse it without duplicating the
find-replace-between-markers logic a second time. Ch2 is not migrated onto this in the same
change -- it already works, and O3's scope is the serving scorecard, not a refactor of C5.
"""

from __future__ import annotations

from pathlib import Path


def update_readme_section(readme_path: Path | str, marker: str, markdown: str) -> None:
    """Replace the content between `<!-- {marker}_START -->` and `<!-- {marker}_END -->`.

    Idempotent by construction: re-running regenerates the section from the markers
    outward, rather than appending or drifting on repeated runs.
    """
    start = f"<!-- {marker}_START -->"
    end = f"<!-- {marker}_END -->"

    readme_path = Path(readme_path)
    content = readme_path.read_text()
    if content.count(start) != 1 or content.count(end) != 1:
        raise ValueError(
            f"{readme_path} must contain exactly one {start!r}/{end!r} marker pair -- add it "
            "once, by hand, before this can update the section automatically."
        )
    if content.index(start) > content.index(end):
        raise ValueError(
            f"{readme_path}: {end!r} appears before {start!r} -- malformed marker pair."
        )

    before, _, rest = content.partition(start)
    _, _, after = rest.partition(end)
    new_content = f"{before}{start}\n{markdown}\n{end}{after}"
    readme_path.write_text(new_content)
