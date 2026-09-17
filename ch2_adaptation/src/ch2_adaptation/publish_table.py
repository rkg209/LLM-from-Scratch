"""Re-publish the README's head-to-head table from the committed result artifacts.

C5 AC-5: "`/eval` reproduces the table from the saved artifacts". That has to be a code
path, not an instruction someone follows by hand -- a table re-typed from a JSON file is
exactly the step where a number silently drifts from the run that produced it.

This module loads nothing, generates nothing, and scores nothing. It reads two finished
JSON files and renders `evaluate.py`'s table into `README.md`. It never touches the
holdout, so unlike `evaluate.py` it runs anywhere, with no GPU and no `EVAL_CONTEXT=1`.

    uv run python -m ch2_adaptation.publish_table --config ch2_adaptation/configs/publish_table.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from ch2_adaptation.evaluate import render_table, update_readme_results_section
from eval.harness import EvalResult

if TYPE_CHECKING:
    pass


def _as_result(doc: dict) -> EvalResult:
    """Rebuild an `EvalResult` from a results-file section.

    Deliberately strict: a missing key raises rather than defaulting to 0, because a
    silently-zeroed rate is the one failure mode that would publish a wrong table while
    looking completely normal.
    """
    return EvalResult(
        schema_validity_rate=doc["schema_validity_rate"],
        bug_catch_rate=doc["bug_catch_rate"],
        n_samples=doc["n_samples"],
        n_valid=doc["n_valid"],
        n_caught=doc["n_caught"],
        per_sample=doc["per_sample"],
    )


def verify_same_holdout(finetuned: dict, baselines: dict) -> None:
    """Refuse to publish a table whose rows were scored on different inputs (AC-1).

    Both files record the hashes they ran against. If either the holdout or the schema
    differs between them, the rows are not comparable and the table would be a lie told
    in a format that looks authoritative. Files written before those fields existed
    (`finetuned.json` from Rudra job 401990) are skipped rather than guessed at.
    """
    for field, baseline_field in (("holdout_sha256", "stub_set_sha256"), ("schema_sha256",) * 2):
        theirs = baselines.get(baseline_field)
        ours = finetuned.get(field)
        if ours is not None and theirs is not None and ours != theirs:
            raise ValueError(
                f"{field} differs between the two results files: fine-tuned ran against "
                f"{ours}, the baselines against {theirs}. These rows are not comparable — "
                "re-run both systems before publishing a table from them."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-render the README eval table from the committed result artifacts."
    )
    parser.add_argument("--config", type=Path, required=True, help="path to a YAML config")
    return parser.parse_args()


def main() -> None:
    from ch2_adaptation.config import load_publish_table_config

    args = parse_args()
    config = load_publish_table_config(args.config)

    finetuned = json.loads(Path(config.finetuned_path).read_text())
    baselines = json.loads(Path(config.baselines_path).read_text())
    verify_same_holdout(finetuned, baselines)

    table = render_table(
        _as_result(finetuned),
        _as_result(baselines["base_model"]),
        _as_result(baselines["frontier_api"]),
    )
    update_readme_results_section(table, config.readme_path)
    print(table)
    print(f"[C5] republished the eval table into {config.readme_path}")


if __name__ == "__main__":
    main()
