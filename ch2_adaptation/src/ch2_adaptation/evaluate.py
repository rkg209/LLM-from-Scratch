"""C5's head-to-head comparison: the fine-tuned adapter, scored on the frozen holdout by
the identical harness used for the base model and the frontier API (AC-1).

Requires `EVAL_CONTEXT=1` to read the holdout at all -- the leakage guard enforces this
regardless. Never scores anything ad hoc: `eval.harness.score_outputs` is the only path
from raw model output to a rate, same as `baseline.py`'s `score_system`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from ch2_adaptation.baseline import Generator, score_system
from eval.config import seed_everything
from eval.harness import DEFAULT_SCHEMA_PATH, EvalResult

if TYPE_CHECKING:
    from ch2_adaptation.config import EvaluateConfig

# eval_full.yaml's paired baseline run -- re-scores the base model and frontier API on
# the same holdout, per baseline_holdout.yaml. Its schema_sha256 is what a full adapter
# run checks against before trusting the comparison (AC-8).
_BASELINE_HOLDOUT_RESULTS_PATH = "eval/results/baselines_holdout.json"


def score_adapter_on_holdout(
    prompts: list[str], generate: Generator, holdout_path: Path | str
) -> EvalResult:
    """Generate against every holdout prompt and score through the shared harness.

    `generate` is injected (mirrors `baseline.py::score_system`'s shape) so this is
    testable with a fake -- no torch, no adapter load, no network. The real adapter
    generator (`make_adapter_generator`, loading `peft.PeftModel.from_pretrained`) is a
    separate function `main()` wires in; the fine-tuned model needs no few-shot examples
    to be prompted with, so it always uses `prompts.format_zero_shot` -- that is the point
    of fine-tuning.
    """
    return score_system(prompts, generate, holdout_path)


def verify_schema_unchanged(
    recorded_sha256: str, schema_path: Path | str = DEFAULT_SCHEMA_PATH
) -> None:
    """Raise if `eval/schema.json` has changed since the baselines were measured (AC-8).

    A schema that drifted between the baseline run and this one would make the comparison
    meaningless -- the systems would be scored against different contracts without anyone
    noticing.
    """
    schema_path = Path(schema_path)
    if not schema_path.exists():
        raise FileNotFoundError(
            f"cannot verify AC-8: {schema_path} does not exist -- check the path, don't "
            "guess whether the schema changed."
        )
    current_sha256 = hashlib.sha256(schema_path.read_bytes()).hexdigest()
    if current_sha256 != recorded_sha256:
        raise ValueError(
            f"eval/schema.json has changed since the baselines were recorded: expected "
            f"{recorded_sha256}, got {current_sha256}. Re-measure the baselines before "
            "comparing any system against a changed schema."
        )


# The exact row order and column set the eval-table skill locks (README.md's rendered
# table). Every sample counts in `n` (AC-2/AC-6) -- nothing gets excluded from the count.
_SYSTEM_LABELS = (
    ("Fine-tuned (QLoRA, Qwen2.5-Coder-1.5B)", "finetuned"),
    ("Base model (zero-shot)", "base"),
    ("Frontier API (3-shot)", "frontier"),
)

README_EVAL_TABLE_START = "<!-- EVAL_TABLE_START -->"
README_EVAL_TABLE_END = "<!-- EVAL_TABLE_END -->"


def render_table(finetuned: EvalResult, base: EvalResult, frontier: EvalResult) -> str:
    """The exact markdown shape the eval-table skill locks: one row per system, `n`
    always stated so a reader can judge the noise."""
    results = {"finetuned": finetuned, "base": base, "frontier": frontier}
    lines = ["| System | Schema-validity | Bug-catch | n |", "|---|---|---|---|"]
    for label, key in _SYSTEM_LABELS:
        result = results[key]
        lines.append(
            f"| {label} | {result.schema_validity_rate:.2f} | "
            f"{result.bug_catch_rate:.2f} | {result.n_samples} |"
        )
    return "\n".join(lines)


def update_readme_results_section(table_md: str, readme_path: Path | str) -> None:
    """Replace the content between the README's marker comments with `table_md`.

    Idempotent by construction: re-running `/eval` regenerates the section from the
    markers outward, rather than appending or drifting on repeated runs (AC-3, AC-5).
    """
    readme_path = Path(readme_path)
    content = readme_path.read_text()
    if content.count(README_EVAL_TABLE_START) != 1 or content.count(README_EVAL_TABLE_END) != 1:
        raise ValueError(
            f"{readme_path} must contain exactly one {README_EVAL_TABLE_START!r}/"
            f"{README_EVAL_TABLE_END!r} marker pair -- add it once, by hand, before this "
            "can update the section automatically."
        )
    if content.index(README_EVAL_TABLE_START) > content.index(README_EVAL_TABLE_END):
        raise ValueError(
            f"{readme_path}: {README_EVAL_TABLE_END!r} appears before "
            f"{README_EVAL_TABLE_START!r} -- the marker pair is malformed."
        )
    before, _, rest = content.partition(README_EVAL_TABLE_START)
    _, _, after = rest.partition(README_EVAL_TABLE_END)
    new_content = f"{before}{README_EVAL_TABLE_START}\n{table_md}\n{README_EVAL_TABLE_END}{after}"
    readme_path.write_text(new_content)


def make_adapter_generator(config: EvaluateConfig) -> Generator:
    """Load the base model, attach the trained adapter, return a `prompts -> raw
    responses` callable -- zero-shot, since the fine-tuned model needs no few-shot
    examples (that's the point of fine-tuning).
    """
    from peft import PeftModel

    from ch2_adaptation.model import generate_completions, load_base_model

    base_model, tokenizer = load_base_model(config.model_tag, config.use_4bit)
    model = PeftModel.from_pretrained(base_model, config.adapter_path, is_trainable=False)
    model.eval()

    def generate(prompts: list[str]) -> list[str]:
        return generate_completions(model, tokenizer, prompts, config.max_new_tokens)

    return generate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the fine-tuned adapter on the frozen holdout, zero-shot."
    )
    parser.add_argument("--config", type=Path, required=True, help="path to a YAML config")
    return parser.parse_args()


def main() -> None:
    from ch2_adaptation.config import load_evaluate_config
    from ch2_adaptation.prompts import format_zero_shot
    from eval.harness import load_holdout, write_json_atomic

    args = parse_args()
    config = load_evaluate_config(args.config)
    seed_everything(config.seed)
    print(f"[C5] config loaded: {args.config}")

    if not config.is_smoke:
        # The real run must not compare against a schema that drifted since the paired
        # baseline_holdout.yaml run measured the other two systems (AC-8).
        baselines = json.loads(Path(_BASELINE_HOLDOUT_RESULTS_PATH).read_text())
        verify_schema_unchanged(baselines["schema_sha256"])

    holdout_records = load_holdout(config.holdout_path)
    prompts = [format_zero_shot(record["code"]) for record in holdout_records]

    generate = make_adapter_generator(config)
    result = score_adapter_on_holdout(prompts, generate, config.holdout_path)

    doc = {
        "model_tag": config.model_tag,
        "adapter_path": config.adapter_path,
        "mode": "zero-shot",
        **result.to_json(),
    }
    write_json_atomic(doc, config.results_path)

    print(
        f"[C5] validity={result.schema_validity_rate:.2f} "
        f"catch={result.bug_catch_rate:.2f} n={result.n_samples}"
    )
    print(f"[C5] wrote {config.results_path}")


if __name__ == "__main__":
    main()
