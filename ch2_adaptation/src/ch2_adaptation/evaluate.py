"""C5's head-to-head comparison: the fine-tuned adapter, scored on the frozen holdout by
the identical harness used for the base model and the frontier API (AC-1).

Requires `EVAL_CONTEXT=1` to read the holdout at all -- the leakage guard enforces this
regardless. Never scores anything ad hoc: `eval.harness.score_outputs` is the only path
from raw model output to a rate, same as `baseline.py`'s `score_system`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ch2_adaptation.baseline import Generator, score_system
from eval.harness import DEFAULT_SCHEMA_PATH, EvalResult


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
