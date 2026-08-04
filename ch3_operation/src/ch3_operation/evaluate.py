"""O1 T5: score a served GGUF on the holdout, through the exact serving path.

`python -m ch3_operation.evaluate --config ...` builds a real `LlamaCppBackend`, renders
prompts with `ch3_operation.prompts.format_zero_shot` (the serving prompt, not ch2's --
the point is to measure what a caller actually gets), scores with the shared
`eval.harness.score_outputs`, and fails loudly if the quantized model's schema-validity
rate is below the adapter's published rate in `compare_to`.

Requires `EVAL_CONTEXT=1` in the environment whenever `holdout_path` points at
`eval/holdout/` -- the leakage guard enforces this at the tool layer regardless; this
module does not additionally check for it.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from ch3_operation.config import EvalConfig, ServeConfig, load_eval_config
from ch3_operation.model_backend import Backend, LlamaCppBackend
from ch3_operation.prompts import format_zero_shot
from eval.config import seed_everything
from eval.harness import EvalResult, load_holdout, score_outputs, write_result


class QuantizationRegressionError(RuntimeError):
    """The quantized GGUF scored worse than the model it was quantized from."""


def _serve_config_for_eval(config: EvalConfig) -> ServeConfig:
    """`LlamaCppBackend` takes a `ServeConfig`; host/port/metrics_window are unused for an
    offline eval run, so they're filled with placeholders rather than adding a second
    dataclass shape for the backend to accept."""
    return ServeConfig(
        model_path=str(config.resolved_model_path()),
        n_ctx=config.n_ctx,
        n_threads=config.n_threads,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        host="127.0.0.1",
        port=0,
        metrics_window=1,
        max_request_bytes=32768,
        api_version="v1",
    )


def generate_outputs(backend: Backend, holdout: list[dict], max_tokens: int) -> list[str]:
    outputs = []
    for record in holdout:
        prompt = format_zero_shot(record["code"])
        outputs.append(backend.generate(prompt, max_tokens))
    return outputs


def score_sample(outputs: list[str], sample: list[dict]) -> EvalResult:
    """Score `outputs` against `sample` through the shared harness.

    `score_outputs` reads its holdout from a file path, not a list -- `sample` (already a
    possibly-truncated slice of the real holdout, per `n_samples`) is written to a scratch
    JSONL file so scoring still goes through the one shared harness rather than a
    hand-rolled copy of `is_schema_valid`/`is_bug_caught` here.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as scratch:
        for record in sample:
            scratch.write(json.dumps(record) + "\n")
        scratch_path = Path(scratch.name)
    try:
        return score_outputs(outputs, scratch_path)
    finally:
        scratch_path.unlink()


def load_compare_rate(compare_to: Path | str) -> float:
    path = Path(compare_to)
    if not path.exists():
        raise FileNotFoundError(
            f"compare_to result not found: {path} -- run the adapter's own evaluate step "
            "first, or point compare_to at whichever published rate this GGUF must clear."
        )
    return json.loads(path.read_text())["schema_validity_rate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a served GGUF on the holdout via the real serving path."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_eval_config(args.config)
    seed_everything(config.seed)
    print(f"[O1 evaluate] config loaded: {args.config}")

    sample = load_holdout(config.holdout_path)[: config.n_samples]
    backend = LlamaCppBackend(_serve_config_for_eval(config))
    outputs = generate_outputs(backend, sample, config.max_tokens)

    result: EvalResult = score_sample(outputs, sample)
    write_result(result, config.results_path)
    print(
        f"[O1 evaluate] validity={result.schema_validity_rate:.2f} "
        f"catch={result.bug_catch_rate:.2f} n={result.n_samples}"
    )
    print(f"[O1 evaluate] wrote {config.results_path}")

    compare_rate = load_compare_rate(config.compare_to)
    if result.schema_validity_rate < compare_rate:
        raise QuantizationRegressionError(
            f"quantized model's schema-validity ({result.schema_validity_rate:.2f}) is below "
            f"{config.compare_to}'s ({compare_rate:.2f}) -- AC-4's bar was not cleared. Walk "
            "the fallback ladder (Q5_K_M, Q8_0) before publishing this GGUF."
        )


if __name__ == "__main__":
    main()
