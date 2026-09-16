"""Score the base model zero-shot and a frontier API 3-shot on the stub set (spec C1).

Both numbers are frozen and committed to `eval/results/baselines.json` *before* a single
fine-tuning token is spent (CON-11) — otherwise the comparison is measured by someone who
already knows what number they need.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ch2_adaptation.config import BaselineConfig, load_baseline_config
from eval.config import seed_everything
from eval.harness import (
    DEFAULT_SCHEMA_PATH,
    EvalResult,
    load_holdout,
    score_outputs,
    write_json_atomic,
)

# A prompt in, a raw model response out — one call per prompt, in order. `baseline.py::main`
# wires this to model.make_hf_generator / frontier.make_client; tests drive it with a fake.
Generator = Callable[[list[str]], list[str]]


@dataclass(frozen=True)
class Usage:
    """Token usage and recorded cost for one frontier-API scoring run (AC-7).

    On a free tier the honest recorded cost is $0.00, but the token counts are kept
    alongside it — they are what a paid re-run would cost, and hiding them behind a bare
    zero would make the budget line unverifiable.
    """

    input_tokens: int
    output_tokens: int
    cost_usd: float
    tier: str


def score_system(prompts: list[str], generate: Generator, stub_path: Path | str) -> EvalResult:
    """Generate against every prompt and score the outputs through the shared harness.

    Never ad-hoc (AC-3, NFR-7): the only path from raw output to a rate is
    `eval.harness.score_outputs`, so every system — base, frontier, fine-tuned — is scored
    identically.
    """
    outputs = generate(prompts)
    return score_outputs(outputs, stub_path)


def _sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_baselines_doc(
    base: EvalResult,
    frontier: EvalResult,
    cfg: BaselineConfig,
    usage: Usage,
) -> dict[str, Any]:
    """Assemble the `baselines.json` payload.

    `stub_set_sha256` and `schema_sha256` are what make a committed baselines.json
    re-checkable later (X2): if the stub set or schema changes under it, that is
    detectable rather than silent.
    """
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "stub_set": str(cfg.stub_set_path),
        "stub_set_sha256": _sha256_file(cfg.stub_set_path),
        "schema_sha256": _sha256_file(DEFAULT_SCHEMA_PATH),
        "base_model": {
            "model_tag": cfg.base_model_tag,
            "mode": "zero-shot",
            "dtype": "int4" if cfg.use_4bit else "float32",
            **base.to_json(),
        },
        "frontier_api": {
            "model_tag": cfg.frontier_model_tag,
            "mode": f"{cfg.n_few_shot}-shot",
            "provider": cfg.frontier_provider,
            "tier": usage.tier,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": usage.cost_usd,
            **frontier.to_json(),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the base-model and frontier-API baselines on the stub set."
    )
    parser.add_argument("--config", type=Path, required=True, help="path to a YAML config")
    return parser.parse_args()


def _log_to_wandb(cfg: BaselineConfig, base: EvalResult, frontier: EvalResult) -> None:
    """Best-effort W&B logging: never fail a run whose real measurement already
    succeeded and was written to disk just because the *optional* observability step
    couldn't reach W&B (no login, offline machine, revoked key, ...). Reproduced live:
    `wandb.init(mode="online", ...)` raises `CommError: user is not logged in` when the
    environment has no `WANDB_API_KEY` and the interactive prompt picked "don't
    visualize" -- that crashed the whole script *after* `baselines.json` was already
    written, making a successful run look like a failure.
    """
    if cfg.wandb_mode == "disabled":
        return

    import wandb

    try:
        run = wandb.init(project=cfg.wandb_project, mode=cfg.wandb_mode, config=cfg.__dict__)
        wandb.log(
            {
                "base/schema_validity_rate": base.schema_validity_rate,
                "base/bug_catch_rate": base.bug_catch_rate,
                "frontier/schema_validity_rate": frontier.schema_validity_rate,
                "frontier/bug_catch_rate": frontier.bug_catch_rate,
            }
        )
        run.finish()
    except wandb.Error as error:
        print(f"[C1] W&B logging skipped ({error}) -- the measurement above is unaffected.")


def _score_base(cfg: BaselineConfig, prompts: list[str]) -> EvalResult:
    # model.make_hf_generator imports torch/transformers function-locally; importing
    # ch2_adaptation.model here, not at baseline.py's top level, is what keeps `pytest -q`
    # green in the base+dev CI env, which never installs the ch2 extra.
    from ch2_adaptation.model import make_hf_generator

    print(f"[C1] base model: {cfg.base_model_tag} (zero-shot)")
    return score_system(prompts, make_hf_generator(cfg), cfg.stub_set_path)


def _score_frontier(cfg: BaselineConfig, prompts: list[str]) -> tuple[EvalResult, Usage]:
    # Deferred for the same reason as _score_base, and to avoid a baseline.py <-> frontier.py
    # import cycle: frontier.py imports Usage from this module at its own top level.
    from ch2_adaptation.frontier import make_client

    print(
        f"[C1] frontier: {cfg.frontier_model_tag} ({cfg.n_few_shot}-shot, {cfg.frontier_provider})"
    )
    client = make_client(
        cfg.frontier_provider, cfg.frontier_model_tag, cfg.temperature, cfg.max_new_tokens
    )
    usage_holder: list[Usage] = []

    def frontier_generate(batch: list[str]) -> list[str]:
        outputs, usage = client.generate(batch)
        usage_holder.append(usage)
        return outputs

    result = score_system(prompts, frontier_generate, cfg.stub_set_path)
    return result, usage_holder[0]


def main() -> None:
    from ch2_adaptation.prompts import format_three_shot, format_zero_shot

    args = parse_args()
    cfg = load_baseline_config(args.config)
    seed_everything(cfg.seed)
    print(f"[C1] config loaded: {args.config}")

    stub_records = load_holdout(cfg.stub_set_path)
    zero_shot_prompts = [format_zero_shot(record["code"]) for record in stub_records]
    three_shot_prompts = [format_three_shot(record["code"]) for record in stub_records]

    base_result = _score_base(cfg, zero_shot_prompts)
    frontier_result, usage = _score_frontier(cfg, three_shot_prompts)

    doc = build_baselines_doc(base_result, frontier_result, cfg, usage)
    write_json_atomic(doc, cfg.results_path)

    print(
        f"[C1] base   : validity={base_result.schema_validity_rate:.2f} "
        f"catch={base_result.bug_catch_rate:.2f}"
    )
    print(
        f"[C1] frontier: validity={frontier_result.schema_validity_rate:.2f} "
        f"catch={frontier_result.bug_catch_rate:.2f}"
    )
    print(f"[C1] wrote {cfg.results_path}")

    _log_to_wandb(cfg, base_result, frontier_result)


if __name__ == "__main__":
    main()
