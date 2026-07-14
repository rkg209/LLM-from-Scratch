"""Score the base model zero-shot and a frontier API 3-shot on the stub set (spec C1).

Both numbers are frozen and committed to `eval/results/baselines.json` *before* a single
fine-tuning token is spent (CON-11) — otherwise the comparison is measured by someone who
already knows what number they need.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ch2_adaptation.config import BaselineConfig
from eval.harness import DEFAULT_SCHEMA_PATH, EvalResult, score_outputs

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
