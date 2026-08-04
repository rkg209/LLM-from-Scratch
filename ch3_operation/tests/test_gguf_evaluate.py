"""O1 T5: `ch3_operation.evaluate` scores a served GGUF through the real serving path.

No real llama.cpp model is loaded here -- `LlamaCppBackend` is monkeypatched with a fake
that returns canned output, same shape as `test_api.py`'s `FakeBackend`.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import ch3_operation.evaluate as evaluate
import pytest
from ch3_operation.config import load_eval_config

CONFIGS = Path("ch3_operation/configs")


class FakeBackend:
    def __init__(self, config) -> None:  # noqa: ANN001 - mirrors LlamaCppBackend's signature
        self.config = config

    def generate(self, prompt: str, max_tokens: int) -> str:
        return (
            '{"severity": "minor", "category": "x", "line": 1, "issue": "y", "suggested_fix": "z"}'
        )


def test_eval_config_loads_smoke_and_full() -> None:
    smoke = load_eval_config(CONFIGS / "eval_smoke.yaml")
    full = load_eval_config(CONFIGS / "eval_full.yaml")
    assert smoke.is_smoke
    assert not full.is_smoke


def test_generate_outputs_calls_backend_once_per_holdout_record() -> None:
    class RecordingBackend:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt: str, max_tokens: int) -> str:
            self.calls += 1
            return "raw"

    holdout = [{"code": "a();"}, {"code": "b();"}]
    backend = RecordingBackend()

    outputs = evaluate.generate_outputs(backend, holdout, max_tokens=16)

    assert outputs == ["raw", "raw"]
    assert backend.calls == 2


def test_score_sample_matches_the_shared_harness() -> None:
    sample = [{"id": "s1", "code": "a();", "line": 1}]
    valid_output = [
        '{"severity": "minor", "category": "x", "line": 1, "issue": "y", ' '"suggested_fix": "z"}'
    ]

    result = evaluate.score_sample(valid_output, sample)

    assert result.schema_validity_rate == 1.0
    assert result.n_samples == 1


def test_load_compare_rate_reads_the_published_number() -> None:
    rate = evaluate.load_compare_rate("ch3_operation/tests/fixtures/eval_smoke_compare.json")
    assert rate == 0.0


def test_load_compare_rate_requires_the_file_to_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        evaluate.load_compare_rate(tmp_path / "missing.json")


def test_main_writes_results_and_passes_when_compare_rate_is_cleared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evaluate, "LlamaCppBackend", FakeBackend)
    results_path = tmp_path / "gguf_eval.json"

    config = load_eval_config(CONFIGS / "eval_smoke.yaml")
    config = replace(config, results_path=str(results_path))
    monkeypatch.setattr(evaluate, "load_eval_config", lambda _path: config)

    class Args:
        config = CONFIGS / "eval_smoke.yaml"

    monkeypatch.setattr(evaluate, "parse_args", lambda: Args())

    evaluate.main()

    assert results_path.exists()


def test_main_raises_on_a_quantization_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evaluate, "LlamaCppBackend", FakeBackend)
    results_path = tmp_path / "gguf_eval.json"
    compare_path = tmp_path / "compare.json"
    compare_path.write_text('{"schema_validity_rate": 1.0}')

    config = load_eval_config(CONFIGS / "eval_smoke.yaml")
    config = replace(config, results_path=str(results_path), compare_to=str(compare_path))
    monkeypatch.setattr(evaluate, "load_eval_config", lambda _path: config)

    class Args:
        config = CONFIGS / "eval_smoke.yaml"

    monkeypatch.setattr(evaluate, "parse_args", lambda: Args())

    class ZeroValidityBackend(FakeBackend):
        def generate(self, prompt: str, max_tokens: int) -> str:
            return "not json"

    monkeypatch.setattr(evaluate, "LlamaCppBackend", ZeroValidityBackend)

    with pytest.raises(evaluate.QuantizationRegressionError):
        evaluate.main()
