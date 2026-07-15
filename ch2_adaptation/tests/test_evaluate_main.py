"""make_adapter_generator and evaluate.py::main(), end to end against the real cached
SMOKE_MODEL_TAG and a freshly-trained smoke adapter (spec C5, task 5).

Skipped in the base+dev CI environment (no torch/transformers/peft installed) via
pytest.importorskip -- exercised locally and whenever the ch2 extra is present.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from ch2_adaptation.config import load_evaluate_config, load_finetune_config  # noqa: E402
from ch2_adaptation.evaluate import make_adapter_generator, score_adapter_on_holdout  # noqa: E402
from ch2_adaptation.finetune import _build_trainer  # noqa: E402

CONFIGS = Path("ch2_adaptation/configs")


@pytest.fixture
def smoke_adapter_dir(tmp_path: Path) -> Path:
    """Train the real smoke recipe for a few steps and save the adapter -- reuses C4's
    already-tested training loop rather than hand-rolling a second one just for fixtures."""
    finetune_cfg = load_finetune_config(CONFIGS / "smoke.yaml")
    output_dir = tmp_path / "adapter-smoke"
    finetune_cfg = replace(finetune_cfg, output_dir=str(output_dir))

    trainer, peft_model = _build_trainer(finetune_cfg)
    trainer.train()
    peft_model.save_pretrained(str(output_dir))
    return output_dir


def test_make_adapter_generator_produces_one_output_per_prompt(smoke_adapter_dir: Path) -> None:
    cfg = load_evaluate_config(CONFIGS / "eval_smoke.yaml")
    cfg = replace(cfg, adapter_path=str(smoke_adapter_dir), max_new_tokens=4)

    generate = make_adapter_generator(cfg)
    outputs = generate(["review this code", "and this code too"])

    assert len(outputs) == 2
    assert all(isinstance(output, str) for output in outputs)


def test_score_adapter_on_holdout_writes_a_real_result(
    smoke_adapter_dir: Path, tmp_path: Path
) -> None:
    cfg = load_evaluate_config(CONFIGS / "eval_smoke.yaml")
    cfg = replace(cfg, adapter_path=str(smoke_adapter_dir), max_new_tokens=4)

    generate = make_adapter_generator(cfg)
    prompts = ["p1", "p2", "p3"]
    result = score_adapter_on_holdout(prompts, generate, cfg.holdout_path)

    assert result.n_samples == 3
    assert 0.0 <= result.schema_validity_rate <= 1.0
    assert 0.0 <= result.bug_catch_rate <= 1.0


def test_smoke_evaluate_main_runs_end_to_end(
    smoke_adapter_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actual CLI path: parse args, load config, score, write results -- mirrors
    `uv run python -m ch2_adaptation.evaluate --config .../eval_smoke.yaml`."""
    import yaml

    from ch2_adaptation import evaluate

    raw_config = yaml.safe_load((CONFIGS / "eval_smoke.yaml").read_text())
    raw_config["adapter_path"] = str(smoke_adapter_dir)
    raw_config["max_new_tokens"] = 4
    results_path = tmp_path / "finetuned-smoke.json"
    raw_config["results_path"] = str(results_path)
    config_path = tmp_path / "eval_smoke.yaml"
    config_path.write_text(yaml.safe_dump(raw_config))

    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--config", str(config_path)])
    evaluate.main()

    doc = json.loads(results_path.read_text())
    assert doc["adapter_path"] == str(smoke_adapter_dir)
    assert doc["mode"] == "zero-shot"
    assert "schema_validity_rate" in doc
    assert "n_samples" in doc
