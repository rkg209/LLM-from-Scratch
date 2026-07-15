"""finetune.py's smoke path, end to end: load -> attach LoRA -> train -> save adapter,
against the real cached SMOKE_MODEL_TAG (spec C4, task 5).

Skipped in the base+dev CI environment (no torch/transformers/peft installed) via
pytest.importorskip -- exercised locally and whenever the ch2 extra is present.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from ch2_adaptation.config import load_finetune_config  # noqa: E402
from ch2_adaptation.finetune import _build_trainer  # noqa: E402

CONFIGS = Path("ch2_adaptation/configs")


def test_smoke_finetune_trains_and_saves_an_adapter_under_the_time_budget(
    tmp_path: Path,
) -> None:
    cfg = load_finetune_config(CONFIGS / "smoke.yaml")
    cfg = replace(cfg, output_dir=str(tmp_path / "adapter-smoke"))

    started = time.monotonic()
    trainer, peft_model = _build_trainer(cfg)
    trainer.train()
    peft_model.save_pretrained(cfg.output_dir)
    elapsed = time.monotonic() - started

    assert elapsed < 60  # comfortable margin under NFR-1's under-120s smoke budget
    adapter_dir = Path(cfg.output_dir)
    assert (adapter_dir / "adapter_config.json").exists()
    assert (adapter_dir / "adapter_model.safetensors").exists()


def test_smoke_finetune_train_loss_is_finite(tmp_path: Path) -> None:
    cfg = load_finetune_config(CONFIGS / "smoke.yaml")
    cfg = replace(cfg, output_dir=str(tmp_path / "adapter-smoke"))

    trainer, _ = _build_trainer(cfg)
    result = trainer.train()

    assert result.training_loss == result.training_loss  # not NaN
    assert result.training_loss < 1000  # sanity bound, not a specific target value


def test_smoke_finetune_actually_runs_validation_during_train(tmp_path: Path) -> None:
    """config.val_path must be exercised automatically during training, not just loaded
    and left unused -- a prior cut of _build_trainer wired eval_dataset in but never set
    an eval_strategy, so no eval_loss was ever computed or logged. Checking
    trainer.train()'s own log history (rather than calling trainer.evaluate() directly,
    which would pass even without an eval_strategy) is what actually verifies this."""
    cfg = load_finetune_config(CONFIGS / "smoke.yaml")
    cfg = replace(cfg, output_dir=str(tmp_path / "adapter-smoke"))

    trainer, _ = _build_trainer(cfg)
    trainer.train()

    eval_entries = [entry for entry in trainer.state.log_history if "eval_loss" in entry]
    assert eval_entries, "training completed without ever logging eval_loss"
    assert eval_entries[-1]["eval_loss"] == eval_entries[-1]["eval_loss"]  # not NaN
