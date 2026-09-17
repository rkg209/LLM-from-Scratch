"""X2 AC-5: same config + same seed -> identical results for deterministic ops.

Two levels, cheap to expensive: `seed_everything` itself, then a full ch1 smoke
training run reproduced end to end. Both are CPU-only and skip cleanly without torch
so the base-env test run (no `--extra ch1`) stays green, same as the rest of ch1's
tests.
"""

from __future__ import annotations

import random

import pytest

from eval.config import seed_everything

torch = pytest.importorskip("torch")


def test_seed_everything_makes_draws_reproducible() -> None:
    seed_everything(123)
    random_draw_a = random.random()
    torch_draw_a = torch.rand(3)

    seed_everything(123)
    random_draw_b = random.random()
    torch_draw_b = torch.rand(3)

    assert random_draw_a == random_draw_b
    assert torch.equal(torch_draw_a, torch_draw_b)


def test_ch1_smoke_train_is_bit_identical_across_runs(tmp_path) -> None:
    from ch1_architecture.config import load_gpt_config
    from ch1_architecture.train import (
        build_model,
        build_optimizer,
        build_scheduler,
        build_tokenizer_and_loaders,
        training_loop,
    )

    base_config = load_gpt_config("ch1_architecture/configs/smoke.yaml")

    def run(tag: str) -> list[float]:
        config = base_config.__class__(
            **{
                **{f: getattr(base_config, f) for f in base_config.__dataclass_fields__},
                "max_steps": 5,
                "tokenizer_path": str(tmp_path / f"tokenizer_{tag}.json"),
                "checkpoint_path": str(tmp_path / f"model_{tag}.pt"),
            }
        )
        seed_everything(config.seed)
        tokenizer, loader, val_loader = build_tokenizer_and_loaders(config)
        model = build_model(config)
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)
        losses, _ = training_loop(
            model, loader, val_loader, optimizer, scheduler, tokenizer, config
        )
        return losses

    losses_a = run("a")
    losses_b = run("b")

    assert losses_a == losses_b
