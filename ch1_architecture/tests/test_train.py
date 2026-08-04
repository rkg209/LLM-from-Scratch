"""Training-loop tests (A3): loss behavior, checkpoint round-trip, LR curve, determinism."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from ch1_architecture.config import GPTConfig
from ch1_architecture.model.gpt import GPTModel
from ch1_architecture.train import (
    build_model,
    build_optimizer,
    build_scheduler,
    build_tokenizer_and_loader,
    lr_lambda,
    save_checkpoint,
    training_loop,
)


def _toy_config(tmp_path: Path, max_steps: int = 3) -> GPTConfig:
    return GPTConfig(
        vocab_size=256,
        d_model=32,
        n_heads=2,
        n_layers=2,
        seq_len=16,
        batch_size=2,
        max_steps=max_steps,
        learning_rate=1e-3,
        clip_grad_norm=1.0,
        log_every=1,
        seed=42,
        device="cpu",
        run_name="train-test",
        corpus_path="ch1_architecture/tests/fixtures/corpus_smoke.txt",
        tokenizer_path=str(tmp_path / "tokenizer.json"),
        warmup_steps=1,
        lr_min_ratio=0.1,
        ckpt_every=2,
        sample_every=2,
        sample_prompt="First",
        max_new_tokens=3,
        temperature=0.8,
        top_k=10,
        checkpoint_path=str(tmp_path / "model.pt"),
        wandb_project="ch1-architecture",
        wandb_mode="disabled",
    )


def _run(tmp_path: Path, max_steps: int = 3) -> tuple[GPTConfig, list[float]]:
    config = _toy_config(tmp_path, max_steps)
    tokenizer, loader = build_tokenizer_and_loader(config)
    model = build_model(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    losses = training_loop(model, loader, optimizer, scheduler, tokenizer, config)
    return config, losses


def test_loss_is_finite_and_moves(tmp_path: Path) -> None:
    _, losses = _run(tmp_path, max_steps=3)
    assert all(math.isfinite(loss) for loss in losses)
    assert len(losses) == 3


def test_checkpoint_round_trips_to_identical_logits(tmp_path: Path) -> None:
    config = _toy_config(tmp_path, max_steps=2)
    tokenizer, loader = build_tokenizer_and_loader(config)
    model = build_model(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    training_loop(model, loader, optimizer, scheduler, tokenizer, config)

    save_checkpoint(model, optimizer, config, step=1, tokenizer=tokenizer)
    checkpoint = torch.load(config.checkpoint_path, weights_only=False)

    reloaded = GPTModel(config)
    reloaded.load_state_dict(checkpoint["model_state"])

    probe = torch.randint(0, config.vocab_size, (1, 8))
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        assert torch.allclose(model(probe), reloaded(probe), atol=1e-6)


def test_lr_curve_matches_closed_form() -> None:
    warmup_steps, max_steps, lr_min_ratio = 5, 20, 0.1
    for step in [0, 2, 4, 5, 10, 19]:
        value = lr_lambda(step, warmup_steps, max_steps, lr_min_ratio)
        if step < warmup_steps:
            assert value == (step + 1) / warmup_steps
        else:
            progress = min(1.0, (step - warmup_steps) / (max_steps - warmup_steps))
            expected = lr_min_ratio + (1.0 - lr_min_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
            assert math.isclose(value, expected)


def test_same_seed_gives_bit_identical_losses(tmp_path: Path) -> None:
    from eval.config import seed_everything

    seed_everything(42)
    _, losses_a = _run(tmp_path / "a", max_steps=3)
    seed_everything(42)
    _, losses_b = _run(tmp_path / "b", max_steps=3)
    assert losses_a == losses_b
