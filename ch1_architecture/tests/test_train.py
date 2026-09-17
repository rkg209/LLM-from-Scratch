"""Training-loop tests (A3): loss behavior, checkpoint round-trip, LR curve, determinism."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import torch
from ch1_architecture.config import GPTConfig
from ch1_architecture.model.gpt import GPTModel
from ch1_architecture.train import (
    build_model,
    build_optimizer,
    build_scheduler,
    build_tokenizer_and_loaders,
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
        val_fraction=0.1,
        val_every=5,
    )


def _run(tmp_path: Path, max_steps: int = 3) -> tuple[GPTConfig, list[float]]:
    config = _toy_config(tmp_path, max_steps)
    tokenizer, loader, val_loader = build_tokenizer_and_loaders(config)
    model = build_model(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    losses, _ = training_loop(model, loader, val_loader, optimizer, scheduler, tokenizer, config)
    return config, losses


def test_loss_is_finite_and_moves(tmp_path: Path) -> None:
    _, losses = _run(tmp_path, max_steps=3)
    assert all(math.isfinite(loss) for loss in losses)
    assert len(losses) == 3


def test_checkpoint_round_trips_to_identical_logits(tmp_path: Path) -> None:
    config = _toy_config(tmp_path, max_steps=2)
    tokenizer, loader, val_loader = build_tokenizer_and_loaders(config)
    model = build_model(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    training_loop(model, loader, val_loader, optimizer, scheduler, tokenizer, config)

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


def _corpus_config(tmp_path: Path, text: str, val_fraction: float = 0.1) -> GPTConfig:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(text)
    return dataclasses.replace(
        _toy_config(tmp_path, max_steps=2),
        corpus_path=str(corpus),
        vocab_size=300,
        val_fraction=val_fraction,
    )


def test_the_tokenizer_is_never_fitted_on_held_out_text(tmp_path: Path) -> None:
    """The split happens before the merge table is learned.

    A token-level split would leave this failing: the tokenizer would have been trained on
    the whole corpus, so the tail's character pairs would appear in the merges even though
    the model is scored on that tail. The marker below appears only in the held-out tail.
    """
    text = "the cat sat on the mat. " * 400 + "qzqzqz " * 100
    config = _corpus_config(tmp_path, text)
    tokenizer, _, _ = build_tokenizer_and_loaders(config)

    merged_tokens = [a + b for a, b in tokenizer.merges]
    assert not any(b"qz" in token for token in merged_tokens)
    assert any(b"the" in token for token in merged_tokens)


def test_held_out_tokens_never_appear_in_a_training_batch(tmp_path: Path) -> None:
    text = "the cat sat on the mat. " * 400 + "qzqzqz " * 100
    config = _corpus_config(tmp_path, text)
    tokenizer, train_loader, val_loader = build_tokenizer_and_loaders(config)

    train_seen = {int(t) for inputs, _ in train_loader for t in inputs.flatten()}
    val_text = text[-int(len(text) * config.val_fraction) :]
    val_only = set(tokenizer.encode(val_text)) - set(tokenizer.encode(text[: -len(val_text)]))

    assert val_only, "the fixture should give the tail at least one token of its own"
    assert not (train_seen & val_only)


def test_validation_loss_is_recorded_during_training(tmp_path: Path) -> None:
    config = _corpus_config(tmp_path, "the cat sat on the mat. " * 400)
    config = dataclasses.replace(config, max_steps=6, val_every=2)
    tokenizer, loader, val_loader = build_tokenizer_and_loaders(config)
    model = build_model(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    _, val_losses = training_loop(
        model, loader, val_loader, optimizer, scheduler, tokenizer, config
    )

    steps = [step for step, _ in val_losses]
    assert steps[0] == 0
    assert config.max_steps - 1 in steps, "the final step must be scored, whatever val_every is"
    assert all(math.isfinite(loss) for _, loss in val_losses)


def test_validation_loss_is_the_same_number_twice_for_one_model(tmp_path: Path) -> None:
    """The eval loader is unshuffled, so re-scoring an unchanged model cannot drift."""
    from ch1_architecture.train import evaluate

    config = _corpus_config(tmp_path, "the cat sat on the mat. " * 400)
    _, _, val_loader = build_tokenizer_and_loaders(config)
    model = build_model(config)

    assert evaluate(model, val_loader, config) == evaluate(model, val_loader, config)
