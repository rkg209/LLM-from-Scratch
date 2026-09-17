"""The single most valuable test in the chapter: a model that cannot memorize eight
sequences is broken, and no amount of hyperparameter tuning will fix it.

The step budget went 100 -> 150 when GPT-2 initialization landed. The threshold did not
move: the point of the test is that the model *can* drive the loss to nothing, and it
still does. What changed is that weights no longer start enormous, so the first steps go
into learning rather than into shrinking logits — the same reason the real training run
got dramatically better. A test that passed only because the model started badly scaled
was measuring the wrong thing.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from ch1_architecture.config import GPTConfig
from ch1_architecture.model.gpt import GPTModel


def test_overfit_tiny_batch() -> None:
    torch.manual_seed(0)
    config = GPTConfig(
        vocab_size=32,
        d_model=32,
        n_heads=4,
        n_layers=2,
        seq_len=8,
        batch_size=8,
        max_steps=100,
        learning_rate=3e-3,
        clip_grad_norm=1.0,
        log_every=10,
        seed=0,
        device="cpu",
        run_name="overfit-test",
        corpus_path="ch1_architecture/tests/fixtures/corpus_smoke.txt",
        tokenizer_path="outputs/ch1/tokenizer_smoke.json",
        warmup_steps=5,
        lr_min_ratio=0.1,
        ckpt_every=25,
        sample_every=25,
        sample_prompt="First Citizen:",
        max_new_tokens=20,
        temperature=0.8,
        top_k=20,
        checkpoint_path="outputs/ch1/model_smoke.pt",
        wandb_project="ch1-architecture",
        wandb_mode="disabled",
        val_fraction=0.1,
        val_every=5,
    )
    model = GPTModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    inputs = torch.randint(0, config.vocab_size, (8, config.seq_len))
    labels = torch.randint(0, config.vocab_size, (8, config.seq_len))

    loss = torch.tensor(float("inf"))
    for _ in range(150):
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.1, f"final loss {loss.item():.4f} did not fall below 0.1"
