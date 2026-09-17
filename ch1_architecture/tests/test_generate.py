"""Uncached generation: shapes, determinism at top_k=1, and the use_cache=True stub (A3)."""

from __future__ import annotations

import torch
from ch1_architecture.config import GPTConfig
from ch1_architecture.generate import _generate_uncached, generate
from ch1_architecture.model.gpt import GPTModel


def _config() -> GPTConfig:
    return GPTConfig(
        vocab_size=32,
        d_model=16,
        n_heads=2,
        n_layers=2,
        seq_len=20,
        batch_size=2,
        max_steps=10,
        learning_rate=1e-3,
        clip_grad_norm=1.0,
        log_every=10,
        seed=42,
        device="cpu",
        run_name="gen-test",
        corpus_path="ch1_architecture/tests/fixtures/corpus_smoke.txt",
        tokenizer_path="outputs/ch1/tokenizer_smoke.json",
        warmup_steps=1,
        lr_min_ratio=0.1,
        ckpt_every=10,
        sample_every=10,
        sample_prompt="a",
        max_new_tokens=5,
        temperature=0.8,
        top_k=0,
        checkpoint_path="outputs/ch1/model_smoke.pt",
        wandb_project="ch1-architecture",
        wandb_mode="disabled",
        val_fraction=0.1,
        val_every=5,
    )


def test_generate_uncached_shape() -> None:
    torch.manual_seed(0)
    config = _config()
    model = GPTModel(config)
    prompt = torch.randint(0, config.vocab_size, (1, 3))

    tokens, step_logits = _generate_uncached(model, prompt, max_new=4, temperature=1.0, top_k=0)
    assert tokens.shape == (1, 7)
    assert len(step_logits) == 4
    assert step_logits[0].shape == (1, config.vocab_size)


def test_generate_top_k_1_is_deterministic_greedy() -> None:
    torch.manual_seed(0)
    config = _config()
    model = GPTModel(config)
    prompt = torch.randint(0, config.vocab_size, (1, 3))

    first = generate(model, prompt, max_new=5, use_cache=False, temperature=0.8, top_k=1)
    second = generate(model, prompt, max_new=5, use_cache=False, temperature=0.8, top_k=1)
    assert torch.equal(first, second)


def test_generate_use_cache_true_returns_expected_shape() -> None:
    # The cached path itself is exercised in depth by test_kv_cache.py's identity
    # test; this just confirms the public `generate()` entry point wires it up.
    config = _config()
    model = GPTModel(config)
    prompt = torch.randint(0, config.vocab_size, (1, 3))
    tokens = generate(model, prompt, max_new=2, use_cache=True)
    assert tokens.shape == (1, 5)
