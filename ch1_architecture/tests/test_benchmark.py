"""Benchmark tests (A5). Build a tiny model in-memory (D-8) — `outputs/` is gitignored
and no checkpoint exists in CI, so these tests must not depend on a training run.
"""

from __future__ import annotations

from pathlib import Path

import torch
from ch1_architecture.benchmark import (
    _write_plots,
    compute_perplexity,
    load_checkpoint_model_and_tokenizer,
    measure_tokens_per_sec,
    measure_tokens_per_sec_cached,
)
from ch1_architecture.config import GPTConfig
from ch1_architecture.model.gpt import GPTModel
from ch1_architecture.tokenizer import BPETokenizer

CORPUS = Path("ch1_architecture/tests/fixtures/corpus_smoke.txt").read_text()


def _tiny_checkpoint(tmp_path: Path) -> tuple[Path, GPTConfig]:
    tokenizer = BPETokenizer()
    tokenizer.train(CORPUS, vocab_size=260)

    config = GPTConfig(
        vocab_size=260,
        d_model=16,
        n_heads=2,
        n_layers=2,
        seq_len=16,
        batch_size=2,
        max_steps=1,
        learning_rate=1e-3,
        clip_grad_norm=1.0,
        log_every=1,
        seed=0,
        device="cpu",
        run_name="bench-test",
        corpus_path="ch1_architecture/tests/fixtures/corpus_smoke.txt",
        tokenizer_path=str(tmp_path / "tokenizer.json"),
        warmup_steps=1,
        lr_min_ratio=0.1,
        ckpt_every=1,
        sample_every=1,
        sample_prompt="a",
        max_new_tokens=3,
        temperature=1.0,
        top_k=1,
        checkpoint_path=str(tmp_path / "model.pt"),
        wandb_project="ch1-architecture",
        wandb_mode="disabled",
    )
    model = GPTModel(config)

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": {},
        "config": {f: getattr(config, f) for f in config.__dataclass_fields__},
        "step": 0,
        "tokenizer_vocab": {k: v.hex() for k, v in tokenizer.vocab.items()},
        "tokenizer_merges": [[a.hex(), b.hex()] for a, b in tokenizer.merges],
    }
    ckpt_path = tmp_path / "model.pt"
    torch.save(checkpoint, ckpt_path)
    return ckpt_path, config


def test_load_checkpoint_model_and_tokenizer(tmp_path: Path) -> None:
    ckpt_path, config = _tiny_checkpoint(tmp_path)
    model, tokenizer, loaded_config = load_checkpoint_model_and_tokenizer(str(ckpt_path), "cpu")
    assert loaded_config == config
    assert tokenizer.decode(tokenizer.encode("First Citizen")) == "First Citizen"
    ids = torch.randint(0, config.vocab_size, (1, 5))
    with torch.no_grad():
        logits = model(ids)
    assert logits.shape == (1, 5, config.vocab_size)


def test_measure_tokens_per_sec_is_positive(tmp_path: Path) -> None:
    _, config = _tiny_checkpoint(tmp_path)
    model = GPTModel(config)
    prompt = torch.randint(0, config.vocab_size, (1, 3))
    tps = measure_tokens_per_sec(model, prompt, n_steps=5, warmup_steps=1)
    assert tps > 0


def test_measure_tokens_per_sec_cached_is_positive(tmp_path: Path) -> None:
    _, config = _tiny_checkpoint(tmp_path)
    model = GPTModel(config)
    prompt = torch.randint(0, config.vocab_size, (1, 3))
    tps = measure_tokens_per_sec_cached(model, prompt, n_steps=5, warmup_steps=1)
    assert tps > 0


def test_compute_perplexity_is_finite_and_positive(tmp_path: Path) -> None:
    _, config = _tiny_checkpoint(tmp_path)
    model = GPTModel(config)
    token_ids = list(range(200)) * 2
    ppl = compute_perplexity(model, token_ids, seq_len=config.seq_len)
    assert ppl > 0
    assert ppl == ppl  # not nan


def test_write_plots_creates_both_pngs(tmp_path: Path) -> None:
    results = {
        "fp32": {"tokens_per_sec": 10.0, "perplexity": 5.0},
        "int8": {"tokens_per_sec": 15.0, "perplexity": 6.0},
    }
    _write_plots(results, str(tmp_path / "plots"))
    assert (tmp_path / "plots" / "speedup_curve.png").exists()
    assert (tmp_path / "plots" / "perplexity_tradeoff.png").exists()
