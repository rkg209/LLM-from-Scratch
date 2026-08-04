"""KV-cache correctness (A4). The identity test is the gate: no speedup number may be
reported until cached and uncached generation produce the same logits.
"""

from __future__ import annotations

import pytest
import torch
from ch1_architecture.config import GPTConfig
from ch1_architecture.generate import _generate_cached, _generate_uncached
from ch1_architecture.kv_cache import KVCache
from ch1_architecture.model.gpt import GPTModel

N_LAYERS, MAX_SEQ_LEN, N_HEADS, HEAD_DIM = 2, 10, 2, 8


def _config() -> GPTConfig:
    return GPTConfig(
        vocab_size=32,
        d_model=N_HEADS * HEAD_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        seq_len=MAX_SEQ_LEN,
        batch_size=1,
        max_steps=10,
        learning_rate=1e-3,
        clip_grad_norm=1.0,
        log_every=10,
        seed=0,
        device="cpu",
        run_name="kv-test",
        corpus_path="ch1_architecture/tests/fixtures/corpus_smoke.txt",
        tokenizer_path="outputs/ch1/tokenizer_smoke.json",
        warmup_steps=1,
        lr_min_ratio=0.1,
        ckpt_every=10,
        sample_every=10,
        sample_prompt="a",
        max_new_tokens=4,
        temperature=1.0,
        top_k=1,
        checkpoint_path="outputs/ch1/model_smoke.pt",
        wandb_project="ch1-architecture",
        wandb_mode="disabled",
    )


def test_cache_lazy_allocation() -> None:
    cache = KVCache(N_LAYERS, MAX_SEQ_LEN, N_HEADS, HEAD_DIM)
    assert not cache.is_allocated
    k = torch.randn(1, N_HEADS, 3, HEAD_DIM)
    cache.update(0, k, k)
    assert cache.is_allocated


def test_update_returns_a_view_not_a_copy() -> None:
    cache = KVCache(N_LAYERS, MAX_SEQ_LEN, N_HEADS, HEAD_DIM)
    k = torch.randn(1, N_HEADS, 2, HEAD_DIM)
    k_out, _ = cache.update(0, k, k)
    assert k_out.data_ptr() == cache._k[0].data_ptr()


def test_write_pointer_advances_once_per_forward_not_per_layer() -> None:
    cache = KVCache(N_LAYERS, MAX_SEQ_LEN, N_HEADS, HEAD_DIM)
    k = torch.randn(1, N_HEADS, 3, HEAD_DIM)
    cache.update(0, k, k)
    cache.update(1, k, k)
    assert cache.current_len == 0  # unchanged until advance() is called
    cache.advance(3)
    assert cache.current_len == 3


def test_reset_keeps_buffers_but_zeroes_length() -> None:
    cache = KVCache(N_LAYERS, MAX_SEQ_LEN, N_HEADS, HEAD_DIM)
    k = torch.randn(1, N_HEADS, 3, HEAD_DIM)
    cache.update(0, k, k)
    cache.advance(3)
    buf_ptr = cache._k[0].data_ptr()

    cache.reset()
    assert cache.current_len == 0
    assert cache._k[0].data_ptr() == buf_ptr


def test_update_past_max_seq_len_raises() -> None:
    cache = KVCache(N_LAYERS, MAX_SEQ_LEN, N_HEADS, HEAD_DIM)
    k = torch.randn(1, N_HEADS, MAX_SEQ_LEN, HEAD_DIM)
    cache.update(0, k, k)
    cache.advance(MAX_SEQ_LEN)
    with pytest.raises(IndexError):
        cache.update(0, torch.randn(1, N_HEADS, 1, HEAD_DIM), torch.randn(1, N_HEADS, 1, HEAD_DIM))


def test_cached_and_uncached_generation_produce_identical_logits() -> None:
    torch.manual_seed(0)
    config = _config()
    model = GPTModel(config)
    prompt = torch.randint(0, config.vocab_size, (1, 4))

    cache = KVCache(config.n_layers, config.seq_len, config.n_heads, config.head_dim)
    _, cached_logits = _generate_cached(
        model, prompt, max_new=5, cache=cache, temperature=1.0, top_k=1
    )
    _, uncached_logits = _generate_uncached(model, prompt, max_new=5, temperature=1.0, top_k=1)

    assert len(cached_logits) == len(uncached_logits)
    for cached, uncached in zip(cached_logits, uncached_logits, strict=True):
        assert torch.allclose(cached, uncached, atol=1e-5), (cached - uncached).abs().max()
