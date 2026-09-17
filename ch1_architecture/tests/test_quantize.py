"""Quantization round-trip and integration tests (A5)."""

from __future__ import annotations

import torch
from ch1_architecture.config import GPTConfig
from ch1_architecture.model.gpt import GPTModel
from ch1_architecture.quantize import (
    QuantizedLinear,
    _pack_int4,
    _quantize_absmax,
    _unpack_int4,
    quantize_model,
)
from torch import nn


def _config() -> GPTConfig:
    return GPTConfig(
        vocab_size=32,
        d_model=16,
        n_heads=2,
        n_layers=2,
        seq_len=10,
        batch_size=2,
        max_steps=10,
        learning_rate=1e-3,
        clip_grad_norm=1.0,
        log_every=10,
        seed=0,
        device="cpu",
        run_name="quant-test",
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
        val_fraction=0.1,
        val_every=5,
    )


def test_int8_round_trip_within_quantization_step() -> None:
    torch.manual_seed(0)
    weight = torch.randn(8, 8)
    w_q, scale = _quantize_absmax(weight, qmax=127)
    dequantized = w_q * scale
    assert (dequantized - weight).abs().max() <= scale.item() + 1e-6


def test_int4_pack_unpack_is_exactly_invertible() -> None:
    torch.manual_seed(0)
    values = torch.randint(-8, 8, (5, 7))  # odd number of elements -> exercises padding
    packed, shape = _pack_int4(values.to(torch.int8))
    restored = _unpack_int4(packed, shape)
    assert torch.equal(restored, values.to(torch.int64))


def test_all_zero_weight_does_not_divide_by_zero() -> None:
    weight = torch.zeros(4, 4)
    w_q, scale = _quantize_absmax(weight, qmax=127)
    assert scale.item() != 0
    assert torch.equal(w_q, torch.zeros(4, 4))


def test_quantized_linear_every_mode_runs() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(6, 4)
    x = torch.randn(3, 6)
    for mode in ["fp16", "int8", "int4"]:
        q = QuantizedLinear(linear.weight.data, linear.bias.data, mode)
        out = q(x)
        assert out.shape == (3, 4)
        assert torch.isfinite(out).all()


def test_quantize_model_fp32_is_a_no_op_copy() -> None:
    config = _config()
    model = GPTModel(config)
    quantized = quantize_model(model, "fp32")
    assert quantized is not model
    for module in quantized.modules():
        assert not isinstance(module, QuantizedLinear)


def test_quantize_model_replaces_linears_and_runs_every_mode() -> None:
    config = _config()
    model = GPTModel(config)
    ids = torch.randint(0, config.vocab_size, (1, 5))

    for mode in ["fp16", "int8", "int4"]:
        quantized = quantize_model(model, mode)
        found_quantized = any(isinstance(m, QuantizedLinear) for m in quantized.modules())
        assert found_quantized
        with torch.no_grad():
            logits = quantized(ids)
        assert logits.shape == (1, 5, config.vocab_size)
        assert torch.isfinite(logits).all(), f"mode={mode} produced non-finite logits"


def test_int8_perplexity_is_within_a_sane_factor_of_fp32() -> None:
    config = _config()
    torch.manual_seed(0)
    model = GPTModel(config)
    ids = torch.randint(0, config.vocab_size, (2, 8))
    labels = torch.randint(0, config.vocab_size, (2, 8))

    fp32_model = quantize_model(model, "fp32")
    int8_model = quantize_model(model, "int8")

    import torch.nn.functional as F

    with torch.no_grad():
        fp32_loss = F.cross_entropy(fp32_model(ids).view(-1, config.vocab_size), labels.view(-1))
        int8_loss = F.cross_entropy(int8_model(ids).view(-1, config.vocab_size), labels.view(-1))
    assert torch.isfinite(int8_loss)
    assert int8_loss.item() < fp32_loss.item() * 3 + 5
