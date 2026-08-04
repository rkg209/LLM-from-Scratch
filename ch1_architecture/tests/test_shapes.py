"""Shape tests for every A2 module — the cheapest bug-catcher in the chapter."""

from __future__ import annotations

import torch
from ch1_architecture.config import GPTConfig
from ch1_architecture.model.attention import MultiHeadSelfAttention
from ch1_architecture.model.block import TransformerBlock
from ch1_architecture.model.embeddings import PositionalEmbedding, TokenEmbedding
from ch1_architecture.model.gpt import GPTModel
from ch1_architecture.model.mlp import MLP
from ch1_architecture.model.norm import LayerNorm

B, T, D, H, V, MAX_SEQ_LEN = 2, 5, 16, 4, 32, 10


def _config() -> GPTConfig:
    return GPTConfig(
        vocab_size=V,
        d_model=D,
        n_heads=H,
        n_layers=2,
        seq_len=MAX_SEQ_LEN,
        batch_size=B,
        max_steps=10,
        learning_rate=1e-3,
        clip_grad_norm=1.0,
        log_every=10,
        seed=42,
        device="cpu",
        run_name="test",
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
    )


def test_layer_norm_shape() -> None:
    x = torch.randn(B, T, D)
    out = LayerNorm(D)(x)
    assert out.shape == (B, T, D)


def test_token_embedding_shape() -> None:
    ids = torch.randint(0, V, (B, T))
    out = TokenEmbedding(V, D)(ids)
    assert out.shape == (B, T, D)


def test_positional_embedding_shape_and_offset() -> None:
    pos_emb = PositionalEmbedding(MAX_SEQ_LEN, D)
    out = pos_emb(T)
    assert out.shape == (1, T, D)
    offset_out = pos_emb(1, offset=T)
    assert torch.equal(offset_out[0, 0], pos_emb(MAX_SEQ_LEN)[0, T])


def test_mlp_shape() -> None:
    x = torch.randn(B, T, D)
    out = MLP(D)(x)
    assert out.shape == (B, T, D)


def test_attention_shape() -> None:
    x = torch.randn(B, T, D)
    out = MultiHeadSelfAttention(D, H, MAX_SEQ_LEN)(x)
    assert out.shape == (B, T, D)


def test_block_shape() -> None:
    x = torch.randn(B, T, D)
    out = TransformerBlock(D, H, MAX_SEQ_LEN)(x)
    assert out.shape == (B, T, D)


def test_gpt_model_shape() -> None:
    config = _config()
    model = GPTModel(config)
    ids = torch.randint(0, V, (B, T))
    logits = model(ids)
    assert logits.shape == (B, T, V)


def test_get_num_params_counts_tied_weight_once() -> None:
    config = _config()
    model = GPTModel(config)
    manual_count = sum(p.numel() for p in model.parameters())
    assert model.get_num_params() == manual_count
