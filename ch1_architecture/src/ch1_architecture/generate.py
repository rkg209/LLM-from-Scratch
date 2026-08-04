"""Autoregressive sampling: the uncached path (A3) and the KV-cached path (A4)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ch1_architecture.kv_cache import KVCache
from ch1_architecture.model.gpt import GPTModel


def _sample_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    # logits: [B, V] -> next_token: [B, 1]
    logits = logits / temperature
    if top_k > 0:
        top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        threshold = top_values[:, -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)  # [B, 1]


def _generate_uncached(
    model: GPTModel,
    prompt_ids: torch.Tensor,
    max_new: int,
    temperature: float,
    top_k: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Re-run the whole sequence on every step. Returns (tokens, per-step logits).

    Per-step logits (not just sampled tokens) are what A4's cache-identity test needs
    to compare against the cached path — kept here so that test does not have to
    retrofit this function later.
    """
    model.eval()
    tokens = prompt_ids
    step_logits: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(tokens)  # [B, T, V]
            last_logits = logits[:, -1, :]  # [B, V]
            step_logits.append(last_logits)
            next_token = _sample_next_token(last_logits, temperature, top_k)  # [B, 1]
            tokens = torch.cat([tokens, next_token], dim=1)
    return tokens, step_logits


def _generate_cached(
    model: GPTModel,
    prompt_ids: torch.Tensor,
    max_new: int,
    cache: KVCache,
    temperature: float,
    top_k: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """One prefill forward over the prompt (fills the cache), then one-token steps.

    Sampling is shared with `_generate_uncached` so the two paths differ only in how
    the logits were produced — which is exactly what the cache-identity test checks.
    """
    model.eval()
    cache.reset()
    tokens = prompt_ids
    step_logits: list[torch.Tensor] = []

    with torch.no_grad():
        logits = model(tokens, cache=cache)  # [B, T_prompt, V], fills the cache
        last_logits = logits[:, -1, :]
        step_logits.append(last_logits)
        next_token = _sample_next_token(last_logits, temperature, top_k)  # [B, 1]
        tokens = torch.cat([tokens, next_token], dim=1)

        for _ in range(max_new - 1):
            logits = model(next_token, cache=cache)  # [B, 1, V]
            last_logits = logits[:, -1, :]
            step_logits.append(last_logits)
            next_token = _sample_next_token(last_logits, temperature, top_k)
            tokens = torch.cat([tokens, next_token], dim=1)

    return tokens, step_logits


def generate(
    model: GPTModel,
    prompt_ids: torch.Tensor,
    max_new: int,
    use_cache: bool = True,
    temperature: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    if use_cache:
        cache = KVCache(
            n_layers=model.config.n_layers,
            max_seq_len=model.config.seq_len,
            n_heads=model.config.n_heads,
            head_dim=model.config.head_dim,
        )
        tokens, _ = _generate_cached(model, prompt_ids, max_new, cache, temperature, top_k)
        return tokens

    tokens, _ = _generate_uncached(model, prompt_ids, max_new, temperature, top_k)
    return tokens
