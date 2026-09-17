"""A5: measure tokens/sec and perplexity across quantization modes, plus the A4
KV-cache speedup, and publish both to `eval/results/` and the README.

Every mode is reported honestly, including the ones that got worse (fp16 on CPU is
expected to be slower than fp32; int4 perplexity is expected to climb) — the tradeoff
curve is the deliverable, not just the good end of it.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn

from ch1_architecture.config import BenchmarkConfig, GPTConfig, load_benchmark_config
from ch1_architecture.data import split_corpus_text
from ch1_architecture.kv_cache import KVCache
from ch1_architecture.model.gpt import GPTModel
from ch1_architecture.quantize import quantize_model
from ch1_architecture.tokenizer import BPETokenizer
from eval.config import seed_everything
from eval.harness import write_json_atomic
from eval.report import update_readme_section

README_PATH = Path("README.md")
README_MARKER = "CH1_BENCHMARK"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark quantized Chapter 1 models.")
    parser.add_argument("--config", type=Path, required=True, help="path to a YAML config")
    return parser.parse_args()


def load_checkpoint_model_and_tokenizer(
    checkpoint_path: str, device: str
) -> tuple[GPTModel, BPETokenizer, GPTConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    gpt_config = GPTConfig(**checkpoint["config"])

    model = GPTModel(gpt_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    tokenizer = BPETokenizer()
    tokenizer.vocab = {int(k): bytes.fromhex(v) for k, v in checkpoint["tokenizer_vocab"].items()}
    tokenizer.merges = [
        (bytes.fromhex(a), bytes.fromhex(b)) for a, b in checkpoint["tokenizer_merges"]
    ]
    tokenizer.encoder = {v: k for k, v in tokenizer.vocab.items()}
    tokenizer._trained = True

    return model, tokenizer, gpt_config


def measure_tokens_per_sec(
    model: nn.Module, prompt_ids: torch.Tensor, n_steps: int, warmup_steps: int
) -> float:
    """Wall-clock tokens/sec over `n_steps` greedy decode steps, re-running the whole
    growing sequence each step (no cache). The first `warmup_steps` are discarded.
    """
    model.eval()
    tokens = prompt_ids
    durations: list[float] = []
    with torch.no_grad():
        for step in range(n_steps):
            start = time.perf_counter()
            logits = model(tokens)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            elapsed = time.perf_counter() - start
            if step >= warmup_steps:
                durations.append(elapsed)
    return (len(durations) * tokens.size(0)) / sum(durations)


def measure_tokens_per_sec_cached(
    model: GPTModel, prompt_ids: torch.Tensor, n_steps: int, warmup_steps: int
) -> float:
    """Same measurement as `measure_tokens_per_sec`, but decoding through a KVCache:
    one prefill forward, then one-token steps against the cached history.
    """
    model.eval()
    cache = KVCache(
        model.config.n_layers, model.config.seq_len, model.config.n_heads, model.config.head_dim
    )
    cache.reset()
    durations: list[float] = []

    with torch.no_grad():
        start = time.perf_counter()
        logits = model(prompt_ids, cache=cache)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        elapsed = time.perf_counter() - start
        if warmup_steps <= 0:
            durations.append(elapsed)

        for step in range(1, n_steps):
            start = time.perf_counter()
            logits = model(next_token, cache=cache)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            elapsed = time.perf_counter() - start
            if step >= warmup_steps:
                durations.append(elapsed)

    return (len(durations) * prompt_ids.size(0)) / sum(durations)


def compute_perplexity(model: nn.Module, token_ids: list[int], seq_len: int) -> float:
    """exp(mean cross-entropy) over non-overlapping windows of `token_ids`.

    Callers pass the held-out tail from `split_token_ids`, never the whole corpus — this
    function scores whatever it is handed and cannot tell the difference, which is
    exactly how the published number came to be in-sample while this docstring claimed
    otherwise. Chapter 2's frozen set is a different thing entirely and is off-limits
    here; the leakage hook blocks it regardless.
    """
    model.eval()
    total_loss, n_windows = 0.0, 0
    vocab_size = None
    # The windows must be built on the model's own device. Reading them off the model
    # rather than taking a device argument keeps the call sites honest: there is no way
    # to pass a device that disagrees with where the weights actually are.
    device = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, len(token_ids) - seq_len, seq_len):
            inputs = torch.tensor([token_ids[i : i + seq_len]], device=device)
            labels = torch.tensor([token_ids[i + 1 : i + seq_len + 1]], device=device)
            logits = model(inputs)
            vocab_size = logits.size(-1)
            loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
            total_loss += loss.item()
            n_windows += 1
    if n_windows == 0:
        raise ValueError("eval corpus too short for even one perplexity window")
    return math.exp(total_loss / n_windows)


def _write_plots(results: dict[str, dict[str, float]], plots_dir: str) -> None:
    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)
    modes = list(results.keys())

    fig, ax = plt.subplots()
    ax.bar(modes, [results[m]["tokens_per_sec"] for m in modes])
    ax.set_ylabel("tokens/sec")
    ax.set_title("Chapter 1 — inference speed by quantization mode")
    fig.savefig(plots_path / "speedup_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.bar(modes, [results[m]["perplexity"] for m in modes])
    ax.set_ylabel("perplexity")
    ax.set_title("Chapter 1 — quality cost by quantization mode")
    fig.savefig(plots_path / "perplexity_tradeoff.png")
    plt.close(fig)


def _update_readme(results: dict[str, dict[str, float]], kv_cache: dict[str, float]) -> None:
    lines = ["| Mode | tokens/sec | perplexity |", "|---|---|---|"]
    for mode, row in results.items():
        lines.append(f"| {mode} | {row['tokens_per_sec']:.1f} | {row['perplexity']:.2f} |")
    lines.append("")
    lines.append(
        f"KV-cache: {kv_cache['cached_tokens_per_sec']:.1f} tok/s cached vs "
        f"{kv_cache['uncached_tokens_per_sec']:.1f} tok/s uncached "
        f"({kv_cache['speedup']:.2f}x)."
    )
    lines.append("")
    lines.append(
        "Perplexity is measured on a held-out tail of the corpus that the training run "
        "never saw, so it is a generalization number rather than a memorization one."
    )
    lines.append("")
    lines.append("![speedup](eval/results/plots/speedup_curve.png)")
    lines.append("![perplexity](eval/results/plots/perplexity_tradeoff.png)")
    update_readme_section(README_PATH, README_MARKER, "\n".join(lines))


def _is_smoke_run(config_path: Path, checkpoint_path: str) -> bool:
    return "smoke" in Path(config_path).stem or "smoke" in checkpoint_path


def main() -> None:
    args = parse_args()
    config: BenchmarkConfig = load_benchmark_config(args.config)
    seed_everything(config.seed)

    fp32_model, tokenizer, gpt_config = load_checkpoint_model_and_tokenizer(
        config.checkpoint_path, config.device
    )
    corpus_text = Path(config.eval_corpus_path).read_text()
    # Score the same held-out tail the training run never saw. Measuring perplexity on
    # the corpus the model was trained on would report memorization: at the full config's
    # token budget the training set is seen about ten times over, so an in-sample number
    # would look good for the wrong reason and could not be defended as a quality claim.
    _, eval_text = split_corpus_text(corpus_text, config.val_fraction)
    if not eval_text:
        raise ValueError(
            f"val_fraction={config.val_fraction} holds out nothing from "
            f"{len(corpus_text)} characters; perplexity would be measured in-sample"
        )
    eval_token_ids = tokenizer.encode(eval_text)
    print(
        f"[ch1-benchmark] perplexity on the held-out tail: {len(eval_token_ids)} tokens "
        f"({config.val_fraction:.0%} of {config.eval_corpus_path})"
    )
    prompt_ids = torch.tensor([tokenizer.encode(config.prompt)], device=config.device)

    results: dict[str, dict[str, float]] = {}
    for mode in config.modes:
        quantized = quantize_model(fp32_model, mode)
        tokens_per_sec = measure_tokens_per_sec(
            quantized, prompt_ids, config.n_steps, config.warmup_steps
        )
        perplexity = compute_perplexity(quantized, eval_token_ids, gpt_config.seq_len)
        results[mode] = {
            "tokens_per_sec": tokens_per_sec,
            "perplexity": perplexity,
            "n_steps": config.n_steps,
        }
        print(f"[ch1-benchmark] {mode}: {tokens_per_sec:.1f} tok/s, ppl={perplexity:.2f}")

    cached_tps = measure_tokens_per_sec_cached(
        fp32_model, prompt_ids, config.n_steps, config.warmup_steps
    )
    uncached_tps = measure_tokens_per_sec(
        fp32_model, prompt_ids, config.n_steps, config.warmup_steps
    )
    kv_cache_block = {
        "cached_tokens_per_sec": cached_tps,
        "uncached_tokens_per_sec": uncached_tps,
        "speedup": cached_tps / uncached_tps,
    }
    print(f"[ch1-benchmark] kv_cache speedup: {kv_cache_block['speedup']:.2f}x")

    payload = {
        **results,
        "kv_cache": kv_cache_block,
        "seed": config.seed,
        "checkpoint_path": config.checkpoint_path,
        "device": config.device,
        "eval_corpus_path": config.eval_corpus_path,
        "val_fraction": config.val_fraction,
        "n_eval_tokens": len(eval_token_ids),
        "perplexity_split": "held-out tail",
    }
    write_json_atomic(payload, config.results_path)
    _write_plots(results, config.plots_dir)
    if _is_smoke_run(args.config, config.checkpoint_path):
        print(
            "[ch1-benchmark] smoke run detected — skipping README update; these numbers "
            "are a toy model on a tiny corpus and are not meaningful (CLAUDE.md honest-metrics)."
        )
    else:
        _update_readme(results, kv_cache_block)
    print(f"[ch1-benchmark] wrote {config.results_path} and plots to {config.plots_dir}")


if __name__ == "__main__":
    main()
