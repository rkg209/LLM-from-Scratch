"""Benchmark tests (A5). Build a tiny model in-memory (D-8) — `outputs/` is gitignored
and no checkpoint exists in CI, so these tests must not depend on a training run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from ch1_architecture.benchmark import (
    _write_kv_cache_plot,
    _write_plots,
    compute_perplexity,
    load_checkpoint_model_and_tokenizer,
    measure_kv_cache_curve,
    measure_tokens_per_sec,
    measure_tokens_per_sec_cached,
)
from ch1_architecture.benchmark import (
    main as benchmark_main,
)
from ch1_architecture.config import BenchmarkConfig, GPTConfig
from ch1_architecture.model.gpt import GPTModel
from ch1_architecture.tokenizer import BPETokenizer

from ch1_architecture.data import split_corpus_text

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
        val_fraction=0.1,
        val_every=5,
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


def _accelerator() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


@pytest.mark.skipif(_accelerator() is None, reason="needs a non-CPU device to be meaningful")
def test_compute_perplexity_runs_on_a_non_cpu_model(tmp_path: Path) -> None:
    """The windows must be built where the weights are.

    This is not a hypothetical: `compute_perplexity` built its tensors on the CPU while
    the full-config run puts the model on `cuda`, so the GPU benchmark would have died
    on its first window. A CPU-only test cannot see that — both tensors land on the CPU
    either way — so this one runs on whatever accelerator the machine has (MPS on the
    dev laptop, CUDA on the GPU box).
    """
    device = _accelerator()
    assert device is not None
    _, config = _tiny_checkpoint(tmp_path)
    model = GPTModel(config).to(device)
    ppl = compute_perplexity(model, list(range(200)) * 2, seq_len=config.seq_len)
    assert ppl > 0
    assert ppl == ppl


def test_perplexity_on_the_held_out_tail_differs_from_the_in_sample_number(
    tmp_path: Path,
) -> None:
    """The held-out score must actually be a different measurement.

    If someone reverts the split, the benchmark would score the whole corpus again and
    these two numbers would coincide. They are computed here from the same model on the
    same corpus, differing only in which slice is scored.
    """
    _, config = _tiny_checkpoint(tmp_path)
    model = GPTModel(config)
    tokenizer = BPETokenizer()
    tokenizer.train(CORPUS, vocab_size=260)

    train_text, val_text = split_corpus_text(CORPUS, 0.1)
    in_sample = compute_perplexity(model, tokenizer.encode(CORPUS), config.seq_len)
    held_out = compute_perplexity(model, tokenizer.encode(val_text), config.seq_len)

    assert train_text and val_text
    assert held_out != in_sample


def test_benchmark_main_records_which_slice_it_scored(tmp_path: Path, monkeypatch) -> None:
    ckpt_path, _ = _tiny_checkpoint(tmp_path)
    results_path = tmp_path / "ch1_benchmark.json"
    config_path = tmp_path / "benchmark_test.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"checkpoint_path: {ckpt_path}",
                "modes: [fp32]",
                'prompt: "a"',
                "n_steps: 3",
                "warmup_steps: 1",
                "seed: 42",
                "device: cpu",
                "eval_corpus_path: ch1_architecture/tests/fixtures/corpus_smoke.txt",
                "val_fraction: 0.1",
                "kv_cache_steps: [3, 5]",
                f"results_path: {results_path}",
                f"plots_dir: {tmp_path / 'plots'}",
            ]
        )
    )
    monkeypatch.setattr("sys.argv", ["benchmark", "--config", str(config_path)])
    # The config name and checkpoint path carry no "smoke" marker, so `main` takes this
    # for a real run and publishes to the README. Point that at a throwaway file: a test
    # must never rewrite the repository's own results section.
    readme = tmp_path / "README.md"
    readme.write_text("<!-- CH1_BENCHMARK_START -->\n<!-- CH1_BENCHMARK_END -->\n")
    monkeypatch.setattr("ch1_architecture.benchmark.README_PATH", readme)
    benchmark_main()
    assert "tokens/sec" in readme.read_text()

    payload = json.loads(results_path.read_text())
    assert payload["val_fraction"] == 0.1
    assert payload["perplexity_split"] == "held-out tail"
    # The artifact has to be able to prove on its own that something was held out: a
    # tenth of the corpus, not all of it.
    tokenizer = BPETokenizer()
    tokenizer.train(CORPUS, vocab_size=260)
    assert 0 < payload["n_eval_tokens"] < len(tokenizer.encode(CORPUS)) // 2


def test_benchmark_refuses_a_config_that_holds_nothing_out(tmp_path: Path) -> None:
    _, config = _tiny_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="val_fraction"):
        BenchmarkConfig(
            checkpoint_path=str(tmp_path / "model.pt"),
            modes=["fp32"],
            prompt="a",
            n_steps=3,
            warmup_steps=1,
            seed=42,
            device="cpu",
            eval_corpus_path="ch1_architecture/tests/fixtures/corpus_smoke.txt",
            val_fraction=1.5,
            kv_cache_steps=[3],
            results_path=str(tmp_path / "r.json"),
            plots_dir=str(tmp_path / "plots"),
        )


def test_kv_cache_curve_covers_every_requested_length(tmp_path: Path) -> None:
    _, config = _tiny_checkpoint(tmp_path)
    model = GPTModel(config)
    prompt = torch.randint(0, config.vocab_size, (1, 2))

    curve = measure_kv_cache_curve(model, prompt, [3, 5], warmup_steps=1)

    assert sorted(curve) == ["3", "5"]
    for row in curve.values():
        assert row["cached_tokens_per_sec"] > 0
        assert row["uncached_tokens_per_sec"] > 0
        assert row["speedup"] == row["cached_tokens_per_sec"] / row["uncached_tokens_per_sec"]


def test_kv_cache_curve_plot_is_written(tmp_path: Path) -> None:
    curve = {
        "32": {"cached_tokens_per_sec": 10.0, "uncached_tokens_per_sec": 10.0, "speedup": 1.0},
        "250": {"cached_tokens_per_sec": 20.0, "uncached_tokens_per_sec": 10.0, "speedup": 2.0},
    }
    _write_kv_cache_plot(curve, str(tmp_path / "plots"))
    assert (tmp_path / "plots" / "kv_cache_curve.png").exists()


def test_a_decode_length_at_or_below_warmup_is_rejected(tmp_path: Path) -> None:
    """Otherwise the measurement silently averages zero timed steps."""
    with pytest.raises(ValueError, match="kv_cache_steps"):
        BenchmarkConfig(
            checkpoint_path=str(tmp_path / "model.pt"),
            modes=["fp32"],
            prompt="a",
            n_steps=10,
            warmup_steps=5,
            seed=42,
            device="cpu",
            eval_corpus_path="ch1_architecture/tests/fixtures/corpus_smoke.txt",
            val_fraction=0.1,
            kv_cache_steps=[5],
            results_path=str(tmp_path / "r.json"),
            plots_dir=str(tmp_path / "plots"),
        )
