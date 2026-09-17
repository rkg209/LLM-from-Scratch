"""Chapter 1 training entrypoint — the real training loop (A3).

Replaces the F2 scaffold. `main()` stays small by delegating to the `build_*` helpers
and `training_loop` below; the loop itself is a hand-written forward/backward/step, no
`Trainer` class from anywhere.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ch1_architecture.config import GPTConfig, load_gpt_config
from ch1_architecture.data import (
    CorpusDataset,
    make_dataloader,
    make_eval_dataloader,
    split_corpus_text,
)
from ch1_architecture.generate import generate
from ch1_architecture.model.gpt import GPTModel
from ch1_architecture.tokenizer import BPETokenizer
from eval.config import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Chapter 1 GPT model.")
    parser.add_argument("--config", type=Path, required=True, help="path to a YAML config")
    return parser.parse_args()


def _corpus_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_tokenizer_and_loaders(
    config: GPTConfig,
) -> tuple[BPETokenizer, DataLoader, DataLoader]:
    """Train (or load a cached) tokenizer, then build the train and validation loaders.

    The merge loop is a plain Python loop by design (A1 out-of-scope: not optimized).
    Retraining it every run would spend the 120s smoke budget on the wrong thing, so
    the result is cached and reused whenever the corpus is unchanged (keyed by hash).
    """
    corpus_text = Path(config.corpus_path).read_text()
    train_text, val_text = split_corpus_text(corpus_text, config.val_fraction)
    # Keyed on the training text and the vocabulary size, not the whole corpus: those are
    # the two inputs that determine the merge table, and a cache hit on a tokenizer fitted
    # to a different split would silently reintroduce the leak the split exists to close.
    corpus_hash = _corpus_hash(f"{config.vocab_size}:{config.val_fraction}:{train_text}")

    tok_path = Path(config.tokenizer_path)
    hash_path = tok_path.with_suffix(tok_path.suffix + ".hash")

    if tok_path.exists() and hash_path.exists() and hash_path.read_text().strip() == corpus_hash:
        tokenizer = BPETokenizer.load(tok_path)
    else:
        tokenizer = BPETokenizer()
        tokenizer.train(train_text, config.vocab_size)
        tokenizer.save(tok_path)
        hash_path.parent.mkdir(parents=True, exist_ok=True)
        hash_path.write_text(corpus_hash)

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text) if val_text else []
    train_loader = make_dataloader(CorpusDataset(train_ids, config.seq_len), config)
    # Non-overlapping windows, so every held-out token is scored exactly once and this
    # loss is directly comparable to the perplexity `benchmark.py` publishes.
    val_loader = make_eval_dataloader(
        CorpusDataset(val_ids, config.seq_len, stride=config.seq_len), config
    )
    print(
        f"[ch1] corpus: {len(train_ids)} train tokens / {len(val_ids)} held-out tokens "
        f"({config.val_fraction:.0%} tail, split before the tokenizer was fitted)"
    )
    return tokenizer, train_loader, val_loader


def build_model(config: GPTConfig) -> GPTModel:
    return GPTModel(config).to(config.device)


def build_optimizer(model: GPTModel, config: GPTConfig) -> AdamW:
    return AdamW(model.parameters(), lr=config.learning_rate)


def lr_lambda(step: int, warmup_steps: int, max_steps: int, lr_min_ratio: float) -> float:
    """Linear warmup, then cosine decay to `learning_rate * lr_min_ratio`."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    decay_steps = max(1, max_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min_ratio + (1.0 - lr_min_ratio) * cosine


def build_scheduler(optimizer: AdamW, config: GPTConfig) -> LambdaLR:
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_lambda(
            step, config.warmup_steps, config.max_steps, config.lr_min_ratio
        ),
    )


def save_checkpoint(
    model: GPTModel,
    optimizer: AdamW,
    config: GPTConfig,
    step: int,
    tokenizer: BPETokenizer,
) -> None:
    """Atomic checkpoint write: temp file + `os.replace`, so an interrupted save can
    never corrupt the previous checkpoint (AC-5 needs a reload that always works).
    """
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": {f: getattr(config, f) for f in config.__dataclass_fields__},
        "step": step,
        "tokenizer_vocab": {k: v.hex() for k, v in tokenizer.vocab.items()},
        "tokenizer_merges": [[a.hex(), b.hex()] for a, b in tokenizer.merges],
    }
    path = Path(config.checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model: GPTModel, loader: DataLoader, config: GPTConfig) -> float:
    """Mean cross-entropy over the held-out tail.

    The model is put back in train mode by the caller's `model.train()` below; doing it
    here as well would hide a missing one somewhere else.
    """
    model.eval()
    total_loss, n_batches = 0.0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(config.device), labels.to(config.device)
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))
        total_loss += loss.item()
        n_batches += 1
    if n_batches == 0:
        raise ValueError(
            "validation split produced no batches — val_fraction is too small for this "
            f"corpus and seq_len ({config.seq_len})"
        )
    return total_loss / n_batches


def training_loop(
    model: GPTModel,
    loader: DataLoader,
    val_loader: DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    tokenizer: BPETokenizer,
    config: GPTConfig,
    wandb_run: Any | None = None,
) -> tuple[list[float], list[tuple[int, float]]]:
    model.train()
    losses: list[float] = []
    val_losses: list[tuple[int, float]] = []
    data_iter = iter(loader)

    for step in range(config.max_steps):
        try:
            inputs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inputs, labels = next(data_iter)
        inputs, labels = inputs.to(config.device), labels.to(config.device)

        start = time.perf_counter()
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
        optimizer.step()
        scheduler.step()
        elapsed = time.perf_counter() - start

        losses.append(loss.item())

        if wandb_run is not None and step % config.log_every == 0:
            tokens_per_sec = config.batch_size * config.seq_len / max(elapsed, 1e-9)
            wandb_run.log(
                {
                    "step": step,
                    "train_loss": loss.item(),
                    "perplexity": math.exp(min(loss.item(), 20.0)),
                    "lr": scheduler.get_last_lr()[0],
                    "tokens_per_sec": tokens_per_sec,
                }
            )

        if step % config.val_every == 0 or step == config.max_steps - 1:
            val_loss = evaluate(model, val_loader, config)
            model.train()
            val_losses.append((step, val_loss))
            print(
                f"[ch1] step {step}: train_loss={loss.item():.4f} "
                f"val_loss={val_loss:.4f} val_ppl={math.exp(min(val_loss, 20.0)):.2f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "step": step,
                        "val_loss": val_loss,
                        "val_perplexity": math.exp(min(val_loss, 20.0)),
                    }
                )

        if step % config.sample_every == 0:
            sample_ids = torch.tensor(
                [tokenizer.encode(config.sample_prompt)], device=config.device
            )
            generated = generate(
                model,
                sample_ids,
                max_new=config.max_new_tokens,
                use_cache=False,
                temperature=config.temperature,
                top_k=config.top_k,
            )
            try:
                sample_text = tokenizer.decode(generated[0].tolist())
            except UnicodeDecodeError:
                # An early, near-random model can sample raw bytes that do not form a
                # valid UTF-8 sequence on their own — not a tokenizer bug, just not
                # yet English. Log it as such rather than crashing the run over it.
                sample_text = "<undecodable byte sequence>"
            # Printed as well as logged: the samples are A3 AC-7's evidence, and on the
            # GPU box W&B runs offline, so until someone syncs it the job log is the only
            # record that exists.
            print(f"[ch1] step {step} sample: {sample_text!r}", flush=True)
            if wandb_run is not None:
                wandb_run.log({"step": step, "sample": sample_text})
            model.train()

        if step % config.ckpt_every == 0 or step == config.max_steps - 1:
            save_checkpoint(model, optimizer, config, step, tokenizer)

    return losses, val_losses


def main() -> None:
    args = parse_args()
    config: GPTConfig = load_gpt_config(args.config)
    seed_everything(config.seed)

    print(f"[ch1] config loaded: {args.config}")
    print(f"[ch1] run={config.run_name} device={config.device} seed={config.seed}")

    tokenizer, loader, val_loader = build_tokenizer_and_loaders(config)
    model = build_model(config)
    print(f"[ch1] model: {model.get_num_params() / 1e6:.2f}M parameters")

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    wandb_run = None
    if config.wandb_mode != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=config.wandb_project,
            mode=config.wandb_mode,
            config={f: getattr(config, f) for f in config.__dataclass_fields__},
        )

    losses, val_losses = training_loop(
        model, loader, val_loader, optimizer, scheduler, tokenizer, config, wandb_run
    )
    print(f"[ch1] trained {config.max_steps} steps, final loss={losses[-1]:.4f}")
    best_step, best_val = min(val_losses, key=lambda item: item[1])
    print(
        f"[ch1] final val_loss={val_losses[-1][1]:.4f} "
        f"(best {best_val:.4f} at step {best_step} of {config.max_steps})"
    )
    print(f"[ch1] checkpoint saved to {config.checkpoint_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
