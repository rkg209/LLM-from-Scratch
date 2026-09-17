"""Sliding-window dataset over tokenized text, and its DataLoader factory (A1)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from ch1_architecture.config import GPTConfig


def split_corpus_text(text: str, val_fraction: float) -> tuple[str, str]:
    """Split the raw corpus into a training head and a held-out validation tail.

    The split happens on the text, before tokenization, and deliberately not after it.
    A token-level split would still leave the tokenizer trained on the whole corpus, so
    the BPE merge table — which is part of the model — would have been fitted to text the
    model is later scored on. It is a small leak next to training on it outright, but it
    is the same kind, and the point of holding anything out is that nothing sees it.

    The split is contiguous and takes the tail, so it is fully determined by the corpus
    and one number: no seed, no shuffle, nothing that could quietly differ between the
    training run and the benchmark that has to score the same held-out text. Contiguous
    also means no training window can overlap a validation window, which an interleaved
    split would allow for any window straddling the boundary.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0.0, 1.0), got {val_fraction}")

    n_val = int(len(text) * val_fraction)
    if n_val == 0:
        return text, ""
    return text[:-n_val], text[-n_val:]


class CorpusDataset(Dataset):
    """Sliding-window views over a token stream.

    `stride=1` (training) gives every offset as a sample; `stride=seq_len` gives
    non-overlapping windows, which is what validation and `benchmark.compute_perplexity`
    both use — each held-out token is then scored exactly once, so the validation loss
    and the published perplexity are the same measurement on the same tokens.
    """

    def __init__(self, token_ids: list[int], seq_len: int, stride: int = 1) -> None:
        if len(token_ids) <= seq_len:
            raise ValueError(
                f"corpus has {len(token_ids)} tokens, needs more than seq_len ({seq_len})"
            )
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.token_ids = token_ids
        self.seq_len = seq_len
        self.stride = stride

    def __len__(self) -> int:
        return (len(self.token_ids) - self.seq_len - 1) // self.stride + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.stride
        inputs = self.token_ids[start : start + self.seq_len]
        labels = self.token_ids[start + 1 : start + self.seq_len + 1]
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def make_dataloader(dataset: CorpusDataset, config: GPTConfig) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )


def make_eval_dataloader(dataset: CorpusDataset, config: GPTConfig) -> DataLoader:
    """Unshuffled, so the validation loss is a fixed number for a given model and split —
    two runs of the same checkpoint must not disagree because the batches were drawn in a
    different order.
    """
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
