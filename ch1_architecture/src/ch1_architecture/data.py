"""Sliding-window dataset over tokenized text, and its DataLoader factory (A1)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from ch1_architecture.config import GPTConfig


class CorpusDataset(Dataset):
    def __init__(self, token_ids: list[int], seq_len: int) -> None:
        if len(token_ids) <= seq_len:
            raise ValueError(
                f"corpus has {len(token_ids)} tokens, needs more than seq_len ({seq_len})"
            )
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.token_ids[index : index + self.seq_len]
        labels = self.token_ids[index + 1 : index + self.seq_len + 1]
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def make_dataloader(dataset: CorpusDataset, config: GPTConfig) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
