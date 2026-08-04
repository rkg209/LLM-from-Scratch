"""CorpusDataset and make_dataloader tests (A1)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from ch1_architecture.config import load_gpt_config

from ch1_architecture.data import CorpusDataset, make_dataloader

CONFIG_PATH = Path("ch1_architecture/configs/smoke.yaml")


def test_shift_by_one() -> None:
    ids = list(range(20))
    dataset = CorpusDataset(ids, seq_len=5)
    inputs, labels = dataset[0]
    assert inputs.tolist() == [0, 1, 2, 3, 4]
    assert labels.tolist() == [1, 2, 3, 4, 5]


def test_length_never_runs_off_the_end() -> None:
    ids = list(range(20))
    dataset = CorpusDataset(ids, seq_len=5)
    assert len(dataset) == 15
    inputs, labels = dataset[len(dataset) - 1]
    assert inputs.tolist() == ids[-6:-1]
    assert labels.tolist() == ids[-5:]


def test_too_short_corpus_raises() -> None:
    with pytest.raises(ValueError, match="seq_len"):
        CorpusDataset(list(range(5)), seq_len=5)


def test_dtype_is_long() -> None:
    dataset = CorpusDataset(list(range(20)), seq_len=5)
    inputs, labels = dataset[0]
    assert inputs.dtype == torch.long
    assert labels.dtype == torch.long


def test_dataloader_batches_from_config() -> None:
    config = load_gpt_config(CONFIG_PATH)
    dataset = CorpusDataset(list(range(1000)), seq_len=config.seq_len)
    loader = make_dataloader(dataset, config)

    inputs, labels = next(iter(loader))
    assert inputs.shape == (config.batch_size, config.seq_len)
    assert labels.shape == (config.batch_size, config.seq_len)


def test_dataloader_seed_is_reproducible() -> None:
    config = load_gpt_config(CONFIG_PATH)
    dataset = CorpusDataset(list(range(1000)), seq_len=config.seq_len)

    first = next(iter(make_dataloader(dataset, config)))[0]
    second = next(iter(make_dataloader(dataset, config)))[0]
    assert torch.equal(first, second)
