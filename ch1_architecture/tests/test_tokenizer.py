"""Round-trip and lifecycle tests for the hand-rolled BPE tokenizer (A1)."""

from __future__ import annotations

from pathlib import Path

import pytest
from ch1_architecture.tokenizer import BPETokenizer

CORPUS = Path("ch1_architecture/tests/fixtures/corpus_smoke.txt").read_text()

ROUND_TRIP_STRINGS = [
    "First Citizen:\nBefore we proceed any further, hear me speak.",
    "whitespace   runs\t\tand\nnewlines",
    "non-ascii: café, über, \U0001f600 emoji, éèê",
    "",
    "a",
    "   leading and trailing spaces   ",
]


def _trained_tokenizer(vocab_size: int = 300) -> BPETokenizer:
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size)
    return tok


@pytest.mark.parametrize("text", ROUND_TRIP_STRINGS)
def test_round_trip(text: str) -> None:
    tok = _trained_tokenizer()
    assert tok.decode(tok.encode(text)) == text


def test_train_produces_exact_vocab_size() -> None:
    tok = _trained_tokenizer(vocab_size=300)
    assert len(tok.vocab) == 300
    assert len(tok.merges) == 300 - 256


def test_train_with_byte_only_vocab_size_does_no_merges() -> None:
    tok = _trained_tokenizer(vocab_size=256)
    assert len(tok.vocab) == 256
    assert tok.merges == []


def test_encode_before_train_raises() -> None:
    tok = BPETokenizer()
    with pytest.raises(RuntimeError):
        tok.encode("hello")


def test_decode_before_train_raises() -> None:
    tok = BPETokenizer()
    with pytest.raises(RuntimeError):
        tok.decode([0])
