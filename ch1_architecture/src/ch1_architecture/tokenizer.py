"""A hand-rolled byte-level BPE tokenizer.

Deliberately unoptimized (A1 out-of-scope): a plain Python pair-count-and-merge loop,
run once on a small corpus, is the point — the merge loop is the part of the tokenizer
story worth being able to explain, not something to hide behind a C++ binding.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


class BPETokenizer:
    def __init__(self) -> None:
        # Seeded with all 256 byte values so encode/decode round-trips even on bytes
        # the training corpus never contained (A1 AC-1).
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: list[tuple[bytes, bytes]] = []
        self.encoder: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        self._trained: bool = False

    def train(self, corpus: str, vocab_size: int) -> None:
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be >= 256 (the byte-level base), got {vocab_size}")

        words: list[list[bytes]] = [
            [bytes([b]) for b in word.encode("utf-8")] for word in corpus.split(" ")
        ]

        while len(self.vocab) < vocab_size:
            pair_counts: Counter[tuple[bytes, bytes]] = Counter()
            for word in words:
                for a, b in zip(word, word[1:], strict=False):
                    pair_counts[(a, b)] += 1
            if not pair_counts:
                break

            best_pair = max(pair_counts.items(), key=lambda item: item[1])[0]
            merged = best_pair[0] + best_pair[1]
            new_id = len(self.vocab)
            self.vocab[new_id] = merged
            self.merges.append(best_pair)
            self.encoder[merged] = new_id

            new_words: list[list[bytes]] = []
            for word in words:
                new_word: list[bytes] = []
                i = 0
                while i < len(word):
                    if (
                        i < len(word) - 1
                        and word[i] == best_pair[0]
                        and word[i + 1] == best_pair[1]
                    ):
                        new_word.append(merged)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_words.append(new_word)
            words = new_words

        self._trained = True

    def encode(self, text: str) -> list[int]:
        if not self._trained:
            raise RuntimeError("tokenizer must be trained (or loaded) before encode()")

        tokens: list[bytes] = [bytes([b]) for b in text.encode("utf-8")]
        for a, b in self.merges:
            merged = a + b
            new_tokens: list[bytes] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return [self.encoder[t] for t in tokens]

    def decode(self, ids: list[int]) -> str:
        if not self._trained:
            raise RuntimeError("tokenizer must be trained (or loaded) before decode()")

        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8")

    def save(self, path: Path | str) -> None:
        if not self._trained:
            raise RuntimeError("tokenizer must be trained before save()")

        payload = {
            "vocab": {str(k): v.hex() for k, v in self.vocab.items()},
            "merges": [[a.hex(), b.hex()] for a, b in self.merges],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as handle:
            json.dump(payload, handle)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path | str) -> BPETokenizer:
        with Path(path).open() as handle:
            payload = json.load(handle)

        tok = cls()
        tok.vocab = {int(k): bytes.fromhex(v) for k, v in payload["vocab"].items()}
        tok.merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in payload["merges"]]
        tok.encoder = {v: k for k, v in tok.vocab.items()}
        tok._trained = True
        return tok
