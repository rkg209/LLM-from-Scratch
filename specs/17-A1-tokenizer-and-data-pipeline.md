# A1 — Tokenizer + data pipeline

| | |
|---|---|
| **State** | draft |
| **Depends on** | F2 |
| **Requirements** | FR-9, CON-5, NFR-4 |
| **W&B run** | — |

## Problem

Chapter 1 claims the model was built from first principles. That claim starts at the tokenizer: `sentencepiece` or HF `tokenizers` would make the merge loop — the part that is actually interesting to explain in an interview — invisible. A hand-rolled BPE is a few dozen lines, and writing it is the difference between knowing that BPE merges frequent pairs and knowing what happens to a byte that never appears in training.

Everything downstream needs it: no tokenizer, no dataset, no training loop.

## Scope

`BPETokenizer` (train / encode / decode / save / load) in pure Python, plus `CorpusDataset` and a DataLoader factory that produce `(input_ids, labels)` pairs by sliding window.

## Acceptance criteria

1. `decode(encode(text)) == text` for every string in the test corpus, including non-ASCII bytes and whitespace runs — the round-trip property (FR-9).
2. `train(corpus, vocab_size)` produces a vocabulary of exactly `vocab_size` entries and an ordered merge list.
3. `encode()` or `decode()` before `train()` raises `RuntimeError` rather than returning garbage.
4. `save()` then `load()` round-trips a tokenizer: the reloaded one encodes an identical corpus to identical ids.
5. `CorpusDataset[i]` returns `(token_ids[i : i+seq_len], token_ids[i+1 : i+seq_len+1])` — labels are inputs shifted by one.
6. No import of `tokenizers`, `sentencepiece`, or `transformers` appears anywhere in `ch1_architecture/src/` (CON-5).
7. Vocab size, sequence length, and batch size all come from the config; nothing is hardcoded.

## Out of scope

- The model → **A2**. The training loop → **A3**.
- Tokenizer efficiency. A plain Python merge loop is expected to be slow; it runs once on a small corpus. Optimizing it is not the lesson.

## Clarifications

*(filled by `/clarify`)*

## Technical plan

*(filled by `/plan`)*

## Tasks

*(filled by `/tasks`)*
