# A1 — Tokenizer + data pipeline

| | |
|---|---|
| **State** | done |
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

- **Corpus source.** A committed public-domain text file (TinyShakespeare slice, ~400 KB) at `ch1_architecture/data/corpus.txt`, plus a ~5 KB slice at `tests/fixtures/corpus_smoke.txt` for the smoke config and tests. A network fetch in `data.py` would make `/smoke ch1` and CI flaky; every other smoke path in this repo runs offline.
- **`save()` byte encoding.** JSON has no bytes type, so `vocab`/`merges` are hex-encoded on save and decoded on load.
- **Merge-loop performance.** Deliberately unoptimized plain-Python pair counting (AC out-of-scope). It runs once per distinct corpus and its result is cached by A3 keyed on a corpus hash, so retraining it every smoke run does not eat into the 120s budget.

## Technical plan

See `.claude/plans/AfterOperation.md` §A1. `BPETokenizer` (`tokenizer.py`) holds `vocab: dict[int, bytes]`, `merges: list[tuple[bytes, bytes]]`, `encoder: dict[bytes, int]`, `_trained: bool`; seeded with all 256 byte values so the round-trip property holds on bytes never seen in training. `train()` repeatedly merges the most frequent adjacent byte-pair until `vocab_size` is reached (word-boundary pre-split on spaces, standard BPE practice — does not affect encode/decode correctness). `encode`/`decode`/`save` raise `RuntimeError` before `train()`/`load()`. `CorpusDataset` (`data.py`) is a sliding window: `__getitem__(i)` returns `(ids[i:i+seq_len], ids[i+1:i+seq_len+1])`; `make_dataloader` wraps it with a `torch.Generator` seeded from config. `vocab_size`/`seq_len`/`batch_size` are config fields; `corpus_path`/`tokenizer_path` were added to `GPTConfig`.

## Tasks

- **A1-T1** — `tests/test_purity.py` (ban-list grep over `src/**/*.py`) + `data/corpus.txt` + `tests/fixtures/corpus_smoke.txt` + `GPTConfig.corpus_path`/`tokenizer_path` + both YAMLs + `test_gpt_config.py` update. **Done.**
- **A1-T2** — `tokenizer.py`: `train`/`encode`/`decode`, `RuntimeError` guards, round-trip tests (ASCII, non-ASCII, whitespace runs). **Done.**
- **A1-T3** — `save`/`load` + reload-encodes-identically test. **Done.**
- **A1-T4** — `data.py`: `CorpusDataset`, `make_dataloader`, shift-by-one and shape tests. **Done.**
