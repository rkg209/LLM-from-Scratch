# `tiny.gguf` — CI test fixture

| | |
|---|---|
| **Source** | `stories260K.gguf` from [`ggml-org/models-moved`](https://huggingface.co/ggml-org/models-moved) on Hugging Face Hub — the moved location of `ggml-org/models`, the canonical llama.cpp CI fixture repo. |
| **Upstream file** | `tinyllamas/stories260K.gguf` |
| **Original model** | Karpathy's `stories260K` — a 260K-parameter llama2.c toy model trained on TinyStories, converted to GGUF. |
| **Licence** | MIT (llama2.c / TinyStories). |
| **Size** | 1,185,376 bytes (~1.1 MB). |
| **sha256** | `270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d` |
| **Fetched by** | `scripts/fetch_tiny_gguf.py` (idempotent, sha256-verified). |

## Why this file is committed to git

CLAUDE.md non-negotiable #6 forbids weights, datasets, and `.gguf` files in git. This is the
one sanctioned exception (spec `10-O0-serving-thin-slice.md`): CI has no GPU/network budget to
download a real model, and `n_gpu_layers=0` + FastAPI startup + request wiring all need a real,
loadable GGUF to exercise `llama-cpp-python` honestly — a mock would let a broken `Llama(...)`
call pass silently. `stories260K` is small enough (~1 MB, under the hook's 5 MB cap) and
permissively licensed enough (MIT) that committing it is the sanctioned exception, not a
workaround.

`.gitignore` whitelists `!ch3_operation/tests/fixtures/tiny.gguf` and
`.claude/hooks/commit_hygiene.py` exempts `tests/fixtures/*.gguf` under 5 MB from the
no-weights block — both pre-date this file and were written to allow exactly this.

Output quality is irrelevant here: `stories260K` writes children's stories, not JSON code
reviews. O0 only proves the serving path carries bytes end to end; a real (or fine-tuned)
model is what O1 swaps in through the same `Backend` seam.
