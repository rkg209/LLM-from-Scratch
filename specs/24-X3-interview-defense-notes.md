# X3 — Interview-defense notes

| | |
|---|---|
| **State** | draft |
| **Depends on** | A5, C5 |
| **Requirements** | FR-29, BG-1 |
| **W&B run** | — |

## Problem

The project's purpose is to survive a conversation. An interviewer who is any good will not ask "did you build a transformer" — they will ask **why naive attention is slow**, and then they will ask what the KV-cache actually stores, and then they will ask why it does not help at prefill. That is where a portfolio project either proves understanding or reveals that the code was assembled rather than written.

Writing the explanations down is not a documentation task; it is a **test of whether the understanding is real**. An explanation that cannot be written clearly is an explanation that does not exist yet, and the time to discover that is now, not in the interview.

## Scope

A short, technically precise write-up of the key insight from each chapter, in `docs/`.

## Acceptance criteria

1. **Why naive attention is slow, and what the KV-cache fixes** (FR-29a). Covers: what is recomputed on every step and why that is quadratic in sequence length; what the cache stores and what it does *not* (it does not help prefill); why the cached and uncached paths must produce identical logits. Cites **this project's own measured numbers** from A4.
2. **Why a small fine-tuned specialist can beat few-shot prompting on a narrow task** (FR-29b). Covers: why structured-output conformance is where a specialist wins first; what few-shot prompting cannot teach that gradient updates can; the honest limits — where the frontier model still won in *this project's* table, and why; and the cost/latency/privacy argument that holds regardless.
3. **The quantization tradeoff** (FR-29c). Covers: what absmax quantization actually does to the weights; why speed improves; why quality degrades and which capability degrades first (structured output); **this project's measured perplexity delta per precision level** from A5.
4. Each explanation is **self-contained** — readable without the code open — and grounded in the project's own measurements rather than in restated textbook claims.
5. Each names at least one thing the author got wrong or found surprising while building it. This is the most credible paragraph in the document, and the one an interviewer will remember.

## Out of scope

- A tutorial or blog series. This is a defense document — dense, honest, technically precise, and written for one reader who is trying to find the gap.

## Clarifications

*(filled by `/clarify`)*

## Technical plan

*(filled by `/plan`)*

## Tasks

*(filled by `/tasks`)*
