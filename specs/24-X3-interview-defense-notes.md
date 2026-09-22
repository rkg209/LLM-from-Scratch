# X3 — Interview-defense notes

| | |
|---|---|
| **State** | done (2026-09-22 audit; see specs/STATUS.md) |
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

None open. `.claude/plans/Extension.md` D-6 fixed the file shape: three separate files
under `docs/` plus a one-paragraph `docs/README.md` index, rather than one long document —
AC-4's self-containment requirement is what a single cross-referenced document would
quietly violate.

## Technical plan

See `.claude/plans/Extension.md`, section "X3 — Interview-defense notes". Each note is
written against the code it defends, re-read immediately before writing (the re-read *is*
the work, per the plan) — `attention.py`/`kv_cache.py`/`test_kv_cache.py` for the
KV-cache note, `baseline.py`/`eval/schema.json`/the qlora-recipe skill for the
finetune-vs-prompting note, `quantize.py`/`benchmark.py`/the gguf-export skill for the
quantization note. "What I got wrong" is a section within each note (T4), not a fourth
file, mined from real implementation surprises rather than invented for the occasion.

## Tasks

All five tasks from the plan are committed:

1. ✅ T1 — `docs/kv-cache.md`: the `O(n²)` cost of naive decoding, what the cache stores
   (K/V, not Q) and why not Q, why it does nothing for prefill, the bit-identical-logits
   test as the actual gate rather than eyeballed samples, the memory cost the speed buys.
2. ✅ T2 — `docs/finetune-vs-prompting.md`: why structured-output conformance is where a
   specialist wins first, what few-shot prompting structurally cannot do, the honest limit
   (frontier model's win left as a pending, named slot rather than guessed), the
   cost/latency/privacy argument that holds regardless of the accuracy table.
3. ✅ T3 — `docs/quantization.md`: what absmax quantization does to a tensor concretely,
   why speed is a bandwidth story rather than a compute story (and why fp16 breaks that on
   CPU), why structured output is expected to degrade first under aggressive quantization.
4. ✅ T4 — each note ends with a genuine, specific surprise: the `seq_offset` mask-slice
   bug (kv-cache), the `Usage`-accounting rationale (finetune-vs-prompting), the
   fp16-slower-than-fp32-on-CPU result (quantization).
5. ✅ T5 — `docs/README.md` index, linked from `README.md`'s "How this repo is built" and
   inline from the Results section next to the metric each note explains.

**Pending, gated on A4/A5/C5's full runs (cannot happen from a session):** the specific
measured numbers each note cites — cached-vs-uncached tokens/sec (kv-cache), the frontier
model's specific winning metric (finetune-vs-prompting), the perplexity-per-precision-level
table (quantization). Each is an explicit `*(pending — ... full run)*` note pointing at the
`eval/results/*.json` file that will hold it, not a placeholder number.
