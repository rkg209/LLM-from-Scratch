# X1 — README + architecture diagram

| | |
|---|---|
| **State** | building |
| **Depends on** | A5, C5, O5 |
| **Requirements** | FR-27, BG-1, NFR-7 |
| **W&B run** | — |

## Problem

Almost everyone who evaluates this project will read the README and nothing else. Six months of work is judged in ninety seconds, by someone scrolling on a phone between interviews.

So the README is not documentation — it is the artifact. It has to answer, above the fold, what was built, what the numbers were, and where the live demo is. A reader who has to scroll past a setup section to find the eval table will not find the eval table.

## Scope

The README: one architecture diagram covering all three chapters, the Chapter 2 eval table, the Chapter 1 speedup and quantization plots, the live links, and a short honest description of each chapter.

## Acceptance criteria

1. **One architecture diagram** covering all three chapters and how they connect, rendering correctly on GitHub (FR-27a).
2. The **Chapter 2 head-to-head eval table** — fine-tuned vs base vs frontier API, schema-validity and bug-catch, with `n` (FR-27b).
3. The **Chapter 1 speedup curve and perplexity-vs-quantization plot**, both embedded (FR-27c).
4. The **live HF Spaces link**, plus a worked example: Java in, JSON review out (FR-27d).
5. A brief description of each chapter and what it demonstrates (FR-27e).
6. The **cost and compute are stated plainly** — ≈$1 of API spend, free GPU tiers — because the constraint is part of what makes the result impressive.
7. **The honest framing is stated, not buried**: Chapter 2 is distillation of a frontier model into a small specialist, the eval set is independent and partly real, and any metric on which the fine-tuned model lost is shown as a loss (NFR-7).
8. Results appear **before** setup instructions.

## Out of scope

- The interview-defense notes → **X3**. The reproduction instructions → **X2** (a `REPRODUCING.md` linked from here).
- A landing page, blog post, or video walkthrough.

## Clarifications

Diagram tooling resolved: **Mermaid, inline in `README.md`** (`.claude/plans/Extension.md`
D-1). Renders natively on GitHub, lives in git as text, diffs alongside the code it
describes, and cannot become a stale PNG claiming the system does something it stopped
doing.

## Technical plan

See `.claude/plans/Extension.md`, section "X1 — README + architecture diagram". Number-
bearing slots are filled only by the existing generators (`eval.report.update_readme_section`
via `benchmark.py`/`evaluate.py`/`scripts/benchmark_serving.py`) — X1 adds no new
number-writing code (D-3). Prose slots that cannot be marker-driven (the live link, the
worked example) carry an explicit `*(pending — O5 deploy)*` placeholder rather than a
plausible-looking invented value (D-4).

## Tasks

T1, T3, T4, T5 from the plan are committed this session:

1. ✅ T1 — one Mermaid `flowchart` covering all three chapters: Chapter 1 drawn as a
   disconnected parallel subgraph (no edge into Ch2/Ch3, a deliberate design fact); Chapter
   2's synthetic-data → QLoRA → adapter → HF Hub → eval path; Chapter 3's
   merge → GGUF → FastAPI → Docker → Spaces path; `eval/schema.json`/`eval/harness.py`
   drawn as the shared spine touching both Ch2 and Ch3.
2. ✅ T3 — Results section moved above Getting started (already true structurally, now
   also true in spirit: Live demo and Architecture precede it too); holdout description
   (n=40, 30 synthetic + 10 mined, frozen and deduplicated) added under the Ch2 table;
   plot-meaning prose added under `CH1_BENCHMARK`, including the expected fp16-on-CPU
   result; the smoke-numbers caveat kept verbatim.
3. ✅ T4 — `## What this cost` added (≈$1 API spend, free GPU tiers, CPU-only serving).
4. ✅ T5 — `## How to read these numbers` added: distillation framing, independent/
   partly-real holdout, losses shown next to wins, Chapter 1 is a toy.

**Deferred, per the plan's own gating (steps 4–5 of its build-order table), not an
oversight:**

- T2 (live demo link + worked example) — gated on the O5 deploy. `## Live demo` carries an
  explicit `*(pending — O5 deploy)*` placeholder, pointing at the local API example in the
  meantime.
- T6 (fill every number slot) — gated on A5 + C5 + O5 all landing. Cannot run from this
  session; the markers, provenance table, and docs/ placeholders are all wired so T6 is a
  fill-and-link pass, not further design work, once the runs exist.

AC-1 (diagram renders on GitHub) needs verification by looking, after push — noted rather
than claimed sight-unseen.
