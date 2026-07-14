# X1 — README + architecture diagram

| | |
|---|---|
| **State** | draft |
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

*(filled by `/clarify` — open question: diagram tooling. Mermaid renders natively on GitHub and stays in git; Excalidraw looks better but is a committed PNG that drifts from reality.)*

## Technical plan

*(filled by `/plan`)*

## Tasks

*(filled by `/tasks`)*
