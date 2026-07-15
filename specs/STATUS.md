# Spec backlog — live state

States: `draft` (written, unclarified) → `planned` (clarified, no open questions) → `building` (planned + tasks) → `done` (every acceptance criterion met).

## How to read this file

Every spec file is named `NN-ID-slug.md`, where **`NN` is the global build order** and **`ID` is the track**:

| Track | Meaning | Package |
|---|---|---|
| **F** | Foundation — scaffolding everything else stands on | repo-wide |
| **C** | Chapter 2 — Adaptation (QLoRA fine-tune) | `ch2_adaptation/` |
| **O** | Chapter 3 — Operation (GGUF, serving, deploy) | `ch3_operation/` |
| **A** | Chapter 1 — Architecture (from-scratch GPT) | `ch1_architecture/` |
| **X** | Cross-cutting portfolio work | repo-wide |

`ls specs/` now lists the specs in the order they should be built. **Work top to bottom.** The
critical path to a demoable artifact is **F1→F4 → C1→C5 → O0→O5**. Chapter 1 (**A1–A5**) is a
*parallel track*: nothing on the critical path depends on it, so it is numbered after Chapter 3
(17–21) and can be picked up at any time — but it is still required by the definition of done.

Spec IDs (`F1`, `C1`, `A1`, …) are unchanged and are what the slash commands take:
`/plan C1`, `/implement C1 T2`. The `NN-` prefix is for humans reading the directory.

## Progress

**4 of 24 done (17%)** — all foundation. Chapter 2 is underway (C1 building, C2 paused on a
GitHub-token blocker, C3 building, C4 building); Chapters 1 and 3 are unstarted: `train.py` and
`serve.py` are config-loading stubs with no model code behind them.

| Track | Done | Total |
|---|---|---|
| F — Foundation | 4 | 4 |
| C — Chapter 2 (Adaptation) | 0 | 5 |
| O — Chapter 3 (Operation) | 0 | 7 |
| A — Chapter 1 (Architecture) | 0 | 5 |
| X — Cross-cutting | 0 | 3 |

## Phase 0 — Foundation

| # | ID | Spec | State | Depends on | W&B run |
|---|---|---|---|---|---|
| 01 | F1 | repo-tooling-skeleton | done | — | — |
| 02 | F2 | config-convention | done | F1 | — |
| 03 | F3 | eval-harness-and-schema | done | F1 | — |
| 04 | F4 | claude-code-scaffold | done | F1 | — |

## Chapter 2 — Adaptation *(critical path)*

| # | ID | Spec | State | Depends on | W&B run |
|---|---|---|---|---|---|
| 05 | C1 | task-schema-and-base-model | building | F3 | — |
| 06 | C2 | independent-eval-set | building (paused: T4-T7 need `GITHUB_TOKEN`) | C1 | — |
| 07 | C3 | synthetic-training-data | building | C1 | — |
| 08 | C4 | qlora-finetune | building (code done; full GPU run pending) | C2, C3 | — |
| 09 | C5 | head-to-head-eval-table | draft | C4 | — |

## Chapter 3 — Operation *(critical path)*

| # | ID | Spec | State | Depends on | W&B run |
|---|---|---|---|---|---|
| 10 | O0 | serving-thin-slice | draft | F3 | — |
| 11 | O1 | merge-and-gguf-quantize | draft | C5, O0 | — |
| 12 | O2 | fastapi-serving-guardrails | draft | O1 | — |
| 13 | O3 | monitoring | draft | O2 | — |
| 14 | O4 | containerize | draft | O3 | — |
| 15 | O5 | deploy | draft | O4 | — |
| 16 | O6 | in-browser-webllm *(stretch)* | draft | O5 | — |

## Chapter 1 — Architecture *(parallel track — no critical-path spec depends on it)*

| # | ID | Spec | State | Depends on | W&B run |
|---|---|---|---|---|---|
| 17 | A1 | tokenizer-and-data-pipeline | draft | F2 | — |
| 18 | A2 | core-transformer-modules | draft | A1 | — |
| 19 | A3 | custom-training-loop | draft | A2 | — |
| 20 | A4 | kv-cache | draft | A3 | — |
| 21 | A5 | quantization | draft | A4 | — |

## Cross-cutting / portfolio

| # | ID | Spec | State | Depends on | W&B run |
|---|---|---|---|---|---|
| 22 | X1 | readme-and-architecture-diagram | draft | A5, C5, O5 | — |
| 23 | X2 | reproducibility | draft | F2 | — |
| 24 | X3 | interview-defense-notes | draft | A5, C5 | — |

## Cut-lines

Drop these before the critical path if the schedule slips: **O6** (in-browser), deep Chapter 1 extensions, extra quantization levels beyond int8/int4.

**Definition of done:** F1–F4 · Chapter 1 through A5 with both plots · Chapter 2 through C5 with the eval table · Chapter 3 through O5 with a live HF Spaces link · X1 README with the diagram and results.
