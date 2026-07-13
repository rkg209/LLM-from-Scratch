# Spec backlog — live state

States: `draft` (written, unclarified) → `planned` (clarified, no open questions) → `building` (planned + tasks) → `done` (every acceptance criterion met).

Build top to bottom. The critical path to a demoable artifact is **F1→F3 → C1→C5 → O0→O5**; Chapter 1 hangs off the side and can be worked in parallel.

## Phase 0 — Foundation

| ID | Spec | State | Depends on | W&B run |
|---|---|---|---|---|
| F1 | repo-tooling-skeleton | done | — | — |
| F2 | config-convention | done | F1 | — |
| F3 | eval-harness-and-schema | done | F1 | — |
| F4 | claude-code-scaffold | done | F1 | — |

## Chapter 1 — Architecture

| ID | Spec | State | Depends on | W&B run |
|---|---|---|---|---|
| A1 | tokenizer-and-data-pipeline | draft | F2 | — |
| A2 | core-transformer-modules | draft | A1 | — |
| A3 | custom-training-loop | draft | A2 | — |
| A4 | kv-cache | draft | A3 | — |
| A5 | quantization | draft | A4 | — |

## Chapter 2 — Adaptation

| ID | Spec | State | Depends on | W&B run |
|---|---|---|---|---|
| C1 | task-schema-and-base-model | draft | F3 | — |
| C2 | independent-eval-set | draft | C1 | — |
| C3 | synthetic-training-data | draft | C1 | — |
| C4 | qlora-finetune | draft | C2, C3 | — |
| C5 | head-to-head-eval-table | draft | C4 | — |

## Chapter 3 — Operation

| ID | Spec | State | Depends on | W&B run |
|---|---|---|---|---|
| O0 | serving-thin-slice | draft | F3 | — |
| O1 | merge-and-gguf-quantize | draft | C5, O0 | — |
| O2 | fastapi-serving-guardrails | draft | O1 | — |
| O3 | monitoring | draft | O2 | — |
| O4 | containerize | draft | O3 | — |
| O5 | deploy | draft | O4 | — |
| O6 | in-browser-webllm *(stretch)* | draft | O5 | — |

## Cross-cutting / portfolio

| ID | Spec | State | Depends on | W&B run |
|---|---|---|---|---|
| X1 | readme-and-architecture-diagram | draft | A5, C5, O5 | — |
| X2 | reproducibility | draft | F2 | — |
| X3 | interview-defense-notes | draft | A5, C5 | — |

## Cut-lines

Drop these before the critical path if the schedule slips: **O6** (in-browser), deep Chapter 1 extensions, extra quantization levels beyond int8/int4.

**Definition of done:** F1–F4 · Chapter 1 through A5 with both plots · Chapter 2 through C5 with the eval table · Chapter 3 through O5 with a live HF Spaces link · X1 README with the diagram and results.
