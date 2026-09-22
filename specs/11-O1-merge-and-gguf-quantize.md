# O1 — Merge + GGUF quantize

| | |
|---|---|
| **State** | building |
| **Depends on** | C5, O0 |
| **Requirements** | FR-21, CON-12, NFR-13 |
| **W&B run** | — |

## Problem

The adapter that won the eval table is a set of low-rank matrices sitting on top of a 3 GB fp16 model. Nothing on a free CPU tier can serve that. Getting it to the edge means merging the adapter into the base weights, converting to GGUF, and quantizing to 4-bit — and each of those three steps can silently degrade the model.

The step that gets skipped is the last one: **checking that the quantized model still does what the eval table says it does.** Quantization damages structured-output ability before it damages anything else — a model that emitted flawless JSON at fp16 starts dropping closing braces at Q4. If nobody re-measures, the README publishes a number for a model that is not the one being served, and the entire Chapter 3 claim is quietly false.

## Scope

`scripts/merge_adapter.py`, `scripts/export_gguf.py`, `scripts/quantize_gguf.py`, and the re-validation of the quantized model against the eval contract.

## Acceptance criteria

1. The adapter is merged onto the **same base tag it was trained on** (`Qwen/Qwen2.5-Coder-1.5B-Instruct`) via `merge_and_unload()`, producing `outputs/merged/`.
2. Conversion goes to **f16 GGUF first**, then quantizes to `Q4_K_M` — a two-step path, so a failure can be localized to conversion or to quantization.
3. **The quantized, GGUF-served model is scored by `eval/harness.py` on the same holdout sample as C5, and its schema-validity rate is no lower than the adapter's** (FR-21b). This is the acceptance criterion the spec exists for.
4. If the rate drops, a less aggressive quantization (Q5_K_M, Q8_0) is tried and re-measured. If the drop persists at every level, **it is published in the README as a finding**, not hidden.
5. The `.gguf` file is **never committed** — it lives on HF Hub, and the commit-hygiene hook blocks it (FR-21a, NFR-13).
6. The served GGUF model is the one the README's serving section describes, and the README states which quantization level it is.

## Out of scope

- Guardrails and the repair strategy → **O2**. Metrics → **O3**.
- Fine-tuning a better model in response to a quantization loss. That is a C4 change with a full re-eval, not a patch here.

## Clarifications

*(filled by `/clarify`)*

## Technical plan

See `.claude/plans/Operation.md` §O1 for the full umbrella plan. Summary: `scripts/merge_adapter.py` (base-tag guard via the adapter's own `adapter_config.json`, `peft.merge_and_unload()`, sanity-generate on non-holdout examples), `scripts/export_gguf.py` (`convert_hf_to_gguf.py --outtype f16`), `scripts/quantize_gguf.py` (`llama-quantize`, ladder-selectable `quant_type`), `ch3_operation/src/ch3_operation/evaluate.py` (real `LlamaCppBackend`, scores via `eval.harness.score_outputs`, raises `QuantizationRegressionError` below the adapter's rate). `ExportConfig`/`EvalConfig` added to `ch3_operation/config.py`, with `export_{smoke,full}.yaml`/`eval_{smoke,full}.yaml`. T1–T6 done and tested (23 tests); T7 (the real merge → GGUF → re-eval on a real adapter) is a manual, out-of-session run per `scripts/README.md`'s runbook, blocked on C4/C5.

## Tasks

*(filled by `/tasks`)*
