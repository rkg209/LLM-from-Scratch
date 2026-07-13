---
name: gguf-export
description: The merge → convert → quantize → re-validate checklist for turning the Chapter 2 LoRA adapter into a served GGUF model. Use when working on ch3_operation model preparation, the scripts/ export helpers, llama.cpp conversion, or when the served model's output quality is in question.
---

# Adapter → GGUF → served

Four steps, in order. Each one can silently degrade the model, so each one ends with a check.

## 1. Merge the adapter

```
peft.PeftModel.from_pretrained(base, adapter_path).merge_and_unload()  →  outputs/merged/
```

The merged model must be loaded from the **same base tag** the adapter was trained on (`Qwen/Qwen2.5-Coder-1.5B-Instruct`). Merging onto a different revision produces a model that loads, runs, and is quietly wrong.

Check: generate on ~10 training examples. The merged model must produce schema-valid JSON at roughly the adapter's rate. If it collapsed, the merge is wrong — stop, do not convert.

## 2. Convert to GGUF

```
python llama.cpp/convert_hf_to_gguf.py outputs/merged/ --outfile outputs/model-f16.gguf --outtype f16
```

Convert at f16 first, *then* quantize. Converting straight to a quantized format hides which step broke things.

Check: `llama-cli` loads the f16 GGUF and produces coherent output for one prompt.

## 3. Quantize

```
llama.cpp/llama-quantize outputs/model-f16.gguf outputs/model-Q4_K_M.gguf Q4_K_M
```

`Q4_K_M` is the project default — the standard quality/size tradeoff for CPU serving, and it fits the free HF Spaces tier.

## 4. Re-validate against the eval contract — the step that is always skipped

**Quantization is lossy, and structured-output ability is exactly what it degrades first.** A model that emitted perfect JSON at fp16 can start dropping closing braces at Q4. The whole Chapter 3 story collapses if the served model is worse than the model in the eval table and nobody checked.

So: run `eval/harness.py` against the **quantized, GGUF-served** model, on the same holdout sample:

```
EVAL_CONTEXT=1 uv run python -m ch3_operation.evaluate --config ch3_operation/configs/full.yaml
```

Acceptance (FR-21): **the GGUF model's schema-validity rate is no lower than the adapter's** on the same sample. If it is lower, do not ship it — try a less aggressive quantization (Q5_K_M, Q8_0) and re-measure. If the drop persists, that is a finding and it goes in the README, not under the rug.

## Serving invariants

- `n_gpu_layers=0`, always. CPU-only is a hard project constraint (CON-12), not a default to tune.
- `llama-cpp-python` is **not thread-safe** — every `generate()` call is wrapped in a `threading.Lock`.
- The model loads once at process startup, never per request.
- Output is validated by `validator.py` before it reaches the caller: parse → Pydantic → one repair attempt (extract from a ```json fence) → `422`. **Two attempts, then fail.** The validator never invents a missing field to make a response validate.

## Artifacts

The `.gguf` never goes in git (CLAUDE.md #6 — the commit-hygiene hook blocks it). It lives on HF Hub, and the container downloads it at startup.
