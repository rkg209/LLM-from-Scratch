---
name: qlora-recipe
description: The project's locked QLoRA fine-tuning recipe for Chapter 2 — LoRA config, 4-bit setup, smoke-vs-full deltas, prompt format, and the baseline-comparison requirement. Use when working on ch2_adaptation (finetune.py, lora.py, model.py, data_loader.py, baseline.py) or when deciding fine-tuning hyperparameters.
---

# Chapter 2: the QLoRA recipe

Every value here is locked in `planning/02-architecture.md` §2.2 and `planning/03-system-design.md` §1.2. Use them; do not re-derive them from a blog post.

## The model

`Qwen/Qwen2.5-Coder-1.5B-Instruct` — the *instruct* variant, because the task is instruction-following (emit JSON matching a schema), and 1.5B fits a free Colab GPU under 4-bit. Substituting a different base model means re-running every baseline, so it needs a spec change, not a config tweak (CON-11).

## The LoRA config

```python
LoraConfig(
    r=config.lora_r,                                   # 16 full · 4 smoke
    lora_alpha=config.lora_alpha,                      # 32 full · 8 smoke
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=config.lora_dropout,                  # 0.05
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

After `get_peft_model`, call `print_trainable_parameters()` and log it. **Expect roughly 0.5% of parameters trainable.** If it prints 100%, LoRA did not attach and you are about to full-fine-tune a 1.5B model on a free GPU — stop.

## Smoke versus full: the deltas

The smoke profile exists so the code path can be proven on a CPU in seconds. It is not a small training run; it is a correctness check.

| | smoke (CPU, session, CI) | full (GPU box, manual) |
|---|---|---|
| `use_4bit` | `False` — bitsandbytes needs CUDA | `True` |
| load dtype | `torch.float32`, `device_map="cpu"` | 4-bit NF4, `device_map="auto"` |
| `max_steps` | 5 | 125 (was 1000; overfit the 203-record set — val-loss sweep, spec 08-C4) |
| `batch_size` | 1 | 4 |
| `lora_r` / `alpha` | 4 / 8 | 16 / 32 |

**Never run the full config from a session.** The GPU-budget hook blocks it, and that hook is right (CLAUDE.md #1). The full run is launched by hand on Colab / Kaggle / the cluster; its metrics come back through W&B.

Full-config 4-bit setup:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

## The prompt format

Qwen2.5 chat template, and **loss is computed on the assistant turn only** — mask the system and user tokens with `-100`. Training on the prompt tokens teaches the model to reproduce buggy Java, which is the opposite of the goal.

```
<|im_start|>system\n...<|im_end|>\n
<|im_start|>user\n{code}<|im_end|>\n
<|im_start|>assistant\n{review_json}<|im_end|>
```

## The output is a contract, not a suggestion

Every generated review must validate against `eval/schema.json` — `severity` ∈ {critical, major, minor, info}, plus `category`, `line`, `issue`, `suggested_fix`, and **no additional properties**. Training records that fail validation are dropped at generation time, not repaired by hand.

## A fine-tune without baselines is not a result

The deliverable of Chapter 2 is not a trained adapter. It is the **comparison** (BG-2, FR-19). Three systems, one frozen holdout set, one harness:

1. the fine-tuned model,
2. the base model, zero-shot,
3. a frontier API, 3-shot.

Measure 2 and 3 **before** fine-tuning (`baseline.py`, spec C1). A baseline measured afterward is a baseline you can unconsciously tune against.

Report schema-validity rate and bug-catch rate for each. If the fine-tune loses, the honest table is still the deliverable — the cost, latency, and privacy story stands on its own, and a fabricated win is worth nothing in an interview.

## Never touch the holdout

`finetune.py`, `data_gen.py`, and `data_loader.py` must not read `eval/holdout/` — not to check data quality, not to dedupe, not "just to look." The leakage guard enforces it. Deduplication against the holdout happens once, at curation time, inside spec C2.
