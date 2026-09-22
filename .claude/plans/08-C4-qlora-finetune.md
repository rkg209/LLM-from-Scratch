# Plan: C4 — QLoRA fine-tune

## Context

C4 is the project's central demonstrated skill: turning `Qwen/Qwen2.5-Coder-1.5B-Instruct` into
a Java review specialist via QLoRA, with code that is provable on CPU in seconds and correct on a
GPU for an hour, since whoever wrote it won't be watching the GPU run. Unlike C2/C3, almost
nothing here is an open design decision — the LoRA config, the 4-bit setup, the prompt/masking
format, and the smoke-vs-full deltas are all locked verbatim in the `qlora-recipe` skill and
`planning/03-system-design.md` §1.2, and `ch2_adaptation/configs/{smoke,full}.yaml` already exist
(written under spec F2, before any C4 code). C4's job is almost purely mechanical: write the four
modules the recipe already specifies, against configs that already exist.

**One real problem surfaced while planning, not previously flagged anywhere:** `smoke.yaml`
currently sets `model_tag: Qwen/Qwen2.5-Coder-1.5B-Instruct` — the real 1.5B model — and
`FinetuneConfig.__post_init__` (`ch2_adaptation/src/ch2_adaptation/config.py:41-45`) requires
`model_tag == LOCKED_MODEL_TAG` unconditionally, with **no smoke carve-out**. That's inconsistent
with `BaselineConfig`, which already solved exactly this problem in the same file
(`SMOKE_MODEL_TAG`, a tiny stand-in, because "fp32 Qwen-1.5B on CPU is ~6GB resident and
40-120s of generation alone" — C1's own AC-6 amendment). A first-time load of the real 1.5B model
(a ~3GB download) cannot reliably finish inside AC-1's 120-second budget. This plan applies the
same fix C1 already established, as a mechanical extension rather than a new decision:

**Amendment (recorded in the spec, not applied silently):** `FinetuneConfig.__post_init__` gets
the same smoke/full split `BaselineConfig` already has — smoke requires `model_tag ==
SMOKE_MODEL_TAG`, full requires `model_tag == LOCKED_MODEL_TAG`. `smoke.yaml`'s `model_tag`
changes to `SMOKE_MODEL_TAG`. This only touches config validation; the recipe itself (LoRA
r/alpha, 4-bit setup, prompt format) is unchanged.

## Files

**Modified — `ch2_adaptation/src/ch2_adaptation/config.py`**
`FinetuneConfig.__post_init__` gains an `is_smoke` branch mirroring `BaselineConfig`'s (lines
74-92); `smoke.yaml`'s `model_tag` → `SMOKE_MODEL_TAG`.

**New — `ch2_adaptation/src/ch2_adaptation/lora.py`**
- `build_lora_config(config: FinetuneConfig) -> LoraConfig` — the locked recipe verbatim:
  `r=config.lora_r, lora_alpha=config.lora_alpha, target_modules=["q_proj","v_proj","k_proj","o_proj"], lora_dropout=config.lora_dropout, bias="none", task_type=TaskType.CAUSAL_LM`.
- `attach_lora(model, config) -> PeftModel` — `get_peft_model`, then
  `print_trainable_parameters()` (logged), then a hard check: if the trainable fraction is not
  roughly 0.5% (say, outside 0.1%–5%), raise — AC-3's "the run must be stopped," not just logged.

**New — `ch2_adaptation/src/ch2_adaptation/data_loader.py`**
- `format_prompt(record: dict) -> str` — exact Qwen2.5 chat-template shape from the recipe:
  `<|im_start|>system...` / `user...` / `assistant...{review_json}`.
- `ReviewDataset(torch.utils.data.Dataset)` — `__init__(jsonl_path, tokenizer, config)`,
  `__len__`, `__getitem__(idx) -> {"input_ids","attention_mask","labels"}`. Labels for the
  system+user span are set to `-100`; only the assistant JSON span keeps real token ids
  (AC-4). Implementation tokenizes the prompt-only prefix and the full sequence separately to
  find the split boundary precisely, rather than guessing an offset.

**Modified — `ch2_adaptation/src/ch2_adaptation/model.py`**
Add `load_base_model_for_training(config: FinetuneConfig) -> tuple[model, tokenizer]` alongside
the existing `make_hf_generator` (inference-only, built for C1's baseline — untouched). Smoke
branch: `torch.float32`, `device_map="cpu"`, no `bitsandbytes` import anywhere on that path (it's
a Linux-only extra per `pyproject.toml`, and the dev machine here is Darwin — the import must
stay inside the `if config.use_4bit:` branch so smoke never touches it). Full branch:
`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)`,
`device_map="auto"`.

**Modified — `ch2_adaptation/src/ch2_adaptation/finetune.py`**
Replace the scaffold body of `main()`: load train/val via `ReviewDataset`, load model+tokenizer
via `load_base_model_for_training`, attach LoRA via `attach_lora`, build `trl.SFTTrainer` with
`transformers.DataCollatorForSeq2Seq`, `report_to="wandb"` (`wandb.init(project=config.wandb_project)` —
reuses the existing `wandb` MCP server wiring in `.mcp.json`, no new integration code), train,
save the adapter to `config.output_dir`. **Never reads `eval/holdout/`** (CON-6) — validation
comes only from `data/val.jsonl`; the leakage guard would block it anyway if it tried.

**New — tests:**
- `ch2_adaptation/tests/test_lora.py` — `build_lora_config` field values match the locked recipe
  exactly; `attach_lora` raises on a mock 100%-trainable model (AC-3's stop condition).
- `ch2_adaptation/tests/test_data_loader.py` — `format_prompt` output shape; the **label-masking
  assertion AC-4 explicitly requires**: the label tensor is `-100` across the full prompt span
  and non-`-100` across the assistant span, for a known fixture record.
- `ch2_adaptation/tests/test_finetune_config.py` (existing file) — extended with the new
  smoke/full `model_tag` branch.

No new dependencies: `torch`, `transformers`, `peft`, `trl`, `bitsandbytes` (Linux-only marker)
are already declared under the `ch2` extra in `pyproject.toml`.

## Task sequence

1. `FinetuneConfig` smoke/full amendment + `smoke.yaml` fix + test. Verify:
   `uv run pytest ch2_adaptation/tests/test_finetune_config.py -q`.
2. `lora.py` + tests. Verify: `uv run pytest ch2_adaptation/tests/test_lora.py -q`.
3. `data_loader.py` + tests (the AC-4 masking assertion is the one that matters most here).
   Verify: `uv run pytest ch2_adaptation/tests/test_data_loader.py -q`.
4. `model.py::load_base_model_for_training` (both branches; smoke path only, since full needs
   real GPU hardware to exercise bitsandbytes).
5. `finetune.py::main()` real training loop wired to all of the above. Verify:
   `time uv run python -m ch2_adaptation.finetune --config ch2_adaptation/configs/smoke.yaml`
   completes under 120s, prints the trainable-parameter fraction, and saves an adapter to
   `outputs/adapter-smoke/`.
6. Spec/STATUS updates — record the `FinetuneConfig` amendment (parallel to how C1 recorded its
   own AC-1/AC-6 amendments), note that the full GPU run is manual-only per CON-4/the GPU-budget
   hook, and that the W&B run ID gets recorded in `specs/STATUS.md` **after** that manual run
   (NFR-16) — same deferred-completion shape as C1's still-open AC-5.

## Verification

- `uv run ruff check . && uv run black --check . && uv run pytest -q` after every task.
- Smoke run timed end-to-end, confirmed under 120s, confirmed it never imports `bitsandbytes`.
- `attach_lora`'s trainable-parameter log inspected by hand once on the smoke path to sanity-check
  it lands near the expected ~0.5%-of-smoke-scale figure (smoke uses `r=4` so the exact fraction
  differs from full's `r=16`, but it must not be anywhere near 100%).
- Full run remains out of scope for in-session verification by design (CLAUDE.md #1) — the
  runbook for the manual GPU launch and the W&B-run-ID bookkeeping step is the deliverable, not
  the run itself.
