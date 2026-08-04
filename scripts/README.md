# scripts/ — standalone model-preparation helpers

Not an importable package. One-shot scripts run by hand between chapters, not imported at runtime.

| Script | Purpose | Spec |
|---|---|---|
| `fetch_tiny_gguf.py` | Idempotent, sha256-verified download of the `tiny.gguf` CI fixture into `ch3_operation/tests/fixtures/` | O0 |
| `merge_adapter.py` | `peft.merge_and_unload()` → `outputs/merged/`, then a schema-valid sanity check on non-holdout examples | O1 |
| `export_gguf.py` | llama.cpp `convert_hf_to_gguf.py --outtype f16` → `outputs/model-f16.gguf` | O1 |
| `quantize_gguf.py` | llama.cpp `llama-quantize` → `outputs/model.gguf` (`Q4_K_M` by default) | O1 |

Each of the three is independently runnable and idempotent — it skips its step if the
output already exists, unless `--force` is passed. `ch3_operation.evaluate` (in
`ch3_operation/src/ch3_operation/evaluate.py`, not here — it needs the package's
`LlamaCppBackend` and `prompts.format_zero_shot`) is what scores the result.

## The manual GPU-box sequence (O1's real numbers)

This is a **manual, out-of-session** run. The GPU-budget hook blocks any Bash command that
touches `configs/full.yaml` or `load_in_4bit` from inside a Claude Code session on purpose —
do not work around it with `ALLOW_FULL_RUN=1` from a session; run this by hand on the GPU box,
once a real adapter exists from C4/C5:

```bash
uv run python scripts/merge_adapter.py   --config ch3_operation/configs/export_full.yaml
uv run python scripts/export_gguf.py     --config ch3_operation/configs/export_full.yaml
uv run python scripts/quantize_gguf.py   --config ch3_operation/configs/export_full.yaml
EVAL_CONTEXT=1 uv run python -m ch3_operation.evaluate \
  --config ch3_operation/configs/eval_full.yaml
```

Acceptance is one number: the quantized GGUF's schema-validity rate on the holdout must be
**≥** the adapter's rate in `eval/results/finetuned.json` (AC-4). `ch3_operation.evaluate`
raises `QuantizationRegressionError` and exits non-zero if it isn't — re-run
`quantize_gguf.py` with `quant_type: Q5_K_M`, then `Q8_0`, in `export_full.yaml` before
concluding the ladder failed. If it never clears the bar, that goes in the README as a
finding, not a quiet re-run with the bar moved.

Before any of the above, `llama_cpp_dir` in `export_full.yaml` must point at a built
llama.cpp checkout (`cmake -B build && cmake --build build --config Release`) so
`convert_hf_to_gguf.py` and `llama-quantize` exist where the scripts expect them.

The CPU smoke path proves the code path only, with the tiny stand-in model, and is safe to
run from a session:

```bash
uv run python scripts/merge_adapter.py --config ch3_operation/configs/export_smoke.yaml
```
