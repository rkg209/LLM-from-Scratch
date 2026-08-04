# Reproducing this project

There are two audiences here, with two different budgets, and both are served below:
someone who will spend **five minutes** (they watch the whole code path run, on their
laptop, and see it work) and someone who will spend **a GPU afternoon** (they re-run the
full fine-tune / training / export and reproduce the numbers in the README). The
five-minute path is first, and it is the one that must never ask for a GPU, an API key,
or a W&B account.

## Five minutes, no GPU

```bash
git clone <this repo> && cd 3_LLM_from_scratch
make setup
make smoke
```

`make setup` runs `uv sync --extra dev --extra ch1 --extra ch2 --extra ch3` — every
heavy dependency (torch, transformers/peft/trl, llama-cpp-python), all pure CPU wheels.
`make smoke` then runs all three packages' smoke configs and must exit `0`:

```bash
uv run python -m ch1_architecture.train  --config ch1_architecture/configs/smoke.yaml
uv run python -m ch2_adaptation.finetune --config ch2_adaptation/configs/smoke.yaml
uv run python -m ch3_operation.serve     --config ch3_operation/configs/smoke.yaml --check
```

None of the three needs anything beyond what `make setup` installed. Specifically, no
GPU, no API key, and no W&B account:

- Every `*_smoke.yaml` sets `wandb_mode: disabled` — W&B is never contacted.
- `ch2_adaptation/configs/baseline_smoke.yaml` and `datagen_smoke.yaml` set
  `frontier_provider: stub` — no `GEMINI_API_KEY` is read.
- `ch2_adaptation/configs/smoke.yaml` points at
  `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`, a few-KB stand-in model, and sets
  `use_4bit: false` (bitsandbytes needs CUDA) — the smoke fine-tune runs in fp32 on CPU.
- `ch3_operation/configs/smoke.yaml` serves `ch3_operation/tests/fixtures/tiny.gguf`, a
  committed CI fixture, not a downloaded model.

To confirm this for yourself rather than take the paragraph above on faith:

```bash
unset GEMINI_API_KEY WANDB_API_KEY HF_TOKEN
make smoke   # still exits 0
```

If you only care about one chapter, sync just its extra —
`uv sync --extra dev --extra ch1` — and run that chapter's `smoke-ch*` target alone.
`make` is a convenience here, never a dependency: every target is a literal `uv run`
command, spelled out above and in `CLAUDE.md`, so nothing is hidden behind Make syntax.

## What the smoke path does and does not prove

It exercises **every code path end to end**: tokenizer training, model forward/backward,
checkpointing, the KV-cache, quantization, the QLoRA collator and adapter save, the
FastAPI app boot and one real inference. If it breaks, something in the pipeline itself
is broken, independent of any model quality question.

It proves **no meaningful metric**. The ch1 smoke model is 50 gradient steps on a 5 KB
corpus; the ch2 smoke fine-tune is 5 steps on a toy model at `lora_r=4`; the ch3 smoke
server serves a stand-in GGUF that talks but does not code-review. None of these numbers
belong in the README, and none are committed there — see `CH1_BENCHMARK` in `README.md`
for the placeholder this discipline produces. Do not quote a smoke-path perplexity or
latency number as if it were a result.

## A GPU afternoon

Each chapter's full run is launched **by hand, on your own GPU box** (Colab, Kaggle, a
cluster) — never from an agent session; see [`ALLOW_FULL_RUN`](#allow_full_run-and-why-it-exists)
below for why. Environment and commands, per chapter:

**Chapter 1 — from-scratch GPT** (`uv sync --extra ch1`)

```bash
uv run python -m ch1_architecture.train     --config ch1_architecture/configs/full.yaml
uv run python -m ch1_architecture.benchmark --config ch1_architecture/configs/benchmark_full.yaml
```

`configs/full.yaml` trains 5000 steps, `d_model=512`, on the committed TinyShakespeare
corpus (`ch1_architecture/data/corpus.txt`) — expect on the order of an hour on a single
free-tier GPU. `benchmark.py` writes `eval/results/ch1_benchmark.json`, the two plots
under `eval/results/plots/`, and fills the `CH1_BENCHMARK` README marker directly.

**Chapter 2 — QLoRA fine-tune** (`uv sync --extra ch2`, needs `GEMINI_API_KEY` for data
generation and baselines — see `.claude/skills/qlora-recipe/SKILL.md` for the full
recipe)

```bash
uv run python -m ch2_adaptation.data_gen  --config ch2_adaptation/configs/datagen_full.yaml
uv run python -m ch2_adaptation.baseline  --config ch2_adaptation/configs/baseline_full.yaml
uv run python -m ch2_adaptation.finetune  --config ch2_adaptation/configs/full.yaml
EVAL_CONTEXT=1 uv run python -m ch2_adaptation.evaluate \
  --config ch2_adaptation/configs/eval_full.yaml
```

`configs/full.yaml` fine-tunes `Qwen/Qwen2.5-Coder-1.5B-Instruct` with 4-bit NF4 QLoRA
(`lora_r=16`) for 1000 steps — expect roughly 30–60 minutes on a free-tier T4/A100 and
under $1 of total Gemini API spend across data generation and the frontier baseline (both
capped and logged — see `Usage` in `ch2_adaptation/baseline.py`). `evaluate.py` writes
the head-to-head table into the `EVAL_TABLE` README marker.

**Chapter 3 — GGUF export and serving** (`uv sync --extra ch3`, plus a built `llama.cpp`
checkout — see `scripts/README.md`)

```bash
uv run python scripts/merge_adapter.py   --config ch3_operation/configs/export_full.yaml
uv run python scripts/export_gguf.py     --config ch3_operation/configs/export_full.yaml
uv run python scripts/quantize_gguf.py   --config ch3_operation/configs/export_full.yaml
EVAL_CONTEXT=1 uv run python -m ch3_operation.evaluate \
  --config ch3_operation/configs/eval_full.yaml
```

Then, to actually deploy — a separate, human decision, not part of reproducing the
numbers — `scripts/deploy_space.sh` and `scripts/benchmark_serving.py` per
`docs/DEPLOY.md`, which fills the `SERVING_METRICS` marker against the live Space, not a
laptop.

### `ALLOW_FULL_RUN` and why it exists

`.claude/hooks/gpu_budget_guard.py` is a `PreToolUse` hook that blocks any Bash command
in an agent session matching a full-run pattern — `configs/full.yaml`,
`accelerate launch`, `torchrun`, `load_in_4bit`, a `--num_train_epochs` flag, and so on.
It exists because a full run launched from inside a session burns a scarce free-GPU
budget and hangs the session for hours waiting on something an agent cannot meaningfully
supervise. Reproducing this project's full-run numbers is not something an agent session
does; it is something a human does, once, by hand, on their own box. If you are that
human running these commands yourself outside of a session, the guard does not apply —
it only intercepts Bash tool calls from inside Claude Code. Do not set
`ALLOW_FULL_RUN=1` to work around it from within a session; that defeats the point.

## Determinism and its limits

Same config, same seed, same deterministic result — verified by a real test, not
asserted in prose: `uv run pytest eval/tests/test_determinism.py -q` runs the ch1 smoke
training loop twice from the same seed and asserts the loss curve is bit-identical, plus
a direct check that `eval.config.seed_everything(n)` makes two `random`/`torch` draws
identical. `seed_everything` is the single seeding entrypoint every training/eval script
calls first, before touching any RNG.

That guarantee has real limits, and the project does not pretend otherwise:

- **CUDA kernel nondeterminism.** Several CUDA ops (notably scatter/gather-style
  reductions used inside attention and Adam) are not bit-deterministic across runs even
  with a fixed seed, unless `torch.use_deterministic_algorithms(True)` is set — which
  this project does not set, because it measurably slows the full GPU runs and the
  headline metric is a rate (tokens/sec, schema-validity), not a bit-exact loss curve.
  Two full runs of `configs/full.yaml` with the same seed will be close, not identical.
- **Different hardware, different results.** A GPU run reproduced on a different GPU
  model, driver version, or CUDA/cuDNN build is not claimed to be bit-reproducible — only
  the CPU-only smoke path is.
- **Untested, not unsupported.** Windows and non-CUDA GPUs (Apple MPS, ROCm) have not
  been run against this codebase in either configuration. They may work; nobody has
  checked, and this document does not claim they do.

## Provenance — every published number, traced

| Number | Config | Command | W&B run |
|---|---|---|---|
| Ch1 tokens/sec by quantization mode, KV-cache speedup | `ch1_architecture/configs/benchmark_full.yaml` | `python -m ch1_architecture.benchmark --config ...` | *(pending — A5 full run)* |
| Ch1 perplexity by quantization mode | `ch1_architecture/configs/benchmark_full.yaml` | `python -m ch1_architecture.benchmark --config ...` | *(pending — A5 full run)* |
| Ch2 schema-validity / bug-catch — fine-tuned | `ch2_adaptation/configs/full.yaml` + `eval_full.yaml` | `python -m ch2_adaptation.finetune` then `evaluate.py` | *(pending — C4/C5 full run)* |
| Ch2 schema-validity / bug-catch — base (zero-shot) | `ch2_adaptation/configs/baseline_full.yaml` | `python -m ch2_adaptation.baseline --config ...` | *(pending — C1 full run)* |
| Ch2 schema-validity / bug-catch — frontier (3-shot) | `ch2_adaptation/configs/baseline_full.yaml` | `python -m ch2_adaptation.baseline --config ...` | *(pending — C1 full run)* |
| Ch3 p50/p99 latency, throughput | `ch3_operation/configs/export_full.yaml` (model) | `scripts/benchmark_serving.py <space-url>` | *(pending — O5 deploy)* |

This table stays in sync with `specs/STATUS.md`'s own W&B column, which tracks state at
the spec level rather than the number level — check both if one looks stale.

## Pinned artifacts

The fine-tuned adapter and the exported GGUF are pulled from HF Hub **by repo ID plus a
pinned revision SHA**, never `main` — "the latest adapter" is a moving target that makes
last week's eval table describe a different model than the one currently deployed. Until
the adapter from C4/C5 is published, `ch3_operation`'s config points at a bare repo ID
with no revision pin; that is a to-do, resolved the same session the adapter is first
published, not a permanent exception.
