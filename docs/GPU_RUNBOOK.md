# GPU runbook — what's left, in order

**Code-complete, run-pending.** Every module across all three chapters is written,
tested, and smoke-verified end to end (`make smoke` exits 0 with no GPU, no API key, no
W&B account — see `REPRODUCING.md`). What's left is executing the real configs — GPU
time, API keys, a GitHub token, and one human deploy decision — none of which a Claude
Code session performs itself (`CLAUDE.md` non-negotiable #1, enforced by
`.claude/hooks/gpu_budget_guard.py`).

This file is the single ordered sequence to run those steps, outside any agent session,
on your own GPU box. It consolidates `HANDOFF.md`'s runbook (written mid-Chapter-2) and
`REPRODUCING.md`'s "GPU afternoon" section (written after O1–O5/A1–A5 landed) into one
current path, plus — at the end — the parts of the repo that are genuinely **not
implemented yet**, which running configs will not fix.

Check `specs/STATUS.md` before starting; it is the live source of truth and may have
moved since this was written.

## Step 0 — credentials (once)

```bash
export GEMINI_API_KEY=...              # C1 baselines, C3 data generation
export PRICE_PER_1K_INPUT_USD=...      # cost tracking (data-curator budget guard)
export PRICE_PER_1K_OUTPUT_USD=...
export GITHUB_TOKEN=...                # C2 mining, or use `gh auth login` instead
export WANDB_API_KEY=...               # optional — W&B logging on real runs
export HF_TOKEN=...                    # O5 deploy, pinning adapter/GGUF revisions
```

## Step 1 — C1: real baselines

```bash
uv run python -m ch2_adaptation.baseline --config ch2_adaptation/configs/baseline_full.yaml
```
→ `eval/results/baselines.json`. Commit it.

## Step 2 — C2: mine, label, freeze the holdout set

10 real Java/Spring snippets need mining — the `github` MCP server in `.mcp.json` 400s
on this endpoint regardless of token shape (see `HANDOFF.md`'s "GitHub blocker" section
for the full diagnosis). Cheapest fix: use `gh search code` / `gh api` directly instead
of MCP, or mine by hand and paste the 10 `(code, source_repo, license, commit_url)`
tuples into a session for it to write to `eval/staging/mined_raw.jsonl`.

Once mined:
```bash
uv run python -m ch2_adaptation.holdout_curator label-mined \
  --config ch2_adaptation/configs/baseline_full.yaml
uv run python -m ch2_adaptation.holdout_curator freeze --curator "<your name>"
```

Then, **in your own terminal only** — the leakage guard blocks this from any session,
unconditionally, no override:
```bash
mkdir -p eval/holdout
mv eval/staging/holdout.jsonl eval/holdout/holdout.jsonl
mv eval/staging/manifest.json eval/holdout/manifest.json
git add eval/holdout/holdout.jsonl eval/holdout/manifest.json eval/frozen_hashes.txt
git commit -m "C2: freeze the 40-record independent eval set"
```

## Step 3 — C3: generate real training data

Needs C2's `eval/frozen_hashes.txt`.
```bash
uv run python -m ch2_adaptation.data_gen --config ch2_adaptation/configs/datagen_full.yaml
```
→ `ch2_adaptation/data/{train.jsonl,val.jsonl}` (gitignored) +
`ch2_adaptation/data/provenance.json` (commit this one). Expect well under $1 of Gemini
spend — capped and logged, see `ch2_adaptation/baseline.py::Usage`.

## Step 4 — C4: the real QLoRA fine-tune (needs an actual GPU)

```bash
uv run python -m ch2_adaptation.finetune --config ch2_adaptation/configs/full.yaml
```
`lora_r=16`, 4-bit NF4, 125 steps (1000 overfit — spec 08-C4) — ~1 min of training on an A100. → adapter at
`outputs/adapter/`. Record the W&B run ID in `specs/STATUS.md`.

## Step 5 — C5: score and publish the head-to-head table

```bash
EVAL_CONTEXT=1 uv run python -m ch2_adaptation.baseline --config ch2_adaptation/configs/baseline_holdout.yaml
EVAL_CONTEXT=1 uv run python -m ch2_adaptation.evaluate --config ch2_adaptation/configs/eval_full.yaml
```
Writes the `EVAL_TABLE` marker in `README.md`. Pure post-processing once C1's baselines
and C4's adapter exist — a Claude Code session can do this part.

## Step 6 — A1–A5: the from-scratch GPT full run (independent of C2–C5, any order)

```bash
uv run python -m ch1_architecture.train     --config ch1_architecture/configs/full.yaml
uv run python -m ch1_architecture.benchmark --config ch1_architecture/configs/benchmark_full.yaml
```
5000 steps, `d_model=512`, on the committed TinyShakespeare corpus — on the order of an
hour on a single free-tier GPU. Writes `eval/results/ch1_benchmark.json`, the two plots,
and fills the `CH1_BENCHMARK` README marker (this is what unblocks A3/A4/A5's remaining
acceptance criteria — "recognizably English" samples, a real KV-cache speedup number, the
headline tokens/sec/perplexity table).

## Step 7 — O1: merge, export, quantize (needs C4/C5's real adapter)

```bash
uv run python scripts/merge_adapter.py --config ch3_operation/configs/export_full.yaml
uv run python scripts/export_gguf.py   --config ch3_operation/configs/export_full.yaml
uv run python scripts/quantize_gguf.py --config ch3_operation/configs/export_full.yaml
EVAL_CONTEXT=1 uv run python -m ch3_operation.evaluate --config ch3_operation/configs/eval_full.yaml
```
Needs a built `llama.cpp` checkout — see `scripts/README.md`. This is O1's real
remaining gap (T1–T6 are already done and tested against the base GGUF; only T7, the
real merge/re-eval, was blocked on a real adapter).

## Step 8 — O5: the actual deploy (human decision, not a script)

Tooling is built and tested (`space/README.md`, `docs/DEPLOY.md`,
`scripts/deploy_space.sh`, `scripts/smoke_deployed.py`) but T2–T6 haven't run. Needs:
`hf auth login`, the real GGUF from step 7, a second-host choice, and someone deciding to
publish a public URL. Then:
```bash
scripts/deploy_space.sh          # per docs/DEPLOY.md
uv run python scripts/smoke_deployed.py <space-url>
uv run python scripts/benchmark_serving.py <space-url>   # fills SERVING_METRICS marker
```

## Step 9 — pin the published artifacts

Once the adapter (step 4) and GGUF (step 7) are on HF Hub, point `ch3_operation`'s config
at the repo ID **plus the pinned revision SHA**, not a bare repo ID / `main`. **Done
2026-09-22 (A13):** `full.yaml` pins the GGUF, `export_full.yaml` pins the adapter, and the
container fetches from that revision only. The SHAs are in `REPRODUCING.md`'s "Pinned
artifacts" table. A new model means new SHAs in both places, updated in the same session it
is published.

---

## Running steps 4 and 6 on PARAM Rudra (IITB cluster)

If you're using the PARAM Rudra cluster instead of Colab/Kaggle for the two GPU-bound
steps (C4's QLoRA fine-tune, A-series' full `ch1_architecture` run), see
your local PARAM Rudra quickstart notes (account/login/quota basics: your cluster account,
`ssh <user>@<cluster-host> -p <port>`, `/scratch/<user>/` for job I/O, 3-month
purge on untouched scratch files) and `PARAM_Rudra_Training_Workflow.md` (a worked
example of the same login → conda env → rsync → SLURM → pull-back pattern for a
different project — reuse its shape, not its commands, since that one targets a
different repo and Python 3.10; this repo needs `>=3.12,<3.13` per `pyproject.toml`).

Concrete adaptation for **this** repo:

```bash
# 1. one-time env setup, on the Rudra login node (never run training there)
ssh <user>@<cluster-host> -p <port>
module load miniconda
conda create --name llm_edge python=3.12 -y
conda activate llm_edge

# 2. from your Mac: sync the repo (skip venv/outputs/git — regenerated or irrelevant)
rsync -avz -e "ssh -p <port>" \
  --exclude '.venv' --exclude '__pycache__' --exclude '.git' \
  --exclude 'outputs' --exclude '.ruff_cache' --exclude '.pytest_cache' \
  ~/Placement/Project/3_LLM_from_scratch/ \
  <user>@<cluster-host>:/scratch/<user>/3_LLM_from_scratch/

# 3. on Rudra: install this repo's heavy extras (bitsandbytes needs Linux+CUDA,
#    which is exactly what Rudra provides and your Mac does not)
cd /scratch/<user>/3_LLM_from_scratch
pip install uv
uv sync --extra dev --extra ch1 --extra ch2
```

SLURM batch script for **C4** (`train_c4.sh`, needs `GEMINI_API_KEY`/`WANDB_API_KEY`
exported or sourced before `sbatch` if you want online logging):

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --job-name=c4_qlora
#SBATCH --output=/scratch/<user>/3_LLM_from_scratch/logs/slurm.%J.out
#SBATCH --error=/scratch/<user>/3_LLM_from_scratch/logs/slurm.%J.err

module purge
module load miniconda
conda activate llm_edge
cd /scratch/<user>/3_LLM_from_scratch
mkdir -p logs
uv run python -m ch2_adaptation.finetune --config ch2_adaptation/configs/full.yaml
```

Same shape for **A-series** (`train_a.sh`, swap the last line and bump `--time` — the
ch1 full run is CPU/GPU-portable but 5000 steps is much faster with a GPU):

```bash
uv run python -m ch1_architecture.train     --config ch1_architecture/configs/full.yaml
uv run python -m ch1_architecture.benchmark --config ch1_architecture/configs/benchmark_full.yaml
```

Submit and pull results back the same way as the Pallet-CC workflow:

```bash
sbatch train_c4.sh && squeue --me
tail -f logs/slurm.<jobid>.out
# once done, from your Mac:
scp -r -P <port> <user>@<cluster-host>:/scratch/<user>/3_LLM_from_scratch/outputs/adapter ~/Placement/Project/3_LLM_from_scratch/outputs/
rsync -avz -e "ssh -p <port>" \
  <user>@<cluster-host>:/scratch/<user>/3_LLM_from_scratch/eval/results/ \
  ~/Placement/Project/3_LLM_from_scratch/eval/results/
```

Scratch is not backed up and purges files untouched for 3 months — pull the adapter,
checkpoint, and `eval/results/*.json` back to the Mac (or push straight to HF Hub with
`HF_TOKEN`) promptly after each job finishes, then continue with step 5 (C5 eval table)
or step 7 (O1 GGUF export) locally on the Mac as documented above.

---

## To-do: components not yet implemented (beyond running the above)

Everything above is "run the code that already exists." The items below are gaps in the
repo itself — no amount of GPU time closes them.

1. **O6 — in-browser WebLLM demo** (`specs/16-O6-in-browser-webllm.md`, still `draft`,
   explicitly gated on O5 being live). Nothing has been built: no WebLLM/MLC conversion
   of the quantized model, no static client-side-inference page, no WebGPU-availability
   fallback, no client-side schema validation against `eval/schema.json`. This is a
   listed **stretch goal / cut-line** — drop first if time is short — not a silent gap.
2. **C2 GitHub mining path** — the `github` MCP server in `.mcp.json` is broken against
   `api.githubcopilot.com` regardless of token shape (see step 2 above and `HANDOFF.md`).
   Either fix that MCP config or standardize on the `gh` CLI workaround; right now it's
   an ad hoc per-session decision.
3. **X1 T2/T6** — the README's live-demo link and headline number slots are placeholders
   until O5 is actually deployed and A5/C5's full runs land (steps 4–8 above resolve
   this automatically once run; no separate code work needed).
4. **Artifact revision pinning** (step 9) — closed 2026-09-22 (A13); see step 9.
