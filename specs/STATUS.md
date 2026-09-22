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

**20 of 24 done (83.3%)** as of 2026-09-22: all foundation (F1–F4), all of Chapter 2 (C1–C5),
all of Chapter 1 (A1–A5), O0, O1, O2, O4, X2 and X3. **Two more are done except one criterion
each** — O3 (AC-6/7) and X1 (AC-4) — and both of those criteria need a live HF Spaces
deployment. That is also what blocks **O5** (`blocked-by-cost`: HF put Docker Spaces behind a
paid plan in 2026, and free 512 MB hosts cannot hold the 1.15+ GiB service). **O6** is a
declared stretch cut-line, still `draft`. Nothing on the critical path is left to execute; the
remaining gap is hosting money, not work.
C4 in progress on PARAM Rudra (2026-09-17): environment working, 4-bit path verified (0.488% trainable),
step sweep showed 1000 steps overfits the 203 records, so `max_steps` is now 125 (`ab42114`). Headline adapter
re-trained from the committed config (job 401897, val loss 0.679), W&B synced (`5wwfz38t`), adapter downloaded and
checksum-verified: **C4 done**.
**C5 is now done (2026-09-18) — the project's headline table is published.** Two runs on the
identical frozen 40-record holdout: `baseline_holdout.yaml` on the laptop (base model fp32 CPU
0.00/0.00; `gemini-3.5-flash-lite` 3-shot 1.00/0.825, $0.00 free tier) and `eval_full.yaml` as
Rudra job 401990 on one A100 (4-bit base + the C4 adapter, 0.975/0.80). The fine-tune **lost both
metrics to the frontier API by one record each** and beat its own base model 0.00 -> 0.97 on
schema-validity; the loss is stated in the README in words per AC-7, with the per-record
breakdown in `docs/finetune-vs-prompting.md`. Its single invalid output is unescaped quotes from
an inlined Java snippet, not the markdown fence that costs the base model all 40.
Next on the critical path: O1's real merge -> GGUF -> re-eval, which was only ever blocked on
C4/C5 existing.

**Action 9 (Chapter 1's full run) is DONE (2026-09-18).** Six Rudra jobs, ~25 minutes of A100
time total, and two real bugs found by running the thing:
- `4fcf0a4` — the benchmark built perplexity windows on the CPU while the model was on `cuda`.
  Every test ran on CPU, so the mismatch was invisible; it would have killed the job minutes in.
- `72a7409` — **the model had no weight initialization at all.** Torch's `nn.Embedding` default
  is N(0, 1) and the output head is tied to it, so loss at init was 386 against ln(4096)=8.32.
  Job 402127 spent its whole 200-step budget climbing back to 9.59 — worse than random — while
  its samples read as English, because unigram frequency is the first thing any model learns.
- `0d70ac2` / `38ef5b4` — perplexity moved onto a held-out tail (it was in-sample), and the
  KV-cache reported as a curve over decode length rather than one ratio.
Headline: **perplexity 61.45 on 16,287 held-out tokens**, 21.0M params, W&B `9gxft2pt`. The
sweep that set `max_steps: 3000` and `lr: 1e-3` is W&B `2rd56pci` / `kan8xdyv`.

A seventh job (402179, `small` partition) re-ran the same checkpoint on CPU, which turned two
flat-looking results into the chapter's best finding: the KV-cache goes 1.03x -> **4.45x** and
int8 goes 0.93x -> **1.51x** when only `device` changes. Both artifacts are committed side by
side (`ch1_benchmark.json`, `ch1_benchmark_cpu.json`).
C2's 40-record holdout (30 synthetic + 10 mined from real `spring-projects` bug-fix commits,
labels taken from the maintainer fixes) was frozen and committed on 2026-09-16 (`b216df7`).
C1's real baselines were produced against `eval/stub/stub_eval.jsonl` (24 records) on
2026-09-16 and committed to `eval/results/baselines.json`: base model (Qwen2.5-Coder-1.5B-
Instruct, zero-shot) scores 0.00 schema-validity (every response is markdown-fenced, and the
harness's no-repair rule correctly refuses to strip fences — a real finding, not a bug);
frontier API (`gemini-3.5-flash-lite`, 3-shot — not `gemini-2.5-flash` as C1's own spec names,
see progress_report.md's Action Plan item 4 entries for why) scores 1.00 schema-validity,
0.917 bug-catch (22/24), at $0.00 real spend. Chapter 2's remaining specs are underway (C2
building, 6/8 tasks done, only the actual GitHub mining paused on `GITHUB_TOKEN`; C3/C4/C5
building — all with code complete, each waiting on a manual run that depends on the others).
Chapter 3's O0 (serving thin slice, base model) is done: `serve.py` and
`uvicorn ch3_operation.api.main:app` both serve a real base GGUF (`tiny.gguf` in CI, `MODEL_PATH`
in production) end to end. O1–O4 are now `building`: code complete and tested against the base
GGUF per `.claude/plans/Operation.md`'s D-3 decision. O4's image is now build-verified end to
end (`docker build` + `docker run` + `DOCKER=1 pytest`, both a build-time bug and a
runtime-only bug found and fixed against the real container — see progress_report.md). O1's
real gap remains: the merge→GGUF→re-eval needs a real adapter from C4/C5. O5's deploy
tooling (T1) is built and tested (`space/README.md`, `docs/DEPLOY.md`,
`scripts/deploy_space.sh`, `scripts/smoke_deployed.py`), but the actual deploy (T2–T6) is
untouched — it needs `hf auth login`, a real GGUF from O1 T7, a second-host choice, and a
human decision to publish a public URL, none of which this session can do unilaterally. O6
is untouched (stretch, explicitly gated on O5 being live).
Chapter 1 (A1–A5) now has a real model behind `train.py`: a hand-rolled BPE tokenizer, a
from-scratch GPT (embeddings, attention, MLP, norm, pre-norm blocks, weight tying), a
custom training loop with W&B logging and atomic checkpointing, a KV-cache whose
cached-vs-uncached logits are provably identical, and absmax quantization (fp16/int8/int4)
with a benchmark that publishes tokens/sec and perplexity per mode. A1/A2 are `done` — every
acceptance criterion is testable and met on CPU. A3/A4/A5 are `building`: all code and tests
pass (63/63 ch1 tests, full smoke train-then-benchmark verified end to end at ~2s total,
well under the 120s budget), but the criteria that need a meaningful, non-toy model — A3's
"recognizably English" samples, A4's actual speedup, A5's headline tokens/sec/perplexity
table — wait on the full-config GPU run, same as C4/C5/O1. The smoke benchmark's numbers
(an undertrained model on a 5 KB corpus) were deliberately not committed to the README.

| Track | Done | Total |
|---|---|---|
| F — Foundation | 4 | 4 |
| C — Chapter 2 (Adaptation) | 4 | 5 |
| O — Chapter 3 (Operation) | 1 | 7 (5 building) |
| A — Chapter 1 (Architecture) | 5 | 5 |
| X — Cross-cutting | 1 | 3 (2 building) |

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
| 05 | C1 | task-schema-and-base-model | done | F3 | — |
| 06 | C2 | independent-eval-set | done (frozen `b216df7`: 40 records, 10 mined, 0 dedup removals; licences `094d2c9`) | C1 | — |
| 07 | C3 | synthetic-training-data | done (`d78179f`: 226 records, 203 train / 23 val, $0.00) | C1 | — |
| 08 | C4 | qlora-finetune | done (job 401897 on PARAM Rudra: 125 steps from `ab42114`, val loss 0.679; adapter in `outputs/adapter`, checksum-verified on laptop, peft-load-verified; HF Hub `rkg209/qwen2.5-coder-1.5b-java-review-qlora` (private) @ `f8b1cf7`) | C2, C3 | [`5wwfz38t`](https://wandb.ai/rahulyk09-iit-bombay/ch2-adaptation/runs/5wwfz38t) |
| 09 | C5 | head-to-head-eval-table | done (2026-09-18: fine-tuned 0.97/0.80, base 0.00/0.00, frontier 1.00/0.82, n=40 each; Rudra jobs 401990 + 402095, byte-identical) | C4 | [`5wwfz38t`](https://wandb.ai/rahulyk09-iit-bombay/ch2-adaptation/runs/5wwfz38t) |

## Chapter 3 — Operation *(critical path)*

| # | ID | Spec | State | Depends on | W&B run |
|---|---|---|---|---|---|
| 10 | O0 | serving-thin-slice | done | F3 | — |
| 11 | O1 | merge-and-gguf-quantize | done (T1–T7 done: Q4_K_M validity 1.00 / catch 0.80 vs adapter 0.975 / 0.80, n=40, `0f3b2dd`; backend chat-template fix `483cb30`. GGUF public on HF Hub `rkg209/qwen2.5-coder-1.5b-java-review-gguf` @ `a2133f6`) | C5, O0 | — |
| 12 | O2 | fastapi-serving-guardrails | done (2026-09-22 audit: all 9 ACs covered by `test_validator.py` / `test_api.py` / `test_errors.py` — 422 `VALIDATION_FAILED`/`SCHEMA_VIOLATION` with Pydantic detail, exactly two parse attempts, 413 over 32 KB, 503 `MODEL_NOT_READY`, `request_id` on every response; served the real GGUF at schema-validity 1.0 in O5) | O1 | — |
| 13 | O3 | monitoring | done except AC-6/AC-7, blocked-by-cost like O5 (2026-09-22): AC-1–5 and AC-8 met and tested; live gauge `review_schema_validity_rate` 1.0 over 91 real requests. AC-6/7 require HF Spaces, which is paywalled, so p50 2.16 s / p99 3.68 s / 0.44 req/s were measured on the production image under `--cpus=2 --memory=4g` instead, and the NFR-3 1 req/s miss is reported with its cause | O2 | — |
| 14 | O4 | containerize | done (2026-09-22 audit: clean-checkout build + pinned first-boot download + `/health` + `/metrics` + schema-valid `/review` verified locally; `linux/amd64` build green in the Docker CI job at `5e12262`; CPU-only slim final stage, model not baked in; two real bugs found and fixed, see progress_report.md) | O3 | — |
| 15 | O5 | deploy | blocked-by-cost: HF Docker Spaces paywalled (402), 512 MB free hosts too small for 1.15 GiB. Measured instead: prod image under --cpus=2 --memory=4g, p50 2.16 s / p99 3.68 s / 0.44 req/s / 0 errors after the n_threads 4→2 fix (2026-09-22; was 2.97 / 4.64 / 0.32 in `567c05a`); still misses NFR-3 1 req/s, Q4_0 rejected on quality; temporary demos via cloudflared. No permanent URL | O4 | — |
| 16 | O6 | in-browser-webllm *(stretch)* | draft | O5 | — |

## Chapter 1 — Architecture *(parallel track — no critical-path spec depends on it)*

| # | ID | Spec | State | Depends on | W&B run |
|---|---|---|---|---|---|
| 17 | A1 | tokenizer-and-data-pipeline | done | F2 | — |
| 18 | A2 | core-transformer-modules | done | A1 | — |
| 19 | A3 | custom-training-loop | done (2026-09-18: 3000 steps on one A100 from committed config `c4335d3`, val_loss 4.1186, best 4.1155 @ 2800; `max_steps`/`lr` set by a measured sweep, not a guess. **Deviations:** `0d70ac2` held-out tail + val-loss logging (A3 had no overfitting signal at all); `72a7409` the model had no weight init — loss at init was 386 vs ln(V)=8.32, see below) | A2 | [`9gxft2pt`](https://wandb.ai/rahulyk09-iit-bombay/ch1-architecture/runs/9gxft2pt) |
| 20 | A4 | kv-cache | done (2026-09-18, job 402166: 1.008x @ 32 decode steps rising to 1.030x @ 250; cached throughput flat at ~273 tok/s while uncached decays 271.6 -> 265.0, which is the mechanism. **Action-plan item 9's ''larger than the toy 1.26x'' expectation is met on CPU (job 402179: 2.44x @ 32 steps -> 4.45x @ 250, same checkpoint) but NOT on the A100 (1.03x)** — a 21M model at batch 1 on an A100 is launch-latency bound (~4.3 ms/step vs ~0.08 ms of attention compute), so the cache removes work that is already free. Published as a curve on both devices: cached throughput is flat everywhere (~273 A100 / ~220 CPU) while uncached collapses only on CPU (90.4 -> 49.4), which is the mechanism. A KV-cache's value is a property of the hardware's prefix-re-encoding cost, not of the cache. **Deviation `38ef5b4`:** adds a third plot where A5 names two) | A3 | [`9gxft2pt`](https://wandb.ai/rahulyk09-iit-bombay/ch1-architecture/runs/9gxft2pt) |
| 21 | A5 | quantization | done (2026-09-18, job 402166: fp32 237.4 tok/s @ ppl 61.45 on 16,287 held-out tokens; fp16 200.2, int8 219.8 @ 61.70, int4 107.6 @ 84.11. **Every mode is slower than fp32 on the A100, but int8 reverses to 1.51x on CPU** (job 402179: 60.5 vs 40.2 tok/s, +0.42% perplexity) — a CPU fp32 matmul is bandwidth-bound so smaller weights repay the dequantization, while an A100 with memory to spare pays it for nothing. int4 loses on both (0.45x / 0.57x, +36.9% ppl). **Deviations:** `0d70ac2` perplexity moved off the training corpus onto a held-out tail (the spec measured it in-sample); `4fcf0a4` fixed a CPU/CUDA device bug that would have killed the run) | A4 | [`9gxft2pt`](https://wandb.ai/rahulyk09-iit-bombay/ch1-architecture/runs/9gxft2pt) |

## Cross-cutting / portfolio

| # | ID | Spec | State | Depends on | W&B run |
|---|---|---|---|---|---|
| 22 | X1 | readme-and-architecture-diagram | done except AC-4, blocked-by-cost like O5 (2026-09-22): diagram, head-to-head table, Ch1 plots, per-chapter summaries, $0.00 cost statement, honest framing (frontier wins shown as losses) and results-before-setup all in the README; AC-4 wants a live HF Spaces link, which is paywalled — the README carries a real worked example, the public model link and the one-command Docker run instead | A5, C5, O5 | — |
| 23 | X2 | reproducibility | done (AC-2 through AC-7 verified this session; AC-1 — handing `REPRODUCING.md` to a real second person — cannot be self-certified in a session, noted as such) | F2 | — |
| 24 | X3 | interview-defense-notes | done (2026-09-22 audit: all three notes cite this project's own numbers — KV-cache 1.03x A100 / 4.45x CPU, perplexity 61.45 / 61.70 / 84.11 by precision, head-to-head 0.97 vs 1.00 and 0.80 vs 0.82 with the frontier win stated — and each has a "What I got wrong" section) | A5, C5 | — |

## Cut-lines

Drop these before the critical path if the schedule slips: **O6** (in-browser), deep Chapter 1 extensions, extra quantization levels beyond int8/int4.

**Definition of done:** F1–F4 · Chapter 1 through A5 with both plots · Chapter 2 through C5 with the eval table · Chapter 3 through O5 with a live HF Spaces link · X1 README with the diagram and results.
