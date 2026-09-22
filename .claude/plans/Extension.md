# Plan — Cross-cutting / portfolio: X1 → X3

> Covers `specs/22-X1-readme-and-architecture-diagram.md`, `specs/23-X2-reproducibility.md`,
> `specs/24-X3-interview-defense-notes.md`. Written against the repo as of `185c89c`
> (A1–A5 code complete). Companion to `.claude/plans/Operation.md` (O1–O5) and
> `.claude/plans/AfterOperation.md` (A1–A5). Each per-spec section still gets pasted into
> its spec's **Technical plan** block before `/tasks <ID>`; this file is the umbrella design
> so the three specs are built as one coherent portfolio artifact.

---

## Context

Every other spec in this repo builds a *capability*. These three build the **evidence** that
the capability exists — and they are the last three, because everything they describe had to
be built first.

The state that makes this plan necessary:

- `README.md` already has the right skeleton — three marker pairs (`EVAL_TABLE`,
  `CH1_BENCHMARK`, `SERVING_METRICS`), an `## Architecture` heading holding
  `*(spec X1 — ... goes here)*`, and a `## Reproducing` section holding `*(spec X2)*`. The
  scaffolding is there; the content is not, and the section **order is wrong** — Getting
  started currently precedes nothing, but there is no live link and no diagram at all.
- There is **no `REPRODUCING.md`**, no `Makefile`, and `ALLOW_FULL_RUN` — the guard a
  reproducer must know about — is documented in exactly one place (`scripts/README.md:24`)
  that nobody reproducing the project will read.
- `docs/` contains only `DEPLOY.md`. The three interview-defense notes X3 asks for do not
  exist in any form.

**The blocker that shapes everything:** X1 depends on A5/C5/O5 and X3 on A5/C5, and all
three of those wait on manual GPU-box runs that cannot happen in a session (CLAUDE.md
non-negotiable #1). Per the decision taken up front, this plan **builds everything except
the numbers**: diagram, prose, `REPRODUCING.md`, `Makefile`, and all three defense notes get
written now, with every number-bearing slot left as either an existing marker pair (filled
mechanically by the tooling that already exists) or an explicit `*(pending — produced by
`<command>`)*` placeholder. A final task fills every slot in one pass once the runs land.
**No number is ever invented, and no smoke number is ever published unlabelled** — the
README already sets that precedent for `CH1_BENCHMARK` and this plan keeps it.

**Intended outcome:** a reader who lands on the repo sees, above the fold, what was built,
what it cost, what the numbers were, and where the live demo is; a skeptic can run
`make smoke` from a clean checkout with no GPU, no API key, and no W&B account and watch the
whole code path work; and an interviewer who probes the three hardest questions in the
project finds them already answered, in writing, in `docs/`, with this project's own
measurements attached.

### Decisions taken up front

| # | Decision | Why |
|---|---|---|
| **D-1** | The architecture diagram is **Mermaid, inline in `README.md`**. No committed image. | Renders natively on GitHub (AC-1's actual requirement), lives in git as text, diffs alongside the code it describes, and can never become the stale PNG that says the system does something it stopped doing. This closes X1's open Clarifications question. |
| **D-2** | Add a **`Makefile`** at the repo root with `setup`, `check`, `smoke`, `smoke-ch1/ch2/ch3` — thin wrappers over the existing `uv run` commands, nothing new invented. | X2 AC-3 names `make smoke` literally. One command is the difference between a skeptic reproducing the project and a skeptic bouncing. The `uv` equivalents stay documented verbatim next to it, so `make` is a convenience, never a dependency. |
| **D-3** | Number-bearing README slots are filled **only** by the existing generators — `eval.report.update_readme_section` via `ch1_architecture/benchmark.py` (`CH1_BENCHMARK`), `ch2_adaptation/evaluate.py` (`EVAL_TABLE`), `scripts/benchmark_serving.py` (`SERVING_METRICS`). X1 adds **no new number-writing code**. | Three writers already exist and are tested. A fourth path for hand-editing numbers into the README is exactly how a number that no run produced ends up published (NFR-7). |
| **D-4** | Prose slots that cannot be marker-driven (the live link, the worked example, the cost statement) are written now with an explicit `*(pending — O5 deploy)*` marker in the text, and resolved in **X1 T6**. | An honest placeholder is a to-do; a plausible-looking invented URL is a lie. The placeholder is also the thing that makes T6 impossible to forget. |
| **D-5** | `REPRODUCING.md` at the repo root, not under `docs/`. `README.md#reproducing` links to it. | X2's spec names the file `REPRODUCING.md`; GitHub surfaces root-level docs, and a reproducer looks at the root. `docs/` stays for the defense notes and `DEPLOY.md`. |
| **D-6** | The three defense notes are **three separate files** under `docs/` — `docs/kv-cache.md`, `docs/finetune-vs-prompting.md`, `docs/quantization.md` — plus a one-paragraph `docs/README.md` index. | X3 AC-4 requires each to be self-contained and readable without the code open. Three focused files satisfy that; one long document invites cross-references between sections, which is how "self-contained" quietly dies. |
| **D-7** | Determinism (X2 AC-5) is **verified by a real test**, `eval/tests/test_determinism.py`, not asserted in prose: run the ch1 smoke train twice with the same seed and assert the final loss is bit-identical. | "Re-running produces identical results" is a claim; a passing test is evidence. It also catches the day someone adds an unseeded RNG. CPU-only, seconds, safe in CI. |
| **D-8** | Numbers-to-provenance (X2 AC-6, NFR-16) is served by a **single table in `REPRODUCING.md`** mapping each published number → config file → command → W&B run id, with the run-id column empty until the runs happen. `specs/STATUS.md` keeps its own W&B column as today. | One table a reader can scan beats a run id buried in a sentence next to each number. The empty column is itself honest signalling. |
| **D-9** | X2 is built **first**, before X1 and X3. | It is the only one of the three with no blocked dependency (it depends on F2, which is done), and `README.md`'s Reproducing section links to its output — so building it first means X1 never writes a link to a file that does not exist. |

### Constraints that shape everything below

- **Never invent a number.** Every figure in the README or `docs/` traces to a config file and
  a run. Where the run has not happened, the slot says so.
- **Honest framing is not optional** (NFR-7, X1 AC-7): Chapter 2 is distillation of a frontier
  model into a small specialist; the holdout is independent and only partly real; any metric on
  which the fine-tune lost is shown as a loss, in the table, in words.
- The **leakage guard** blocks reads of `eval/holdout/` without `EVAL_CONTEXT=1`. No doc-writing
  task needs to read it — and none of these tasks should.
- The **GPU-budget guard** blocks anything naming `configs/full.yaml`. Every full-path command
  in this plan is a manual, out-of-session GPU-box run and is written as such.
- One task = one commit; append a `progress_report.md` entry per task; update `specs/STATUS.md`
  on any state change. **Never** a `Co-Authored-By:` trailer.
- `uv run ruff check . && uv run black --check . && uv run pytest -q` before every commit.

---

## X2 — Reproducibility *(built first, per D-9)*

**Files:** `REPRODUCING.md` *(new)*, `Makefile` *(new)*, `eval/tests/test_determinism.py`
*(new)*, `README.md` *(edit — Reproducing section)*, `.github/workflows/ci.yml` *(edit)*.

### T1 — `Makefile`

Thin wrappers, no logic:

| Target | Wraps |
|---|---|
| `setup` | `uv sync --extra dev` (plus a commented note on `--extra ch1/ch2/ch3`) |
| `check` | `uv run ruff check . && uv run black --check . && uv run pytest -q` |
| `smoke-ch1` | `uv run python -m ch1_architecture.train --config ch1_architecture/configs/smoke.yaml` |
| `smoke-ch2` | `uv run python -m ch2_adaptation.finetune --config ch2_adaptation/configs/smoke.yaml` |
| `smoke-ch3` | `uv run python -m ch3_operation.serve --config ch3_operation/configs/smoke.yaml --check` |
| `smoke` | all three, `set -e` semantics so a failure exits non-zero (AC-3) |

`.PHONY` on every target. `ch3` needs `--extra ch3` and `ch1` needs `--extra ch1`; the
Makefile must not silently succeed on a base-only env — let the import error speak, and say
so in `REPRODUCING.md`.

### T2 — `REPRODUCING.md`

Structured around the spec's two audiences, in this order:

1. **Five minutes, no GPU** — clone, `make setup`, `make smoke`, what you should see. States
   explicitly what is *not* required: no GPU, no API key, no W&B account (AC-2), and *why*
   that holds: `wandb_mode: disabled` in every `*_smoke.yaml`, `frontier_provider: stub` in
   `baseline_smoke.yaml`/`datagen_smoke.yaml`, `ch3_operation/tests/fixtures/tiny.gguf` as the
   CI model. Name those files — a reader who can verify the claim will.
2. **What the smoke path does and does not prove** — it exercises every code path end to end;
   it produces no meaningful metric. This paragraph is what stops someone quoting a toy
   perplexity back at the project.
3. **A GPU afternoon** — per-chapter full-run runbook (Colab / Kaggle / cluster): environment
   (`uv sync --extra ch1|ch2`), the exact commands with `configs/full.yaml`, expected
   wall-clock, expected cost (≈$1 total API spend, free GPU tier), and where the artifacts
   land. Cross-links `scripts/README.md` (O1 export runbook) and `docs/DEPLOY.md` (O5) rather
   than restating them.
4. **`ALLOW_FULL_RUN=1`** (AC-4) — what the guard is, that it lives in
   `.claude/hooks/gpu_budget_guard.py`, exactly which patterns trip it, and *why* it exists:
   a full run started from an agent session burns a scarce free-GPU budget and hangs the
   session for hours. Reproducers running by hand on their own box set it once and move on.
5. **Determinism and its limits** (AC-5) — same config + same seed ⇒ identical results for
   deterministic ops, verified by `eval/tests/test_determinism.py`; `eval.config.seed_everything`
   is the single seeding entrypoint. Then the honest half: CUDA kernel nondeterminism and
   differing hardware mean the **full GPU run is not bit-reproducible across machines**, and
   the project does not claim it is. Also list the untested platforms from Out-of-scope
   (Windows, non-CUDA GPUs) as untested rather than unsupported.
6. **Provenance table** (AC-6, D-8) — one row per published number: number → config file →
   command → W&B run id (blank until the run happens).
7. **Pinned artifacts** (AC-7) — HF Hub model/adapter pulled by repo id **plus revision SHA**,
   never `main`. Where the repo currently resolves a bare id, note it as a to-do resolved when
   the adapter is published.

### T3 — determinism test

`eval/tests/test_determinism.py`: run the ch1 smoke train twice in-process with the same seed
(reusing `ch1_architecture.train`'s existing entry path and `tmp_path` output dirs), assert
final loss is bit-identical; and a cheap direct test that `seed_everything(n)` makes two
`random`/`torch` draws identical. Skips cleanly (`pytest.importorskip("torch")`) so the
base-env test run stays green, matching how the other ch1 tests behave.

### T4 — wire into README + CI

- `README.md#reproducing` — replace `*(spec X2)*` with the five-minute command block and a
  link to `REPRODUCING.md`.
- `ci.yml` — replace the three inline smoke steps with `make smoke`, so the documented command
  and the tested command are the same command. This is the mechanism that stops
  `REPRODUCING.md` from rotting: if the documented path breaks, CI goes red.

**Acceptance check:** every X2 AC maps to something checkable — AC-3 to a green `make smoke`,
AC-5 to a passing test, AC-1 to the actual test of handing it to someone (call this out in the
progress report as the one AC a session cannot self-certify).

---

## X1 — README + architecture diagram

**Files:** `README.md` *(edit, substantial)*. No new modules (D-3).

### Target section order (AC-8: results before setup)

```
# Title + one-line thesis
## What this is        (three-chapter table — already exists, keep)
## Live demo           (link + worked example: Java in, JSON review out)     ← new
## Architecture        (the Mermaid diagram)                                  ← new
## Results             (Ch2 table · Ch1 plots · Ch3 serving)  — exists, move up
## What this cost      (≈$1 API spend, free GPU tiers, CPU-only serving)     ← new
## How to read these numbers  (the honest framing)                            ← new
## Getting started / API / Docker / Reproducing / How this repo is built     — exists
```

### T1 — the Mermaid diagram (AC-1)

One `flowchart` covering all three chapters and — the part that matters — **how they connect**,
which is the thing the current README only states in prose. The connections are file-shaped,
not import-shaped, and the diagram should show that:

- Ch1 is a self-contained subgraph (corpus → BPE → GPT → checkpoint → KV-cache/quantize →
  `eval/results/ch1_benchmark.json` + plots). Drawn deliberately as a **parallel track** with
  no edge into Ch2/Ch3 — that separation is a design fact, and an interviewer will ask.
- Ch2: base model + synthetic data → QLoRA → adapter → HF Hub; adapter + frozen holdout →
  eval harness → `eval/results/*.json` → the README table.
- Ch3: adapter → merge → GGUF quantize → FastAPI (+ schema validation, `/metrics`) → Docker →
  HF Spaces.
- The shared spine: `eval/schema.json` and `eval/harness.py` touching both Ch2 and Ch3 — the
  structured-output contract is the one thing the whole project hangs on.

Keep it to one flowchart, top-to-bottom, `subgraph` per chapter. Verify it renders on GitHub
after push (Mermaid is unforgiving about labels containing `(`, `)`, and `/` — quote them).

### T2 — live demo + worked example (AC-4)

The HF Spaces link, then a real request/response pair: a Java snippet in, the validated
`ReviewOutput` JSON out. Reuse the exact snippet already in the README's API section so the
two never drift. **Pending until O5 deploys** — placeholder per D-4, resolved in T6.

### T3 — results section (AC-2, AC-3)

Move `## Results` above `## Getting started`. Keep all three marker pairs byte-identical so
the existing writers keep working. Add, around the markers:

- Under the Ch2 table: one line naming the holdout (`n`, how it was built, that it is frozen
  and independent) — the reader needs to know what the number is *of*.
- Under `CH1_BENCHMARK`: the plot embeds are already emitted by
  `benchmark.py::_update_readme`; the surrounding prose says what the two curves mean and that
  fp16-on-CPU being slower is an expected, reported result, not a bug.
- Keep the current "smoke numbers are not meaningful" caveat verbatim until T6 replaces it.

### T4 — chapter descriptions + cost (AC-5, AC-6)

The existing three-row table stays as the chapter descriptions (it is good and it is short).
Add a short `## What this cost` section stating plainly: ≈$1 total LLM-API spend, free GPU
tiers (Colab/Kaggle) for every training run, CPU-only serving on a free HF Spaces tier. The
constraint is part of the result — a reader who does not know the budget cannot see what was
achieved inside it.

### T5 — the honest framing, stated not buried (AC-7)

A short `## How to read these numbers`, placed immediately after Results, saying in plain
words:

1. Chapter 2 is **distillation** — a frontier model generated the training labels; the
   fine-tuned model is a small specialist trained on that signal, not an independent discovery.
2. The eval set is **independent of the training data and only partly real** — mostly
   synthetic, some mined from real Java/Spring code — with the dedup guarantee named.
3. **Where the fine-tune lost, it says so**, next to where it won, with the cost/latency/privacy
   argument that holds regardless.
4. Chapter 1's model is a **toy trained on a small corpus**; its purpose is to demonstrate the
   mechanics, and its perplexity is not comparable to anything.

This section is the project's differentiator. Write it as prose, not bullets-of-hedges.

### T6 — fill the slots *(gated on A5 + C5 + O5 completing)*

The one task that cannot run until the manual runs land. Then, in order: run
`ch2_adaptation/evaluate.py` and `benchmark.py` (they write their own markers),
`scripts/benchmark_serving.py` against the deployed Space, commit the generated plot PNGs,
replace the T2/T4 placeholders with the real link and real cost, delete the smoke-numbers
caveat, and record the W&B run ids in `specs/STATUS.md` and `REPRODUCING.md`'s provenance
table.

---

## X3 — Interview-defense notes

**Files:** `docs/kv-cache.md`, `docs/finetune-vs-prompting.md`, `docs/quantization.md`,
`docs/README.md` *(all new)*, `README.md` *(edit — link them)*.

These are written **against the code in this repo**, with file and line references, not from
memory of a textbook. Before writing each, re-read the module it defends — that re-read *is*
the work; AC-4 and AC-5 cannot be satisfied any other way.

### T1 — `docs/kv-cache.md` (AC-1)

Sources: `ch1_architecture/src/ch1_architecture/model/attention.py`, `kv_cache.py`,
`benchmark.py::measure_tokens_per_sec_cached`, `tests/test_kv_cache.py` (the logit-identity
test).

Must cover: what is recomputed at every decode step without a cache and why the total work is
quadratic in sequence length; what the cache stores (the K and V projections of every prior
position, per layer, per head) and what it explicitly does *not* store (Q — recomputed each
step from the one new token) ; **why it does not help prefill** — prefill already computes every
position once, in parallel, so there is nothing to reuse and the cache is being *filled*, not
read; why the cached and uncached paths must produce **bit-identical logits**, and that this
repo asserts exactly that in a test rather than eyeballing the samples; the memory cost that
buys the speed (`2 · n_layers · n_heads · head_dim · seq_len` per sequence) and why that is
the real constraint at scale. Ends with this project's measured numbers from A4/A5.

### T2 — `docs/finetune-vs-prompting.md` (AC-2)

Sources: `specs/08-C4`, `specs/09-C5`, `.claude/skills/qlora-recipe/SKILL.md`, `eval/schema.json`,
`ch2_adaptation/baseline.py`.

Must cover: why **structured-output conformance** is where a specialist wins first — schema
adherence is a distributional property of the output format, which gradient updates move
directly and a prompt can only ask for; what few-shot prompting structurally cannot do (the
format constraint competes with the task for attention, and every request pays the few-shot
tokens again); the honest limits — **where the frontier model still won in this project's own
table and why** (breadth of reasoning about unfamiliar bugs, not format); and the
cost/latency/privacy argument that holds even when the frontier model wins on accuracy: a
1.5B model on a CPU box, no per-request API cost, no code leaving the network.

### T3 — `docs/quantization.md` (AC-3)

Sources: `ch1_architecture/src/ch1_architecture/quantize.py`, `benchmark.py`,
`eval/results/ch1_benchmark.json`, and `.claude/skills/gguf-export/SKILL.md` for the Ch3 tie-in.

Must cover: what absmax quantization does to a weight tensor concretely — per-tensor scale
`max(|W|)/qmax`, round, store as int8 or as packed int4 nibbles, dequantize on forward — and
what that discards (outlier resolution; every weight in the tensor pays for the largest one);
why speed improves (memory bandwidth, not arithmetic — the model is bandwidth-bound at
batch 1) and the honest counterexample this project measured, **fp16 on CPU being slower than
fp32** because there is no fast CPU fp16 matmul path; why quality degrades and **which
capability degrades first — structured output**, because valid JSON is a long chain of
low-entropy token choices where a small logit perturbation compounds; and the measured
perplexity delta per precision level from A5.

### T4 — the surprises (AC-5)

Not a separate file — a **"What I got wrong"** section at the end of each of the three notes.
Candidates already recorded in `progress_report.md` and worth mining rather than inventing:
the fp16-slower-on-CPU result; the KV-cache giving no prefill benefit; the C2 byte-identical
duplicate that dedup caught; the two Docker bugs only a real `docker build` found. Each note
names at least one thing that was genuinely surprising **to the author, while building this**.
This is AC-5's whole point — a generic "I learned tests are important" fails it.

### T5 — index + link

`docs/README.md` — a paragraph on what these documents are (a defense document, dense and
honest, written for one reader trying to find the gap) plus links to the three notes and to
`DEPLOY.md`. Link the three from `README.md`'s "How this repo is built" section.

---

## Build order and gating

| Order | Task | Gated on |
|---|---|---|
| 1 | X2 T1–T4 | nothing — fully unblocked |
| 2 | X1 T1, T3, T4, T5 | nothing (structure and prose only) |
| 3 | X3 T1–T5 | nothing — the *explanations* do not need the numbers; each cites a slot |
| 4 | X1 T2 | O5 deploy (live URL) |
| 5 | X1 T6, and the number slots inside X3 T1/T3 | A5 + C5 full GPU runs |

Steps 1–3 are this session's work; 4–5 are a single short follow-up session after the manual
runs. Mark X2 `done` when its ACs pass; X1 and X3 go to `building` with the pending ACs named
explicitly in `specs/STATUS.md` — the same convention A3/A4/A5 and O1–O5 already use.

---

## Verification

**X2 — the whole point is that it works from cold:**

```bash
git clone <repo> /tmp/repro && cd /tmp/repro    # a genuinely clean checkout
make setup
make smoke                                       # must exit 0, no GPU/API key/W&B
make check
```

Then, deliberately: unset every credential (`GEMINI_API_KEY`, `WANDB_API_KEY`, `HF_TOKEN`) and
run `make smoke` again. If it still passes, AC-2 is real. AC-1 is only truly verified by handing
`REPRODUCING.md` to another person — say so in the progress report rather than claiming it.

```bash
uv run pytest eval/tests/test_determinism.py -q   # AC-5
uv run ruff check . && uv run black --check . && uv run pytest -q
```

**X1 — the diagram and layout are verified by looking, on GitHub:**

- Push and open the repo page: the Mermaid block must render, not show as a code fence.
- Scroll test: on a phone-width viewport, the live link, the diagram, and the eval table all
  appear before `## Getting started` (AC-8).
- Marker integrity: `grep -c "EVAL_TABLE_START\|CH1_BENCHMARK_START\|SERVING_METRICS_START"`
  README.md → each exactly once, or `update_readme_section` raises.
- After T6, re-run each generator twice and confirm the README is unchanged the second time
  (the writers are idempotent by construction — verify it, since T3 moved sections around).

**X3 — the test is whether it survives being read cold:**

- Read each note with the code **closed**. Anything that only makes sense with the source open
  fails AC-4 and gets rewritten.
- Every number in the notes has a citation to `eval/results/*.json` or a named config. Grep for
  bare numerals and check each one.
- Each note has a non-generic "What I got wrong" section (AC-5).

**Repo-wide, before each commit:** `uv run ruff check . && uv run black --check . &&
uv run pytest -q`, one task per commit, a `progress_report.md` entry per task, and
`specs/STATUS.md` updated on state change.
