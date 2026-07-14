# C1 — Task + schema + base model fixed, baselines measured

| | |
|---|---|
| **State** | draft |
| **Depends on** | F3 |
| **Requirements** | FR-14, FR-15, CON-11, NFR-7 |
| **W&B run** | — |

## Problem

Two things must be nailed down before a single training token is spent, and both are cheap now and expensive later.

**The base model tag.** Changing it after data generation means re-running every baseline and every comparison (CON-11). The prompt format, the chat template, and the LoRA target modules all follow from it.

**The baselines.** This is the one that gets skipped, and skipping it quietly ruins the project. A baseline measured *after* the fine-tune exists is a baseline measured by someone who knows what number they need it to be. Even with no bad intent, the prompt gets tweaked, the parse gets a little more forgiving, and the comparison stops being a comparison. Measure the base model and the frontier API **first**, freeze the numbers, and the eval table means something.

## Scope

Fix `Qwen/Qwen2.5-Coder-1.5B-Instruct` in config; write the `ReviewOutput` Pydantic model mirroring `eval/schema.json`; write `baseline.py`; measure base-model zero-shot and frontier-API 3-shot on a stub set and record them.

## Acceptance criteria

1. `ch2_adaptation/configs/full.yaml` contains `model.name_or_path: Qwen/Qwen2.5-Coder-1.5B-Instruct`, and this spec is the only place the choice is made (FR-14).
2. `ch2_adaptation/schema.py::ReviewOutput` is a Pydantic model with `extra="forbid"`, whose fields and constraints match `eval/schema.json` exactly. **On import it structurally compares itself against `eval/schema.json` and raises if they have diverged** (NFR-20) — the two can never silently drift apart.
3. `baseline.py` scores the base model zero-shot **and** a frontier API 3-shot on the same stub set, through `eval/harness.py` — not through ad-hoc scoring code.
4. Both scores (schema-validity rate, bug-catch rate) are written to `eval/results/baselines.json` with the frontier model tag and an ISO timestamp, **atomically** (write to `.tmp`, rename).
5. `baselines.json` exists and is committed **before spec C4 moves to `building`** (FR-15). This ordering is the point of the spec.
6. The smoke path runs on CPU in under 120 seconds: fp32, no bitsandbytes, a handful of stub examples.
7. The frontier API spend for this spec is recorded and counts against the ≈$1 total budget (CON-3).

## Out of scope

- The frozen holdout set → **C2**. C1's baselines run against a small **stub** set precisely because the real holdout does not exist yet — and once it does, the baselines are re-run against it under **C5**.
- Generating training data → **C3**. Fine-tuning → **C4**.

## Clarifications

Resolved during `/plan` (full reasoning: `.claude/plans/05-C1-task-schema-and-base-model.md`):

- **Frontier model = Gemini 2.5 Flash**, free tier, `GEMINI_API_KEY`. No Claude API key is
  available in this environment; Gemini's free tier is rate-limited rather than credit-limited
  and is a genuine frontier-lab model, so "Frontier API, 3-shot" stays an honest row label.
  Native JSON mode (`response_mime_type="application/json"`) sidesteps the markdown-fence
  failure mode entirely on this path.
- **C3's generator model is not yet decided.** If C3 also picks Gemini, the fine-tune is
  distilled from the same model it is benchmarked against, which makes "beating the frontier
  baseline" circular by construction. Survivable — the honest framing becomes "approaches the
  teacher at a fraction of the serving cost" — but C3's spec must decide this explicitly rather
  than inherit it by accident.

## Amendments (recorded here per plan, not applied silently)

- **AC-1** originally required `model.name_or_path`. The shipped config uses a flat `model_tag`
  (`ch2_adaptation/src/ch2_adaptation/config.py:12`, `LOCKED_MODEL_TAG`, locked by
  `planning/03-system-design.md:339`); `eval/config.py::load_config` rejects unknown keys, so a
  nested key would break `FinetuneConfig` and all six existing ch2 tests. **AC-1 is amended to
  `model_tag`.** F2 already landed the lock; C1's new surface is the schema, prompt, stub set,
  baseline scorer and frontier client built on top of it.
- **AC-6** originally implied running the real fp32 1.5B model in smoke. fp32 Qwen-1.5B on CPU is
  ~6 GB resident and ~40-120s of generation alone — it cannot honour the under-120s NFR reliably.
  **AC-6 is amended**: smoke uses a named tiny stand-in tag (`SMOKE_MODEL_TAG`), proving the code
  path (load → prompt → generate → parse → score → atomic write); the full config produces the
  real number.
- **AC-7 / CON-3.** CON-3 says the API budget is "used exclusively for synthetic training-data
  generation." C1's 3-shot baseline call is an *eval* use, not data generation. This is recorded
  as an explicit carve-out to CON-3 rather than assumed to already be covered.

## Technical plan

See `.claude/plans/05-C1-task-schema-and-base-model.md` for full detail (files, approach per
module, risks, spec-amendment rationale). Summary:

**New files** — `ch2_adaptation/src/ch2_adaptation/{schema.py,prompts.py,model.py,frontier.py,baseline.py}`;
`eval/stub/{stub_eval.jsonl,stub_eval_smoke.jsonl,README.md}`;
`ch2_adaptation/configs/{baseline_smoke.yaml,baseline_full.yaml}`;
`ch2_adaptation/tests/{test_review_schema.py,test_baseline.py,test_baseline_config.py,test_stub_eval_set.py}`.

**Modified** — `config.py` (`BaselineConfig`, `load_baseline_config`, `SMOKE_MODEL_TAG`);
`eval/harness.py` (extract `write_json_atomic`); `pyproject.toml` (pydantic → base deps,
`google-genai` → `ch2` extra); `.env.example` (`GEMINI_API_KEY=`).

**Key design points:**

- `schema.py`'s drift guard raises a dedicated `SchemaDriftError(RuntimeError)`, not `assert`
  (stripped under `python -O` — deliberate deviation from `planning/03-system-design.md:371`).
  It compares only the load-bearing keys (`required`, `additionalProperties`, `severity.enum`,
  per-field length/minimum constraints) since Pydantic's `model_json_schema()` emits extra
  `title` keys that a naive equality check would false-positive on.
- `prompts.py` is locked and reused verbatim by C4 inference and C5's comparison. No
  fence-repair in the baseline scoring path — a fenced response is a real schema-validity
  failure, which is the exact number the fine-tune exists to move.
- `baseline.py::score_system` takes an injected `Generator = Callable[[list[str]], list[str]]`
  so every test drives it with a fake — no torch, no network, in the base+dev CI env. Heavy
  imports (`torch`/`transformers`, `google.genai`) stay function-local in `main()`.
- `BaselineConfig.__post_init__` enforces the CON-11 invariant: full run requires
  `base_model_tag == LOCKED_MODEL_TAG`; smoke run requires `base_model_tag == SMOKE_MODEL_TAG`.
  Smoke writes only to gitignored `outputs/baselines-smoke.json`, never to
  `eval/results/baselines.json`.
- `frontier.py::estimate_cost_usd` reads prices from env, never hardcoded (mirrors C3's planned
  budget guard). Free tier records real token counts alongside an honest `$0.00`.

## Tasks

- [x] **T1** — `ReviewOutput` Pydantic schema + import-time drift guard against `eval/schema.json`; promote `pydantic` to base `dependencies` in `pyproject.toml`. · files: `ch2_adaptation/src/ch2_adaptation/schema.py`, `ch2_adaptation/tests/test_review_schema.py`, `pyproject.toml` · verify: `uv run pytest ch2_adaptation/tests/test_review_schema.py -q`
- [x] **T2** — Hand-written stub eval set + smoke subset + README; locked review prompt module. · files: `eval/stub/stub_eval.jsonl`, `eval/stub/stub_eval_smoke.jsonl`, `eval/stub/README.md`, `ch2_adaptation/src/ch2_adaptation/prompts.py`, `ch2_adaptation/tests/test_stub_eval_set.py` · verify: `uv run pytest ch2_adaptation/tests/test_stub_eval_set.py -q`
- [x] **T3** — `BaselineConfig` + `SMOKE_MODEL_TAG` + `load_baseline_config()`; `baseline_smoke.yaml` / `baseline_full.yaml`. · files: `ch2_adaptation/src/ch2_adaptation/config.py`, `ch2_adaptation/configs/baseline_smoke.yaml`, `ch2_adaptation/configs/baseline_full.yaml`, `ch2_adaptation/tests/test_baseline_config.py` · verify: `uv run pytest ch2_adaptation/tests/test_baseline_config.py -q`
- [x] **T4** — `write_json_atomic` extracted in `eval/harness.py`; `baseline.py` core (`score_system`, `build_baselines_doc`) driven by fakes. · files: `eval/harness.py`, `ch2_adaptation/src/ch2_adaptation/baseline.py`, `ch2_adaptation/tests/test_baseline.py` · verify: `uv run pytest ch2_adaptation/tests/test_baseline.py eval/tests -q`
- [x] **T5** — `model.py::make_hf_generator`; `frontier.py` (`GeminiClient`, `StubFrontierClient`, `estimate_cost_usd`); `baseline.py::main`; `ch2` extra deps; `.env.example`. Smoke run green under 120s. · files: `ch2_adaptation/src/ch2_adaptation/model.py`, `ch2_adaptation/src/ch2_adaptation/frontier.py`, `ch2_adaptation/src/ch2_adaptation/baseline.py`, `pyproject.toml`, `.env.example` · verify: `time uv run python -m ch2_adaptation.baseline --config ch2_adaptation/configs/baseline_smoke.yaml`
- [x] **T6** — Spec amendments recorded (this file), `specs/STATUS.md` C1 → `planned`/`building` as applicable, `progress_report.md` entries for T1-T5. Full baseline run + `eval/results/baselines.json` commit happens outside this session (GPU-budget hook blocks `--config *full*` in-session by design). · files: `specs/05-C1-task-schema-and-base-model.md`, `specs/STATUS.md`, `progress_report.md` · verify: `uv run ruff check . && uv run black --check . && uv run pytest -q`

## Remaining before this spec can move to `done`

All six tasks are committed and the full gate is green. **AC-5 is still open**: the full baseline
run has not happened. It cannot happen in-session — the GPU-budget hook blocks any command
matching `--config .../full.yaml` by design, and this is the correct behavior (a full run
downloads ~3GB and calls a paid-tier-capable API). Run it by hand, then commit the result:

```bash
GEMINI_API_KEY=... PRICE_PER_1K_INPUT_USD=... PRICE_PER_1K_OUTPUT_USD=... \
  uv run python -m ch2_adaptation.baseline --config ch2_adaptation/configs/baseline_full.yaml
```

Then inspect `eval/results/baselines.json` (both `base_model` and `frontier_api` entries, a real
`n_samples`), commit it, and move C1 to `done` in `specs/STATUS.md`. **Only then may C2/C3 begin
in earnest** — per FR-15, `baselines.json` must exist and be committed before C4 moves to
`building`.
