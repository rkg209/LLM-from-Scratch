# C1 — Task + schema + base model fixed, baselines measured

*Plan for `specs/05-C1-task-schema-and-base-model.md`. Written 2026-07-15, before implementation.
The spec's `Technical plan` section is filled from this during task 6.*

## Context

C1 is the first non-foundation spec and the gate on all of Chapter 2. Its thesis is that two
things are cheap now and ruinously expensive later: **the base model tag** (changing it after
data generation means re-running every baseline, CON-11) and **the baselines themselves**. A
baseline measured *after* the fine-tune exists is a baseline measured by someone who already
knows what number they need — the prompt gets tweaked, the parse gets forgiving, and the
comparison quietly stops being a comparison. So: measure the base model and a frontier API
**first**, freeze the numbers, and the C5 eval table means something.

F2 already landed most of acceptance criterion 1 — `LOCKED_MODEL_TAG` in
`ch2_adaptation/src/ch2_adaptation/config.py:12`, enforced in `FinetuneConfig.__post_init__`.
C1's real new surface is: the Pydantic schema with a drift guard, the review prompt, a
hand-written stub eval set, a base-model generator, a frontier client, and `baselines.json`.

### Decisions taken during planning (these resolve the spec's open `/clarify` question)

| Decision | Choice | Why |
|---|---|---|
| Frontier model | **Gemini 2.5 Flash**, free tier, `GEMINI_API_KEY` | No Claude API available. Gemini's free tier is rate-limited, not credit-limited, and it is a real frontier-lab model — so "Frontier API, 3-shot" stays an honest row label. Native JSON mode matters (see Risk 1). |
| Smoke base model | **Tiny stand-in tag**, not the real 1.5B | fp32 Qwen-1.5B on CPU is ~6 GB resident and ~40–120s of generation alone. It cannot honour NFR-1 reliably. Smoke proves the *code path*; the full config produces the *number*. |
| Stub set location | **`eval/stub/stub_eval.jsonl`** | It is eval data, not ch2 training data. `ch2_adaptation/data/*.jsonl` is gitignored (`.gitignore:11`) *and* hard-blocked by `commit_hygiene.py:51` — and this file must be committed, since it is the provenance of a published number. `eval/stub/` touches no safety rail: the leakage guard is scoped to `eval/holdout/` only. |

## Files

**New — `ch2_adaptation/src/ch2_adaptation/`**

- `schema.py` — Pydantic `ReviewOutput` (`extra="forbid"`) + the import-time drift guard against `eval/schema.json`.
- `prompts.py` — the locked review prompt: system message, the 3 few-shot examples, `format_zero_shot()` / `format_three_shot()`. **C4 inference and C5's comparison must reuse this verbatim** or the head-to-head is not apples-to-apples.
- `model.py` — `make_hf_generator(cfg)`, returning a `list[str] -> list[str]` callable. Pulled forward from C4; `baseline.py` cannot score the base model without it.
- `frontier.py` — `FrontierClient` protocol, `GeminiClient`, `StubFrontierClient`, `estimate_cost_usd()`.
- `baseline.py` — `score_system()`, `build_baselines_doc()`, `main()`.

**New — data & config**

- `eval/stub/stub_eval.jsonl` — ~24 hand-written buggy-Java records, `{id, code, line, category, severity}`.
- `eval/stub/stub_eval_smoke.jsonl` — a strict 3-record subset of the above.
- `eval/stub/README.md` — what this set is, and that it is *not* the holdout (C2 builds that).
- `ch2_adaptation/configs/baseline_smoke.yaml`, `ch2_adaptation/configs/baseline_full.yaml`.

**Modified**

- `ch2_adaptation/src/ch2_adaptation/config.py` — add `BaselineConfig`, `load_baseline_config()`, `SMOKE_MODEL_TAG`.
- `eval/harness.py` — extract `write_json_atomic(payload, path)` as a public primitive; `write_result` becomes a one-liner over it. One atomic-write implementation, reused by C3/C5.
- `pyproject.toml` — move `pydantic>=2.7` into base `dependencies` (see Risk 3); add `google-genai` to the `ch2` extra.
- `.env.example` — add a `GEMINI_API_KEY=` placeholder (NFR-14: keys live in env, never in a tracked file).
- `specs/05-C1-task-schema-and-base-model.md` — fill Technical plan; amend AC-1/AC-6/AC-7; record the clarification answers.
- `specs/STATUS.md` — C1 `draft` → `planned`.
- `progress_report.md` — one entry per task, per CLAUDE.md.

**Tests — `ch2_adaptation/tests/`** (basenames must stay globally unique; `test_schema.py` is taken by `eval/tests/`)

`test_review_schema.py`, `test_baseline.py`, `test_baseline_config.py`, `test_stub_eval_set.py`.

## Approach

### `schema.py` — the drift guard is the whole point

`ReviewOutput` mirrors `planning/03-system-design.md:357-371` exactly: `severity: Literal[...]`,
`category/issue/suggested_fix: str` with min/max length, `line: int = Field(ge=1)`,
`model_config = ConfigDict(extra="forbid")`.

The guard is `_assert_matches_eval_schema(schema: dict | None = None) -> None`, called at module
import with the real `eval/schema.json`, and taking an injectable dict so a test can hand it a
mutated schema and assert it raises. A naive `model_json_schema() == json.load(...)` **will not
work** — Pydantic emits `title` keys per property and no `$schema`. So compare the load-bearing
keys only, the same set `eval/tests/test_schema.py:69-74` already treats as the contract:

- `required` (as a set), `additionalProperties is False`
- `properties.severity.enum` exact list
- the `minLength`/`maxLength`/`minimum` constraints per field

Raise a dedicated `SchemaDriftError(RuntimeError)` — **not** `AssertionError` as
`planning/03-system-design.md:371` suggests. `assert` is stripped under `python -O`, which would
silently disable the one guard whose entire job is to not be silent. This is a deliberate
deviation from the design doc; record it in `progress_report.md`.

### `prompts.py` — one prompt, locked

The system message instructs: identify exactly one bug, respond with a **single bare JSON object,
no markdown fences, no prose**, and inlines `eval/schema.json`. The fence instruction is
load-bearing: `eval/metrics.py:20-26` does a plain `json.loads` with no fence-stripping, so a
` ```json ` wrapper scores as **invalid**.

**Do not add fence-repair to the baseline path.** If the base model wraps its output, that is a
real schema-validity failure and it is precisely the number the fine-tune is supposed to move.
(ch3's validator repairs fences at *serving* time — that is a different contract.)

The 3 few-shot examples are hand-written and **must not appear in the stub set** — few-shot
examples drawn from the eval set is leakage, and it is the exact failure mode this spec exists to
prevent. `test_stub_eval_set.py` asserts the disjointness.

### `baseline.py` — inject the generator, so scoring is testable without torch

```python
Generator = Callable[[list[str]], list[str]]

def score_system(prompts: list[str], generate: Generator, stub_path: Path) -> EvalResult
def build_baselines_doc(base: EvalResult, frontier: EvalResult, cfg: BaselineConfig, usage: Usage) -> dict
def main() -> None   # --config; wires generators, scores both, writes, logs to W&B
```

`score_system` is pure and takes the generator as an argument, so every test in CI drives it with
a fake that returns canned strings — no torch, no network. `main()` builds the real generators:
`model.make_hf_generator(cfg)` and `frontier.make_client(cfg)`, whose heavy imports
(`torch`/`transformers`, `google.genai`) are **function-local, never module-level** — otherwise
`pytest -q` breaks in CI, which syncs `--extra dev` only (`.github/workflows/ci.yml:21`).

Scoring goes through `eval/harness.py::score_outputs` — never ad-hoc (AC-3, NFR-7).

`baselines.json` shape (both entries + provenance, atomic via the new `write_json_atomic`):

```json
{
  "timestamp": "2026-07-15T…Z",
  "stub_set": "eval/stub/stub_eval.jsonl",
  "stub_set_sha256": "…",
  "schema_sha256": "…",
  "base_model":   { "model_tag": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "mode": "zero-shot",
                    "dtype": "float32", "…": "EvalResult fields" },
  "frontier_api": { "model_tag": "gemini-2.5-flash", "mode": "3-shot", "provider": "google-genai",
                    "tier": "free", "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                    "…": "EvalResult fields" }
}
```

The two sha256 fields are what make the number re-checkable later (X2): if the stub set or the
schema changes under a committed `baselines.json`, that is detectable rather than silent.

### `frontier.py` — cost recording (AC-7)

`GeminiClient` requests `response_mime_type="application/json"` plus a `response_schema` derived
from `ReviewOutput`, so the model emits bare JSON and the fence problem never arises on this path.
It records `usage_metadata` token counts and computes cost via
`estimate_cost_usd(in_tokens, out_tokens)`, which reads `PRICE_PER_1K_INPUT_USD` /
`PRICE_PER_1K_OUTPUT_USD` **from the environment, never hardcoded** — the same rule C3's budget
guard is specified with (`planning/03-system-design.md:400-405`). C3 imports this helper rather
than writing a second one.

On the free tier the honest recorded figure is `$0.00`, with `tier: "free"` and the real token
counts alongside it — the tokens are what a paid re-run would cost, and hiding them behind a zero
would make the budget line unverifiable.

`StubFrontierClient` returns canned valid JSON and touches no network; it is what
`baseline_smoke.yaml` selects.

### `BaselineConfig` — and the invariant that keeps CON-11 honest

```yaml
# baseline_full.yaml
base_model_tag: Qwen/Qwen2.5-Coder-1.5B-Instruct
frontier_provider: gemini            # smoke: stub
frontier_model_tag: gemini-2.5-flash
stub_set_path: eval/stub/stub_eval.jsonl        # smoke: eval/stub/stub_eval_smoke.jsonl
results_path: eval/results/baselines.json       # smoke: outputs/baselines-smoke.json
max_new_tokens: 256                  # smoke: 32
temperature: 0.0
n_few_shot: 3
use_4bit: false
seed: 42
wandb_mode: online                   # smoke: disabled
wandb_project: ch2-adaptation
```

`is_smoke` is derived from `frontier_provider == "stub"`. `__post_init__` enforces:

- **not smoke** → `base_model_tag` must equal `LOCKED_MODEL_TAG`. This is what stops someone
  quietly baselining a *different* real model and publishing the number (CON-11).
- **smoke** → `base_model_tag` must equal `SMOKE_MODEL_TAG` (an allowlist of exactly one tiny
  stand-in). The escape hatch is a single named tag, not "anything goes in smoke".

**Smoke must never write `eval/results/baselines.json`.** It writes to gitignored
`outputs/baselines-smoke.json`. A smoke run silently overwriting the committed, published numbers
with tiny-random-model garbage is the worst outcome this spec has.

`eval/config.py::load_config` rejects both unknown *and* missing keys, so every field above must
exist in both YAMLs.

## Tests

One test per acceptance criterion; all run in the base+dev CI env with no torch and no network.

| AC | Test |
|---|---|
| 1 | `test_baseline_config.py` — `baseline_full.yaml` loads with `base_model_tag == LOCKED_MODEL_TAG`; a non-locked tag on the full path raises `ValueError` (via `dataclasses.replace`, the idiom already in `test_finetune_config.py`); smoke's tag must equal `SMOKE_MODEL_TAG`. |
| 2 | `test_review_schema.py` — valid record accepted; extra key rejected; bad severity rejected; `line=0` rejected; empty `category` rejected; **drift guard raises `SchemaDriftError` when handed a mutated schema dict**; `model_json_schema()` agrees with `eval/schema.json` on every load-bearing key. |
| 3 | `test_baseline.py` — `score_system` with a hand-built fake generator produces *exact* rates (mirror `eval/tests/test_harness.py`'s arithmetic-fixture style: N samples, known-valid, known-caught). `StubFrontierClient` makes no network call. |
| 4 | `test_baseline.py` — `build_baselines_doc` yields both `base_model` and `frontier_api` keys, a frontier `model_tag`, and a timestamp that `datetime.fromisoformat` parses; `write_json_atomic` leaves no `.tmp` behind and the result re-reads as valid JSON. |
| 6 | The smoke command itself, timed (below). Not in CI — CI syncs `--extra dev` only, and the tiny-model path needs `--extra ch2` (~800 MB of torch). Covered locally and by the fake-generator unit tests. |
| 7 | `test_baseline.py` — `estimate_cost_usd` reads prices from env and raises if they are unset; it never carries a hardcoded price. |
| — | `test_stub_eval_set.py` — every record has `{id, code, line}`; ids unique; `line >= 1`; smoke set is a strict subset of the full set; **no few-shot example code appears anywhere in the stub set**. |

## Risks

1. **Fences sink the base-model score.** An instruct model asked for JSON often returns
   ` ```json … ``` `, which `eval/metrics.py` scores as invalid — so the base-model
   schema-validity rate may land near zero. That is a *real* finding, not a bug, and it is the
   headline the fine-tune exists to fix. The temptation to "fix" it by relaxing the parser is
   exactly what NFR-7 forbids. The test that catches a silent relaxation is
   `eval/tests/test_harness.py`, which already pins the strict behaviour.
2. **The comparison may become circular.** C3 must now also pick an API, and with no Claude key
   the obvious choice is Gemini again — in which case the fine-tune is distilled from the same
   model it is benchmarked against, and "beating the frontier baseline" becomes near-impossible
   by construction. That is survivable (the honest framing is "we approach the teacher at a
   fraction of the serving cost"), but **C3's spec must decide it explicitly** rather than
   inherit it by accident. Flag on the C3 spec when C1 lands.
3. **`pydantic` must move to base deps.** `pytest -q` runs `ch2_adaptation/tests` in the base+dev
   CI env. The moment `schema.py` imports pydantic, CI breaks unless pydantic is a base
   dependency. It is light and it is the runtime home of the eval contract, which CLAUDE.md wants
   importable everywhere — so promoting it is right, not a workaround.
4. **The GPU-budget hook will block the full baseline command in-session** — `gpu_budget_guard.py:21`
   matches `--config[= ]\S*full`, and `baseline_full.yaml` matches. This is *correct*: the full
   baseline downloads 3 GB and calls a paid-tier-capable API. The human runs it outside the
   session, or with `ALLOW_FULL_RUN=1`. `baseline_smoke.yaml` is waved through by `SMOKE_OVERRIDE`.
5. **AC-5 ordering.** `baselines.json` must be committed *before* C4 moves to `building`. Nothing
   mechanically enforces this — it is a discipline point, and it is the entire reason the spec
   exists. `/checkpoint` on C1 should verify the file is committed and non-empty.

## Spec amendments required (surface these; do not apply silently)

- **AC-1** says `full.yaml` must contain `model.name_or_path`. The shipped config uses a flat
  `model_tag` (`config.py:17`, locked by `planning/03-system-design.md:339`), and
  `eval/config.py::load_config` rejects unknown keys — a nested key would break `FinetuneConfig`
  and all six existing ch2 tests. **Amend AC-1 to `model_tag`.** FR-14's acceptance text carries
  the same stale wording.
- **AC-6** implies the real fp32 model in smoke. **Amend** to name the locked tiny stand-in tag,
  with the NFR-1 arithmetic as the reason.
- **AC-7 / CON-3.** CON-3 says the API budget is "used exclusively for synthetic training-data
  generation". C1's 3-shot baseline is an *eval* use. Record it as an explicit carve-out rather
  than pretending the sentence already covers it.
- **Clarifications** — record: frontier = Gemini 2.5 Flash (free tier); C3's generator model is
  **not yet decided** and inherits the circularity question in Risk 2.

## Tasks (for `/tasks`; one task = one commit)

1. `schema.py` + drift guard + `test_review_schema.py`; promote `pydantic` to base deps.
2. `eval/stub/` — stub set, smoke subset, README; `prompts.py`; `test_stub_eval_set.py`.
3. `BaselineConfig` + both YAMLs + `test_baseline_config.py`.
4. `write_json_atomic` in `eval/harness.py`; `baseline.py` core (`score_system`,
   `build_baselines_doc`); `test_baseline.py` with fakes.
5. `model.py` + `frontier.py` (Gemini + stub clients, cost estimator); `--extra ch2` deps;
   `.env.example`. Smoke run green.
6. Spec amendments, `STATUS.md` → `planned`/`done`, `progress_report.md`. **Then** the human runs
   the full baseline and commits `baselines.json` — that commit closes AC-5.

## Verification

```bash
# the default check — must be green before every commit
uv run ruff check . && uv run black --check . && uv run pytest -q

# smoke: proves load → prompt → generate → parse → score → atomic write, on CPU, in seconds.
# Writes to outputs/baselines-smoke.json, NEVER to eval/results/baselines.json.
time uv run python -m ch2_adaptation.baseline --config ch2_adaptation/configs/baseline_smoke.yaml
#   → must finish well under 120s (NFR-1) and print both rates.

# the real run — outside the session (the GPU-budget hook blocks it in-session, by design):
GEMINI_API_KEY=… uv run python -m ch2_adaptation.baseline \
    --config ch2_adaptation/configs/baseline_full.yaml
#   → writes eval/results/baselines.json; inspect it, then commit it.
```

Then confirm by hand that `eval/results/baselines.json` carries **both** a `base_model` and a
`frontier_api` entry with a real `n_samples`, and that git tracks it — that, and nothing less, is
what lets C4 move to `building`.
