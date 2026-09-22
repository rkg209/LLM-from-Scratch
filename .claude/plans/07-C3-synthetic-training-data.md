# Plan: C3 — Synthetic training data

## Context

C4's fine-tune needs a few hundred (buggy Java/Spring snippet → structured review JSON) pairs
to train on. `specs/07-C3-synthetic-training-data.md` was `draft` with three open
clarifications — C1's own spec explicitly flagged one of them ("C3's generator model is not yet
decided... must decide this explicitly rather than inherit it by accident") as a real fork, not
a detail to assume. Resolved with the user before this plan was written:

- **Snippet source**: synthetic bug-injection into a curator-collected pool of clean Java/Spring
  code, not live GitHub-MCP mining. C2 already mines real snippets for the holdout; reusing that
  same live pool for ~300 training records doesn't scale within a session and risks source-pool
  overlap that dedup then has to clean up after the fact. Synthetic injection is disjoint by
  construction and scales to hundreds of records.
- **Generator model**: Gemini 2.5 Flash — the same frontier model C1 already benchmarks as the
  baseline. This is deliberately circular in the narrow sense C1 flagged, and the honest framing
  is recorded as-is: the project's claim becomes "distillation — the specialist approaches its
  teacher at a fraction of the size/cost/latency," not "beats a bigger model." Reusing
  `frontier.GeminiClient` verbatim also means $0 real cost (free tier) and no new API key.
- **Target size**: ~300 records, 270 train / 90%-val split → 270/30, generated from a seed pool
  of clean snippets with up to ~2 bug variants each, budget-guarded and deduped down from there.

## Design decisions made in this plan (not asked, mechanical extensions of existing patterns)

- **One frontier call injects and labels together.** Rather than one call to inject a bug and a
  second to describe it (which risks the description drifting from what was actually inserted),
  a single prompt asks Gemini to take a clean snippet, introduce exactly one realistic bug of a
  named category, and return `{"code": "...", "severity", "category", "line", "issue",
  "suggested_fix"}` — `code` plus the five `ReviewOutput` fields in one JSON response. This is a
  **new** prompt template (`BUG_INJECTION_PROMPT_TEMPLATE` in `data_gen.py`), distinct from
  `prompts.py`'s locked review-only prompt — `prompts.py` stays untouched and reserved for C4
  inference / C5 comparison, per its own docstring.
- **Cost tracking reuses `frontier.estimate_cost_usd`/`GeminiClient` unmodified** — no new client
  code, no `PRICE_PER_1K_TOKENS` env var (the old `planning/03-system-design.md` pseudocode named
  one, but C1 shipped `PRICE_PER_1K_INPUT_USD`/`PRICE_PER_1K_OUTPUT_USD` instead; C3 follows what
  actually shipped, and this gets recorded as a spec amendment like C1 recorded its own AC-1/AC-6
  amendments).
- **Cross-spec dedup closes the loop C2 opened.** C2's plan (`.claude/plans/06-C2-...md`)
  publishes `eval/holdout_hashes.txt` — a normalized-code-hash index living outside
  `eval/holdout/`, so reading it doesn't trip the leakage guard. `data_gen.py` reads that file
  (never `eval/holdout/` itself — CON-6/AC-7 stay intact) and drops any generated record whose
  `eval.dedup.code_hash` collides with a holdout hash, before it ever reaches `train.jsonl`.
- **The GPU-budget hook already gates the full run correctly, by accident of naming.** Its
  `FULL_RUN_PATTERNS` regex matches any `--config` value containing `full` (e.g.
  `configs/datagen_full.yaml`), so a real full-budget generation run is blocked from inside a
  session exactly like a GPU training run is — consistent with the project's "smoke only in a
  session" rule applying to real spend, not just GPU time. The full run is therefore launched by
  hand, same shape as C1's `baseline_full.yaml` and C2's freeze step.

## Files

**New — `ch2_adaptation/src/ch2_adaptation/data_gen.py`**
- `BUG_INJECTION_PROMPT_TEMPLATE` — locked for this spec's own generation run (hashed into
  provenance as `prompt_template_hash`, per FR-17b).
- `inject_and_label(snippets: list[str], client: FrontierClient) -> list[dict]` — calls the
  client once per snippet, parses `{"code", "severity", "category", "line", "issue",
  "suggested_fix"}`, validates the five label fields against `ch2_adaptation.schema.ReviewOutput`
  (reused, not re-implemented). A response that fails validation is **dropped and logged**, per
  AC-1 — never hand-repaired.
- `dedup_against_holdout(records, holdout_hashes: set[str]) -> list[dict]` — uses
  `eval.dedup.code_hash` (the same primitive C2 defines) to drop any record whose code hash
  appears in `eval/holdout_hashes.txt`.
- `dedup_within_training_set(records) -> list[dict]` — same primitive, drops internal duplicates
  (AC-6).
- `split_train_val(records, frac=0.10, seed) -> tuple[list, list]` — seeded split (AC-5).
- `build_provenance(...) -> dict` — `run_id`, `generated_at`, `frontier_model`,
  `prompt_template_hash`, `n_requested`, `n_valid`, `n_skipped`, `estimated_cost_usd`,
  `actual_cost_usd`, `split_seed` (FR-17b, exact field list).
- `main()` — budget guard (abort before spending if `estimate_cost_usd(...) > 1.50`, AC-3),
  smoke path (`StubFrontierClient`, a handful of seed snippets, no network — CI-safe per AC-9),
  full path (real `GeminiClient`, gated behind the GPU-budget hook as above). Writes
  `data/train.jsonl` / `data/val.jsonl` (gitignored) and `data/provenance.json` (committed) via
  `eval.harness.write_json_atomic`.

**New — `ch2_adaptation/data/seed_snippets/clean_pool.jsonl` + `README.md`**
~150 hand-collected/-written clean Java/Spring methods (small, committed — this is source
material, not generated output), each used for up to 2 bug-injection attempts to reach the ~300
target after budget/dedup losses.

**New — config:** `DataGenConfig` + `load_data_gen_config()` in
`ch2_adaptation/src/ch2_adaptation/config.py`, mirroring `BaselineConfig`'s shape (a smoke/full
split enforced in `__post_init__`); `ch2_adaptation/configs/{datagen_smoke.yaml,datagen_full.yaml}`.

**New — `eval/dedup.py`** — only if not already landed by C2's plan (task 1 there). If C2 has
already shipped it, C3 imports it as-is; this plan does not re-specify it.

**New — tests:** `ch2_adaptation/tests/{test_data_gen.py,test_data_gen_config.py}` — all pure
functions driven by fixtures and a fake `FrontierClient`, no network, no torch.

**Modified:** `.gitignore` (confirm `ch2_adaptation/data/*.jsonl` is ignored — `commit_hygiene.py`
already blocks committing it as defense-in-depth, but the gitignore should also say so
explicitly so `git status` stays clean).

## Task sequence

1. `DataGenConfig` + configs + tests. Verify: `uv run pytest ch2_adaptation/tests/test_data_gen_config.py -q`.
2. Seed snippet pool (`clean_pool.jsonl`, ~150 entries) — curation content, not code.
3. `data_gen.py` core (`inject_and_label`, `dedup_against_holdout`, `dedup_within_training_set`,
   `split_train_val`, `build_provenance`) + tests with a fake client. Verify:
   `uv run pytest ch2_adaptation/tests/test_data_gen.py -q`.
4. `data_gen.py::main()` wiring — budget guard, smoke path exercised
   (`uv run python -m ch2_adaptation.data_gen --config ch2_adaptation/configs/datagen_smoke.yaml`),
   full path documented as a manual, out-of-session run (mirrors C1's `baseline_full.yaml`
   runbook).
5. `.gitignore` update; confirm `commit_hygiene.py` still blocks a stray `data/train.jsonl`
   staged by accident.
6. Spec/STATUS updates — record the three resolved clarifications and the
   `PRICE_PER_1K_TOKENS` → `PRICE_PER_1K_INPUT_USD`/`PRICE_PER_1K_OUTPUT_USD` amendment in
   `specs/07-C3-synthetic-training-data.md`, move C3 to `planned`/`building` in `specs/STATUS.md`.

## Verification

- `uv run ruff check . && uv run black --check . && uv run pytest -q` after every task.
- Smoke run: under a few seconds, `StubFrontierClient`, writes only to a gitignored path, no
  `GEMINI_API_KEY` needed.
- Full run (by hand, outside the session, after C2's `eval/holdout_hashes.txt` exists): inspect
  `data/provenance.json` for `n_requested`/`n_valid`/`n_skipped` and confirm `actual_cost_usd`
  reflects real token counts (free tier → $0.00, honestly recorded, not assumed).
- Confirm zero overlap: every hash in the frozen `train.jsonl` is absent from
  `eval/holdout_hashes.txt` (spot-check script, not just trust the dedup step ran).
