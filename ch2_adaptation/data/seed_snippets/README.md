# Seed snippet pool

`clean_pool.jsonl` is 150 hand-curated, **bug-free** Java/Spring methods — the raw material
`data_gen.py` injects a synthetic bug into (spec C3). Each line is `{"code": "...", "context":
"..."}`; there are no label fields here because these snippets have no bug yet — the frontier
API's job (`BUG_INJECTION_PROMPT_TEMPLATE`) is to introduce exactly one and describe it.

## Provenance

Generated from 11 common Spring/JPA method shapes (getter-by-id, `@Transactional` create,
guarded delete, active-only filter, DTO mapper, REST GET/POST endpoint, a correct
`equals`/`hashCode` pair, a builder step, a blank-field validator, a derived-query search),
each instantiated against 15 domain entities (User, Order, Product, Invoice, Account, Customer,
Payment, Shipment, Employee, Ticket, Comment, Review, Notification, Category, Subscription).
Templating over entity names — rather than hand-writing 150 independent snippets — is what made
150 genuinely distinct, syntactically valid methods tractable to author in one sitting; each
combination produces different field names, different repository calls, and different logic
shape, not just a renamed copy.

Every snippet is confirmed:
- **Unique** — no two snippets share a normalized code hash (`eval.dedup.code_hash`).
- **Disjoint from the holdout and stub sets** — zero hash overlap against
  `eval/stub/stub_eval.jsonl` and the C2 synthetic holdout records, checked directly against
  those files before this pool was committed.

## Use

`data_gen.py` reads this file, sends each snippet (up to `variants_per_snippet` times, per
`DataGenConfig`) to the frontier API asking it to inject one realistic bug and label it, and
writes the results to `data/train.jsonl` / `data/val.jsonl` (gitignored — only the generation
provenance is committed, not the generated data itself).
