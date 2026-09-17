# Why a small fine-tuned specialist can beat few-shot prompting on a narrow task

Source: `specs/08-C4-qlora-finetune.md`, `specs/09-C5-head-to-head-eval-table.md`,
`.claude/skills/qlora-recipe/SKILL.md`, `eval/schema.json`, `ch2_adaptation/baseline.py`.

**Read the measured result first (2026-09-18).** The mechanism this note argues for is real and
showed up exactly where predicted *against the base model*: schema-validity went 0.00 → 0.97 on
the same 40 records, from the same weights. Against the **frontier API** it did not win — 0.97 vs
1.00 on validity and 0.80 vs 0.82 on bug-catch, one record each. The title's "can beat" is a
statement about the mechanism, not a claim about this table; see *The honest limits* below for
what actually happened and why.

## Where a specialist wins first: structured-output conformance

The task here is not "review Java code well" in the open-ended sense — it is "review
Java code **and emit output that validates against `eval/schema.json`**: a `severity`
enum, bounded-length `category`/`issue`/`suggested_fix` strings, a `line` integer,
`additionalProperties: false`." That second half is a distributional property of the
*output format*, not of the model's reasoning, and it is exactly what gradient updates
are good at moving.

Fine-tuning on records that are all schema-valid by construction (`data_gen.py` drops any
generated record that fails `eval/schema.json` validation before it ever reaches the
training set — see the qlora-recipe skill) directly reshapes the model's output
distribution toward "the next token is what a valid JSON object at this position looks
like." A prompt can only *ask* for that shape; it cannot make the model's underlying
token distribution favor it. That is the mechanical reason format conformance is the
first place a small specialist catches up to, and can pass, a much larger model prompted
zero- or few-shot.

### What the base model's 0.00 is actually made of (measured, 2026-09-18)

The zero is not the single tidy "it wraps everything in a fence" story it is easy to tell. Of the
40 zero-shot base-model outputs in `eval/results/baselines_holdout.json`:

- **40/40** are markdown-fenced, which fails them immediately under the no-repair rule.
- **12/40** would validate if the fence were stripped. The other 28 fail anyway: **27 on the
  `severity` enum** — the model emits `"error"` (25) or `"warning"` (2), which is what a linter
  calls those levels, against a schema that specifies `critical|major|minor|info` — and one on
  malformed JSON (a raw control character inside a string).

So format conformance here is two separate learned things: the *envelope* (no prose, no fence)
and the *vocabulary* (this enum, not the one the model's pretraining makes obvious). A prompt can
state both; the base model, read zero-shot, obeys neither reliably. The fine-tune fixed both at
once, which is the real content of 0.00 → 0.97 — and it is a stronger claim than "it stopped
adding fences", which is what the raw number lets you assume if you never open the per-sample
outputs.

**And the base model's 0.00 bug-catch is entailed, not observed.** `eval/harness.py` scores
`caught = valid and is_bug_caught(...)`: validity gates catch by construction, so 0.00 validity
forces 0.00 catch regardless of what the model saw. Its outputs do name real defects. Whether it
would have *located* them within ±2 lines is unmeasured — it cannot be measured through this
harness without relaxing the contract, which is not on the table. The honest reading of that cell
is "no scorable answer", and the honest comparison it supports is against the fine-tune's
*format* competence, not its *code-reading* competence.

## What few-shot prompting structurally cannot do

Two shots or three shots of the target format compete with the actual task for the
model's limited attention within one context window — the model has to simultaneously
infer the schema *from examples* and solve the code-review problem, every single request,
starting from zero. It never accumulates that inference into anything permanent. Worse,
every request pays for it again: the few-shot examples are real input tokens, on every
call, forever, which is a recurring cost a fine-tune pays exactly once (at training time)
and never again at inference time. `ch2_adaptation/baseline.py`'s `Usage` dataclass exists
specifically to make that recurring per-request cost visible rather than hidden behind a
"free tier" that quietly isn't free at scale.

## The honest limits — where the frontier model still won here, and why

**The fine-tune lost on both published metrics.** On the frozen 40-record holdout
(`eval/results/finetuned.json` vs `eval/results/baselines_holdout.json`):

| System | Schema-validity | Bug-catch | n |
|---|---|---|---|
| Fine-tuned (QLoRA, Qwen2.5-Coder-1.5B) | 0.97 (39/40) | 0.80 (32/40) | 40 |
| Frontier API (`gemini-3.5-flash-lite`, 3-shot) | 1.00 (40/40) | 0.82 (33/40) | 40 |

Both losses are one record wide. On n = 40 a single record is 2.5 points, so neither gap
supports a claim that the frontier model is meaningfully better at this task — and, just as
importantly, the fine-tune's near-parity is not a claim that it is as good either. The honest
statement is that the two are indistinguishable at this sample size, and the sample size is
small because the holdout was built by hand and frozen rather than scraped.

The prediction this project wrote down in advance was that the specialist would **win
schema-validity outright** and lose bug-catch. Half of that was wrong, and the way it was wrong
is worth more than the prediction: the fine-tune's single invalid output is not a markdown fence
(the base model's failure mode, 40 times out of 40) but **unescaped double quotes** — it emitted
a Java snippet, `result += name + ", ";`, straight into the `suggested_fix` string, and the JSON
died at character 337. Fine-tuning moved the output distribution onto "emit a bare JSON object"
completely and reliably; what it did not teach is the one place JSON and Java collide, which is
also the place a code-review model is most likely to go. The frontier model avoids it by
construction, not by being smarter: its native JSON mode serializes the string for it. That is a
real advantage of a constrained-decoding API over a fine-tune with free-form decoding, and the
fix is the same class of thing (grammar-constrained decoding at serving time, which Chapter 3's
GGUF path can do) rather than more training.

On bug-catch the two models disagree on five of the forty records: three the fine-tune missed
and the frontier caught, two the reverse. The three it missed are the predicted failure — it
finds *a* plausible defect rather than *the* defect. On one record it reported a null-pointer
risk at line 28 where the actual bug was `SimpleDateFormat` being shared across threads; on
another it flagged a null return at line 13 where the real defect was an off-by-one in a
`parts.length >= 2` bounds check. Both times the output was well-formed, confident, and about a
bug class heavily represented in the synthetic training set, aimed at code whose real defect was
not. That is the shape of a narrow training distribution, not of a model that cannot read code.

A fine-tune trained on synthetic data generated by that same frontier model has a hard
ceiling: it can approach the teacher's behavior on the *distribution the teacher was asked
to generate*, but it has no path to exceed the teacher's judgment on cases the teacher
itself was never asked to produce examples of. Where the frontier model wins, that is
almost certainly why — not a fluke, and not something more training steps on the same
synthetic data would fix.

## The argument that holds regardless of who wins on accuracy

Even on the metric where the frontier API wins, the case for the fine-tuned specialist is
a cost/latency/privacy argument, not an accuracy argument, and it does not depend on the
accuracy table going the specialist's way:

- **Cost.** The fine-tuned model runs at zero marginal per-request API cost, forever,
  after the one-time ≈$1 training spend. The frontier baseline pays a token bill on every
  single call, indefinitely.
- **Latency and availability.** A 1.5B model quantized to GGUF and served on CPU (Chapter
  3) has no network round-trip to a third-party API, no rate limit, no outage dependency.
- **Privacy.** Code sent to a frontier API leaves the network. Code reviewed by the
  self-hosted 1.5B model never does — which matters specifically for the kind of code
  (internal, proprietary, security-sensitive) that a code-review tool exists to look at.

None of these three depend on the fine-tune winning the accuracy table. They are the
reason a company would deploy the smaller model even in a world where the frontier model
is measurably better at the task — which is also, honestly, the more common world.

## What I got wrong

I expected the baseline comparison to be an afterthought — measure it, report it, move
on. It turned out to be structurally load-bearing: `baseline.py`'s `Usage` accounting
(recording input/output tokens and cost even when the recorded cost is $0.00 on a free
tier) exists because a bare "$0.00" is unverifiable and would have let the cost argument
above collapse into an assertion nobody could check. The surprising part was realizing
that *the honesty of the cost story* depended on instrumenting something that, on paper,
cost nothing.
