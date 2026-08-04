# docs/

Two kinds of document live here. `DEPLOY.md` is an operational runbook — steps to follow.
The other three are interview-defense notes — not a tutorial, not a blog post, but a
dense, honest, technically precise answer to the question an interviewer who has actually
read the code will ask next. Each is self-contained (readable without the source open),
cites this project's own measured numbers rather than restating textbook claims, and ends
with something the author got wrong or found surprising while building it — the part most
worth reading, and the part a generic writeup would omit.

| Document | Answers |
|---|---|
| [`kv-cache.md`](kv-cache.md) | Why naive attention is `O(n²)` in sequence length, what a KV-cache stores (and doesn't), why it does nothing for prefill, and why the cached and uncached paths are tested for bit-identical logits rather than eyeballed. |
| [`finetune-vs-prompting.md`](finetune-vs-prompting.md) | Why structured-output conformance is where a small fine-tuned specialist beats few-shot prompting first, what few-shot prompting structurally cannot teach, where the frontier model still wins in this project's own table, and the cost/latency/privacy argument that holds regardless. |
| [`quantization.md`](quantization.md) | What absmax quantization does to a weight tensor, why speed comes from memory bandwidth rather than arithmetic (and the fp16-on-CPU counterexample this project measured), and why structured output is the first capability to degrade under aggressive quantization. |
| [`DEPLOY.md`](DEPLOY.md) | The manual, credentialed runbook for actually deploying Chapter 3 to HF Spaces and a second host. |

The three defense notes reference numbers in `eval/results/*.json` that are filled by the
full GPU runs described in [`../REPRODUCING.md`](../REPRODUCING.md); where a run hasn't
happened yet, the note says so explicitly rather than guessing at what the number will be.
