---
title: Java Code Reviewer
emoji: 🧐
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# Java Code Reviewer

A QLoRA fine-tune of `Qwen/Qwen2.5-Coder-1.5B-Instruct`, quantized to GGUF and served on CPU,
that reads one Java method and returns a structured bug review — no prose, a single JSON
object matching a locked schema every time (or an honest `422` if it can't).

This Space *is* the model: no API key, no external calls, nothing running but this container.

**Cold start:** free-tier CPU Spaces sleep after inactivity. The first request after a sleep
can take a minute or more while the Space wakes and the model reloads — that's expected, not
broken. Once warm, a review typically takes a few seconds.

## Try it

```bash
curl -s -X POST https://rkg209-java-code-reviewer.hf.space/v1/review \
  -H 'content-type: application/json' \
  -d '{"code": "public int sum(int[] a) {\n    int s = 0;\n    for (int i = 0; i <= a.length; i++) s += a[i];\n    return s;\n}"}'
```

A real response from this model, copied verbatim:

```json
{"severity":"critical","category":"Logic Error","line":3,"issue":"The loop condition checks i <= a.length, which will cause an ArrayIndexOutOfBoundsException when i equals a.length because the array index is zero-based.","suggested_fix":"Change the loop condition to i < a.length to avoid accessing out-of-bounds elements."}
```

`GET /health` and `GET /metrics` are also live. Full API docs, source, and the training story
(from-scratch transformer → QLoRA fine-tune → this container) are in the project repo:
https://github.com/rkg209/LLM-from-Scratch — see its README for the head-to-head eval table this model
was measured against before it was quantized and deployed here.
