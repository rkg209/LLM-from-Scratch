# project-summary.md

---

# LLM Engineering: From Architecture to Edge — Executive Summary

## What This Project Is

This is a solo engineering portfolio project that demonstrates end-to-end mastery of large language model (LLM) technology across three connected chapters: building a transformer from scratch, fine-tuning a real open-source model for a specialized task, and deploying that model as a production-grade edge service. The work spans approximately six months of weekend development and is designed to be fully reproducible on free or near-free compute.

The three chapters share a single narrative arc — "I understand a model from its internals all the way to a deployed edge service" — while remaining technically independent so each can be evaluated on its own merits.

**Chapter 1 — Architecture.** A small GPT-style language model built entirely from first principles in pure PyTorch: hand-written embeddings, multi-head attention, feed-forward layers, and a custom training loop. A KV-cache and quantization are then added, and the performance impact of each is measured precisely — producing a tokens-per-second speedup curve and a perplexity-versus-quantization tradeoff chart.

**Chapter 2 — Adaptation.** A Qwen2.5-Coder-1.5B open-source model is fine-tuned using QLoRA on a Java/Spring code-review task. The model learns to emit structured JSON reviews (`severity`, `category`, `line`, `issue`, `suggested_fix`). Training data is generated synthetically for approximately one dollar in API costs. The result is evaluated head-to-head against the base model and a frontier API in few-shot mode, on an independent held-out set that includes real code diffs — so the comparison is not circular.

**Chapter 3 — Operation.** The fine-tuned model is quantized to GGUF format, served behind a FastAPI endpoint with JSON-schema validation and output guardrails, instrumented with latency and quality-drift monitoring, containerized with Docker, and deployed to a live public demo on Hugging Face Spaces. An optional stretch goal adds in-browser inference via WebGPU.

---

## Who This Is For

This project is built as a **senior-level engineering portfolio artifact** aimed at three audiences:

- **Hiring managers and technical interviewers** evaluating candidates for ML engineering, GenAI engineering, or applied AI roles. The project demonstrates not just that the builder can run existing tools, but that they understand what is happening inside them — and can take a model from a blank file to a live production endpoint.

- **Peers and collaborators** in the ML and software engineering community who want a reference implementation of the full stack: from transformer internals through QLoRA fine-tuning to edge deployment, with honest evaluation methodology and reproducible results.

- **The builder's own future projects.** The Java code-review specialist, the evaluation harness, the serving infrastructure, and the Claude Code development scaffold are all designed for reuse in sibling projects (a Career Copilot and an SWE agent).

---

## Why It Matters

Most LLM portfolio projects demonstrate one of three things in isolation: "I understand transformers theoretically," "I can call a fine-tuning API," or "I can wrap a model in a web app." This project demonstrates all three in a single coherent artifact, with the connective tissue that practitioners actually care about.

**It proves depth, not just familiarity.** Writing multi-head attention by hand and measuring the exact cost of removing the KV-cache is a different signal than importing `nn.Transformer`. The speedup curve and perplexity tradeoff are numbers the builder had to earn.

**It demonstrates the most in-demand GenAI skill.** Turning a generalist open-source model into a cheap, private specialist that outperforms a frontier API on a narrow task — at a fraction of the inference cost — is the core value proposition of applied LLM engineering in 2024 and beyond. Chapter 2 makes that case with a real eval table, not a claim.

**It closes the loop to production.** "I trained a model" is table stakes. "I operated one" — with schema validation, latency monitoring, drift detection, containerization, and a live public endpoint — is what separates an ML practitioner from an ML engineer.

**It is honest.** The evaluation set is independent and partly real. The synthetic data provenance is documented. Failures are reported. The cost is stated. The cut-lines are explicit. In a field where benchmark manipulation is common, methodological honesty is itself a differentiator.

**It fits real constraints.** The entire project runs on free compute (Google Colab, Kaggle, a college GPU cluster) with approximately one dollar in API spend. Every component has a CPU smoke configuration that runs in seconds. This is not a project that requires a research lab — it is a project that requires rigor.

---

*Total estimated build time: ~6 months of weekend work. Total estimated API cost: ~$1. Compute: free GPU tiers + college cluster. Live artifact: public HF Spaces demo + reproducible codebase.*