"""Chapter 3 — serving the quantized model on CPU, with guardrails and monitoring.

Two invariants hold everywhere in this package: inference is CPU-only (`n_gpu_layers=0`,
CON-12), and no model output reaches a caller without validating against `eval/schema.json`
(CON-7). A service that returns unvalidated model output is a service that hands its
callers a crash.
"""
