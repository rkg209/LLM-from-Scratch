# O4 — Containerize

| | |
|---|---|
| **State** | draft |
| **Depends on** | O3 |
| **Requirements** | FR-24, NFR-19, CON-12 |
| **W&B run** | — |

## Problem

"It runs on my machine" is not a deployment, and the gap between the two is almost entirely made of things that are invisible until they break somewhere else: a system library llama.cpp needs, a Python version, a CPU instruction set the wheel was not built for, a model file that was on disk locally and is nowhere in the image.

The container is also what makes the reproducibility claim (X2) checkable by a stranger. `docker run` is the shortest possible distance between a hiring manager reading the README and the model actually answering them.

## Scope

A multi-stage Dockerfile producing a self-contained CPU image that serves the quantized GGUF model, plus the entrypoint and `.dockerignore`.

## Acceptance criteria

1. `docker build` succeeds from a clean checkout with no prior local state (FR-24).
2. `docker run -p 8000:8000 <image>` starts the server, and a test request to `/review` returns a schema-valid response.
3. `/health` and `/metrics` respond correctly inside the container.
4. **The image is self-contained**: it requires nothing on the host beyond Docker itself — no pre-installed ML framework, no local Python (NFR-19).
5. The image is **CPU-only**; no CUDA runtime, no GPU base image (CON-12).
6. **The `.gguf` model is not baked into the image.** It is downloaded from HF Hub at container start — a multi-GB layer makes the image unpushable on free tiers and un-cacheable on rebuild.
7. The final stage is a slim base (`python:3.12-slim`), and the build stage's toolchain does not survive into it.
8. The image builds on the architecture HF Spaces actually runs (`linux/amd64`) — an image built only for `arm64` on an Apple Silicon laptop will not start there.

## Out of scope

- Deploying it → **O5**.
- Multi-worker or multi-replica serving. `llama-cpp-python` is lock-serialized (O0) and the free tier is single-instance; concurrency is not a claim this project makes.

## Clarifications

*(filled by `/clarify`)*

## Technical plan

*(filled by `/plan`)*

## Tasks

*(filled by `/tasks`)*
