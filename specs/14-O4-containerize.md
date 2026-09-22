# O4 — Containerize

| | |
|---|---|
| **State** | done (2026-09-22 audit; see specs/STATUS.md) |
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

See `.claude/plans/Operation.md` §O4. Summary: `docker/Dockerfile` (multi-stage, `python:3.12-slim` builder compiles the `ch3` extra's wheels including `llama-cpp-python`'s CPU build; final stage copies only the venv plus a `libgomp1` runtime shared library, non-root user, no compiler survives), `docker/entrypoint.sh` (downloads the `.gguf` from `MODEL_REPO`/`MODEL_FILE` at container start, optional sha256 verify, `exec uvicorn`), `docker/.dockerignore`, `ch3_operation/tests/test_docker.py` (skipped unless `DOCKER=1`), `.github/workflows/docker.yml` (separate `linux/amd64` job, not folded into the main `check` job).

**Build-verified** (once a Docker daemon became available): `docker buildx build --platform linux/amd64` succeeds, `docker run` with `tiny.gguf` mounted in serves real traffic (`/health` 200, `/metrics` Prometheus text, `POST /v1/review` a correct 422 on the tiny stand-in model), and `DOCKER=1 uv run pytest ch3_operation/tests/test_docker.py -q` is 3/3 green. Two real bugs were found and fixed by that verification, not by inspection: `README.md` wasn't copied into the builder stage (hatchling's metadata step needs the file `pyproject.toml`'s `readme` field names), and the final stage was missing `libgomp1` — `llama-cpp-python`'s compiled `libllama.so` needs it at runtime even though the compiler toolchain that built it correctly does not survive into the final image. See `progress_report.md`'s "O4 follow-up" entry for the full story.

## Tasks

*(filled by `/tasks`)*
