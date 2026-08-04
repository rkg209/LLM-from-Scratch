# Deploying (spec O5)

Two live public URLs, real numbers, a worked example — see `specs/15-O5-deploy.md`'s
acceptance criteria. This is a **manual, credentialed runbook**: nothing in this file runs
itself, and none of it should run from an unattended session (it creates public,
externally-visible state — a Space, a deployed service — that this project's own
conventions treat as the kind of action that needs a human's go-ahead, not automation).

## Prerequisite: a real GGUF on HF Hub

O5 serves whatever O1 produced. Before any of the below:

1. O1's manual GPU-box run has produced `outputs/model.gguf` (or whichever quant level
   cleared AC-4's bar) and it has been uploaded to a **public** HF Hub model repo — see
   `scripts/README.md`'s O1 runbook. Public, so the Space needs no `HF_TOKEN` at runtime
   (AC-7).
2. Note the repo id (`MODEL_REPO`) and filename (`MODEL_FILE`) — both are Space
   configuration, not secrets.

Until that exists, everything below can still be exercised against the committed
`ch3_operation/tests/fixtures/tiny.gguf` fixture as a rehearsal (it won't produce schema-valid
reviews, but it proves the deploy mechanics).

## Host 1 — Hugging Face Spaces (Docker SDK, CPU basic)

```bash
hf auth login   # once, needs a token with write access: https://huggingface.co/settings/tokens

SPACE_ID=<your-hf-username>/java-code-reviewer \
MODEL_REPO=<your-hf-username>/<gguf-repo> \
MODEL_FILE=model.gguf \
  ./scripts/deploy_space.sh
```

`scripts/deploy_space.sh` creates the Space if needed and uploads the build context
(`eval/`, `monitoring/`, `ch3_operation/`, `pyproject.toml`, `docker/entrypoint.sh`) plus
`docker/Dockerfile` renamed to the Space's root `Dockerfile` and `space/README.md` renamed to
the Space's root `README.md` — no second, hand-maintained Dockerfile copy to drift from the
one O4 already build-verified. After it prints, set `MODEL_REPO`/`MODEL_FILE` in the Space's
**Settings → Variables** (not Secrets — the model repo is public) if the script's own
`hf repos create` didn't already carry them over, then wait for the Space to report
"Running" in its own build logs.

Before editing `space/README.md`, replace the `https://github.com/<owner>/<repo>` placeholder
with this repo's real URL.

Verify:

```bash
uv run python scripts/smoke_deployed.py --url https://<your-hf-username>-java-code-reviewer.hf.space
```

## Host 2 — a second free tier

Deliberately left as a decision made **at deploy time** (spec O5's own clarification note):
Render, Fly.io, and Railway all shift their free offerings often enough that locking one in
here would just go stale. Whichever is picked:

1. Point it at the same `docker/Dockerfile` (the build context is this repo's root — no
   Space-specific renaming needed off HF Spaces, most platforms accept
   `--dockerfile docker/Dockerfile` or an equivalent build setting).
2. Set the same two env vars, `MODEL_REPO` and `MODEL_FILE`.
3. Verify the same way:
   ```bash
   uv run python scripts/smoke_deployed.py --url https://<host-2-url>
   ```
4. Record which host was picked, and why, in `progress_report.md` — the reasoning matters
   more here than the specific choice, since the choice itself is expected to age.

## After both are live

```bash
# cold start: hit the URL once first and let that request finish before benchmarking, or
# the p99 will just be measuring "how long does this host take to wake up" (Risk 6)
uv run python scripts/benchmark_serving.py --url <url> --requests 30 \
  --config ch3_operation/configs/benchmark_smoke.yaml
```

Paste the real numbers (not a laptop's) into the README's `<!-- SERVING_METRICS_START -->`
block (`eval.report.update_readme_section` does the mechanical replace). Add both links, the
cold-start note, and a worked example — a real snippet in, the actual returned JSON out,
copied from a real response, not written by hand. Flip `specs/STATUS.md`'s O5 row to `done`
only once every AC is actually true, not once the mechanics are in place.
