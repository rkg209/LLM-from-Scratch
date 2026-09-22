# O5 — Deploy

| | |
|---|---|
| **State** | building |
| **Depends on** | O4 |
| **Requirements** | FR-25, NFR-3, BG-3 |
| **W&B run** | — |

## Problem

A live URL is the single highest-leverage artifact in the entire portfolio. A hiring manager will not clone the repo, will not run `docker build`, and will not read `benchmark.py` — but they will click a link, paste in a Java method, and watch a model *they can see the training story for* return structured JSON in a few seconds. Everything else in this project exists to make that moment credible.

This is also the spec that closes the loop on the project's central claim (BG-3): not "I trained a model" but "I operated one" — something that is running right now, on public infrastructure, that someone else can break.

## Scope

Deploy the container to Hugging Face Spaces (CPU tier) and to one cloud free tier (Railway / Render / Fly.io); measure real serving metrics; publish both links.

## Acceptance criteria

1. A **public HF Spaces URL** is live, responds to `POST /review` with a schema-valid response, and is linked from the README (FR-25a).
2. A **public cloud free-tier URL** is live, responds correctly, and is linked from the README (FR-25b). Two hosts, because a free tier that sleeps or disappears leaves the portfolio's best artifact dead at the worst possible moment.
3. The model is pulled from HF Hub at container start; **no weights are in git** (NFR-13).
4. **Real p50/p99 latency and throughput, measured against the deployed Space**, are recorded in the README's metrics scorecard (NFR-3, FR-23b). Numbers from a laptop do not count.
5. Cold-start behavior is documented — a sleeping free-tier Space takes time to wake, and a reader who clicks the link needs to know that rather than concluding the demo is broken.
6. The README shows a **worked example**: a Java snippet in, the actual JSON review out. A link without an example asks the reader to do work.
7. No secret is required to use the demo, and no secret is embedded in the Space (NFR-14).

## Out of scope

- In-browser WebGPU inference → **O6** (stretch).
- Autoscaling, custom domains, uptime SLAs. It is a free-tier demo, and the README says so.
- Authentication (`planning/05-api-design.md` §2).

## Clarifications

*(filled by `/clarify` — open question: which cloud free tier. Railway, Render, and Fly.io all shift their free offerings; pick at deploy time based on what actually still exists.)*

## Technical plan

See `.claude/plans/Operation.md` §O5, and `docs/DEPLOY.md` for the actual runbook. T1 is
done: `space/README.md` (the HF Spaces card: front matter, cold-start note, curl example),
`docs/DEPLOY.md`, `scripts/deploy_space.sh`, `scripts/smoke_deployed.py` (tested, 6 tests).

One deliberate deviation from the umbrella plan's literal file list: no `space/Dockerfile`.
The plan's own reasoning was "one Dockerfile of record, not two that drift" but a *copied*
`space/Dockerfile` is still a second file that can silently diverge from `docker/Dockerfile`
(the one O4 actually build-verified). `deploy_space.sh` instead uploads `docker/Dockerfile`
straight to the Space's required root `Dockerfile` path via `hf upload ... Dockerfile`, so
there is exactly one Dockerfile in the whole repo, ever. Similarly `space/README.md` becomes
the Space's root `README.md` only at upload time, not by being copied ahead of time.

T2–T6 (the actual deploy, both hosts, live benchmark numbers, README demo section, STATUS
flip to `done`) are **not started**: they require a real GGUF on HF Hub (O1 T7, blocked on
C4/C5) and an authenticated `hf auth login` this session does not have (`hf auth whoami`
fails with no token found) — plus a second-host account, and the judgment call of actually
publishing a public URL, which is exactly the kind of externally-visible, hard-to-reverse
action this project's own conventions ask to get explicit go-ahead for rather than run
unattended.

## Tasks

*(filled by `/tasks`)*
