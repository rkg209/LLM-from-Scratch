#!/usr/bin/env bash
# O5 T1/T2: create (if needed) and push to an HF Spaces Docker SDK Space.
#
# Uses `hf upload` rather than a git remote: it avoids the "Dockerfile must live at the
# repo root" vs. "our Dockerfile lives at docker/Dockerfile" mismatch entirely, by uploading
# docker/Dockerfile to the *target* path "Dockerfile" and space/README.md to "README.md" --
# no second, hand-maintained copy of either file to drift from the original (the umbrella
# plan's own words: "one Dockerfile of record, not two that drift"). Everything else is
# uploaded from its real path unchanged, so the Space's build context is exactly what
# `docker buildx build -f docker/Dockerfile .` already builds and O4 already verified.
#
# Requires: `hf auth login` (or HF_TOKEN in the environment) with write access, and a
# public model repo already holding the exported GGUF (see scripts/README.md's O1 runbook --
# this script does not build or upload the model itself).
#
# Usage:
#   SPACE_ID=<hf-username>/<space-name> ./scripts/deploy_space.sh
#
# The image serves the GGUF pinned in ch3_operation's serving config (A13), so no model
# variables are needed. Serving a different model means MODEL_REPO + MODEL_FILE +
# MODEL_REVISION (a full commit SHA) as Space variables -- see docker/entrypoint.sh.

set -eu

: "${SPACE_ID:?SPACE_ID must be set, e.g. someuser/java-code-reviewer}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "[deploy_space] ensuring $SPACE_ID exists (Docker SDK, public, cpu-basic)"
hf repos create "$SPACE_ID" --type space --sdk docker --exist-ok --public

echo "[deploy_space] uploading build context (eval/, monitoring/, ch3_operation/, pyproject.toml)"
hf upload "$SPACE_ID" eval eval --repo-type space
hf upload "$SPACE_ID" monitoring monitoring --repo-type space
hf upload "$SPACE_ID" ch3_operation ch3_operation --repo-type space
hf upload "$SPACE_ID" pyproject.toml pyproject.toml --repo-type space
hf upload "$SPACE_ID" docker/entrypoint.sh docker/entrypoint.sh --repo-type space

echo "[deploy_space] uploading Dockerfile (renamed to repo root, per HF Spaces' own requirement)"
hf upload "$SPACE_ID" docker/Dockerfile Dockerfile --repo-type space

echo "[deploy_space] uploading the Space card (renamed to repo root README.md)"
hf upload "$SPACE_ID" space/README.md README.md --repo-type space

cat <<EOF
[deploy_space] pushed. The image serves the pinned GGUF from its serving config, so no Space
variables are needed (the model repo is public: no HF_TOKEN at runtime, per O5 AC-7). A build
will start automatically; first boot also downloads the model, so expect it to take a few
minutes beyond the image build itself.

Once it reports "Running":
  uv run python scripts/smoke_deployed.py --url https://${SPACE_ID//\//-}.hf.space
EOF
