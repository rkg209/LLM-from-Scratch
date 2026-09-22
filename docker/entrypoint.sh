#!/usr/bin/env sh
# O4 T2: download the served .gguf from HF Hub at container start (never baked into a
# layer, AC-6), pinned to a commit SHA and sha256-verified (A13), then exec uvicorn.
#
# The pin lives in the serving config (model_repo/model_file/model_revision/model_sha256).
# Env vars:
#   MODEL_PATH      where to save it, and what ServeConfig's ${MODEL_PATH} resolves to
#   MODEL_REPO      optional: serve a different HF Hub repo than the config's pin; then
#   MODEL_FILE      ...the filename within it                               (required with it)
#   MODEL_REVISION  ...a full 40-hex commit SHA, never a branch             (required with it)
#   MODEL_SHA256    ...the file's expected sha256                           (optional)
#   CONFIG_PATH     which ServeConfig YAML to load
#   PORT            port uvicorn binds

set -eu

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${CONFIG_PATH:?CONFIG_PATH must be set}"
: "${PORT:=8000}"

if [ ! -f "$MODEL_PATH" ]; then
    python -u -m ch3_operation.fetch_model --config "$CONFIG_PATH"
else
    echo "[entrypoint] $MODEL_PATH already present, skipping download"
fi

echo "[entrypoint] starting uvicorn on :${PORT}"
exec uvicorn ch3_operation.api.main:app --host 0.0.0.0 --port "$PORT"
