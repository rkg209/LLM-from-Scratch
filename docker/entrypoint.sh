#!/usr/bin/env sh
# O4 T2: download the served .gguf from HF Hub at container start (never baked into a
# layer, AC-6), verify it if a sha256 was given, then exec uvicorn.
#
# Env vars:
#   MODEL_REPO    HF Hub repo id, e.g. "someuser/java-reviewer-gguf"      (required to fetch)
#   MODEL_FILE    filename within that repo, e.g. "model-Q4_K_M.gguf"     (required to fetch)
#   MODEL_SHA256  expected sha256 of the downloaded file                  (optional)
#   MODEL_PATH    where to save it, and what ServeConfig's ${MODEL_PATH} resolves to
#   CONFIG_PATH   which ServeConfig YAML to load
#   PORT          port uvicorn binds

set -eu

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${CONFIG_PATH:?CONFIG_PATH must be set}"
: "${PORT:=8000}"

if [ ! -f "$MODEL_PATH" ]; then
    : "${MODEL_REPO:?MODEL_REPO must be set to fetch a model (no local $MODEL_PATH found)}"
    : "${MODEL_FILE:?MODEL_FILE must be set to fetch a model (no local $MODEL_PATH found)}"

    echo "[entrypoint] downloading ${MODEL_REPO}/${MODEL_FILE} -> ${MODEL_PATH}"
    mkdir -p "$(dirname "$MODEL_PATH")"
    python -c "
import hashlib
import os
import sys
import urllib.request

repo = os.environ['MODEL_REPO']
filename = os.environ['MODEL_FILE']
dest = os.environ['MODEL_PATH']
expected_sha256 = os.environ.get('MODEL_SHA256')

url = f'https://huggingface.co/{repo}/resolve/main/{filename}'
urllib.request.urlretrieve(url, dest)
print(f'[entrypoint] downloaded to {dest}')

if expected_sha256:
    digest = hashlib.sha256()
    with open(dest, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        os.remove(dest)
        print(
            f'[entrypoint] sha256 mismatch: expected {expected_sha256}, got {actual_sha256}',
            file=sys.stderr,
        )
        sys.exit(1)
    print('[entrypoint] sha256 verified')
"
else
    echo "[entrypoint] $MODEL_PATH already present, skipping download"
fi

echo "[entrypoint] starting uvicorn on :${PORT}"
exec uvicorn ch3_operation.api.main:app --host 0.0.0.0 --port "$PORT"
