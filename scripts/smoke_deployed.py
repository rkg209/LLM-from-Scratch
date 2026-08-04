"""O5: post-deploy check against a live URL -- `/health`, `/metrics`, one real review.

Run against both hosts after a deploy (docs/DEPLOY.md). Exits non-zero on any failure so it
can gate a deploy script; prints exactly what it checked either way, since a "smoke passed"
with no detail is not something anyone can debug three months from now.

Usage:
    uv run python scripts/smoke_deployed.py --url https://<space>.hf.space
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

_REVIEW_CODE = "public String getUserName(User user) {\n    return user.getProfile().getName();\n}"


class SmokeCheckError(RuntimeError):
    """One of the post-deploy checks failed."""


def _get(url: str, timeout_s: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def check_health(base_url: str, timeout_s: float) -> None:
    status, body = _get(f"{base_url}/health", timeout_s)
    if status != 200:
        raise SmokeCheckError(f"/health returned {status}: {body!r}")
    payload = json.loads(body)
    if payload.get("model_loaded") is not True:
        raise SmokeCheckError(f"/health reports model_loaded != true: {payload}")
    print("[smoke_deployed] /health: ok, model_loaded=true")


def check_metrics(base_url: str, timeout_s: float) -> None:
    status, body = _get(f"{base_url}/metrics", timeout_s)
    if status != 200:
        raise SmokeCheckError(f"/metrics returned {status}")
    if b"review_requests_total" not in body:
        raise SmokeCheckError("/metrics response is missing review_requests_total")
    print("[smoke_deployed] /metrics: ok, review_requests_total present")


def check_review(base_url: str, timeout_s: float) -> dict:
    payload = json.dumps({"code": _REVIEW_CODE}).encode()
    request = urllib.request.Request(
        f"{base_url}/v1/review",
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status, body = response.status, response.read()
    except urllib.error.HTTPError as exc:
        status, body = exc.code, exc.read()

    if status != 200:
        raise SmokeCheckError(f"POST /v1/review returned {status} (not schema-valid): {body!r}")
    result = json.loads(body)
    print(f"[smoke_deployed] POST /v1/review: 200, severity={result.get('severity')!r}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-deploy smoke check for a live URL.")
    parser.add_argument("--url", required=True, help="base URL, e.g. https://x.hf.space")
    parser.add_argument("--timeout", type=float, default=120.0, help="per-request timeout (s)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = args.url.rstrip("/")

    check_health(base_url, args.timeout)
    check_metrics(base_url, args.timeout)
    check_review(base_url, args.timeout)

    print(f"[smoke_deployed] all checks passed against {base_url}")


if __name__ == "__main__":
    try:
        main()
    except SmokeCheckError as exc:
        print(f"[smoke_deployed] FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
