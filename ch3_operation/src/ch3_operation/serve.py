"""Chapter 3 serving entrypoint.

SCAFFOLD (spec F2). Loads and validates the serving config and reports what it would
serve — it does not start a server. The FastAPI app is spec O0 (thin slice, base model),
the guardrails are O2, and the metrics are O3.

O0 is deliberately early in the backlog: standing the serving path up against the base
model, before Chapter 2 finishes, is what stops deployment from becoming the thing that
gets discovered in the last week.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ch3_operation.config import N_GPU_LAYERS, ServeConfig, load_serve_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the Chapter 3 code-review model.")
    parser.add_argument("--config", type=Path, required=True, help="path to a YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config: ServeConfig = load_serve_config(args.config)

    print(f"[ch3] config loaded: {args.config}")
    print(f"[ch3] model_path={config.model_path} (n_gpu_layers={N_GPU_LAYERS}, CPU-only)")
    print(
        f"[ch3] llama.cpp: n_ctx={config.n_ctx} n_threads={config.n_threads} "
        f"max_tokens={config.max_tokens} temperature={config.temperature}"
    )
    print(f"[ch3] would serve on {config.host}:{config.port}")
    print(f"[ch3] drift window: last {config.metrics_window} requests")
    print("[ch3] SCAFFOLD: no FastAPI app yet — that is spec O0. Config path verified.")


if __name__ == "__main__":
    main()
