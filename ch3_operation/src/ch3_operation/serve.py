"""Chapter 3 serving entrypoint (spec O0): loads config, builds the real backend and
FastAPI app, and runs uvicorn -- or, with `--check`, loads the model and exits 0 without
serving, so CI can prove the model loads without keeping a server alive."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from ch3_operation.api.main import create_app
from ch3_operation.config import N_GPU_LAYERS, ServeConfig, load_serve_config
from ch3_operation.model_backend import LlamaCppBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the Chapter 3 code-review model.")
    parser.add_argument("--config", type=Path, required=True, help="path to a YAML config")
    parser.add_argument(
        "--check",
        action="store_true",
        help="load the config and model, report, and exit 0 without starting a server",
    )
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

    backend = LlamaCppBackend(config)  # loads once, here -- never per request (FR-20)
    print("[ch3] model loaded successfully")

    if args.check:
        print("[ch3] --check: exiting without starting a server")
        return

    app = create_app(backend, config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
