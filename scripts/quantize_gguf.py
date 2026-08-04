"""O1 T4: quantize an f16 GGUF with llama.cpp's `llama-quantize` binary.

Step two of the mandatory two-step conversion (AC-2). `quant_type` is a config value, not a
flag baked into this script -- AC-4's fallback ladder (Q4_K_M -> Q5_K_M -> Q8_0) is walked by
re-running with a different `export_*.yaml`, never by editing this file.

Usage:
    uv run python scripts/quantize_gguf.py --config ch3_operation/configs/export_smoke.yaml
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from ch3_operation.config import load_export_config

_MIN_PLAUSIBLE_GGUF_BYTES = 1024


def quantize(
    f16_path: Path | str, out_path: Path | str, quant_type: str, llama_cpp_dir: Path | str
) -> Path:
    binary = _find_quantize_binary(llama_cpp_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run([str(binary), str(f16_path), str(out_path), quant_type], check=True)

    if not out_path.exists() or out_path.stat().st_size < _MIN_PLAUSIBLE_GGUF_BYTES:
        raise RuntimeError(
            f"llama-quantize ran but {out_path} is missing or implausibly small -- treat "
            "this as a failed quantization, not a usable file."
        )
    return out_path


def _find_quantize_binary(llama_cpp_dir: Path | str) -> Path:
    llama_cpp_dir = Path(llama_cpp_dir)
    for candidate in ("llama-quantize", "build/bin/llama-quantize", "quantize"):
        path = llama_cpp_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"no llama-quantize binary found under {llama_cpp_dir} -- build llama.cpp there "
        "first (see scripts/README.md)."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize an f16 GGUF.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--force", action="store_true", help="re-quantize even if the output exists"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_export_config(args.config)

    out_path = Path(config.gguf_quant_path)
    if out_path.exists() and not args.force:
        print(f"[quantize_gguf] {out_path} already exists, skipping (pass --force to redo)")
        return

    print(f"[quantize_gguf] quantizing {config.gguf_f16_path} -> {out_path} ({config.quant_type})")
    quantize(config.gguf_f16_path, out_path, config.quant_type, config.llama_cpp_dir)
    print(f"[quantize_gguf] wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
