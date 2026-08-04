"""O1 T3: convert the merged HF model to f16 GGUF via llama.cpp's own converter.

Wraps `llama.cpp/convert_hf_to_gguf.py` rather than reimplementing GGUF export -- llama.cpp
is the format's reference implementation, and hand-rolling a second converter is exactly the
kind of parallel-path drift the shared eval harness (FR-5) exists to avoid elsewhere. This is
step one of two (AC-2): f16 first, quantized second, so a failure localizes to conversion or
to quantization, never both at once.

Usage:
    uv run python scripts/export_gguf.py --config ch3_operation/configs/export_smoke.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ch3_operation.config import load_export_config

_MIN_PLAUSIBLE_GGUF_BYTES = 1024  # a real f16 GGUF is many MB; a truncated write is not


def convert_to_f16_gguf(
    merged_dir: Path | str, llama_cpp_dir: Path | str, out_path: Path | str
) -> Path:
    converter = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
    if not converter.exists():
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found under {llama_cpp_dir} -- clone/build llama.cpp "
            "there first (see scripts/README.md)."
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(converter),
            str(merged_dir),
            "--outtype",
            "f16",
            "--outfile",
            str(out_path),
        ],
        check=True,
    )

    if not out_path.exists() or out_path.stat().st_size < _MIN_PLAUSIBLE_GGUF_BYTES:
        raise RuntimeError(
            f"convert_hf_to_gguf.py ran but {out_path} is missing or implausibly small -- "
            "treat this as a failed conversion, not a usable file."
        )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a merged HF model to f16 GGUF.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="re-convert even if the output exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_export_config(args.config)

    out_path = Path(config.gguf_f16_path)
    if out_path.exists() and not args.force:
        print(f"[export_gguf] {out_path} already exists, skipping (pass --force to redo)")
        return

    print(f"[export_gguf] converting {config.merged_dir} -> {out_path}")
    convert_to_f16_gguf(config.merged_dir, config.llama_cpp_dir, out_path)
    print(f"[export_gguf] wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
