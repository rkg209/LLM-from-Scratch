"""O1 T2: merge the trained LoRA adapter into the base model, then sanity-check it.

`load_base_model(..., use_4bit=False)` — fp32/CPU — is exactly what a merge needs; merging
onto a 4-bit base is not supported by `peft.merge_and_unload()`. This script imports both
`ch2_adaptation` and `ch3_operation`, which is fine here: `scripts/` is the one place that is
allowed to see across the chapter seam (plan D-4). The two packages themselves still never
import each other.

The sanity generation runs on `sanity_sample_path` — training/staging examples, never
`eval/holdout/` (the leakage guard blocks that unconditionally at the tool layer, but this
script does not even try). It is a collapse check, not a scored comparison: that is O1's
`ch3_operation.evaluate`, run later against the exported GGUF.

Usage:
    uv run python scripts/merge_adapter.py --config ch3_operation/configs/export_smoke.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ch2_adaptation.config import LOCKED_MODEL_TAG, SMOKE_MODEL_TAG
from ch2_adaptation.model import generate_completions, load_base_model
from ch2_adaptation.prompts import format_zero_shot
from ch3_operation.config import ExportConfig, load_export_config

from eval.harness import load_schema
from eval.metrics import is_schema_valid


class BaseModelMismatchError(RuntimeError):
    """The adapter was trained on a different base model than the config names."""


def _expected_base_tag(config: ExportConfig) -> str:
    return SMOKE_MODEL_TAG if config.is_smoke else LOCKED_MODEL_TAG


def check_adapter_base_tag(adapter_path: Path | str, expected_tag: str) -> None:
    """Refuse to merge if the adapter was trained on a different base model (AC-1)."""
    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"no adapter_config.json found under {adapter_path}")
    adapter_config = json.loads(adapter_config_path.read_text())
    recorded_tag = adapter_config.get("base_model_name_or_path")
    if recorded_tag != expected_tag:
        raise BaseModelMismatchError(
            f"adapter at {adapter_path} was trained on {recorded_tag!r}, but this export "
            f"config expects {expected_tag!r}. Merging would silently attach the adapter to "
            "the wrong base model -- fix the config or point at the right adapter."
        )


def load_sanity_prompts(sample_path: Path | str, n: int) -> list[str]:
    path = Path(sample_path)
    if not path.exists():
        raise FileNotFoundError(f"sanity_sample_path not found: {path}")
    records = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
            if len(records) >= n:
                break
    if not records:
        raise ValueError(f"{path} has no usable records for a sanity check")
    return [format_zero_shot(record["code"]) for record in records]


def merge_and_save(config: ExportConfig) -> Path:
    """Load base + adapter, merge, save the merged model + tokenizer to `merged_dir`."""
    from peft import PeftModel

    expected_tag = _expected_base_tag(config)
    check_adapter_base_tag(config.adapter_path, expected_tag)

    base_model, tokenizer = load_base_model(config.base_model_tag, use_4bit=False)
    model = PeftModel.from_pretrained(base_model, config.adapter_path, is_trainable=False)
    merged = model.merge_and_unload()
    merged.eval()

    merged_dir = Path(config.merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    return merged_dir


def sanity_check(merged_dir: Path, config: ExportConfig) -> float:
    """Generate on a handful of non-holdout examples; return the schema-valid fraction."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(merged_dir)
    model = AutoModelForCausalLM.from_pretrained(merged_dir)
    model.eval()

    prompts = load_sanity_prompts(config.sanity_sample_path, config.sanity_n)
    outputs = generate_completions(model, tokenizer, prompts, max_new_tokens=256)

    schema = load_schema()
    n_valid = sum(is_schema_valid(output, schema) for output in outputs)
    return n_valid / len(outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--force", action="store_true", help="re-merge even if merged_dir already exists"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_export_config(args.config)

    merged_dir = Path(config.merged_dir)
    if merged_dir.exists() and any(merged_dir.iterdir()) and not args.force:
        print(f"[merge_adapter] {merged_dir} already populated, skipping (pass --force to redo)")
        return

    print(f"[merge_adapter] merging {config.adapter_path} onto {config.base_model_tag}")
    merge_and_save(config)
    print(f"[merge_adapter] wrote merged model to {merged_dir}")

    valid_fraction = sanity_check(merged_dir, config)
    print(
        f"[merge_adapter] sanity check: {valid_fraction:.2f} schema-valid on "
        f"{config.sanity_n} samples"
    )
    if valid_fraction == 0.0:
        print(
            "[merge_adapter] every sanity sample failed schema validation -- the merge "
            "likely collapsed the model. Refusing to treat this as a usable checkpoint."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
