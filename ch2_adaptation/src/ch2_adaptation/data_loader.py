"""Prompt formatting and the training `Dataset` for the QLoRA fine-tune (spec C4).

Loss is computed only on the assistant turn -- the system/user tokens are masked with
`-100`, per the `qlora-recipe` skill. Training on the prompt tokens would teach the model
to reproduce buggy Java, the opposite of the goal. Never reads `eval/holdout/` (CON-6) --
this module only ever opens whatever JSONL path it is given (`data/train.jsonl` /
`data/val.jsonl`), and the leakage guard would block it anyway if it tried.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ch2_adaptation.prompts import SYSTEM_PROMPT

if TYPE_CHECKING:
    import torch

_LABEL_FIELDS = ("severity", "category", "line", "issue", "suggested_fix")


def _review_json(record: dict[str, Any]) -> str:
    return json.dumps({key: record[key] for key in _LABEL_FIELDS})


def _prompt_prefix(record: dict[str, Any]) -> str:
    """Everything up to (not including) the assistant's response -- the span that gets
    masked with `-100`."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{record['code']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def format_prompt(record: dict[str, Any]) -> str:
    """The exact Qwen2.5 chat-template shape the recipe locks: system, user, assistant."""
    return f"{_prompt_prefix(record)}{_review_json(record)}<|im_end|>"


def _load_records(path: Path | str) -> list[dict[str, Any]]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


class ReviewDataset:
    """A `torch.utils.data.Dataset` of (buggy Java snippet -> review JSON) pairs.

    Tokenizes the prompt-only prefix and the full sequence separately to find the mask
    boundary precisely, rather than guessing an offset -- a guessed offset silently drifts
    whenever the prefix or response text changes length.
    """

    def __init__(self, jsonl_path: Path | str, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.records = _load_records(jsonl_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        import torch

        record = self.records[idx]
        prefix_ids = self.tokenizer(_prompt_prefix(record), add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(format_prompt(record), add_special_tokens=False)["input_ids"]

        # BPE merge rules are context-sensitive: a prefix tokenized alone is not
        # guaranteed to equal the same span once more text follows it, especially right
        # at a newline boundary like the one before the assistant's JSON. If this ever
        # drifts, masking the wrong span silently would teach the model on the wrong
        # tokens -- fail loudly instead (AC-4 is a correctness contract, not a best effort).
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                f"record {idx}: the tokenized prompt prefix is not a prefix of the full "
                "tokenized sequence -- the label mask boundary would be wrong. This means "
                "the tokenizer's BPE merges differ across the prefix/full-sequence split; "
                "fix the split point rather than masking with an unverified offset."
            )

        labels = list(full_ids)
        for i in range(len(prefix_ids)):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
