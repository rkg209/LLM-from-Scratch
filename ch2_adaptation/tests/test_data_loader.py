"""format_prompt's output shape, and the AC-4 label-masking assertion: the labels tensor
must be -100 across the full prompt span and real token ids across the assistant span
(spec C4, task 3)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch  # noqa: E402
from ch2_adaptation.data_loader import ReviewDataset, format_prompt  # noqa: E402

RECORD = {
    "code": "public int add(int a, int b) { return a + b; }",
    "severity": "major",
    "category": "npe-risk",
    "line": 1,
    "issue": "Possible null dereference.",
    "suggested_fix": "Guard with Objects.requireNonNull.",
}


def test_format_prompt_has_the_locked_qwen_chat_template_shape() -> None:
    prompt = format_prompt(RECORD)

    assert prompt.startswith("<|im_start|>system\n")
    assert "<|im_start|>user\n" in prompt
    assert "<|im_start|>assistant\n" in prompt
    assert prompt.count("<|im_end|>") == 3
    assert RECORD["code"] in prompt


def test_format_prompt_assistant_turn_is_the_label_json() -> None:
    prompt = format_prompt(RECORD)
    assistant_turn = prompt.split("<|im_start|>assistant\n")[1]
    review_json = assistant_turn.removesuffix("<|im_end|>")

    parsed = json.loads(review_json)
    assert parsed == {
        "severity": "major",
        "category": "npe-risk",
        "line": 1,
        "issue": "Possible null dereference.",
        "suggested_fix": "Guard with Objects.requireNonNull.",
    }


@pytest.fixture
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")


@pytest.fixture
def dataset_path(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(RECORD) + "\n")
    return path


def test_review_dataset_reports_correct_length(dataset_path, tokenizer) -> None:
    dataset = ReviewDataset(dataset_path, tokenizer)
    assert len(dataset) == 1


def test_review_dataset_masks_the_full_prompt_span_and_keeps_the_assistant_span(
    dataset_path, tokenizer
) -> None:
    dataset = ReviewDataset(dataset_path, tokenizer)
    item = dataset[0]

    assert set(item) == {"input_ids", "attention_mask", "labels"}
    assert item["input_ids"].shape == item["labels"].shape == item["attention_mask"].shape
    assert torch.all(item["attention_mask"] == 1)

    from ch2_adaptation.data_loader import _prompt_prefix

    n_prefix_tokens = len(tokenizer(_prompt_prefix(RECORD), add_special_tokens=False)["input_ids"])

    # The entire prompt span (system + user + "assistant\n") is masked...
    assert torch.all(item["labels"][:n_prefix_tokens] == -100)
    # ...and the assistant's actual response is not.
    assert torch.all(item["labels"][n_prefix_tokens:] != -100)
    assert torch.equal(item["labels"][n_prefix_tokens:], item["input_ids"][n_prefix_tokens:])


def test_review_dataset_label_span_decodes_back_to_the_review_json(dataset_path, tokenizer) -> None:
    """A stronger version of the masking assertion: decoding the unmasked label span
    round-trips to exactly the assistant's review JSON, not merely "some real tokens"."""
    dataset = ReviewDataset(dataset_path, tokenizer)
    item = dataset[0]

    unmasked = item["labels"][item["labels"] != -100]
    decoded = tokenizer.decode(unmasked, skip_special_tokens=True)

    assert json.loads(decoded) == {
        key: RECORD[key] for key in ("severity", "category", "line", "issue", "suggested_fix")
    }


class _DriftingFakeTokenizer:
    """A tokenizer whose BPE-merge-like behavior is context-sensitive: tokenizing the
    prefix alone gives different ids than the same span gets inside the full sequence --
    simulating the real sharp edge this masking approach has to guard against."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        if text.endswith("<|im_start|>assistant\n"):
            # The prefix alone: pretend the trailing newline merges into its own token.
            return {"input_ids": [1, 2, 3]}
        # The full sequence: the same prefix span merges differently once more text
        # follows it, so the first three ids no longer match the prefix-alone tokenization.
        return {"input_ids": [1, 2, 99, 4, 5]}


def test_review_dataset_raises_if_the_tokenized_prefix_is_not_a_true_prefix(dataset_path) -> None:
    dataset = ReviewDataset(dataset_path, _DriftingFakeTokenizer())

    with pytest.raises(ValueError, match="not a prefix"):
        dataset[0]
