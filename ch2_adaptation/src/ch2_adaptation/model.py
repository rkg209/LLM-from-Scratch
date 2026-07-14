"""The base-model generator (pulled forward from spec C4): `baseline.py` cannot score the
base model zero-shot without it.

`transformers`/`torch` stay function-local inside `make_hf_generator`, never module-level —
otherwise importing this module breaks `pytest -q` in the base+dev CI environment, which
never installs the `ch2` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ch2_adaptation.baseline import Generator

if TYPE_CHECKING:
    from ch2_adaptation.config import BaselineConfig


def make_hf_generator(cfg: BaselineConfig) -> Generator:
    """Load the base model once and return a `prompts -> raw responses` callable.

    fp32 on CPU (the only path this spec exercises): the smoke config uses
    `SMOKE_MODEL_TAG` so this stays fast; the full config uses the real 1.5B model and is
    never launched from a session (the GPU-budget hook blocks `--config .../full.yaml`).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_tag)
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model_tag, dtype=torch.float32)
    model.eval()

    def generate(prompts: list[str]) -> list[str]:
        outputs: list[str] = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=cfg.temperature > 0.0,
                    temperature=cfg.temperature if cfg.temperature > 0.0 else None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0][encoded["input_ids"].shape[-1] :]
            outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
        return outputs

    return generate
