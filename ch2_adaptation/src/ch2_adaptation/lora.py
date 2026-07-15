"""LoRA attachment for the QLoRA fine-tune (spec C4).

The recipe (r/alpha/targets/dropout, `bias="none"`, `TaskType.CAUSAL_LM`) is locked in the
`qlora-recipe` skill and `planning/03-system-design.md` §1.2 — this module applies it, it
does not choose it. Heavy imports (`peft`) stay function-local so this module imports
cleanly in the base+dev CI environment, which never installs the `ch2` extra; `TYPE_CHECKING`
keeps real type hints on the public functions without paying for that import at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ch2_adaptation.config import FinetuneConfig

if TYPE_CHECKING:
    import torch
    from peft import LoraConfig, PeftModel

# The upper bound is AC-3's real stop condition: near-100% trainable means the attach
# silently fell back to full fine-tuning instead of LoRA. The lower bound only needs to
# rule out "every parameter frozen, nothing trains at all" -- it is deliberately far below
# the "roughly 0.5%" full-recipe figure (r=16 on a 1.5B model), because the *same* recipe
# on SMOKE_MODEL_TAG's ~2.4M parameters lands at ~0.018% (measured directly): the smoke
# stand-in is so much smaller that the same four target modules are a much larger share of
# its total, but still small in absolute terms. A floor calibrated to the full model's
# fraction would falsely fail every smoke run.
_TRAINABLE_FRACTION_MIN = 0.00001
_TRAINABLE_FRACTION_MAX = 0.05


def build_lora_config(config: FinetuneConfig) -> LoraConfig:
    """The locked recipe, applied to whichever `r`/`alpha`/`dropout`/`target_modules` the
    smoke or full config specifies."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.target_modules),
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def compute_trainable_fraction(model: torch.nn.Module) -> float:
    """Fraction of parameters with `requires_grad=True`, by element count.

    A model with tied weights (e.g. tied input/output embeddings) counts that shared
    tensor twice via `model.parameters()` -- once under each name. Both the numerator and
    denominator inflate together, so the fraction itself stays representative; this does
    not attempt to de-duplicate tied parameters.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if total == 0:
        raise ValueError("model has no parameters to compute a trainable fraction from")
    return trainable / total


def check_trainable_fraction(fraction: float) -> None:
    """Raise if the trainable fraction is nowhere near what a LoRA attach should produce."""
    if not (_TRAINABLE_FRACTION_MIN <= fraction <= _TRAINABLE_FRACTION_MAX):
        raise RuntimeError(
            f"trainable parameter fraction {fraction:.4%} is outside the expected "
            f"[{_TRAINABLE_FRACTION_MIN:.3%}, {_TRAINABLE_FRACTION_MAX:.1%}] range for a LoRA "
            "attach. This usually means the attach silently fell back to full fine-tuning -- "
            "stop the run rather than train on a broken attach (AC-3)."
        )


def attach_lora(model: torch.nn.Module, config: FinetuneConfig) -> PeftModel:
    """Attach LoRA adapters and verify the attach actually happened (AC-3)."""
    from peft import get_peft_model

    peft_model = get_peft_model(model, build_lora_config(config))
    peft_model.print_trainable_parameters()

    fraction = compute_trainable_fraction(peft_model)
    print(f"[C4] trainable parameter fraction: {fraction:.4%}")
    check_trainable_fraction(fraction)

    return peft_model
