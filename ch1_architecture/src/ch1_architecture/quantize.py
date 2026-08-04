"""Post-training absmax quantization (A5). `bitsandbytes` is banned (CON-5) — every
mode here is plain PyTorch: quantized weights are buffers, not parameters, because
nothing in this module is trained.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

MODES = ("fp32", "fp16", "int8", "int4")


class QuantizedLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, mode: str) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown quantization mode {mode!r}, expected one of {MODES}")
        self.mode = mode
        self.out_features, self.in_features = weight.shape
        bias_dtype = torch.float16 if mode == "fp16" else torch.float32
        self.register_buffer("bias", bias.clone().to(bias_dtype) if bias is not None else None)

        if mode == "fp16":
            self.register_buffer("weight", weight.half())
        elif mode == "int8":
            w_q, scale = _quantize_absmax(weight, qmax=127)
            self.register_buffer("weight", w_q.to(torch.int8))
            self.register_buffer("scale", scale)
        elif mode == "int4":
            w_q, scale = _quantize_absmax(weight, qmax=7)
            packed, shape = _pack_int4(w_q.to(torch.int8))
            self.register_buffer("weight", packed)
            self.register_buffer("scale", scale)
            self.register_buffer("unpacked_shape", torch.tensor(shape))
        else:
            raise ValueError(f"QuantizedLinear does not handle mode {mode!r} directly")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "fp16":
            return F.linear(x.half(), self.weight, self.bias).float()
        if self.mode == "int8":
            w = self.weight.float() * self.scale
            return F.linear(x, w, self.bias)
        if self.mode == "int4":
            shape = tuple(self.unpacked_shape.tolist())
            w_q = _unpack_int4(self.weight, shape)
            w = w_q.float() * self.scale
            return F.linear(x, w, self.bias)
        raise ValueError(f"unreachable mode {self.mode!r}")


def _quantize_absmax(weight: torch.Tensor, qmax: int) -> tuple[torch.Tensor, torch.Tensor]:
    scale = weight.abs().max() / qmax
    if scale == 0:
        scale = torch.tensor(1.0)
    w_q = (weight / scale).round().clamp(-qmax - 1, qmax)
    return w_q, scale


def _pack_int4(w_q: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pack two 4-bit values (range [-8, 7], stored as a 4-bit two's-complement nibble)
    per `torch.uint8`. The original shape is returned so unpack can restore it.
    """
    shape = tuple(w_q.shape)
    flat = w_q.flatten().to(torch.int64)
    if flat.numel() % 2 != 0:
        flat = torch.cat([flat, torch.zeros(1, dtype=flat.dtype)])

    nibbles = (flat & 0xF).to(torch.uint8)
    lo, hi = nibbles[0::2], nibbles[1::2]
    packed = (hi << 4) | lo
    return packed, shape


def _unpack_int4(packed: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    nibbles = torch.stack([lo, hi], dim=1).flatten().to(torch.int64)
    # Nibbles are a 4-bit two's-complement encoding: values >= 8 are negative.
    signed = torch.where(nibbles >= 8, nibbles - 16, nibbles)
    n = 1
    for dim in shape:
        n *= dim
    return signed[:n].reshape(shape)


def quantize_model(model: nn.Module, mode: str) -> nn.Module:
    """Return a deep copy of `model` with every `nn.Linear` replaced by a
    `QuantizedLinear` in `mode`. `mode="fp32"` is a no-op — the benchmark needs a
    baseline row, and a trivial branch here is cleaner than a special case at the
    call site. The tied output head is not an `nn.Linear` and is left untouched in
    every mode, so it stays fp32 regardless of the quantization level.
    """
    if mode not in MODES:
        raise ValueError(f"unknown quantization mode {mode!r}, expected one of {MODES}")

    model = copy.deepcopy(model)
    if mode == "fp32":
        return model

    for _name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                quantized = QuantizedLinear(child.weight.data, child.bias, mode)
                setattr(module, child_name, quantized)
    return model
