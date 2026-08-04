"""KV-cache for autoregressive decoding (A4).

An optimization, not a behavior change: it must alter no output, only how fast the
output arrives. `tests/test_kv_cache.py`'s logit-identity test against the uncached
path is the gate — nothing downstream may report a speedup until that test passes.
"""

from __future__ import annotations

import torch


class KVCache:
    def __init__(self, n_layers: int, max_seq_len: int, n_heads: int, head_dim: int) -> None:
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim

        self._k: list[torch.Tensor | None] = [None] * n_layers
        self._v: list[torch.Tensor | None] = [None] * n_layers
        self.current_len = 0

    def _ensure_allocated(self, layer_idx: int, k: torch.Tensor) -> None:
        # Buffers are [B, H, max_seq_len, Dh], allocated lazily on the first update()
        # so __init__ never needs a batch size — planning/03 gives it none.
        if self._k[layer_idx] is None:
            B = k.size(0)
            shape = (B, self.n_heads, self.max_seq_len, self.head_dim)
            self._k[layer_idx] = torch.zeros(shape, dtype=k.dtype, device=k.device)
            self._v[layer_idx] = torch.zeros(shape, dtype=k.dtype, device=k.device)

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # k, v: [B, H, T_new, Dh] -> returns the full history [B, H, S, Dh] as a view.
        self._ensure_allocated(layer_idx, k)
        new_len = self.current_len + k.size(2)
        if new_len > self.max_seq_len:
            raise IndexError(
                f"KVCache overflow: writing to {new_len} exceeds max_seq_len={self.max_seq_len}"
            )

        k_buf, v_buf = self._k[layer_idx], self._v[layer_idx]
        k_buf[:, :, self.current_len : new_len, :] = k
        v_buf[:, :, self.current_len : new_len, :] = v

        return k_buf[:, :, :new_len, :], v_buf[:, :, :new_len, :]

    def advance(self, n_tokens: int) -> None:
        # Called once per forward pass (after every layer has written), not once per
        # layer — advancing per-layer would make layer 5 read a window layer 0 never
        # wrote, since current_len gates how much of the buffer `update` exposes.
        self.current_len += n_tokens

    def reset(self) -> None:
        self.current_len = 0

    @property
    def is_allocated(self) -> bool:
        return self._k[0] is not None
