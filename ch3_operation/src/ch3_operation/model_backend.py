"""The inference seam: a `Backend` Protocol and the real llama.cpp implementation.

`Backend` is what lets `create_app` take a `FakeBackend` in tests without patching
`llama_cpp`, and it is the seam O1 swaps the fine-tuned GGUF through — same interface, a
different `ServeConfig.model_path`.

`llama_cpp` is imported inside `__init__`, not at module level, so this module — and every
test that only needs the Protocol — imports fine in the base env with no `ch3` extra.
"""

from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable

from ch3_operation.config import N_GPU_LAYERS, ServeConfig


@runtime_checkable
class Backend(Protocol):
    def generate(self, prompt: str, max_tokens: int) -> str: ...


class LlamaCppBackend:
    """CPU-only llama.cpp backend. `n_gpu_layers` is never a config key (CON-12)."""

    def __init__(self, config: ServeConfig) -> None:
        from llama_cpp import Llama  # lazy: base env has no llama_cpp

        self._config = config
        self._lock = threading.Lock()
        self._llm = Llama(
            model_path=str(config.resolved_model_path()),
            n_ctx=config.n_ctx,
            n_threads=config.n_threads,
            n_gpu_layers=N_GPU_LAYERS,
            verbose=False,
        )

    def generate(self, prompt: str, max_tokens: int) -> str:
        # llama-cpp-python is not thread-safe: a concurrent call without this lock crashes
        # the process (AC-5).
        with self._lock:
            result = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=self._config.temperature,
                echo=False,
            )
        return result["choices"][0]["text"]
