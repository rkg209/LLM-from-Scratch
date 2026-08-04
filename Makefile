.PHONY: setup check smoke smoke-ch1 smoke-ch2 smoke-ch3

# Installs everything the smoke path needs — dev tools plus all three chapters' heavy
# extras (torch, transformers/peft/trl, llama-cpp-python). None of it needs a GPU: the
# smoke configs are CPU-only by construction. If you only care about one chapter, sync
# just that extra instead, e.g. `uv sync --extra dev --extra ch1`.
setup:
	uv sync --extra dev --extra ch1 --extra ch2 --extra ch3

check:
	uv run ruff check .
	uv run black --check .
	uv run pytest -q

# Each target is the exact command documented in REPRODUCING.md and CLAUDE.md — no
# extra flags invented here. A missing extra fails loudly with an ImportError rather
# than being caught and hidden; that is the intended behavior, not a bug.
smoke-ch1:
	uv run python -m ch1_architecture.train --config ch1_architecture/configs/smoke.yaml

smoke-ch2:
	uv run python -m ch2_adaptation.finetune --config ch2_adaptation/configs/smoke.yaml

smoke-ch3:
	uv run python -m ch3_operation.serve --config ch3_operation/configs/smoke.yaml --check

smoke: smoke-ch1 smoke-ch2 smoke-ch3
