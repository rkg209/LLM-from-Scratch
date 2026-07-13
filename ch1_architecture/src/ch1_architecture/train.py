"""Chapter 1 training entrypoint.

SCAFFOLD (spec F2). This currently loads and validates the config, seeds the RNGs, and
reports what it would train — it does not train. The tokenizer is spec A1, the model is
A2, and the real training loop is A3, which replaces the body of `main()` below.

It exists now so that `/smoke ch1` and CI have something real to run from day one: a
config that does not load is a bug worth catching before there is a model to blame.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ch1_architecture.config import GPTConfig, load_gpt_config
from eval.config import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Chapter 1 GPT model.")
    parser.add_argument("--config", type=Path, required=True, help="path to a YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config: GPTConfig = load_gpt_config(args.config)
    seed_everything(config.seed)

    n_params_est = config.n_layers * 12 * config.d_model**2
    print(f"[ch1] config loaded: {args.config}")
    print(f"[ch1] run={config.run_name} device={config.device} seed={config.seed}")
    print(
        f"[ch1] model: d_model={config.d_model} n_heads={config.n_heads} "
        f"(head_dim={config.head_dim}) n_layers={config.n_layers} "
        f"vocab={config.vocab_size} seq_len={config.seq_len}"
    )
    print(f"[ch1] would train {config.max_steps} steps at lr={config.learning_rate}")
    print(f"[ch1] ~{n_params_est / 1e6:.1f}M parameters (rough estimate)")
    print("[ch1] SCAFFOLD: no training loop yet — that is spec A3. Config path verified.")


if __name__ == "__main__":
    main()
